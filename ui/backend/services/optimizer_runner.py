"""Background optimization runner with SSE event streaming."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spicexplorer.core.domains import Project_Setup

logger = logging.getLogger(__name__)


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


@dataclass
class RunState:
    run_id: str
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    budget: int
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    is_replay: bool = False
    checkpoint_id: str | None = None
    algorithm: str | None = None
    seed: int | None = None
    done: bool = False


_runs: dict[str, RunState] = {}


# ---------- live optimizer ----------

def _build_spicelib_wrappers(project: Project_Setup):
    from pathlib import Path
    from spicexplorer.spice_engine import NGSpice_Wrapper, Sim_Execution_Type

    output_folder = Path(project.ws_root) / Path(project.outdir)
    sim_execution_t = Sim_Execution_Type.RUN_AND_WAIT
    path_to_simulator = Path(project.simulator)

    wrappers = {}
    for tb in project.testbenches:
        if not tb.enable:
            continue
        wrappers[tb.name] = NGSpice_Wrapper(
            testbench_name=tb.name,
            netlist_filename=Path(project.ws_root) / Path(tb.netlist),
            output_folder=output_folder,
            sim_execution_t=sim_execution_t,
            path_to_simulator=path_to_simulator,
        )
    return wrappers


def _make_streaming_optimizer(project: Project_Setup, state: RunState):
    """Create a Nevergrad optimizer subclass that emits SSE events per step."""
    from spicexplorer.optimization.stochastic.nevergrad import Nevergrad_Spice_Single_Objective

    class _StreamingOpt(Nevergrad_Spice_Single_Objective):
        def optimization_step(self):
            if state.stop_event.is_set():
                raise KeyboardInterrupt("stopped by user")
            result = super().optimization_step()
            params, score, metadata = result
            # optimization_step() returns (params, score, fit_summary); the third
            # element IS the fit_summary dict, keyed by spec name
            # ({spec: {"curr_val", "score"}}) — not a wrapper with a "fit_summary"
            # key. Reading metadata.get("fit_summary") always returned {}, so the
            # right-rail spec status never populated on live runs. (Replay reads
            # the same shape correctly via checkpoint_reader; sanity.py also
            # iterates metadata directly.)
            fit = metadata if isinstance(metadata, dict) else {}
            event = {
                "iter": len(self.optimization_log),
                "score": _safe_float(score),
                "best_score": _safe_float(
                    self.optimization_log[self.global_best_index].point.score
                    if len(self.optimization_log) > 0 else score
                ),
                "metrics": {
                    k: _safe_float(v.get("curr_val"))
                    for k, v in fit.items()
                    if isinstance(v, dict)
                },
                "best_params": {k: _safe_float(v) for k, v in params.items()},
            }
            asyncio.run_coroutine_threadsafe(state.queue.put(event), state.loop)
            return result

    spicelib_wrappers = _build_spicelib_wrappers(project)
    return _StreamingOpt(setup_obj=project, spicelib_wrappers=spicelib_wrappers)


def _apply_overrides(
    project: Project_Setup,
    *,
    run_id: str,
    budget: int | None,
    algorithm: str | None,
    seed: int | None,
) -> None:
    """Apply ephemeral run-config overrides to the in-memory project.

    These mutate the loaded Project_Setup only — the YAML on disk is untouched —
    so the Run popover's algorithm/budget/seed actually take effect on a live run
    (previously they were silently ignored; the optimizer always used the YAML's
    optimizer_config).
    """
    cfg = project.optimizer_config
    if budget and budget > 0 and budget != cfg.budget:
        logger.info("[run %s] override budget %s -> %s", run_id[:8], cfg.budget, budget)
        cfg.budget = budget
    if algorithm and algorithm != cfg.name:
        logger.info("[run %s] override algorithm %s -> %s", run_id[:8], cfg.name, algorithm)
        cfg.name = algorithm
    if seed is not None and seed != cfg.random_seed:
        logger.info("[run %s] override seed %s -> %s", run_id[:8], cfg.random_seed, seed)
        cfg.random_seed = seed


def _run_live(state: RunState, project_path: str) -> None:
    logger.info("[run %s] starting live run — project: %s", state.run_id[:8], project_path)
    try:
        logger.info("[run %s] loading project YAML", state.run_id[:8])
        project = Project_Setup.from_yaml(project_path)
        _apply_overrides(
            project,
            run_id=state.run_id,
            budget=state.budget,
            algorithm=state.algorithm,
            seed=state.seed,
        )
        logger.info("[run %s] building optimizer", state.run_id[:8])
        opt = _make_streaming_optimizer(project, state)
        logger.info("[run %s] parameterizing", state.run_id[:8])
        opt.parameterize()
        logger.info("[run %s] starting optimize() — budget %d", state.run_id[:8], state.budget)
        opt.optimize()
        logger.info("[run %s] optimize() finished", state.run_id[:8])
    except KeyboardInterrupt:
        logger.info("[run %s] stopped by user", state.run_id[:8])
    except Exception as e:
        logger.error("[run %s] optimizer error: %s\n%s", state.run_id[:8], e, traceback.format_exc())
        asyncio.run_coroutine_threadsafe(
            state.queue.put({"error": str(e)}), state.loop
        )
    finally:
        state.done = True
        asyncio.run_coroutine_threadsafe(state.queue.put(None), state.loop)


# ---------- replay mode ----------

async def _run_replay(state: RunState, checkpoint_path: Path) -> None:
    """Drip-feed CSV/JSON trace rows as SSE events at ~50ms per event."""
    from ui.backend.services.checkpoint_reader import read_checkpoint

    data = read_checkpoint(checkpoint_path)
    scores = data["scores"]
    best_scores = data["best_scores"]
    per_metric = data["per_metric"]
    params = data["params"]
    metric_names = list(per_metric.keys())
    param_names = list(params.keys())

    for i, (s, bs) in enumerate(zip(scores, best_scores)):
        if state.stop_event.is_set():
            break
        metrics = {m: per_metric[m][i] for m in metric_names if i < len(per_metric[m])}
        best_p = {p: params[p][i] for p in param_names if i < len(params[p])}
        event = {
            "iter": i,
            "score": s,
            "best_score": bs,
            "metrics": metrics,
            "best_params": best_p,
        }
        await state.queue.put(event)
        await asyncio.sleep(0.05)

    state.done = True
    await state.queue.put(None)


# ---------- public API ----------

def start_run(
    *,
    project_path: str | None = None,
    replay: bool = False,
    checkpoint_id: str | None = None,
    checkpoint_path: Path | None = None,
    budget: int = 200,
    algorithm: str | None = None,
    seed: int | None = None,
    loop: asyncio.AbstractEventLoop,
) -> str:
    run_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    state = RunState(
        run_id=run_id,
        queue=queue,
        loop=loop,
        budget=budget,
        is_replay=replay,
        checkpoint_id=checkpoint_id,
        algorithm=algorithm,
        seed=seed,
    )
    _runs[run_id] = state

    if replay and checkpoint_path:
        asyncio.run_coroutine_threadsafe(_run_replay(state, checkpoint_path), loop)
    elif project_path:
        t = threading.Thread(target=_run_live, args=(state, project_path), daemon=True)
        state.thread = t
        t.start()

    return run_id


def stop_run(run_id: str) -> None:
    state = _runs.get(run_id)
    if state:
        state.stop_event.set()


def get_run(run_id: str) -> RunState | None:
    return _runs.get(run_id)
