"""Background optimization runner with SSE event streaming."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from spicexplorer.core.domains import Project_Setup
from ui.backend.services.num import safe_float as _safe_float

logger = logging.getLogger(__name__)


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
    # Ephemeral PVT corner override (applied in-memory to project.pvt.active_corner).
    active_corner: str | None = None
    # Checkpointing: autosave cadence (trials) and an optional checkpoint to
    # resume from (load_checkpoint + optimize(keep_history=True)).
    autosave_every: int | None = None
    resume_path: str | None = None
    done: bool = False


_runs: dict[str, RunState] = {}


def _prune_finished_runs() -> None:
    """Evict completed runs so the registry doesn't grow unbounded in a long-lived
    backend. A run is safe to drop once its stream has ended (state.done is set
    after the None sentinel is queued and the SSE generator has broken its loop)."""
    for rid in [rid for rid, st in _runs.items() if st.done]:
        _runs.pop(rid, None)


# ---------- live optimizer ----------

def _build_spicelib_wrappers(project: Project_Setup, output_subdir: str | None = None):
    from pathlib import Path
    from spicexplorer.spice_engine import NGSpice_Wrapper, Sim_Execution_Type

    output_folder = Path(project.ws_root) / Path(project.outdir)
    # An optional subfolder isolates these wrappers' outputs from a concurrent live
    # run — the wrapper constructor rmtree's its output_folder, so a manual sim
    # sharing ws_root/outdir would otherwise clobber a running optimization.
    if output_subdir:
        output_folder = output_folder / output_subdir
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


def _load_checkpoint_log(path: str):
    """Rebuild an OptimizationLog from a saved checkpoint JSON.

    The library's Base_Optimizer.load_checkpoint() can't round-trip a real run:
    save_checkpoint() stringifies each entry's ``log_file`` (a Dict[str, Path])
    before serializing, but load_checkpoint() feeds it back through dacite, which
    rejects the string for the Dict-typed field (WrongTypeError). The per-trial
    ngspice log paths aren't needed to resume — only params/score/fit_summary —
    so we drop ``log_file`` and rebuild the log here. (Library fix: serialize
    log_file as a dict, or make load_checkpoint tolerant.)
    """
    from dacite import Config, from_dict
    from spicexplorer.core.domains import OptimizationLog, OptimizationLogEntry

    with open(path) as f:
        data = json.load(f)
    entries = []
    for raw in data.get("optimization_log", []):
        raw = dict(raw)
        raw["log_file"] = None
        entries.append(from_dict(OptimizationLogEntry, raw, Config(strict=False)))
    return OptimizationLog(entries)


def _streaming_optimizer_class(state: RunState):
    """Build a Nevergrad single-objective optimizer that (a) streams an SSE event
    per step, (b) emits a ``checkpoint`` event whenever it autosaves, and (c)
    runs its own ``optimize()`` loop so periodic autosave is actually usable.

    Why override ``optimize()``: the base loop resets ``optimization_log`` after
    every autosave (a memory-bound chunking strategy) and then indexes the now
    empty log — harmless at the default frequency of 2500 (a UI run never reaches
    it) but an IndexError the moment a smaller cadence is configured. Keeping the
    full log in memory is trivial at UI budgets (≤5000) and makes every
    checkpoint a *cumulative* snapshot, so a resume restores the whole run rather
    than a single chunk.

    Streaming counters (absolute iteration, running best) live on the instance
    rather than being derived from the log, so they stay correct across a resume.
    """
    from spicexplorer.core.domains import OptimizationLog
    from spicexplorer.optimization.stochastic.nevergrad import Nevergrad_Spice_Single_Objective

    def _emit(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(state.queue.put(event), state.loop)

    class _StreamingOpt(Nevergrad_Spice_Single_Objective):
        _abs_iter = 0
        _best_score: float | None = None
        _best_params: dict = {}
        _ckpt_count = 0

        def optimization_step(self):
            if state.stop_event.is_set():
                raise KeyboardInterrupt("stopped by user")
            params, score, metadata = super().optimization_step()
            self._abs_iter += 1
            sval = _safe_float(score)
            # metadata IS the fit_summary dict, keyed by spec name
            # ({spec: {"curr_val", "score"}}). Track the running best on the
            # instance so iter/best survive autosave and resume.
            fit = metadata if isinstance(metadata, dict) else {}
            if sval is not None and (self._best_score is None or sval > self._best_score):
                self._best_score = sval
                # `params` is candidate.value in optimizer (normalized) space;
                # denormalize to physical values so the streamed best_params match
                # the checkpoint/resume/replay representation (which is physical).
                phys = self.denormalize_params(params)
                if getattr(self, "_frozen_params", None):
                    phys = {**phys, **self._frozen_params}
                self._best_params = {k: _safe_float(v) for k, v in phys.items()}
                self.global_best_index = len(self.optimization_log) - 1
            _emit({
                "iter": self._abs_iter,
                "score": sval,
                "best_score": self._best_score,
                "metrics": {
                    k: _safe_float(v.get("curr_val"))
                    for k, v in fit.items()
                    if isinstance(v, dict)
                },
                "best_params": self._best_params,
            })
            return params, score, metadata

        def save_checkpoint(self, name):
            super().save_checkpoint(name)
            # Surface the autosave so the UI shows checkpoints accumulating live.
            try:
                files = sorted(
                    self.autosave_checkpoint_dir.glob("*.json"),
                    key=lambda p: p.stat().st_mtime,
                )
                latest = files[-1].stem if files else None
            except OSError:
                latest = None
            self._ckpt_count += 1
            _emit({"checkpoint": {"id": latest, "index": self._ckpt_count, "iter": self._abs_iter}})

        def optimize(self, render_optimization_trace: bool = False, keep_history: bool = False):
            self._create_optimizer_obj()
            if self.optimizer is None:
                _emit({"error": "optimizer object was not created"})
                return self.optimization_log
            if not keep_history:
                self.optimization_log = OptimizationLog()
            trial = -1
            try:
                for trial in range(self.optimizer_config.budget):
                    self.optimization_step()
                    if (not self.disable_autosave) and self.autosave_checkpoint_freqeucny \
                            and ((trial + 1) % self.autosave_checkpoint_freqeucny == 0):
                        self.save_checkpoint(name=self.get_auto_save_name(append_txt=f"trial{self._abs_iter}"))
            except KeyboardInterrupt:
                logger.info("[run %s] interrupted at trial %d", state.run_id[:8], trial + 1)
            # Always leave a FINAL checkpoint (end-of-run or stop) to resume from.
            if (not self.disable_autosave) and (not self.optimization_log.is_empty()):
                self.save_checkpoint(name=self.get_auto_save_name(append_txt=f"trial{self._abs_iter}_FINAL"))
            return self.optimization_log

    return _StreamingOpt


def _apply_overrides(
    project: Project_Setup,
    *,
    run_id: str,
    budget: int | None,
    algorithm: str | None,
    seed: int | None,
    active_corner: str | None = None,
) -> None:
    """Apply ephemeral run-config overrides to the in-memory project.

    These mutate the loaded Project_Setup only — the YAML on disk is untouched —
    so the Run popover's algorithm/budget/seed (and the PVT corner) actually take
    effect on a live run (previously they were silently ignored; the optimizer
    always used the YAML's optimizer_config).
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
    # PVT corner switch (operating mode (ii)): same in-memory, never-rewrite-YAML
    # pattern. Only applies when the project actually defines a `pvt:` block and the
    # requested corner exists; an unknown corner is logged and ignored (the YAML's
    # active_corner stays in effect) rather than failing the run.
    if active_corner and project.pvt is not None and active_corner != project.pvt.active_corner:
        if project.pvt.get(active_corner) is not None:
            logger.info("[run %s] override active_corner %s -> %s",
                        run_id[:8], project.pvt.active_corner, active_corner)
            project.pvt.active_corner = active_corner
        else:
            logger.warning("[run %s] requested active_corner '%s' not defined; keeping '%s'",
                           run_id[:8], active_corner, project.pvt.active_corner)


def _run_live(state: RunState, project_path: str) -> None:
    kind = "resumed" if state.resume_path else "live"
    logger.info("[run %s] starting %s run — project: %s", state.run_id[:8], kind, project_path)
    try:
        logger.info("[run %s] loading project YAML", state.run_id[:8])
        project = Project_Setup.from_yaml(project_path)
        _apply_overrides(
            project,
            run_id=state.run_id,
            budget=state.budget,
            algorithm=state.algorithm,
            seed=state.seed,
            active_corner=state.active_corner,
        )
        wrappers = _build_spicelib_wrappers(project)
        stream_cls = _streaming_optimizer_class(state)
        if state.resume_path:
            logger.info("[run %s] resuming from checkpoint %s", state.run_id[:8], state.resume_path)
            opt = stream_cls(setup_obj=project, spicelib_wrappers=wrappers)
            opt.optimization_log = _load_checkpoint_log(state.resume_path)
            logger.info("[run %s] restored %d prior trials", state.run_id[:8], len(opt.optimization_log))
            keep_history = True
        else:
            logger.info("[run %s] building optimizer", state.run_id[:8])
            opt = stream_cls(setup_obj=project, spicelib_wrappers=wrappers)
            keep_history = False
        if state.autosave_every and state.autosave_every > 0:
            opt.autosave_checkpoint_freqeucny = state.autosave_every
            logger.info("[run %s] autosave every %d trials", state.run_id[:8], state.autosave_every)
        # Seed the streaming counters from any restored history so a resume
        # continues the iteration count and best-so-far rather than restarting.
        opt._abs_iter = len(opt.optimization_log)
        opt._ckpt_count = 0
        best_score: float | None = None
        best_params: dict = {}
        for entry in opt.optimization_log:
            s = _safe_float(entry.point.score)
            if s is not None and (best_score is None or s > best_score):
                best_score = s
                best_params = {k: _safe_float(v) for k, v in entry.point.params.items()}
        opt._best_score = best_score
        opt._best_params = best_params
        logger.info("[run %s] parameterizing", state.run_id[:8])
        opt.parameterize()
        logger.info("[run %s] starting optimize() — budget %d%s", state.run_id[:8], state.budget,
                    " (resume)" if keep_history else "")
        opt.optimize(keep_history=keep_history)
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

    try:
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
                "iter": i + 1,  # 1-based, consistent with the live run's iter counter
                "score": s,
                "best_score": bs,
                "metrics": metrics,
                "best_params": best_p,
            }
            await state.queue.put(event)
            await asyncio.sleep(0.05)
    except Exception as e:
        # Without this, a corrupt/unparseable checkpoint would unwind before the
        # done-sentinel, leaving the SSE stream heartbeating forever and the UI
        # stuck "running". Surface the error and always close the stream.
        logger.error("[run %s] replay error: %s\n%s", state.run_id[:8], e, traceback.format_exc())
        await state.queue.put({"error": str(e)})
    finally:
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
    active_corner: str | None = None,
    autosave_every: int | None = None,
    resume_path: str | None = None,
    loop: asyncio.AbstractEventLoop,
) -> str:
    _prune_finished_runs()  # keep the registry bounded across many runs
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
        active_corner=active_corner,
        autosave_every=autosave_every,
        resume_path=resume_path,
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
