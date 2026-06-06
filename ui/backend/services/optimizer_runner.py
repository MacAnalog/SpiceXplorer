"""Background optimization runner with SSE event streaming."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from spicexplorer.core.domains import Project_Setup
from ui.backend.services import project_service
from ui.backend.services.num import safe_float as _safe_float

logger = logging.getLogger(__name__)


class _QueueLogHandler(logging.Handler):
    """Stream the SpiceXplorer library's own log records to a run's SSE queue.

    Attached to the ``spicexplorer`` logger for the lifetime of ONE run (added at
    run start, removed in `finally`), so the raw, human-readable library log is
    delivered per-run as ``{"log", "level"}`` SSE events alongside the structured
    per-trial events. The frontend renders it in a dedicated bottom-panel tab.
    """

    def __init__(self, state: "RunState"):
        super().__init__(level=logging.INFO)
        self._state = state
        # Mirror the library's console/file formatter so the streamed lines look
        # identical to the log file the user already likes.
        self.setFormatter(logging.Formatter(
            fmt="%(asctime)s - %(name)s: [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            event = {"log": line, "level": record.levelname}
            asyncio.run_coroutine_threadsafe(self._state.queue.put(event), self._state.loop)
        except Exception:  # never let logging break the run
            pass


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
    # Per-run isolation (report.md P2): the owning project (None → unscoped runs/)
    # and the run's self-contained dir (checkpoints/ run.log events.ndjson sim/ …),
    # created once the project + effective algorithm are known.
    project_id: str | None = None
    label: str | None = None
    run_dir: Path | None = None
    done: bool = False


_runs: dict[str, RunState] = {}


def _prune_finished_runs() -> None:
    """Evict completed runs so the registry doesn't grow unbounded in a long-lived
    backend. A run is safe to drop once its stream has ended (state.done is set
    after the None sentinel is queued and the SSE generator has broken its loop)."""
    for rid in [rid for rid, st in _runs.items() if st.done]:
        _runs.pop(rid, None)


# ---------- live optimizer ----------

def _build_spicelib_wrappers(
    project: Project_Setup,
    output_subdir: str | None = None,
    output_folder: Path | None = None,
):
    from pathlib import Path
    from spicexplorer.spice_engine import NGSpice_Wrapper, Sim_Execution_Type

    # An explicit `output_folder` (a live run's per-run `runs/<id>/sim`) takes
    # precedence so the run dir is fully self-contained and the wrapper's `rmtree`
    # can never touch another run. Otherwise default to ws_root/outdir[/subdir] — the
    # optional subfolder isolates outputs (e.g. a concurrent manual sim) from a live run.
    if output_folder is None:
        output_folder = Path(project.ws_root) / Path(project.outdir)
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


def _config_snapshot_yaml(project_path: str, state: "RunState") -> str:
    """The exact config a run used: the original YAML with the ephemeral overrides
    (algorithm/budget/seed/active_corner) BAKED IN — so the run dir reproduces itself
    (report.md §6). Best-effort: on any parse error, persist the original text."""
    try:
        text = Path(project_path).read_text()
        data = yaml.safe_load(text) or {}
    except Exception:
        return ""
    # The DSL nests everything under a top-level `project:` key (optimizer_config,
    # pvt, …); fall back to the document root for robustness.
    roots = [r for r in (data.get("project") if isinstance(data, dict) else None, data)
             if isinstance(r, dict)]
    for root in roots:
        for key in ("optimizer_config", "optimizer"):
            opt = root.get(key)
            if isinstance(opt, dict):
                if state.algorithm:
                    opt["name"] = state.algorithm
                if state.budget:
                    opt["budget"] = state.budget
                if state.seed is not None:
                    opt["random_seed"] = state.seed
        if state.active_corner and isinstance(root.get("pvt"), dict):
            root["pvt"]["active_corner"] = state.active_corner
    try:
        return yaml.safe_dump(data, sort_keys=False)
    except Exception:
        return text


def _tee_event(state: "RunState", event: dict) -> None:
    """Append one SSE event to the run's replayable ``events.ndjson`` (report.md §6)."""
    rd = state.run_dir
    if rd is None:
        return
    try:
        with (rd / "events.ndjson").open("a") as f:
            f.write(json.dumps(event) + "\n")
    except (OSError, TypeError):
        pass


def _write_run_json(state: "RunState", *, status: str, started: str,
                    best_score: float | None = None, ended: str | None = None) -> None:
    """Commit the per-run ``run.json`` (label/algo/seed/budget/corner/status/timing/best)."""
    if state.run_dir is None:
        return
    try:
        (state.run_dir / "run.json").write_text(json.dumps({
            "run_id": state.run_id,
            "project_id": state.project_id,
            "label": state.label,
            "kind": "resume" if state.resume_path else "live",
            "algorithm": state.algorithm,
            "seed": state.seed,
            "budget": state.budget,
            "active_corner": state.active_corner,
            "status": status,
            "best_score": best_score,
            "started": started,
            "ended": ended,
            "resume_from": state.resume_path,
        }, indent=2))
    except OSError:
        pass


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
        _tee_event(state, event)  # persist to events.ndjson for offline replay
        asyncio.run_coroutine_threadsafe(state.queue.put(event), state.loop)

    class _StreamingOpt(Nevergrad_Spice_Single_Objective):
        _abs_iter = 0
        _best_score: float | None = None
        _best_params: dict = {}
        _best_metrics: dict = {}
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
                # Snapshot the BEST trial's metrics so the right-rail / Pipeline pass-fail
                # reflect the design whose params/score they display — not a later, lower-
                # scoring trial's metrics (BUG-A13 / OPT-1 / RAIL-2).
                self._best_metrics = {
                    k: _safe_float(v.get("curr_val"))
                    for k, v in fit.items()
                    if isinstance(v, dict)
                }
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
                "best_metrics": self._best_metrics,
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
    # Stream the SpiceXplorer library's own log to this run's SSE queue (per-run:
    # attached now, detached in `finally`). Ensure the library logger has at least
    # one handler/level so records are produced even if setup_loggers never ran.
    lib_logger = logging.getLogger("spicexplorer")
    if lib_logger.level == logging.NOTSET or lib_logger.level > logging.INFO:
        lib_logger.setLevel(logging.INFO)
    log_handler = _QueueLogHandler(state)
    lib_logger.addHandler(log_handler)
    started = datetime.now().isoformat(timespec="seconds")
    final_status = "done"
    best_for_json: float | None = None
    file_log_handler: logging.Handler | None = None
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
        # Per-run isolation (report.md P2): a self-contained dir under the owning
        # project's runs/ (or work_root/runs/ when unscoped). Carries the config
        # snapshot (overrides baked in), a DEBUG run.log, the replayable
        # events.ndjson, checkpoints/, and sim/ — a folder you can zip and hand over.
        _algo = project.optimizer_config.name
        # Record the EFFECTIVE algorithm (post-override) in run.json so the run list
        # shows what actually ran, not just an absent override.
        if not state.algorithm:
            state.algorithm = _algo
        _ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        rdir = project_service.run_dir(state.project_id, f"{_ts}_{_algo}_{state.run_id[:8]}")
        state.run_dir = rdir
        _snap = _config_snapshot_yaml(project_path, state)
        if _snap:
            (rdir / "config_snapshot.yaml").write_text(_snap)
        _write_run_json(state, status="running", started=started)
        file_log_handler = logging.FileHandler(rdir / "run.log")
        file_log_handler.setLevel(logging.DEBUG)
        file_log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        lib_logger.addHandler(file_log_handler)
        logger.info("[run %s] isolated run dir: %s", state.run_id[:8], rdir)

        # sim outputs live INSIDE the run dir so the wrapper's rmtree can never touch
        # another run (report.md §6), and checkpoints autosave to runs/<id>/checkpoints.
        wrappers = _build_spicelib_wrappers(project, output_folder=rdir / "sim")
        stream_cls = _streaming_optimizer_class(state)
        save_root = rdir / "checkpoints"
        if state.resume_path:
            logger.info("[run %s] resuming from checkpoint %s", state.run_id[:8], state.resume_path)
            opt = stream_cls(setup_obj=project, spicelib_wrappers=wrappers, output_root=save_root)
            opt.optimization_log = _load_checkpoint_log(state.resume_path)
            logger.info("[run %s] restored %d prior trials", state.run_id[:8], len(opt.optimization_log))
            keep_history = True
        else:
            logger.info("[run %s] building optimizer", state.run_id[:8])
            opt = stream_cls(setup_obj=project, spicelib_wrappers=wrappers, output_root=save_root)
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
        best_metrics: dict = {}
        for entry in opt.optimization_log:
            s = _safe_float(entry.point.score)
            if s is not None and (best_score is None or s > best_score):
                best_score = s
                best_params = {k: _safe_float(v) for k, v in entry.point.params.items()}
                # Seed best_metrics from the restored best so the rail/Pipeline show the
                # resumed design's metrics immediately — not {} until the first new best
                # (which on a near-optimal resume may never come). (BUG-A13 resume gap)
                fit = entry.fit_summary or {}
                best_metrics = {
                    k: _safe_float(v.get("curr_val"))
                    for k, v in fit.items()
                    if isinstance(v, dict)
                }
        opt._best_score = best_score
        opt._best_params = best_params
        opt._best_metrics = best_metrics
        logger.info("[run %s] parameterizing", state.run_id[:8])
        opt.parameterize()
        logger.info("[run %s] starting optimize() — budget %d%s", state.run_id[:8], state.budget,
                    " (resume)" if keep_history else "")
        opt.optimize(keep_history=keep_history)
        best_for_json = _safe_float(getattr(opt, "_best_score", None))
        logger.info("[run %s] optimize() finished", state.run_id[:8])
    except KeyboardInterrupt:
        final_status = "stopped"
        best_for_json = _safe_float(getattr(locals().get("opt"), "_best_score", None))
        logger.info("[run %s] stopped by user", state.run_id[:8])
    except Exception as e:
        final_status = "error"
        logger.error("[run %s] optimizer error: %s\n%s", state.run_id[:8], e, traceback.format_exc())
        asyncio.run_coroutine_threadsafe(
            state.queue.put({"error": str(e)}), state.loop
        )
    finally:
        if file_log_handler is not None:
            lib_logger.removeHandler(file_log_handler)
            file_log_handler.close()
        lib_logger.removeHandler(log_handler)
        # Commit the final run.json (status/best/ended) so the run list is honest even
        # if the SSE client disconnected; touch the owning project's manifest.
        _write_run_json(state, status=final_status, started=started, best_score=best_for_json,
                        ended=datetime.now().isoformat(timespec="seconds"))
        if state.project_id:
            try:
                project_service.touch_manifest(state.project_id)
            except Exception:
                pass
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
    project_id: str | None = None,
    label: str | None = None,
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
        project_id=project_id,
        label=label,
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


def stop_runs_for(*, project_id: str | None = None, run_id: str | None = None,
                  timeout: float = 10.0) -> tuple[int, list[str]]:
    """Stop + join any IN-FLIGHT live runs matching a project or a specific run.

    Returns ``(attempted, still_alive)`` — the number of runs signalled to stop, and the ids
    of any whose worker thread was STILL ALIVE after the join. A lifecycle op (delete
    project/run) MUST call this before moving the run dir to trash AND must refuse to proceed
    while ``still_alive`` is non-empty (BUG-B4): the worker holds a cached absolute ``run_dir``,
    so its next ``save_checkpoint`` re-``mkdir``s the moved-away tree at the old path —
    escaping the trash (incomplete soft-delete) and resurrecting a ``projects/P`` dir that then
    blocks restore.

    ``stop_event`` is only checked at a trial boundary, so a single in-flight ngspice trial
    longer than ``timeout`` can outlast the join; in that case the run id is reported in
    ``still_alive`` and the caller should 409 ("retry") rather than move the dir out from under
    a live writer. Once a thread is no longer alive, ``optimize()`` has returned and its
    ``finally`` (the last checkpoint/run.json write) has run, so the dir is genuinely quiescent.
    """
    targets = [
        st for st in _runs.values()
        if not st.done
        and (run_id is None or st.run_id == run_id)
        and (project_id is None or st.project_id == project_id)
    ]
    for st in targets:
        st.stop_event.set()
    for st in targets:
        if st.thread is not None and st.thread.is_alive():
            st.thread.join(timeout=timeout)
    still_alive = [
        st.run_id for st in targets
        if st.thread is not None and st.thread.is_alive()
    ]
    return len(targets), still_alive


def get_run(run_id: str) -> RunState | None:
    return _runs.get(run_id)
