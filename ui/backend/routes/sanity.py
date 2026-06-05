"""Sanity check route — verifies SPICE simulator and runs one trial evaluation."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Bound the per-file log payload so a runaway log doesn't bloat the response.
_LOG_TAIL_LINES = 200
_LOG_TAIL_BYTES = 16 * 1024  # 16 KB cap regardless of line count


class SanityRequest(BaseModel):
    yaml_path: str
    # Optional PVT corner the trial step runs against (must match a corner in the
    # project's `pvt:` block). None keeps the YAML's active_corner.
    active_corner: str | None = None


class TestbenchResult(BaseModel):
    name: str
    ok: bool
    error: str | None = None
    elapsed_ms: float | None = None
    log_path: str | None = None
    log_tail: str | None = None
    log_size_bytes: int | None = None


class TrialResult(BaseModel):
    ok: bool
    score: float | None = None
    metrics: dict[str, float | None] = {}
    error: str | None = None
    elapsed_ms: float | None = None
    log_files: dict[str, str] = {}
    log_tails: dict[str, str] = {}


class SanityResponse(BaseModel):
    ok: bool
    testbenches: list[TestbenchResult]
    trial: TrialResult | None = None
    error: str | None = None
    elapsed_ms_total: float | None = None
    elapsed_ms_load: float | None = None
    elapsed_ms_optimizer_init: float | None = None
    ngspice_path: str | None = None
    # PDK verdict (cheap probe folded in so a sanity run confirms the static /api/env check).
    pdk_ok: bool | None = None
    pdk_detail: str | None = None
    # PVT corner the trial step ran at (None when the project has no `pvt:` block).
    active_corner: str | None = None


def _tail_log(path_str: str | Path | None) -> tuple[str | None, int | None]:
    """Return (tail_text, full_size_bytes) for the given log file, or (None, None)."""
    if not path_str:
        return None, None
    p = Path(path_str)
    if not p.exists() or not p.is_file():
        return None, None
    try:
        size = p.stat().st_size
        # Seek-based tail to avoid loading huge files into memory
        with p.open("rb") as f:
            if size > _LOG_TAIL_BYTES:
                f.seek(size - _LOG_TAIL_BYTES)
            chunk = f.read()
        text = chunk.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) > _LOG_TAIL_LINES:
            lines = lines[-_LOG_TAIL_LINES:]
        return "\n".join(lines), size
    except OSError:
        return None, None


def _run_sanity(yaml_path: str, active_corner: str | None = None) -> dict[str, Any]:
    from spicexplorer.core.domains import Project_Setup
    from spicexplorer.spice_engine import NGSpice_Wrapper, Sim_Execution_Type
    from spicexplorer.optimization.stochastic.nevergrad import Nevergrad_Spice_Single_Objective
    from ui.backend.services.env_probe import probe_pdk

    t_start = time.perf_counter()
    t_load_start = t_start
    # Cheap PDK verdict folded into every return path so the UI can explain a sim
    # failure as "PDK missing" rather than a generic error (this Mac has ngspice but
    # no IHP PDK; on the server both are present).
    pdk = probe_pdk()
    try:
        project = Project_Setup.from_yaml(yaml_path)
    except Exception as e:
        return {
            "ok": False, "testbenches": [], "trial": None,
            "error": f"Failed to load project: {e}",
            "elapsed_ms_total": (time.perf_counter() - t_start) * 1000,
            "pdk_ok": pdk["pdk_ok"],
            "pdk_detail": pdk["pdk_detail"],
        }
    elapsed_ms_load = (time.perf_counter() - t_load_start) * 1000

    # Ephemeral PVT corner override for the trial step (same in-memory, never-rewrite
    # pattern as the live run). The optimizer applies project.pvt.active_corner when
    # constructed below, so setting it here is all that's needed.
    active_corner_used: str | None = None
    if project.pvt is not None:
        if active_corner and project.pvt.get(active_corner) is not None:
            project.pvt.active_corner = active_corner
        active_corner_used = project.pvt.active_corner

    output_folder = Path(project.ws_root) / Path(project.outdir)
    path_to_simulator = Path(project.simulator)
    wrappers: dict[str, NGSpice_Wrapper] = {}
    tb_results: list[dict] = []
    all_ok = True

    for tb in project.testbenches:
        if not tb.enable:
            continue
        tb_t0 = time.perf_counter()
        try:
            wrapper = NGSpice_Wrapper(
                testbench_name=tb.name,
                netlist_filename=Path(project.ws_root) / Path(tb.netlist),
                output_folder=output_folder,
                sim_execution_t=Sim_Execution_Type.RUN_AND_WAIT,
                path_to_simulator=path_to_simulator,
            )
            sim_ok = wrapper.run_sanity_check(
                use_editor=False, sim_execution_t=Sim_Execution_Type.RUN_NOW
            )
            wrappers[tb.name] = wrapper
            tail, size = _tail_log(wrapper.curr_log)
            tb_results.append({
                "name": tb.name,
                "ok": sim_ok,
                "error": None,
                "elapsed_ms": (time.perf_counter() - tb_t0) * 1000,
                "log_path": str(wrapper.curr_log) if wrapper.curr_log else None,
                "log_tail": tail,
                "log_size_bytes": size,
            })
            if not sim_ok:
                all_ok = False
        except Exception as e:
            tb_results.append({
                "name": tb.name,
                "ok": False,
                "error": str(e),
                "elapsed_ms": (time.perf_counter() - tb_t0) * 1000,
                "log_path": None,
                "log_tail": None,
                "log_size_bytes": None,
            })
            all_ok = False

    if not all_ok:
        return {
            "ok": False,
            "testbenches": tb_results,
            "trial": None,
            "error": None,
            "elapsed_ms_total": (time.perf_counter() - t_start) * 1000,
            "elapsed_ms_load": elapsed_ms_load,
            "ngspice_path": str(path_to_simulator),
            "pdk_ok": pdk["pdk_ok"],
            "pdk_detail": pdk["pdk_detail"],
            "active_corner": active_corner_used,
        }

    # One trial optimization step to validate the full pipeline
    trial: dict[str, Any] = {"ok": False, "score": None, "metrics": {}, "error": None}
    elapsed_ms_optimizer_init: float | None = None
    try:
        t_init0 = time.perf_counter()
        opt = Nevergrad_Spice_Single_Objective(setup_obj=project, spicelib_wrappers=wrappers)
        opt.parameterize()
        # _create_optimizer_obj() builds self.optimizer; optimization_step() calls .ask() on it.
        # Skipping this step crashes with "'NoneType' object has no attribute 'ask'".
        if not opt._create_optimizer_obj() or opt.optimizer is None:
            raise RuntimeError(
                f"Failed to instantiate optimizer '{project.optimizer_config.name}' — "
                "check that the algorithm name is valid and optimizer_kwargs are accepted."
            )
        elapsed_ms_optimizer_init = (time.perf_counter() - t_init0) * 1000

        trial_t0 = time.perf_counter()
        _params, score, metadata = opt.optimization_step()
        trial_elapsed = (time.perf_counter() - trial_t0) * 1000

        # NevergradMixin.optimization_step() returns (params, score, fit_summary).
        # fit_summary is a dict keyed by spec name: {spec: {curr_val, score, ...}}.
        metrics: dict[str, float | None] = {}
        if isinstance(metadata, dict):
            for spec_name, entry in metadata.items():
                if isinstance(entry, dict) and entry.get("curr_val") is not None:
                    try:
                        metrics[spec_name] = float(entry["curr_val"])
                    except (TypeError, ValueError):
                        metrics[spec_name] = None
                else:
                    metrics[spec_name] = None

        log_files = {n: str(w.curr_log) for n, w in wrappers.items() if w.curr_log is not None}
        log_tails = {n: (_tail_log(path)[0] or "") for n, path in log_files.items()}

        trial = {
            "ok": True,
            "score": float(score) if score is not None else None,
            "metrics": metrics,
            "error": None,
            "elapsed_ms": trial_elapsed,
            "log_files": log_files,
            "log_tails": log_tails,
        }
    except Exception as e:
        trial = {"ok": False, "score": None, "metrics": {}, "error": str(e),
                 "elapsed_ms": None, "log_files": {}, "log_tails": {}}
        return {
            "ok": False,
            "testbenches": tb_results,
            "trial": trial,
            "error": None,
            "elapsed_ms_total": (time.perf_counter() - t_start) * 1000,
            "elapsed_ms_load": elapsed_ms_load,
            "elapsed_ms_optimizer_init": elapsed_ms_optimizer_init,
            "ngspice_path": str(path_to_simulator),
            "pdk_ok": pdk["pdk_ok"],
            "pdk_detail": pdk["pdk_detail"],
            "active_corner": active_corner_used,
        }

    return {
        "ok": True,
        "testbenches": tb_results,
        "trial": trial,
        "error": None,
        "elapsed_ms_total": (time.perf_counter() - t_start) * 1000,
        "elapsed_ms_load": elapsed_ms_load,
        "elapsed_ms_optimizer_init": elapsed_ms_optimizer_init,
        "ngspice_path": str(path_to_simulator),
        "pdk_ok": pdk["pdk_ok"],
        "pdk_detail": pdk["pdk_detail"],
        "active_corner": active_corner_used,
    }


@router.post("/sanity-check", response_model=SanityResponse)
async def sanity_check(body: SanityRequest):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_sanity, body.yaml_path, body.active_corner)
    return SanityResponse(**result)
