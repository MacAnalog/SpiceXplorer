"""Manual single-simulation route — evaluate ONE chosen design point on demand.

Unlike /sanity-check (which sims a *random* in-bounds point via the optimizer's
`ask()`), this evaluates a *caller-supplied* param vector through the exact same
primitive a real trial uses — `Spice_Constraint_Satisfaction.evaluate(params,
append_to_log=False)` — so the returned score + per-spec `fit_summary` are directly
comparable to checkpoint rows and to Score Shaping.

Two input modes:
  • Mode B (manual): `params` — an engineering-real vector ({"x_dut_W_1": 72e-6, ...}).
    A partial dict is valid; unset params keep their netlist `.param` defaults.
  • Mode A (checkpoint): `checkpoint_id` (+ optional `point` index) — replays a stored
    point's vector. `point` omitted → the best iteration (argmax score). Re-simulating
    the best point should reproduce its stored metrics, doubling as a validation tool.

The active PVT corner (Phase 1) is applied automatically when the optimizer is
constructed; `active_corner` optionally overrides it ephemerally (never rewrites YAML).
"""
from __future__ import annotations

import asyncio
import math
from time import perf_counter
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ui.backend.services.num import safe_float as _safe_float

router = APIRouter()


class SimulateOnceRequest(BaseModel):
    yaml_path: str
    # Mode B — explicit param vector (partial dict allowed). Values may be plain
    # numbers OR engineering strings ("250u", "0.18u"); both are parsed server-side
    # via the project DSL's parse_value, so the UI can pass raw user input.
    params: dict[str, str | float] | None = None
    # Mode A — resolve the vector from a stored checkpoint instead.
    checkpoint_id: str | None = None
    point: int | None = None  # index into the checkpoint; None → best (argmax score)
    # Optional ephemeral PVT corner override (must match a corner in the project).
    active_corner: str | None = None


class SpecMetric(BaseModel):
    curr_val: float | None = None
    score: float | None = None


class SimulateOnceResponse(BaseModel):
    ok: bool
    score: float | None = None
    metrics: dict[str, SpecMetric] = {}
    params_used: dict[str, float] = {}
    active_corner: str | None = None
    # Non-fatal advisories (out-of-range value, unknown param, unknown corner, …).
    warnings: list[str] = []
    log_files: dict[str, str] = {}
    log_tails: dict[str, str] = {}
    error: str | None = None
    elapsed_ms: float | None = None
    elapsed_ms_load: float | None = None
    pdk_ok: bool | None = None
    pdk_detail: str | None = None


def _resolve_params_from_checkpoint(checkpoint_id: str, point: int | None, warnings: list[str]) -> dict[str, float]:
    """Pull a single point's engineering-real param vector out of a stored checkpoint."""
    from ui.backend.routes.checkpoint import _resolve_checkpoint_path
    from ui.backend.services.checkpoint_reader import read_checkpoint

    path = _resolve_checkpoint_path(checkpoint_id)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_id}' not found")
    data = read_checkpoint(path)
    params: dict[str, list] = data.get("params", {})
    scores: list = data.get("scores", [])
    if not params:
        raise ValueError(f"Checkpoint '{checkpoint_id}' has no param vectors to evaluate")

    n = max((len(v) for v in params.values()), default=0)
    if point is None:
        # Best point = argmax of finite scores (checkpoints don't persist a best index).
        idx = 0
        best = -math.inf
        for i, s in enumerate(scores):
            sv = _safe_float(s)
            if sv is not None and sv > best:
                best, idx = sv, i
    else:
        idx = point
        if idx < 0 or idx >= n:
            warnings.append(f"point {point} out of range [0,{n - 1}]; clamped.")
            idx = max(0, min(idx, n - 1))

    out: dict[str, float] = {}
    for name, series in params.items():
        if idx < len(series):
            fv = _safe_float(series[idx])
            if fv is not None:
                out[name] = fv
    return out


def _validate_params(project, params: dict[str, float], warnings: list[str]) -> None:
    """Non-fatal range/identity checks — `evaluate` trusts real values, so flag (not
    reject) an unknown param or a value outside its [min_val, max_val] hint."""
    known = {p.name: p for p in project.dut_params}
    for name, val in params.items():
        p = known.get(name)
        if p is None:
            warnings.append(f"'{name}' is not a declared dut_param; it will be injected verbatim.")
            continue
        lo = float(p.min_val) if p.min_val is not None else None
        hi = float(p.max_val) if p.max_val is not None else None
        if lo is not None and val < lo:
            warnings.append(f"'{name}'={val:g} is below min_val {lo:g}.")
        if hi is not None and val > hi:
            warnings.append(f"'{name}'={val:g} is above max_val {hi:g}.")


def _run_single_sim(req: SimulateOnceRequest) -> dict[str, Any]:
    from spicexplorer.core.domains import Project_Setup, parse_value
    from spicexplorer.optimization.stochastic.nevergrad import Nevergrad_Spice_Single_Objective
    from ui.backend.routes.sanity import _tail_log
    from ui.backend.services.env_probe import probe_pdk
    from ui.backend.services.optimizer_runner import _build_spicelib_wrappers

    t_start = perf_counter()
    pdk = probe_pdk()
    warnings: list[str] = []

    def _fail(error: str, **extra) -> dict[str, Any]:
        return {
            "ok": False, "error": error, "warnings": warnings,
            "elapsed_ms": (perf_counter() - t_start) * 1000,
            "pdk_ok": pdk["pdk_ok"], "pdk_detail": pdk["pdk_detail"], **extra,
        }

    # 1 — load project
    try:
        project = Project_Setup.from_yaml(req.yaml_path)
    except Exception as e:
        return _fail(f"Failed to load project: {e}")
    elapsed_load = (perf_counter() - t_start) * 1000

    # 2 — resolve the param vector (Mode A vs Mode B)
    try:
        if req.params is not None:
            params = {}
            for k, v in req.params.items():
                try:
                    params[k] = float(parse_value(v))
                except (ValueError, TypeError):
                    return _fail(f"Could not parse value for '{k}': {v!r}")
        elif req.checkpoint_id:
            params = _resolve_params_from_checkpoint(req.checkpoint_id, req.point, warnings)
        else:
            return _fail("Provide `params` (manual) or `checkpoint_id` (from a checkpoint).")
    except Exception as e:
        return _fail(str(e))
    if not params:
        return _fail("Resolved an empty param vector — nothing to simulate.")

    # 3 — ephemeral PVT corner override (applied at optimizer construction)
    active_corner: str | None = None
    if project.pvt is not None:
        if req.active_corner and project.pvt.get(req.active_corner) is not None:
            project.pvt.active_corner = req.active_corner
        elif req.active_corner:
            warnings.append(
                f"Requested corner '{req.active_corner}' not defined; using "
                f"'{project.pvt.active_corner}'."
            )
        active_corner = project.pvt.active_corner

    # 4 — soft validation (non-fatal)
    _validate_params(project, params, warnings)

    # 5 — build wrappers (isolated subfolder) + evaluate ONCE. Use a UNIQUE subfolder per call so
    # two overlapping manual sims (or a manual sim + the Explorer "Re-simulate", same route) don't
    # share `manual_sim/` and race the per-wrapper rmtree in NGSpice_Wrapper._validate (BUG-B50).
    import uuid
    try:
        wrappers = _build_spicelib_wrappers(
            project, output_subdir=f"manual_sim/{uuid.uuid4().hex[:8]}")
        opt = Nevergrad_Spice_Single_Objective(setup_obj=project, spicelib_wrappers=wrappers)
        score, fit_summary = opt.evaluate(params, append_to_log=False)
    except Exception as e:
        return _fail(f"Simulation failed: {e}", params_used=params, active_corner=active_corner)

    metrics: dict[str, dict] = {}
    for spec_name, entry in (fit_summary or {}).items():
        if isinstance(entry, dict):
            metrics[spec_name] = {
                "curr_val": _safe_float(entry.get("curr_val")),
                "score": _safe_float(entry.get("score")),
            }

    log_files = {n: str(w.curr_log) for n, w in wrappers.items() if w.curr_log is not None}
    log_tails = {n: (_tail_log(p)[0] or "") for n, p in log_files.items()}

    return {
        "ok": True,
        "score": _safe_float(score),
        "metrics": metrics,
        "params_used": {k: float(v) for k, v in params.items()},
        "active_corner": active_corner,
        "warnings": warnings,
        "log_files": log_files,
        "log_tails": log_tails,
        "error": None,
        "elapsed_ms": (perf_counter() - t_start) * 1000,
        "elapsed_ms_load": elapsed_load,
        "pdk_ok": pdk["pdk_ok"],
        "pdk_detail": pdk["pdk_detail"],
    }


@router.post("/simulate/once", response_model=SimulateOnceResponse)
async def simulate_once(body: SimulateOnceRequest):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_single_sim, body)
    return SimulateOnceResponse(**result)
