"""Sanity check route — verifies SPICE simulator and runs one trial evaluation."""
from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class SanityRequest(BaseModel):
    yaml_path: str


class TestbenchResult(BaseModel):
    name: str
    ok: bool
    error: str | None = None


class TrialResult(BaseModel):
    ok: bool
    score: float | None = None
    metrics: dict[str, float | None] = {}
    error: str | None = None


class SanityResponse(BaseModel):
    ok: bool
    testbenches: list[TestbenchResult]
    trial: TrialResult | None = None
    error: str | None = None


def _run_sanity(yaml_path: str) -> dict[str, Any]:
    from spicexplorer.core.domains import Project_Setup
    from spicexplorer.spice_engine import NGSpice_Wrapper, Sim_Execution_Type
    from spicexplorer.optimization.stochastic.nevergrad import Nevergrad_Spice_Single_Objective

    try:
        project = Project_Setup.from_yaml(yaml_path)
    except Exception as e:
        return {"ok": False, "testbenches": [], "trial": None, "error": f"Failed to load project: {e}"}

    output_folder = Path(project.ws_root) / Path(project.outdir)
    path_to_simulator = Path(project.simulator)
    wrappers: dict[str, NGSpice_Wrapper] = {}
    tb_results: list[dict] = []
    all_ok = True

    for tb in project.testbenches:
        if not tb.enable:
            continue
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
            tb_results.append({"name": tb.name, "ok": sim_ok, "error": None})
            if not sim_ok:
                all_ok = False
        except Exception as e:
            tb_results.append({"name": tb.name, "ok": False, "error": str(e)})
            all_ok = False

    if not all_ok:
        return {"ok": False, "testbenches": tb_results, "trial": None, "error": None}

    # One trial optimization step to validate the full pipeline
    try:
        opt = Nevergrad_Spice_Single_Objective(setup_obj=project, spicelib_wrappers=wrappers)
        opt.parameterize()
        params, score, metadata = opt.optimization_step()
        fit = metadata.get("fit_summary", {}) if metadata else {}
        metrics = {
            k: (float(v["curr_val"]) if isinstance(v, dict) and v.get("curr_val") is not None else None)
            for k, v in fit.items()
        }
        trial = {"ok": True, "score": float(score) if score is not None else None, "metrics": metrics, "error": None}
    except Exception as e:
        trial = {"ok": False, "score": None, "metrics": {}, "error": str(e)}
        return {"ok": False, "testbenches": tb_results, "trial": trial, "error": None}

    return {"ok": True, "testbenches": tb_results, "trial": trial, "error": None}


@router.post("/sanity-check", response_model=SanityResponse)
async def sanity_check(body: SanityRequest):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_sanity, body.yaml_path)
    return SanityResponse(**result)
