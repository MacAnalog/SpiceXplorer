"""Netlist inspection + spec-library routes used by the Setup wizard."""
from __future__ import annotations

import yaml
from fastapi import APIRouter, File, UploadFile, HTTPException

from ui.backend.app_config import REPO_ROOT
from ui.backend.services.netlist_parser import parse_params, parse_meas_candidates

router = APIRouter()

_SPEC_LIBRARY_PATH = REPO_ROOT / "examples" / "spec_library.yaml"


@router.post("/netlist/parse")
async def parse_netlist(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode netlist: {e}")
    params = parse_params(text)
    # Also surface `.meas` result names so the Target-Specs step can auto-discover
    # candidate specs from the same upload (the wizard picks which to enable).
    meas_candidates = parse_meas_candidates(text)
    return {
        "ok": True,
        "filename": file.filename,
        "params": params,
        "meas_candidates": meas_candidates,
    }


@router.get("/spec-library")
def spec_library():
    """Serve the shipped analog-spec templates (examples/spec_library.yaml) for the
    wizard's one-click "Spec library" adds. Returns `{specs: [...]}` (empty if the
    file is missing/unreadable, so the wizard degrades gracefully)."""
    try:
        if not _SPEC_LIBRARY_PATH.exists():
            return {"specs": []}
        data = yaml.safe_load(_SPEC_LIBRARY_PATH.read_text()) or {}
        specs = data.get("specs", []) if isinstance(data, dict) else []
        return {"specs": specs if isinstance(specs, list) else []}
    except yaml.YAMLError:
        return {"specs": []}
