"""Netlist inspection routes used by the Setup wizard."""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, HTTPException

from ui.backend.services.netlist_parser import parse_params

router = APIRouter()


@router.post("/netlist/parse")
async def parse_netlist(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode netlist: {e}")
    params = parse_params(text)
    return {"ok": True, "filename": file.filename, "params": params}
