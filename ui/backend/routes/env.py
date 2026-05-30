"""Environment route — reports simulator + PDK availability for graceful degradation.

The Studio UI calls ``GET /api/env`` on load to decide whether live optimization is
possible. On a machine without the IHP ``ihp-sg13g2`` PDK (e.g. this personal Mac),
``live_runs_enabled`` is False and the frontend steers the user to Replay.
"""
from __future__ import annotations

from fastapi import APIRouter

from ui.backend.services.env_probe import probe_env

router = APIRouter()


@router.get("/env")
def get_env():
    """Cheap, no-simulation probe of ngspice + the IHP PDK.

    Returns ``{ngspice_path, ngspice_ok, pdk_root, pdk_ok, pdk_detail, tech,
    live_runs_enabled}``.
    """
    return probe_env()
