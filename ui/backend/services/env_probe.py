"""Environment probe — detects the SPICE simulator and the IHP PDK.

This is a *cheap, no-simulation* check used by:
  - ``GET /api/env`` to drive the Studio status bar / Run popover degradation, and
  - ``POST /api/sanity-check`` to annotate a real sanity run with the PDK verdict.

The headline use case: this repo was moved to a personal Mac where ``ngspice`` is
installed but the IHP ``ihp-sg13g2`` PDK is **not**. We want the UI to treat that as a
first-class supported mode (live sims disabled, replay enabled) rather than an error,
so the probe must reliably distinguish "simulator missing" from "PDK models missing".
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

# The model library the cascode/5t/folded testbenches pull in via ``.lib cornerMOSlv.lib``.
# Its presence on disk is our proxy for "the IHP sg13g2 device models are installed".
_PDK_MODEL_LIB = "cornerMOSlv.lib"
_PDK_TECH = "ihp-sg13g2"

# Env vars that, by convention, point at a PDK install root.
_PDK_ENV_VARS = ("PDK_ROOT", "PDK", "IHP_PDK_ROOT")

# Sub-paths under a candidate root where IHP ships its ngspice model libs.
_PDK_LIB_SUBPATHS = (
    _PDK_MODEL_LIB,
    f"{_PDK_TECH}/libs.tech/ngspice/{_PDK_MODEL_LIB}",
    f"libs.tech/ngspice/{_PDK_MODEL_LIB}",
    f"libs.tech/ngspice/models/{_PDK_MODEL_LIB}",
)


def probe_ngspice() -> dict[str, Any]:
    """Locate the ngspice binary. Cheap: just a PATH lookup."""
    path = shutil.which("ngspice")
    return {"ngspice_path": path, "ngspice_ok": path is not None}


def _candidate_pdk_roots() -> list[tuple[str, Path]]:
    """Ordered (source-label, path) pairs to search for the PDK model libs."""
    candidates: list[tuple[str, Path]] = []
    for var in _PDK_ENV_VARS:
        val = os.environ.get(var)
        if val:
            candidates.append((var, Path(val).expanduser()))
    # Optional explicit override from app_config.json (kept import-light / best-effort).
    try:
        from ui.backend.app_config import get_app_config

        cfg_root = get_app_config().get("pdk_root")
        if cfg_root:
            candidates.append(("app_config.pdk_root", Path(str(cfg_root)).expanduser()))
    except Exception:  # noqa: BLE001 — config is optional; never let it break the probe
        pass
    return candidates


def _find_model_lib(root: Path) -> Path | None:
    """Return the cornerMOSlv.lib under ``root`` if resolvable, else None.

    Checks a handful of known sub-paths first (fast), then falls back to a single
    bounded ``rglob`` so an unusual layout still resolves without scanning forever.
    """
    if not root.exists():
        return None
    for sub in _PDK_LIB_SUBPATHS:
        hit = root / sub
        if hit.exists():
            return hit
    # Bounded fallback: first match only.
    for hit in root.rglob(_PDK_MODEL_LIB):
        return hit
    return None


def probe_pdk() -> dict[str, Any]:
    """Detect the IHP sg13g2 PDK without running a simulation.

    Returns ``pdk_root`` (resolved install dir or None), ``pdk_ok``, and a human
    ``pdk_detail`` string suitable for the Settings/Diagnostics verdict line.
    """
    candidates = _candidate_pdk_roots()

    for source, root in candidates:
        lib = _find_model_lib(root)
        if lib is not None:
            return {
                "pdk_root": str(root),
                "pdk_ok": True,
                "pdk_detail": (
                    f"IHP {_PDK_TECH} models found via {source} "
                    f"({lib})."
                ),
            }

    if not candidates:
        detail = (
            f"IHP {_PDK_TECH} models not found "
            f"(PDK_ROOT/PDK unset; .lib {_PDK_MODEL_LIB} unresolved). "
            "Live simulation unavailable — replay enabled."
        )
    else:
        searched = ", ".join(f"{s}={p}" for s, p in candidates)
        detail = (
            f"IHP {_PDK_TECH} PDK root set but {_PDK_MODEL_LIB} not found under "
            f"[{searched}]. Live simulation unavailable — replay enabled."
        )
    return {"pdk_root": None, "pdk_ok": False, "pdk_detail": detail}


def probe_env() -> dict[str, Any]:
    """Full environment verdict: simulator + PDK + whether live runs are possible."""
    ng = probe_ngspice()
    pdk = probe_pdk()
    return {
        **ng,
        **pdk,
        "tech": _PDK_TECH,
        # Live SPICE optimization needs BOTH the binary and the device models.
        "live_runs_enabled": bool(ng["ngspice_ok"] and pdk["pdk_ok"]),
    }
