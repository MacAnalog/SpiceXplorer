"""Serve xschem .sch/.sym files for the in-browser viewer."""
from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ui.backend.app_config import REPO_ROOT

router = APIRouter()


@lru_cache(maxsize=1)
def _xschem_share_dirs() -> list[Path]:
    """Locate xschem's bundled symbol libraries.

    xschem's default XSCHEM_LIBRARY_PATH contains both `xschem_library/` and
    `xschem_library/devices/` — symbol refs may be written either as
    `devices/foo.sym` (resolved from xschem_library/) or as bare `foo.sym`
    (resolved from xschem_library/devices/). Add both.
    """
    candidates: list[Path] = []
    binary = shutil.which("xschem")
    if binary:
        prefix = Path(binary).resolve().parent.parent
        share = prefix / "share" / "xschem" / "xschem_library"
        candidates.append(share)
        candidates.append(share / "devices")
    # XSCHEM_LIBRARY_PATH from xschemrc; also honor explicit env override.
    env = os.environ.get("XSCHEM_LIBRARY_PATH", "")
    for part in env.split(":"):
        if part:
            candidates.append(Path(part))
    # Dedup while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        if p.exists() and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _discover_pdk_xschem_dir() -> Path | None:
    """Locate the PDK's xschem symbol library, using the SAME root discovery as the
    env probe so the symbol search path can never be a strict subset of the sim PDK
    path (the cause of "symbols render but sims work" — device symbols turning into
    red "?" placeholders while ``pdk_ok: true``).

    For every candidate PDK root (``PDK_ROOT`` / ``PDK`` / ``IHP_PDK_ROOT`` /
    ``app_config.pdk_root``) it matches both layouts:
      - ``<root>/libs.tech/xschem``         (root already points at the tech dir)
      - ``<root>/<tech>/libs.tech/xschem``  (root is the parent of named tech dirs)
    where ``tech`` comes from ``$PDK`` and the env-probe default tech name. The old
    implementation required BOTH ``PDK_ROOT`` and ``PDK`` and only tried the second
    layout, so a ``PDK_ROOT``-only (or ``IHP_PDK_ROOT`` / app-config) deployment lost
    the PDK from the resolver path while the probe still reported the PDK present.
    """
    roots: list[Path] = []
    techs: list[str] = []
    try:
        from ui.backend.services.env_probe import _PDK_TECH, _candidate_pdk_roots

        roots.extend(p for _, p in _candidate_pdk_roots())
        techs.append(_PDK_TECH)
    except Exception:  # noqa: BLE001 — env_probe is optional; fall back to raw env below.
        pass
    pdk_root = os.environ.get("PDK_ROOT")
    if pdk_root:
        roots.append(Path(pdk_root).expanduser())
    pdk = os.environ.get("PDK")  # conventional tech name, e.g. "ihp-sg13g2"
    if pdk:
        techs.insert(0, pdk)

    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        direct = root / "libs.tech" / "xschem"
        if direct.is_dir():
            return direct
        for tech in techs:
            cand = root / tech / "libs.tech" / "xschem"
            if cand.is_dir():
                return cand
    return None


_pdk_xschem_dir_cached: Path | None = None


def _pdk_xschem_dir() -> Path | None:
    """Locate the PDK xschem dir, memoizing only a SUCCESSFUL result.

    Unlike a plain ``lru_cache``, a ``None`` (PDK not yet reachable at first call) is
    never frozen — so a late-exported ``PDK`` env var or a PDK mounted after startup is
    picked up on the next call without a process restart.
    """
    global _pdk_xschem_dir_cached
    if _pdk_xschem_dir_cached is not None and _pdk_xschem_dir_cached.is_dir():
        return _pdk_xschem_dir_cached
    found = _discover_pdk_xschem_dir()
    if found is not None:
        _pdk_xschem_dir_cached = found
    return found


def _search_roots(base_dir: Path | None) -> list[Path]:
    """Ordered list of dirs to resolve a relative symbol/schematic reference against."""
    roots: list[Path] = []
    if base_dir is not None:
        roots.append(base_dir)
    pdk = _pdk_xschem_dir()
    if pdk is not None:
        roots.append(pdk)
    roots.extend(_xschem_share_dirs())
    return roots


def _allowed_roots() -> list[Path]:
    """Whitelist for absolute-path access. Anything outside these is rejected."""
    roots: list[Path] = [REPO_ROOT]
    pdk = _pdk_xschem_dir()
    if pdk is not None:
        roots.append(pdk)
    roots.extend(_xschem_share_dirs())
    return [p.resolve() for p in roots]


def _validate_under_allowed(path: Path) -> Path:
    resolved = path.resolve()
    for root in _allowed_roots():
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise HTTPException(403, f"Path outside allowed roots: {path}")


@router.get("/xschem/file")
def get_file(path: str = Query(..., description="Absolute path to a .sch or .sym file")):
    """Return the raw text of an xschem file, gated by the allowed-roots whitelist."""
    candidate = Path(path)
    if not candidate.is_absolute():
        raise HTTPException(400, "Path must be absolute; use /xschem/resolve for relative refs")
    resolved = _validate_under_allowed(candidate)
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(404, f"Not found: {resolved}")
    if resolved.suffix not in (".sch", ".sym"):
        raise HTTPException(400, "Only .sch and .sym files are served")
    return {
        "path": str(resolved),
        "content": resolved.read_text(),
    }


@router.get("/xschem/resolve")
def resolve_ref(
    ref: str = Query(..., description="Reference like 'devices/foo.sym' or 'sg13g2_pr/bar.sym'"),
    base: str | None = Query(None, description="Absolute path of the parent .sch/.sym (for sibling refs)"),
):
    """Resolve a symbol/schematic reference against the xschem search path.

    Order matches xschem's lookup: same-dir-as-base, then PDK lib, then xschem share libs.
    """
    base_dir = None
    if base:
        base_path = Path(base)
        if base_path.is_absolute():
            try:
                _validate_under_allowed(base_path)
                base_dir = base_path.parent
            except HTTPException:
                pass
    for root in _search_roots(base_dir):
        candidate = (root / ref).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            # `ref` tried to escape the search root (e.g. ../../etc/passwd).
            continue
        if candidate.exists() and candidate.is_file():
            return {
                "path": str(candidate),
                "content": candidate.read_text(),
                "resolved_from": str(root),
            }
    raise HTTPException(404, f"Could not resolve {ref!r}")


@router.get("/xschem/list")
def list_dir(dir: str = Query(..., description="Absolute path of a directory under the allowed roots")):
    """List .sch files in a directory."""
    d = Path(dir)
    if not d.is_absolute():
        raise HTTPException(400, "Path must be absolute")
    resolved = _validate_under_allowed(d)
    if not resolved.is_dir():
        raise HTTPException(404, f"Not a directory: {resolved}")
    files = sorted(p for p in resolved.glob("*.sch") if p.is_file())
    return {
        "dir": str(resolved),
        "files": [{"path": str(p), "name": p.name} for p in files],
    }


@router.get("/xschem/project")
def project_schematics(yaml_path: str = Query(..., description="Absolute path of the project YAML")):
    """Find the conventional xschem dir for a project YAML and list its .sch files.

    Looks for `{ws_root}/xschem/` first (sibling-of-sizing convention), then for any
    `xschem/` dir near the yaml.
    """
    yaml_p = Path(yaml_path)
    if not yaml_p.is_absolute():
        raise HTTPException(400, "yaml_path must be absolute")
    _validate_under_allowed(yaml_p)
    if not yaml_p.exists():
        raise HTTPException(404, f"YAML not found: {yaml_p}")

    candidates: list[Path] = []
    # Walk up from the yaml until we find a sibling `xschem/` dir.
    cur = yaml_p.parent
    for _ in range(6):
        sibling = cur / "xschem"
        if sibling.is_dir():
            candidates.append(sibling)
            break
        if cur.parent == cur:
            break
        cur = cur.parent

    if not candidates:
        return {"xschem_dir": None, "files": []}

    xschem_dir = candidates[0]
    try:
        _validate_under_allowed(xschem_dir)
    except HTTPException:
        return {"xschem_dir": None, "files": []}

    files = sorted(p for p in xschem_dir.glob("*.sch") if p.is_file())
    return {
        "xschem_dir": str(xschem_dir),
        "files": [{"path": str(p), "name": p.name} for p in files],
    }
