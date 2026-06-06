"""Project encapsulation + per-run isolation bookkeeping (report.md P2/P3).

Owns ALL of ``WORK_ROOT``: the project registry (a *directory convention*, no DB),
per-run dirs, manifests, and the single ``project_id → project.yaml`` resolver. No
filesystem logic leaks into route handlers. Every destructive/copy op is guarded to
paths strictly under ``work_root()`` (same defence-in-depth as ``delete_checkpoint``).

The layout (one directory per project, example-structured so it runs unedited):

    WORK_ROOT/projects/<slug>-<id8>/
        project.yaml        # ws_root: .   outdir: scratch
        manifest.json       # UI/backend metadata only — core never reads it
        spice/  xschem/     # copied netlists/schematics (a "new" project gets empty dirs)
        scratch/            # ephemeral one-off sims (out of the source tree)
        runs/<run_name>/    # per-run isolation (checkpoints/, run.log, events.ndjson, …)
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ui.backend.app_config import (
    REPO_ROOT,
    default_yaml_path,
    projects_root,
    runs_root,
    work_root,
)

SCHEMA_VERSION = 1


# ---------- ids / guards ----------

def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "project").lower()).strip("-")
    return s[:40] or "project"


def new_project_id(name: str) -> str:
    return f"{_slugify(name)}-{uuid.uuid4().hex[:8]}"


def _assert_under_work_root(p: Path) -> Path:
    rp = p.resolve()
    wr = work_root().resolve()
    if rp != wr and wr not in rp.parents:
        raise ValueError(f"refusing to touch a path outside WORK_ROOT: {rp}")
    return rp


def project_dir(project_id: str) -> Path:
    if not project_id or "/" in project_id or "\\" in project_id or ".." in project_id:
        raise ValueError(f"invalid project id: {project_id!r}")
    return projects_root() / project_id


def manifest_path(project_id: str) -> Path:
    return project_dir(project_id) / "manifest.json"


def project_yaml(project_id: str) -> Path:
    return project_dir(project_id) / "project.yaml"


def project_exists(project_id: str) -> bool:
    try:
        return project_yaml(project_id).exists()
    except ValueError:
        return False


# ---------- manifest ----------

def read_manifest(project_id: str) -> dict[str, Any]:
    p = manifest_path(project_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def write_manifest(project_id: str, data: dict[str, Any]) -> None:
    _assert_under_work_root(manifest_path(project_id))
    manifest_path(project_id).write_text(json.dumps(data, indent=2))


def touch_manifest(project_id: str) -> None:
    man = read_manifest(project_id)
    man["updated"] = datetime.now().isoformat(timespec="seconds")
    write_manifest(project_id, man)


# ---------- THE single resolver ----------

def resolve_yaml(project_id: str | None, yaml_path: str | None) -> Path:
    """Resolve the project YAML: ``project_id`` → its ``project.yaml``; else an explicit
    ``yaml_path``; else the default example. This is the ONE place "current project"
    is resolved, so a registry can't resurrect the ``yaml_path=""`` regression."""
    if project_id:
        yp = project_yaml(project_id)
        if not yp.exists():
            raise FileNotFoundError(f"project '{project_id}' not found")
        return yp
    if yaml_path:
        return Path(yaml_path)
    return default_yaml_path()


# ---------- runs ----------

def runs_dir(project_id: str | None) -> Path:
    base = (project_dir(project_id) / "runs") if project_id else runs_root()
    base.mkdir(parents=True, exist_ok=True)
    return base


def run_dir(project_id: str | None, run_name: str) -> Path:
    if "/" in run_name or ".." in run_name:
        raise ValueError(f"invalid run name: {run_name!r}")
    d = runs_dir(project_id) / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def reconcile_stale_runs() -> int:
    """On startup, flip any ``run.json`` left ``status: running`` (a crashed/killed
    backend) to ``error`` so the run list is honest (report.md §6). Returns the count."""
    fixed = 0
    for base in (projects_root(), runs_root()):
        for rj in base.rglob("run.json"):
            try:
                d = json.loads(rj.read_text())
            except Exception:
                continue
            if d.get("status") == "running":
                d["status"] = "error"
                d["ended"] = d.get("ended") or datetime.now().isoformat(timespec="seconds")
                try:
                    rj.write_text(json.dumps(d, indent=2))
                    fixed += 1
                except OSError:
                    pass
    return fixed


def list_runs(project_id: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rd in sorted(runs_dir(project_id).glob("*"), reverse=True):
        rj = rd / "run.json"
        if rj.exists():
            try:
                out.append(json.loads(rj.read_text()))
            except Exception:
                pass
    return out


# ---------- registry ----------

def list_projects() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pd in sorted(projects_root().glob("*")):
        if not (pd / "project.yaml").exists():
            continue
        man = read_manifest(pd.name)
        runs = list_runs(pd.name)
        best: float | None = None
        for r in runs:
            bs = r.get("best_score")
            if isinstance(bs, (int, float)) and (best is None or bs > best):
                best = bs
        out.append({
            "id": pd.name,
            "name": man.get("name", pd.name),
            "updated": man.get("updated") or man.get("created"),
            "run_count": len(runs),
            "best_score": best,
            "source": (man.get("source") or {}).get("kind", "unknown"),
        })
    # Newest-updated first.
    out.sort(key=lambda p: p.get("updated") or "", reverse=True)
    return out


# ---------- YAML rewrite (ws_root: . / outdir: scratch) ----------

def _rewrite_project_yaml(yaml_text: str, *, ws_root: str = ".", outdir: str = "scratch") -> str:
    """Bake the encapsulation contract into a project's YAML: ``ws_root`` becomes the
    project dir itself (portable — copy/move it and it still resolves) and ``outdir``
    moves ephemeral sims out of the source tree. NEVER touched for in-repo examples."""
    data = yaml.safe_load(yaml_text)
    if isinstance(data, dict) and isinstance(data.get("project"), dict):
        data["project"]["ws_root"] = ws_root
        data["project"]["outdir"] = outdir
    return yaml.safe_dump(data, sort_keys=False)


# ---------- examples (load demo as project) ----------

def list_examples() -> list[dict[str, Any]]:
    """Discover in-repo example projects (a ``project_setup.yaml`` under examples/)."""
    examples_root = REPO_ROOT / "examples"
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for yp in sorted(examples_root.rglob("project_setup.yaml")):
        # key = path relative to examples/, slug-ified and stable.
        rel = yp.relative_to(examples_root)
        key = "/".join(rel.parts)
        if key in seen:
            continue
        seen.add(key)
        # A human label from the OTA/<topology>/<pdk> portion.
        label = " · ".join(rel.parts[:-2]) if len(rel.parts) >= 2 else rel.parts[0]
        out.append({"key": key, "name": label, "yaml_path": str(yp)})
    return out


def _example_yaml_path(example_key: str) -> Path:
    examples_root = REPO_ROOT / "examples"
    yp = (examples_root / example_key).resolve()
    # Defence-in-depth: stay inside examples/ and require the canonical filename.
    if examples_root.resolve() not in yp.parents or yp.name != "project_setup.yaml":
        raise ValueError(f"invalid example key: {example_key!r}")
    if not yp.exists():
        raise FileNotFoundError(f"example not found: {example_key}")
    return yp


def _ws_root_dir_of(yaml_path: Path) -> Path:
    """The directory an example's ``ws_root`` resolves to (the subtree to copy)."""
    data = yaml.safe_load(yaml_path.read_text())
    wsr = ((data or {}).get("project") or {}).get("ws_root")
    wsr_s = "" if wsr is None else str(wsr).strip()
    if not wsr_s:
        return yaml_path.parent
    p = Path(wsr_s).expanduser()
    return p if p.is_absolute() else (yaml_path.parent / p).resolve()


def _empty_dirs(pid: str) -> None:
    for sub in ("spice", "xschem", "scratch", "runs"):
        (project_dir(pid) / sub).mkdir(parents=True, exist_ok=True)


def create_project(name: str, yaml_content: str | None = None) -> str:
    """Create a NEW project = an example-structured directory under ``WORK_ROOT/projects``.

    If ``yaml_content`` is given (the wizard's generated YAML) it is written (with
    ``ws_root: .`` / ``outdir: scratch`` baked in); otherwise the default example's YAML
    seeds it. Empty ``spice/``/``xschem/``/``scratch/``/``runs/`` dirs make the layout
    match the examples so the user can drop netlists in and run.
    """
    pid = new_project_id(name)
    pd = project_dir(pid)
    _assert_under_work_root(pd)
    pd.mkdir(parents=True, exist_ok=True)
    src_text = yaml_content if yaml_content else default_yaml_path().read_text()
    project_yaml(pid).write_text(_rewrite_project_yaml(src_text))
    _empty_dirs(pid)
    now = datetime.now().isoformat(timespec="seconds")
    write_manifest(pid, {
        "id": pid, "slug": pid.rsplit("-", 1)[0], "name": name,
        "created": now, "updated": now,
        "source": {"kind": "new"}, "schema_version": SCHEMA_VERSION,
    })
    return pid


def copy_example(example_key: str, name: str | None = None) -> str:
    """Load a demo AS a project: copy its ``ws_root`` subtree into a fresh project dir
    and rewrite ``ws_root: .`` / ``outdir: scratch``. The in-repo example is untouched."""
    yp = _example_yaml_path(example_key)
    ws_root_dir = _ws_root_dir_of(yp)
    proj_name = name or f"{ws_root_dir.parent.name}-{ws_root_dir.name}"
    pid = new_project_id(proj_name)
    pd = project_dir(pid)
    _assert_under_work_root(pd)
    # Copy the entire ws_root subtree (netlists + schematics) into the project dir.
    shutil.copytree(ws_root_dir, pd, dirs_exist_ok=True)
    # The canonical project.yaml lives at the project root with ws_root: . — netlist
    # paths (relative to ws_root) keep resolving since the subtree was copied verbatim.
    project_yaml(pid).write_text(_rewrite_project_yaml(yp.read_text()))
    (pd / "scratch").mkdir(parents=True, exist_ok=True)
    (pd / "runs").mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    write_manifest(pid, {
        "id": pid, "slug": pid.rsplit("-", 1)[0], "name": proj_name,
        "created": now, "updated": now,
        "source": {"kind": "example", "ref": example_key}, "schema_version": SCHEMA_VERSION,
    })
    return pid
