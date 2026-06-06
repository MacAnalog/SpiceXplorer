"""Checkpoint listing and loading routes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ui.backend.app_config import preset_checkpoint_paths, REPO_ROOT, auto_save_root
from ui.backend.services.checkpoint_reader import read_checkpoint, compute_envelope, compute_scatter

router = APIRouter()


def _count_iters(path: Path) -> int | None:
    """Cheap trial count for the catalog listing.

    JSON: length of the top-level ``optimization_log`` array (avoids the
    eval()-based ``Optimization_Log_Visualizer.load_checkpoint``). CSV: data rows
    minus the header. Returns None on any error so the listing never 500s.
    """
    try:
        if path.suffix == ".json":
            import json
            with open(path) as f:
                data = json.load(f)
            return len(data.get("optimization_log", []))
        # CSV: count non-empty data lines after the header.
        with open(path) as f:
            rows = sum(1 for line in f if line.strip())
        return max(0, rows - 1)
    except Exception:
        return None


def _infer_score_fn(path: Path) -> str:
    name = path.name.lower()
    if "sigmoid" in name:
        return "relative-sigmoid"
    if "relabs" in name or "relativeabs" in name or "linear" in name:
        return "relative-absolute"
    return "unknown"


def _target_specs_from_yaml(yaml_path: str) -> list[dict[str, Any]] | None:
    """Load a project's target specs as plain dicts for envelope/scatter feasibility.

    Returns None when no path is given or the project fails to load.
    """
    if not yaml_path:
        return None
    from spicexplorer.core.domains import Project_Setup
    try:
        project = Project_Setup.from_yaml(Path(yaml_path))
    except Exception:
        return None
    return [
        {"name": s.name, "target": float(s.target), "goal": s.goal.value,
         "tolerance": float(s.tolerance) if s.tolerance else None}
        for s in project.optimizer_config.target_specs.targets
    ]


def _resolve_checkpoint_path(checkpoint_id: str) -> Path | None:
    """Resolve a checkpoint id to a file on disk.

    Tries the configured presets first, then any autosave under ``auto_save/``
    (a live run writes a FINAL .json there). Shared by load/envelope/scatter so
    the analysis views work on *live* results, not just the preset demo data.
    """
    path = preset_checkpoint_paths().get(checkpoint_id)
    if path is not None and path.exists():
        return path
    for autosave_root in _autosave_roots():
        candidates = sorted(autosave_root.rglob(f"{checkpoint_id}*.json"))
        if candidates:
            return candidates[0]
    return None


def _autosave_roots() -> list[Path]:
    """Dirs where optimizer autosaves can live. ``Base_Optimizer`` writes ``./auto_save``
    relative to the BACKEND's CWD, while the UI historically only searched
    ``REPO_ROOT/auto_save`` — they coincide only when CWD == REPO_ROOT. Search both
    (deduped) so live checkpoints + resume work regardless of where uvicorn was launched
    (BUG-A9 / OPT-3)."""
    roots: list[Path] = []
    seen: set[Path] = set()
    # WORK_ROOT/auto_save is where runs now persist (report.md P1); the legacy
    # REPO_ROOT/auto_save + cwd/auto_save are kept so older runs aren't orphaned.
    for r in (auto_save_root(), REPO_ROOT / "auto_save", Path.cwd() / "auto_save"):
        rr = r.resolve()
        if rr not in seen and rr.exists():
            seen.add(rr)
            roots.append(rr)
    return roots


def _list_autosave_checkpoints() -> list[dict[str, Any]]:
    results = []
    seen_ids: set[str] = set()
    for autosave_root in _autosave_roots():
        for p in sorted(autosave_root.rglob("*.json")):
            if p.stem in seen_ids:  # same checkpoint reachable via two roots
                continue
            seen_ids.add(p.stem)
            results.append({
                "id": p.stem,
                "label": p.stem,
                "path": str(p),
                "type": "json",
                "score_fn": _infer_score_fn(p),
                "n_iters": _count_iters(p),
                "source": "autosave",
            })
    return results


@router.get("/checkpoint")
def list_checkpoints():
    items = []
    for key, path in preset_checkpoint_paths().items():
        if path.exists():
            items.append({
                "id": key,
                "label": key.replace("_", " ").title(),
                "path": str(path),
                "type": "csv" if path.suffix == ".csv" else "json",
                "score_fn": _infer_score_fn(path),
                "n_iters": _count_iters(path),
                "source": "preset",
            })
    items += _list_autosave_checkpoints()
    return {"checkpoints": items}


@router.get("/checkpoint/{checkpoint_id}")
def load_checkpoint(checkpoint_id: str, limit: int = Query(default=0)):
    path = _resolve_checkpoint_path(checkpoint_id)
    if path is None:
        raise HTTPException(404, f"Checkpoint '{checkpoint_id}' not found")

    data = read_checkpoint(path, limit=limit if limit > 0 else None)
    data["id"] = checkpoint_id
    data["label"] = checkpoint_id.replace("_", " ").title()
    data["type"] = "csv" if path.suffix == ".csv" else "json"
    data["score_fn"] = _infer_score_fn(path)
    return data


@router.delete("/checkpoint/{checkpoint_id}")
def delete_checkpoint(checkpoint_id: str):
    """Delete an autosave checkpoint. Preset checkpoints are protected."""
    presets = preset_checkpoint_paths()
    if checkpoint_id in presets:
        raise HTTPException(
            403,
            f"Checkpoint '{checkpoint_id}' is a preset and cannot be deleted from the UI.",
        )

    roots = _autosave_roots()
    if not roots:
        raise HTTPException(404, "No autosave directory.")

    # Match any file whose stem == checkpoint_id under ANY autosave root — list/load use the
    # same multi-root search, so delete must too, else a CWD-rooted autosave is undeletable.
    candidates = [
        p
        for root in roots
        for p in root.rglob("*")
        if p.is_file() and p.stem == checkpoint_id
    ]
    if not candidates:
        raise HTTPException(404, f"Checkpoint '{checkpoint_id}' not found in auto_save.")

    # Defence-in-depth: confirm every candidate really lives under one of the autosave roots.
    resolved_roots = [r.resolve() for r in roots]
    for path in candidates:
        parents = path.resolve().parents
        if not any(root in parents for root in resolved_roots):
            raise HTTPException(403, f"Refusing to delete file outside auto_save: {path}")

    for path in candidates:
        path.unlink()

    return {"ok": True, "deleted": [str(p) for p in candidates]}


@router.get("/checkpoint/{checkpoint_id}/envelope")
def checkpoint_envelope(checkpoint_id: str, yaml_path: str = Query(default="")):
    path = _resolve_checkpoint_path(checkpoint_id)
    if path is None:
        raise HTTPException(404, f"Checkpoint '{checkpoint_id}' not found")

    data = read_checkpoint(path)
    target_specs = _target_specs_from_yaml(yaml_path)
    return {"envelope": compute_envelope(data, target_specs)}


@router.get("/checkpoint/{checkpoint_id}/scatter")
def checkpoint_scatter(
    checkpoint_id: str,
    metric_x: str = Query(...),
    metric_y: str = Query(...),
    yaml_path: str = Query(default=""),
):
    path = _resolve_checkpoint_path(checkpoint_id)
    if path is None:
        raise HTTPException(404, f"Checkpoint '{checkpoint_id}' not found")

    data = read_checkpoint(path)
    target_specs = _target_specs_from_yaml(yaml_path)
    points = compute_scatter(data, metric_x, metric_y, target_specs)
    return {"metric_x": metric_x, "metric_y": metric_y, "points": points}


@router.get("/checkpoint/{checkpoint_id}/report")
def checkpoint_report(checkpoint_id: str, yaml_path: str = Query(default="")):
    """Bundle a run's artifacts into one downloadable zip: the checkpoint file, the
    project YAML (when a path is given), and a generated `summary.md` (final best
    score + the performance envelope per spec)."""
    import io
    import zipfile
    from fastapi.responses import StreamingResponse

    path = _resolve_checkpoint_path(checkpoint_id)
    if path is None:
        raise HTTPException(404, f"Checkpoint '{checkpoint_id}' not found")

    data = read_checkpoint(path)
    target_specs = _target_specs_from_yaml(yaml_path)
    envelope = compute_envelope(data, target_specs) if target_specs else []

    best_scores = [s for s in data.get("best_scores", []) if isinstance(s, (int, float))]
    final_best = best_scores[-1] if best_scores else None

    lines = [
        f"# Run report — {checkpoint_id}",
        "",
        f"- Score function: `{_infer_score_fn(path)}`",
        f"- Iterations: {data.get('n_iters', '?')}",
        f"- Final best score: {final_best if final_best is not None else 'n/a'}",
        "",
        "## Performance envelope (best per spec)",
        "",
        "| spec | goal | best | target | pass |",
        "|---|---|---|---|---|",
    ]
    for e in envelope:
        passes = e.get("passes")
        mark = "PASS" if passes else ("FAIL" if passes is False else "—")
        lines.append(
            f"| {e['metric']} | {e.get('goal', '')} | {e.get('best_ever')} | "
            f"{e.get('target')} | {mark} |"
        )
    summary_md = "\n".join(lines) + "\n"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(path.name, path.read_text())
        if yaml_path:
            yp = Path(yaml_path)
            if yp.exists() and yp.is_file():
                z.writestr("project_setup.yaml", yp.read_text())
        z.writestr("summary.md", summary_md)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{checkpoint_id}-report.zip"'},
    )
