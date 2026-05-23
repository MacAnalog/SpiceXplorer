"""Checkpoint listing and loading routes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ui.backend.app_config import preset_checkpoint_paths, REPO_ROOT
from ui.backend.services.checkpoint_reader import read_checkpoint, compute_envelope, compute_scatter

router = APIRouter()


def _infer_score_fn(path: Path) -> str:
    name = path.name.lower()
    if "sigmoid" in name:
        return "relative-sigmoid"
    if "relabs" in name or "relativeabs" in name or "linear" in name:
        return "relative-absolute"
    return "unknown"


def _list_autosave_checkpoints() -> list[dict[str, Any]]:
    results = []
    autosave_root = REPO_ROOT / "auto_save"
    if autosave_root.exists():
        for p in sorted(autosave_root.rglob("*.json")):
            results.append({
                "id": p.stem,
                "label": p.stem,
                "path": str(p),
                "type": "json",
                "score_fn": _infer_score_fn(p),
                "n_iters": None,
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
                "source": "preset",
            })
    items += _list_autosave_checkpoints()
    return {"checkpoints": items}


@router.get("/checkpoint/{checkpoint_id}")
def load_checkpoint(checkpoint_id: str, limit: int = Query(default=0)):
    presets = preset_checkpoint_paths()
    path: Path | None = presets.get(checkpoint_id)

    if path is None:
        # Try autosave
        autosave_root = REPO_ROOT / "auto_save"
        candidates = list(autosave_root.rglob(f"{checkpoint_id}*.json")) if autosave_root.exists() else []
        if candidates:
            path = candidates[0]

    if path is None or not path.exists():
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

    autosave_root = REPO_ROOT / "auto_save"
    if not autosave_root.exists():
        raise HTTPException(404, "No autosave directory.")

    # Match any file whose stem starts with the checkpoint_id under auto_save/
    candidates = [
        p for p in autosave_root.rglob("*")
        if p.is_file() and p.stem == checkpoint_id
    ]
    if not candidates:
        raise HTTPException(404, f"Checkpoint '{checkpoint_id}' not found in auto_save.")

    # Defence-in-depth: confirm every candidate really lives under auto_save/
    resolved_root = autosave_root.resolve()
    for path in candidates:
        if resolved_root not in path.resolve().parents:
            raise HTTPException(403, f"Refusing to delete file outside auto_save: {path}")

    for path in candidates:
        path.unlink()

    return {"ok": True, "deleted": [str(p) for p in candidates]}


@router.get("/checkpoint/{checkpoint_id}/envelope")
def checkpoint_envelope(checkpoint_id: str, yaml_path: str = Query(default="")):
    from pathlib import Path as _Path
    from spicexplorer.core.domains import Project_Setup

    presets = preset_checkpoint_paths()
    path = presets.get(checkpoint_id)
    if path is None or not path.exists():
        raise HTTPException(404, f"Checkpoint '{checkpoint_id}' not found")

    data = read_checkpoint(path)

    target_specs = None
    if yaml_path:
        try:
            project = Project_Setup.from_yaml(_Path(yaml_path))
            target_specs = [
                {"name": s.name, "target": float(s.target),
                 "goal": s.goal.value, "tolerance": float(s.tolerance) if s.tolerance else None}
                for s in project.optimizer_config.target_specs.targets
            ]
        except Exception:
            pass

    return {"envelope": compute_envelope(data, target_specs)}


@router.get("/checkpoint/{checkpoint_id}/scatter")
def checkpoint_scatter(
    checkpoint_id: str,
    metric_x: str = Query(...),
    metric_y: str = Query(...),
    yaml_path: str = Query(default=""),
):
    from pathlib import Path as _Path
    from spicexplorer.core.domains import Project_Setup

    presets = preset_checkpoint_paths()
    path = presets.get(checkpoint_id)
    if path is None or not path.exists():
        raise HTTPException(404, f"Checkpoint '{checkpoint_id}' not found")

    data = read_checkpoint(path)

    target_specs = None
    if yaml_path:
        try:
            project = Project_Setup.from_yaml(_Path(yaml_path))
            target_specs = [
                {"name": s.name, "target": float(s.target),
                 "goal": s.goal.value, "tolerance": float(s.tolerance) if s.tolerance else None}
                for s in project.optimizer_config.target_specs.targets
            ]
        except Exception:
            pass

    points = compute_scatter(data, metric_x, metric_y, target_specs)
    return {"metric_x": metric_x, "metric_y": metric_y, "points": points}
