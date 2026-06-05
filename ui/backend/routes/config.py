"""GET /api/config — return app_config.json with resolved paths for the frontend."""
from fastapi import APIRouter
from ui.backend.app_config import preset_checkpoint_paths, schematic_svg_path, default_yaml_path
from ui.backend.routes.checkpoint import _infer_score_fn

router = APIRouter()


@router.get("/config")
def get_config():
    checkpoints = []
    for key, path in preset_checkpoint_paths().items():
        checkpoints.append({
            "id": key,
            "label": key.replace("_", " ").title(),
            "path": str(path),
            "type": "csv" if path.suffix == ".csv" else "json",
            "score_fn": _infer_score_fn(path),  # single source of truth (also handles "linear")
            "exists": path.exists(),
        })
    return {
        "default_yaml": str(default_yaml_path()),
        "preset_checkpoints": checkpoints,
        "schematic_svg_path": str(schematic_svg_path()),
    }
