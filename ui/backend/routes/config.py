"""GET /api/config — return demo_config.json with resolved paths for the frontend."""
from fastapi import APIRouter
from ui.backend.app_config import get_demo_config, demo_checkpoint_paths, schematic_svg_path, default_yaml_path

router = APIRouter()


@router.get("/config")
def get_config():
    raw = get_demo_config()
    checkpoints = []
    for key, path in demo_checkpoint_paths().items():
        name = path.name.lower()
        if "sigmoid" in name:
            score_fn = "relative-sigmoid"
        elif "relabs" in name or "relativeabs" in name:
            score_fn = "relative-absolute"
        else:
            score_fn = "unknown"
        checkpoints.append({
            "id": key,
            "label": key.replace("_", " ").title(),
            "path": str(path),
            "type": "csv" if path.suffix == ".csv" else "json",
            "score_fn": score_fn,
            "exists": path.exists(),
        })
    return {
        "default_yaml": str(default_yaml_path()),
        "demo_checkpoints": checkpoints,
        "schematic_svg_path": str(schematic_svg_path()),
    }
