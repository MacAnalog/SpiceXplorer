"""Backend configuration — resolves repo-relative paths from app_config.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).parent
UI_DIR = BACKEND_DIR.parent
REPO_ROOT = UI_DIR.parent

_app_config: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _app_config
    if _app_config is None:
        cfg_path = UI_DIR / "app_config.json"
        with cfg_path.open() as f:
            raw = json.load(f)
        _app_config = raw
    return _app_config


def get_app_config() -> dict[str, Any]:
    return _load()


def resolve(repo_relative: str) -> Path:
    return REPO_ROOT / repo_relative


def default_yaml_path() -> Path:
    return resolve(get_app_config()["default_yaml"])


def preset_checkpoint_paths() -> dict[str, Path]:
    return {k: resolve(v) for k, v in get_app_config()["preset_checkpoints"].items()}


def schematic_svg_path() -> Path:
    return resolve(get_app_config()["schematic_svg"])
