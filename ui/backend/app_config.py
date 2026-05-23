"""Backend configuration — resolves repo-relative paths from demo_config.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).parent
UI_DIR = BACKEND_DIR.parent
REPO_ROOT = UI_DIR.parent

_demo_config: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _demo_config
    if _demo_config is None:
        cfg_path = UI_DIR / "demo_config.json"
        with cfg_path.open() as f:
            raw = json.load(f)
        _demo_config = raw
    return _demo_config


def get_demo_config() -> dict[str, Any]:
    return _load()


def resolve(repo_relative: str) -> Path:
    return REPO_ROOT / repo_relative


def default_yaml_path() -> Path:
    return resolve(get_demo_config()["default_yaml"])


def demo_checkpoint_paths() -> dict[str, Path]:
    return {k: resolve(v) for k, v in get_demo_config()["demo_checkpoints"].items()}


def schematic_svg_path() -> Path:
    return resolve(get_demo_config()["schematic_svg"])
