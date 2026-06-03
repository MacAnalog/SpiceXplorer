"""POST /api/score — compute sigmoid vs. linear penalties for a project's specs."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spicexplorer.core.domains import Project_Setup
from ui.backend.services.score_service import compute_score

router = APIRouter()

_project_cache: dict[str, Project_Setup] = {}


def _get_project(yaml_path: str) -> Project_Setup:
    if yaml_path not in _project_cache:
        p = Path(yaml_path)
        if not p.exists():
            raise HTTPException(404, f"YAML not found: {yaml_path}")
        _project_cache[yaml_path] = Project_Setup.from_yaml(p)
    return _project_cache[yaml_path]


class ScoreRequest(BaseModel):
    yaml_path: str
    metric_values: dict[str, float]
    selected_spec: str | None = None
    n_curve_points: int = 200


@router.post("/score")
def score_endpoint(body: ScoreRequest):
    project = _get_project(body.yaml_path)
    result = compute_score(
        project,
        body.metric_values,
        selected_spec=body.selected_spec,
        n_curve_points=body.n_curve_points,
    )
    return result
