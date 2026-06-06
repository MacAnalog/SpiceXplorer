"""POST /api/score — compute sigmoid vs. linear penalties for a project's specs."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spicexplorer.core.domains import Project_Setup
from ui.backend.services.score_service import apply_spec_overrides, compute_score

router = APIRouter()


def _get_project(yaml_path: str) -> Project_Setup:
    # Reload from disk every call (Project_Setup.from_yaml is cheap and
    # PDK-independent). A module-level cache previously returned stale specs
    # after the YAML was edited on disk — inconsistent with /sanity-check.
    p = Path(yaml_path)
    if not p.exists():
        raise HTTPException(404, f"YAML not found: {yaml_path}")
    return Project_Setup.from_yaml(p)


class ScoreRequest(BaseModel):
    yaml_path: str
    metric_values: dict[str, float]
    selected_spec: str | None = None
    n_curve_points: int = 200
    # Ephemeral per-spec edits (Score Shaping "what-if"): spec name → partial dict of
    # {target, tolerance, weight, range, goal, enable}. Applied to the freshly-loaded
    # project before scoring; never written back to the YAML.
    spec_overrides: dict[str, dict] | None = None


@router.post("/score")
def score_endpoint(body: ScoreRequest):
    project = _get_project(body.yaml_path)
    apply_spec_overrides(project, body.spec_overrides)
    result = compute_score(
        project,
        body.metric_values,
        selected_spec=body.selected_spec,
        n_curve_points=body.n_curve_points,
    )
    return result
