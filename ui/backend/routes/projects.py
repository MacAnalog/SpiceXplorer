"""Project registry + scaffold / copy-example / per-project runs (report.md P3).

A project IS a directory under WORK_ROOT/projects (the registry is the filesystem,
no DB). "New project" scaffolds an example-structured dir; "load example" copies a
demo into a fresh registered project. All FS bookkeeping lives in project_service.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spicexplorer.core.domains import Project_Setup
from ui.backend.routes.project import _summarise
from ui.backend.services import optimizer_runner, project_service

router = APIRouter()


class CreateProjectRequest(BaseModel):
    name: str
    # The wizard's generated YAML (optional). When omitted, the default example seeds it.
    yaml_content: Optional[str] = None


class FromExampleRequest(BaseModel):
    example_key: str
    name: Optional[str] = None


class RenameRequest(BaseModel):
    name: str


class ForkRequest(BaseModel):
    name: Optional[str] = None


class RenameRunRequest(BaseModel):
    label: str


@router.get("/projects")
def list_projects():
    return {"projects": project_service.list_projects()}


@router.get("/examples")
def list_examples():
    """In-repo demo projects that can be loaded (copied) as a new project."""
    return {"examples": project_service.list_examples()}


@router.post("/projects")
def create_project(body: CreateProjectRequest):
    if not body.name.strip():
        raise HTTPException(400, "project name is required")
    try:
        pid = project_service.create_project(body.name, body.yaml_content)
    except Exception as e:
        raise HTTPException(400, f"create failed: {e}")
    return {"id": pid}


@router.post("/projects/from-example")
def from_example(body: FromExampleRequest):
    try:
        pid = project_service.copy_example(body.example_key, body.name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(400, f"copy failed: {e}")
    return {"id": pid}


@router.get("/projects/{project_id}")
def get_project(project_id: str):
    if not project_service.project_exists(project_id):
        raise HTTPException(404, f"project '{project_id}' not found")
    yp = project_service.project_yaml(project_id)
    try:
        project = Project_Setup.from_yaml(yp)
    except Exception as e:
        raise HTTPException(422, str(e))
    return {
        "id": project_id,
        "yaml_path": str(yp),
        "summary": _summarise(project),
        "manifest": project_service.read_manifest(project_id),
    }


@router.get("/projects/{project_id}/runs")
def project_runs(project_id: str):
    if not project_service.project_exists(project_id):
        raise HTTPException(404, f"project '{project_id}' not found")
    return {"runs": project_service.list_runs(project_id)}


# ---------- lifecycle: rename / fork / soft-delete + trash/restore (report.md P4) ----------


@router.patch("/projects/{project_id}")
def rename_project(project_id: str, body: RenameRequest):
    try:
        man = project_service.rename_project(project_id, body.name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": project_id, "manifest": man}


@router.post("/projects/{project_id}/fork")
def fork_project(project_id: str, body: ForkRequest):
    try:
        new_id = project_service.fork_project(project_id, body.name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(400, f"fork failed: {e}")
    return {"id": new_id}


@router.delete("/projects/{project_id}")
def delete_project(project_id: str):
    # Quiesce any in-flight live run for this project FIRST, else its writer thread
    # re-creates the moved-away run tree and defeats the soft-delete (corruption + leak).
    # If a worker is still mid-trial after the join, do NOT move the dir out from under it
    # (BUG-B4) — 409 and let the caller retry once the trial finishes.
    stopped, still_alive = optimizer_runner.stop_runs_for(project_id=project_id)
    if still_alive:
        raise HTTPException(
            409,
            f"{len(still_alive)} run(s) for '{project_id}' are still stopping; retry shortly.",
        )
    try:
        trash_id = project_service.soft_delete_project(project_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(400, f"delete failed: {e}")
    return {"ok": True, "trash_id": trash_id, "stopped_runs": stopped}


@router.get("/trash")
def list_trash():
    return {"trash": project_service.list_trash()}


@router.post("/trash/{trash_id}/restore")
def restore_trash(trash_id: str):
    try:
        pid = project_service.restore_project(trash_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))
    except RuntimeError as e:  # corrupt trash metadata — a data error, not a conflict
        raise HTTPException(500, str(e))
    return {"id": pid}


@router.patch("/projects/{project_id}/runs/{run_id}")
def rename_run(project_id: str, run_id: str, body: RenameRunRequest):
    if not project_service.project_exists(project_id):
        raise HTTPException(404, f"project '{project_id}' not found")
    try:
        run = project_service.rename_run(project_id, run_id, body.label)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"run": run}


@router.delete("/projects/{project_id}/runs/{run_id}")
def delete_run(project_id: str, run_id: str):
    if not project_service.project_exists(project_id):
        raise HTTPException(404, f"project '{project_id}' not found")
    # Stop the run if it's live (same writer-resurrection hazard as project delete). If it's
    # still mid-trial after the join, 409 rather than move its dir out from under it (BUG-B4).
    _, still_alive = optimizer_runner.stop_runs_for(project_id=project_id, run_id=run_id)
    if still_alive:
        raise HTTPException(409, "run is still stopping; retry shortly.")
    try:
        trash_id = project_service.delete_run(project_id, run_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"ok": True, "trash_id": trash_id}
