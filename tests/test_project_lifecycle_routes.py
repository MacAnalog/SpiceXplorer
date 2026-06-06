"""HTTP route tests for the project/run lifecycle endpoints (report.md P4).

Fast, NO SPICE. A temp WORK_ROOT keeps the registry out of the real ./work. The
routes are thin wrappers over project_service (unit-tested in test_project_service.py),
so these assert the wiring: status codes, the soft-delete → trash → restore round-trip,
and that fork/rename reach the right service call.
"""
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))
pytest.importorskip("fastapi", reason="ui extra not installed")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    from fastapi.testclient import TestClient

    from ui.backend.main import app
    return TestClient(app)


def _new(client, name="Proj") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_rename_project_route(client):
    pid = _new(client, "Before")
    r = client.patch(f"/api/projects/{pid}", json={"name": "After"})
    assert r.status_code == 200, r.text
    assert r.json()["manifest"]["name"] == "After"
    # The renamed name shows up in the listing under the same id.
    listed = {p["id"]: p["name"] for p in client.get("/api/projects").json()["projects"]}
    assert listed[pid] == "After"


def test_rename_missing_project_404(client):
    r = client.patch("/api/projects/missing-00000000", json={"name": "x"})
    assert r.status_code == 404


def test_fork_route_creates_new_id(client):
    pid = _new(client, "Origin")
    r = client.post(f"/api/projects/{pid}/fork", json={"name": "Forked"})
    assert r.status_code == 200, r.text
    new_id = r.json()["id"]
    assert new_id != pid
    assert client.get(f"/api/projects/{new_id}").status_code == 200


def test_delete_then_trash_then_restore_roundtrip(client):
    pid = _new(client, "Trashable")
    # Soft-delete → 200 with a trash id.
    d = client.delete(f"/api/projects/{pid}")
    assert d.status_code == 200, d.text
    trash_id = d.json()["trash_id"]
    # Gone from the registry.
    assert client.get(f"/api/projects/{pid}").status_code == 404
    # Visible in the trash listing.
    trash = client.get("/api/trash").json()["trash"]
    assert any(t["trash_id"] == trash_id for t in trash)
    # Restore → back under the original id.
    rr = client.post(f"/api/trash/{trash_id}/restore")
    assert rr.status_code == 200, rr.text
    assert rr.json()["id"] == pid
    assert client.get(f"/api/projects/{pid}").status_code == 200


def test_restore_conflict_returns_409(client):
    pid = _new(client, "Clash")
    trash_id = client.delete(f"/api/projects/{pid}").json()["trash_id"]
    # Re-create a project that occupies the original id, then restore must 409.
    from ui.backend.services import project_service
    project_service.project_dir(pid).mkdir(parents=True, exist_ok=True)
    project_service.project_yaml(pid).write_text("project: {}\n")
    rr = client.post(f"/api/trash/{trash_id}/restore")
    assert rr.status_code == 409


def test_run_rename_and_delete_routes(client):
    import json

    pid = _new(client, "WithRuns")
    from ui.backend.services import project_service
    rd = project_service.run_dir(pid, "2026_algo_abcd1234")
    (rd / "run.json").write_text(json.dumps({"run_id": "run-1", "status": "done"}))

    pr = client.patch(f"/api/projects/{pid}/runs/run-1", json={"label": "Champion"})
    assert pr.status_code == 200, pr.text
    assert pr.json()["run"]["label"] == "Champion"

    dr = client.delete(f"/api/projects/{pid}/runs/run-1")
    assert dr.status_code == 200, dr.text
    assert client.get(f"/api/projects/{pid}/runs").json()["runs"] == []
    # Deleting an already-gone run is a 404.
    assert client.delete(f"/api/projects/{pid}/runs/run-1").status_code == 404
