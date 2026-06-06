"""Tests for the project registry + per-run isolation service (report.md P2/P3).

Fast, NO SPICE. Uses a temp WORK_ROOT so nothing touches the real ./work.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))
pytest.importorskip("fastapi", reason="ui extra not installed")


@pytest.fixture
def ps(tmp_path, monkeypatch):
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    from ui.backend.services import project_service as _ps
    return _ps


def test_create_project_scaffolds_example_structure(ps):
    pid = ps.create_project("My OTA")
    assert pid.startswith("my-ota-")
    pd = ps.project_dir(pid)
    assert (pd / "project.yaml").exists()
    for sub in ("spice", "xschem", "scratch", "runs"):
        assert (pd / sub).is_dir()
    data = yaml.safe_load((pd / "project.yaml").read_text())
    assert data["project"]["ws_root"] == "."
    assert data["project"]["outdir"] == "scratch"
    man = ps.read_manifest(pid)
    assert man["name"] == "My OTA" and man["source"]["kind"] == "new"
    assert any(p["id"] == pid for p in ps.list_projects())


def test_copy_example_registers_loadable_project(ps):
    examples = ps.list_examples()
    assert examples, "expected in-repo examples"
    cascode = next((e for e in examples if "cascode" in e["key"]), examples[0])
    pid = ps.copy_example(cascode["key"], "Demo Copy")
    pd = ps.project_dir(pid)
    data = yaml.safe_load((pd / "project.yaml").read_text())
    assert data["project"]["ws_root"] == "."
    # The copied subtree resolves — the project parses from its new home.
    from spicexplorer.core.domains import Project_Setup
    Project_Setup.from_yaml(pd / "project.yaml")
    assert ps.read_manifest(pid)["source"]["kind"] == "example"


def test_resolve_yaml_prefers_project_id_and_guards(ps):
    pid = ps.create_project("X")
    assert ps.resolve_yaml(pid, None) == ps.project_yaml(pid)
    # Unknown id raises rather than silently falling through (the yaml_path="" guard).
    with pytest.raises(FileNotFoundError):
        ps.resolve_yaml("nope-00000000", None)
    assert ps.resolve_yaml(None, "/x/y.yaml") == Path("/x/y.yaml")
    assert ps.resolve_yaml(None, None) == ps.default_yaml_path()


def test_run_dir_and_list_runs(ps):
    pid = ps.create_project("R")
    rd = ps.run_dir(pid, "2026_LhsDE_abcd1234")
    (rd / "run.json").write_text(json.dumps({"run_id": "r1", "status": "done", "best_score": 1.5}))
    runs = ps.list_runs(pid)
    assert len(runs) == 1 and runs[0]["status"] == "done"


def test_reconcile_flips_running_to_error(ps):
    pid = ps.create_project("C")
    rd = ps.run_dir(pid, "2026_x_aaaa1111")
    (rd / "run.json").write_text(json.dumps({"run_id": "r", "status": "running"}))
    assert ps.reconcile_stale_runs() == 1
    assert json.loads((rd / "run.json").read_text())["status"] == "error"


def test_guards_reject_path_traversal(ps):
    with pytest.raises(ValueError):
        ps.project_dir("../escape")
    with pytest.raises(ValueError):
        ps.project_dir("a/b")
