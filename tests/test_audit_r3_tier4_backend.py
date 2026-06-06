"""Regression tests for the 2026-06 audit r3 **Tier 4** backend minors (Groups A/B/C).

Core scoring/loader (B17,B18,B21,B22,B24,B29,B32,B34), backend routes/services (B35,B38,B39,B41,B43),
and the route status-code fixes (B36,B37). No ngspice/PDK/sim needed.
"""
import logging
import math
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

CASCODE_YAML = REPO_ROOT / "examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml"


# ---------- Group A: core scoring/loader ----------

def test_b18_parse_value_none_raises_valueerror():
    from spicexplorer.core.domains import parse_value
    with pytest.raises(ValueError):
        parse_value(None)  # was an opaque AttributeError on None.lower()


def test_b17_zero_target_gets_positive_tolerance():
    from spicexplorer.core.domains import TargetSpec
    spec = TargetSpec(name="vos", testbench="tb", target=0.0, goal="exact", sim_type="dc", range=1.0)
    assert spec.tolerance is not None and float(spec.tolerance) > 0  # not the degenerate 0


def test_b21_log_reward_finite_at_exact_match():
    from spicexplorer.core.utils import compute_relative_log_reward, compute_log_reward
    assert math.isfinite(float(compute_relative_log_reward(np.float64(5.0), np.float64(5.0), np.float64(1.0))))
    assert math.isfinite(float(compute_log_reward(np.float64(0.0), np.float64(0.0))))


def test_b22_exponential_error_does_not_overflow():
    from spicexplorer.core.utils import compute_exponential_error, compute_relative_exponential_error
    assert math.isfinite(float(compute_exponential_error(np.float64(5e7), np.float64(0.0))))
    assert math.isfinite(float(compute_relative_exponential_error(np.float64(5e7), np.float64(0.0), np.float64(1.0))))


def test_b24_log_normalization_base10_endpoints():
    """compute_log_normalization must map 0→min, 1→max in base-10 (aligned w/ the Nevergrad path)."""
    from spicexplorer.core.domains import Param
    p = Param(name="x", min_val=np.float64(1e-7), max_val=np.float64(1e-4),
              val=None, init=None, description=None, log_scale=True)
    assert float(p.compute_log_normalization(np.float64(0.0))) == pytest.approx(1e-7, rel=1e-9)
    assert float(p.compute_log_normalization(np.float64(1.0))) == pytest.approx(1e-4, rel=1e-9)


def test_b29_duplicate_target_spec_names_rejected():
    from spicexplorer.core.domains import TargetSpec, ListTargetSpec
    a = TargetSpec(name="ugf", testbench="tb", target=1.0, goal="exceed", sim_type="ac")
    b = TargetSpec(name="ugf", testbench="tb2", target=2.0, goal="exceed", sim_type="ac")
    with pytest.raises(ValueError):
        ListTargetSpec(targets=[a, b])


def test_b32_corner_with_both_process_and_includes_raises():
    from spicexplorer.core.domains import _normalize_pvt_block
    proj = {"pvt": {
        "active_corner": "c",
        "process_bundles": {"tt": [{"lib_file": "m.lib", "section": "tt"}]},
        "corners": [{"name": "c", "process": "tt",
                     "model_includes": [{"lib_file": "m.lib", "section": "tt"}], "temp": 27}],
    }}
    with pytest.raises(ValueError):
        _normalize_pvt_block(proj)


def test_b34_disabled_active_corner_warns(caplog):
    from spicexplorer.core.domains import PVTConfig, Corner
    cfg = PVTConfig(active_corner="ss", corners=[Corner(name="ss", temp=85.0, enabled=False)])
    with caplog.at_level(logging.WARNING):
        c = cfg.get_active()
    assert c.name == "ss"
    assert any("enabled: false" in r.getMessage() or "enabled:false" in r.getMessage()
               for r in caplog.records)


# ---------- Group B/C: backend ----------

def _ui():
    pytest.importorskip("fastapi", reason="ui extra not installed")


def test_b39_b35_target_specs_only_enabled_and_no_none_tolerance(tmp_path):
    _ui()
    import yaml
    from ui.backend.routes.checkpoint import _target_specs_from_yaml
    d = yaml.safe_load(CASCODE_YAML.read_text())
    specs = None
    # find target_specs list and disable the first, blank-tolerance a target:0 one
    def find(o, k):
        if isinstance(o, dict):
            if k in o: return o[k]
            for v in o.values():
                r = find(v, k)
                if r is not None: return r
        elif isinstance(o, list):
            for v in o:
                r = find(v, k)
                if r is not None: return r
        return None
    ts = find(d, "target_specs")
    disabled_name = ts[0]["name"]
    ts[0]["enable"] = False
    p = tmp_path / "spx_uploaded_p.yaml"  # B2 validator only allows spx_uploaded_* temp files
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    out = _target_specs_from_yaml(str(p))
    assert out is not None
    names = {s["name"] for s in out}
    assert disabled_name not in names, "disabled spec must be excluded (B39)"
    assert all(s["tolerance"] is not None for s in out), "tolerance must never be None (B35)"


def test_b38_resolve_checkpoint_exact_stem(tmp_path, monkeypatch):
    _ui()
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "wr"))
    from ui.backend.routes import checkpoint as C
    wr = tmp_path / "wr"
    ck = wr / "runs" / "r1" / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "ota_DE_2000_trial20.json").write_text("{}")  # a longer stem
    # An id that is a PREFIX of the above must NOT resolve to it (no trial2.json exists).
    assert C._resolve_checkpoint_path("ota_DE_2000_trial2") is None
    (ck / "ota_DE_2000_trial2.json").write_text("{}")
    got = C._resolve_checkpoint_path("ota_DE_2000_trial2")
    assert got is not None and got.stem == "ota_DE_2000_trial2"


def test_b41_no_autosave_dir_created_without_checkpoint(tmp_path, monkeypatch):
    """Building an optimizer that never checkpoints must not leak a CWD ./auto_save dir (B41)."""
    _ui()
    monkeypatch.chdir(tmp_path)
    from spicexplorer.core.domains import Project_Setup
    from test_audit_r3_newcas import _NoopOpt  # reuse the no-op concrete optimizer
    proj = Project_Setup.from_yaml(CASCODE_YAML)
    _NoopOpt(proj)  # no output_root, no save_checkpoint call
    assert not (tmp_path / "auto_save").exists(), "eager mkdir leaked an ./auto_save dir"


def test_b43_reconcile_flips_trashed_running_run(tmp_path, monkeypatch):
    _ui()
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "wr"))
    import json
    from ui.backend.services import project_service
    from ui.backend.app_config import trash_root
    rj = trash_root() / "proj__ts" / "runs" / "r1" / "run.json"
    rj.parent.mkdir(parents=True, exist_ok=True)
    rj.write_text(json.dumps({"run_id": "r1", "status": "running"}))
    project_service.reconcile_stale_runs()
    assert json.loads(rj.read_text())["status"] == "error"


def test_b40_pvt_corner_params_round_trip():
    """A corner's per-corner `.param` overrides must survive generate → parse → generate (B40)."""
    _ui()
    from ui.backend.services.yaml_generator import _build_pvt_block, _pvt_block_to_form
    form = {"pvt": {"active_corner": "c", "corners": [
        {"name": "c", "temp": "27", "supply_node": "VDD", "supply_value": "1.5",
         "extra_supplies": [], "enabled": True, "includes": [], "params": {"vbias": 0.7}}]}}
    block = _build_pvt_block(form)
    assert block is not None and block["corners"][0].get("params") == {"vbias": 0.7}
    form2 = _pvt_block_to_form(block)  # raw block → form (re-normalizes)
    assert form2["corners"][0]["params"] == {"vbias": 0.7}


def test_b4_residual_project_delete_tombstone():
    _ui()
    from ui.backend.services import optimizer_runner as R
    assert R.is_project_deleting("proj-x") is False
    R.begin_project_delete("proj-x")
    try:
        assert R.is_project_deleting("proj-x") is True
        assert R.is_project_deleting(None) is False
    finally:
        R.end_project_delete("proj-x")
    assert R.is_project_deleting("proj-x") is False


def test_b4_residual_start_refused_while_deleting():
    _ui()
    pytest.importorskip("httpx", reason="starlette TestClient needs httpx")
    from fastapi.testclient import TestClient
    from ui.backend.main import app
    from ui.backend.services import optimizer_runner as R
    client = TestClient(app)
    R.begin_project_delete("proj-being-deleted")
    try:
        r = client.post("/api/optimize/start",
                        json={"project_id": "proj-being-deleted", "replay": False})
        assert r.status_code == 409
    finally:
        R.end_project_delete("proj-being-deleted")


def test_b36_b37_optimize_route_status_codes():
    _ui()
    pytest.importorskip("httpx", reason="starlette TestClient needs httpx")
    from fastapi.testclient import TestClient
    from ui.backend.main import app
    client = TestClient(app)
    # B36: replay with no checkpoint_id → 400 (not a hung run)
    r = client.post("/api/optimize/start", json={"replay": True})
    assert r.status_code == 400
    # B37: malformed project_id → 400 (not an opaque 500)
    r = client.post("/api/optimize/start", json={"project_id": "../etc", "replay": False})
    assert r.status_code in (400, 409)  # 400 (bad id) — or 409 if PDK-gated check runs first
