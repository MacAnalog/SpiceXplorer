"""Round-trip test for the wizard ⇄ YAML PVT mapping (yaml_generator).

Skipped unless the `ui` extra is installed (the backend imports FastAPI/pydantic).
No ngspice / PDK needed — pure dict transforms + a library re-parse.
"""
import os
import sys
import tempfile

import pytest
import yaml

from conftest import REPO_ROOT

# `ui` lives at the repo root (not under src/), so it isn't on sys.path by default
# under pytest. Add it, then skip cleanly if the `ui` extra (FastAPI/pydantic) is absent.
sys.path.insert(0, str(REPO_ROOT))
pytest.importorskip("fastapi", reason="ui extra not installed")

from spicexplorer.core.domains import Project_Setup
from ui.backend.services.yaml_generator import project_dict_to_form, generate_yaml

FC_YAML = REPO_ROOT / "examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml"


def test_pvt_block_roundtrips_through_the_wizard():
    raw = yaml.safe_load(FC_YAML.read_text())

    # YAML (with process_bundles) → wizard form: bundles flatten to inline includes.
    form = project_dict_to_form(raw)
    assert form["pvt"]["active_corner"] == "tt_27C_1V8"
    names = [c["name"] for c in form["pvt"]["corners"]]
    assert names == ["tt_27C_1V8", "ss_125C_1V62", "ff_m40C_1V98"]
    tt = form["pvt"]["corners"][0]
    assert tt["supply_node"] == "VDD"
    assert [(m["lib_file"], m["section"]) for m in tt["includes"]][0] == ("cornerMOSlv.lib", "mos_tt")
    assert form["pvt"]["corners"][2]["enabled"] is False  # ff is disabled

    # wizard form → YAML → library: re-parses to the same active corner.
    out_yaml = generate_yaml(form)
    assert "pvt:" in out_yaml and "active_corner" in out_yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", dir=str(FC_YAML.parent), delete=False) as f:
        f.write(out_yaml)
        tmp = f.name
    try:
        project = Project_Setup.from_yaml(tmp)
        assert project.pvt is not None
        assert project.pvt.active_corner == "tt_27C_1V8"
        active = project.pvt.get_active()
        assert active.temp == 27.0
        assert active.supplies[0].value == 1.8
        assert [(m.lib_file, m.section) for m in active.model_includes][0] == ("cornerMOSlv.lib", "mos_tt")
    finally:
        os.unlink(tmp)


def test_project_without_pvt_yields_empty_wizard_pvt():
    raw = yaml.safe_load(
        (REPO_ROOT / "examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml").read_text()
    )
    form = project_dict_to_form(raw)
    assert form["pvt"] == {"active_corner": "", "corners": []}
    # …and generating from it emits no `pvt:` block.
    assert "\npvt:" not in generate_yaml(form)
