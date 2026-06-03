"""Shared fixtures and marks for SpiceXplorer tests."""
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_YAML = REPO_ROOT / "examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml"
EXAMPLE_DUT_NETLIST = REPO_ROOT / "examples/OTA/cascode/ihp-sg13g2/spice/ota-improved.spice"
EXAMPLE_TB_NETLIST = REPO_ROOT / "examples/OTA/cascode/ihp-sg13g2/spice/ota-improved_tb-loopgain.spice"


def _ngspice_available() -> bool:
    return shutil.which("ngspice") is not None


requires_ngspice = pytest.mark.skipif(
    not _ngspice_available(),
    reason="ngspice binary not found in PATH",
)

slow = pytest.mark.slow
