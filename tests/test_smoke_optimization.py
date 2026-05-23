"""Smoke tests for spicexplorer.optimization.

Coverage (layered — each layer builds on the previous):
  Layer 1 — no SPICE needed:
    - Project_Setup loads from YAML and has expected structure
    - Orchestrator can be created without auto_load
  Layer 2 — ngspice required:
    - Orchestrator.initialize() creates one wrapper per enabled testbench
    - Optimizer.parameterize() returns a valid Nevergrad parameter dict
  Layer 3 — ngspice required + slow:
    - One optimization_step() runs a real simulation without crashing
"""
import pytest

from conftest import EXAMPLE_YAML, requires_ngspice, slow


# ── Layer 1: no SPICE required ──────────────────────────────────────────────

def test_project_setup_loads():
    """Project_Setup.from_yaml() parses the example OTA project correctly."""
    from spicexplorer.core.domains import Project_Setup

    setup = Project_Setup.from_yaml(EXAMPLE_YAML)

    assert setup.name, "Project name must be non-empty"
    assert len(setup.dut_params) > 0, "Expected at least one DUT parameter"
    assert len(setup.testbenches) > 0, "Expected at least one testbench"
    assert len(setup.optimizer_config.target_specs.targets) > 0, "Expected at least one target spec"
    assert setup.optimizer_config.budget > 0


def test_project_setup_param_bounds():
    """All DUT parameters have valid (non-None, min < max) bounds."""
    from spicexplorer.core.domains import Project_Setup

    setup = Project_Setup.from_yaml(EXAMPLE_YAML)
    for p in setup.dut_params:
        if p.freeze:
            continue
        assert p.min_val is not None, f"Param {p.name} has no min_val"
        assert p.max_val is not None, f"Param {p.name} has no max_val"
        assert float(p.min_val) < float(p.max_val), (
            f"Param {p.name}: min_val ({p.min_val}) >= max_val ({p.max_val})"
        )


def test_orchestrator_no_autoload():
    """Orchestrator can be constructed without calling initialize() (auto_load=False)."""
    from spicexplorer.optimization.orchestrator import (
        Circuit_Optimizer_Orchestrator_with_SPICE,
        Optimizer_Type_Enum,
    )

    orchestrator = Circuit_Optimizer_Orchestrator_with_SPICE(
        project_setup_path=EXAMPLE_YAML,
        optimizer_type=Optimizer_Type_Enum.NEVERGRAD_SINGLE,
        auto_load=False,
    )
    assert orchestrator.project_setup is not None
    assert not hasattr(orchestrator, "spicelib_wrappers") or orchestrator.spicelib_wrappers is None


# ── Layer 2: ngspice required ────────────────────────────────────────────────

@requires_ngspice
def test_orchestrator_initialize_creates_wrappers(tmp_path):
    """initialize() creates one NGSpice_Wrapper per enabled testbench."""
    from spicexplorer.optimization.orchestrator import (
        Circuit_Optimizer_Orchestrator_with_SPICE,
        Optimizer_Type_Enum,
    )

    orchestrator = Circuit_Optimizer_Orchestrator_with_SPICE(
        project_setup_path=EXAMPLE_YAML,
        optimizer_type=Optimizer_Type_Enum.NEVERGRAD_SINGLE,
        auto_load=False,
    )
    orchestrator.initialize()

    enabled_tbs = [tb for tb in orchestrator.project_setup.testbenches if tb.enable]
    assert len(orchestrator.spicelib_wrappers) == len(enabled_tbs), (
        f"Expected {len(enabled_tbs)} wrappers, got {len(orchestrator.spicelib_wrappers)}"
    )


@requires_ngspice
def test_optimizer_parameterize(tmp_path):
    """parameterize() builds the Nevergrad parameter space without running SPICE."""
    import nevergrad as ng
    from spicexplorer.optimization.orchestrator import (
        Circuit_Optimizer_Orchestrator_with_SPICE,
        Optimizer_Type_Enum,
    )

    orchestrator = Circuit_Optimizer_Orchestrator_with_SPICE(
        project_setup_path=EXAMPLE_YAML,
        optimizer_type=Optimizer_Type_Enum.NEVERGRAD_SINGLE,
        auto_load=True,
    )
    optimizer = orchestrator.get_optimizer()
    param_dict = optimizer.parameterize()

    assert isinstance(param_dict, ng.p.Dict), "parameterize() must return a nevergrad Dict"
    assert len(param_dict) > 0, "parameterize() returned an empty parameter dict"
    # All nevergrad keys must correspond to known DUT parameter names
    dut_param_names = {p.name for p in orchestrator.project_setup.dut_params}
    unknown = set(param_dict.keys()) - dut_param_names
    assert not unknown, f"parameterize() produced keys not in dut_params: {unknown}"


# ── Layer 3: slow end-to-end ─────────────────────────────────────────────────

@requires_ngspice
@slow
def test_one_optimization_step():
    """A single optimization step runs a real SPICE simulation and returns a score."""
    from spicexplorer.optimization.orchestrator import (
        Circuit_Optimizer_Orchestrator_with_SPICE,
        Optimizer_Type_Enum,
    )

    orchestrator = Circuit_Optimizer_Orchestrator_with_SPICE(
        project_setup_path=EXAMPLE_YAML,
        optimizer_type=Optimizer_Type_Enum.NEVERGRAD_SINGLE,
        auto_load=True,
    )
    optimizer = orchestrator.get_optimizer()
    optimizer.parameterize()

    params, score, metadata = optimizer.optimization_step()

    assert isinstance(params, dict) and len(params) > 0, "optimization_step() must return a non-empty params dict"
    assert score is not None, "optimization_step() must return a numeric score"
    assert metadata is not None, "optimization_step() must return metadata"
