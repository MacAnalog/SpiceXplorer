# SpiceXplorer

SpiceXplorer is a Python toolkit for analog circuit sizing and SPICE-based optimization research. It combines a YAML project description with Python datamodels, simulation wrappers, and optimizer backends so a design exploration run can be defined from a single setup file.

The main documented workflow today is NGSpice-centered: a `project_setup.yaml` file is parsed into `Project_Setup`, consumed by `Circuit_Optimizer_Orchestrator_with_SPICE`, evaluated across enabled testbenches, and turned into scored optimization traces, checkpoints, logs, and Plotly reports.

`project_setup.yaml` -> `Project_Setup.from_yaml(...)` -> `Circuit_Optimizer_Orchestrator_with_SPICE` -> NGSpice testbench execution -> optimizer scoring -> checkpoints and HTML plots

## Development Setup

This repository is managed with `uv` for environment creation, locking, and command execution. The package build backend remains `hatchling`.

Create the default development environment:

```bash
uv sync
```

This creates a project-local `.venv/`, installs the package in editable mode, installs the default `dev` dependency group, and uses the pinned interpreter from `.python-version`.

Useful commands:

```bash
uv run pytest
uv run python -c "import spicexplorer.optimization as opt; print(opt.Optimizer_Type_Enum.NEVERGRAD_SINGLE)"
uv build
```

To work on the optional Ax backend, install the extra dependencies:

```bash
uv sync --extra ax
uv run python -c "from spicexplorer.optimization import Ax_Spice_Single_Objective; print(Ax_Spice_Single_Objective)"
```

The RL stack is intentionally not normalized in this migration because it still depends on an external `rl_framework` package that is not part of this repository.
## Features

- YAML-based project DSL for circuit setup, optimization variables, testbenches, and target specifications.
- Parameterized DUT sizing across multiple enabled SPICE testbenches, with optional parallel simulation execution.
- Spec-driven scoring based on target values, tolerances, weights, error modes, and reward modes.
- Optimization backends centered on Nevergrad, with Ax and RL modules available as additional or evolving paths.
- Checkpoint saving, logging, and Plotly-based visualization of scores and optimization traces.

## Project Organization

- `src/spicexplorer/core`: typed project schema, YAML loading, engineering-value parsing, normalization helpers, and scoring logic.
- `src/spicexplorer/spice_engine`: NGSpice + `spicelib` wrappers for parameter injection, run management, and raw/log extraction.
- `src/spicexplorer/optimization`: the orchestrator plus stochastic and RL optimizer backends.
- `src/spicexplorer/viz` and `src/spicexplorer/logging`: checkpoint visualization and run logging utilities.
- `examples/OTA/cascode`: the main reference example, organized into `xschem/`, `spice/`, and `sizing/`.

## YAML DSL

The central interface for a sizing run is the YAML project spec used by the cascode OTA example at [`examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml`](examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml). It is loaded through `Project_Setup.from_yaml(...)` into typed dataclasses under `src/spicexplorer/core/domains.py`, then passed into `Circuit_Optimizer_Orchestrator_with_SPICE`.

An abbreviated version of the schema looks like this:

```yaml
project:
  simulator: ngspice
  ws_root: /path/to/examples/OTA/cascode/ihp-sg13g2  # machine-specific
  netlist: spice/ota-improved.spice
  outdir: spice/temp_spice_out

  tech_spec:
    name: ihp-sg13g2
    constraints:
      min_nfet_l: 0.18u
      max_nfet_l: 10u

  pvt_corners:
    - temp: 25
      corner: tt
      supply: 1.5

  dut_params:
    - name: X_DUT_M1M2_L
      min_val: min_nfet_l
      max_val: max_nfet_l
    - name: X_DUT_M5_NG
      min_val: 1
      max_val: 5
      is_integer: true

  testbenches:
    - name: tb_ac
      netlist: spice/ota-improved_tb-loopgain.spice
      enable: true
      params:
        - name: CL
          val: 50f

  optimizer_config:
    type: nevergrad
    name: LogBFGSCMAPlus
    budget: 2000
    target_specs:
      - name: ugf
        testbench: tb_ac
        sim_type: ac
        goal: exceed
        target: 200e6
        tolerance: 10e6
```

The main top-level sections in this DSL are:

- `project`: project metadata plus workspace-relative paths such as the DUT netlist and output directory.
- `tech_spec.constraints`: named technology limits that can be referenced elsewhere in the file.
- `pvt_corners`: captured setup metadata for PVT definitions associated with the project.
- `dut_params`: design variables to be sized or optimized.
- `testbenches`: enabled SPICE testbenches and their parameter overrides.
- `optimizer_config`: optimizer settings plus `target_specs`, which define what metrics are scored and how.

Supported DSL behaviors in the current codebase include:

- Parsing engineering-style numeric strings such as `0.18u`, `50f`, `1e6`, and `100e9`.
- Resolving `dut_params` bounds from named technology constraints such as `min_nfet_l` and `max_nfet_w`.
- Handling integer design variables with `is_integer: true`.
- Handling log-scaled parameters and specs through `log_scale: true` where needed.
- Enabling or disabling individual testbenches with `enable: true` or `false`.
- Scoring target specs with goals such as `exact`, `exceed`, and `minimize`.
- Applying tolerances, weights, `error_type`, and `reward_type` during fitness computation.

In the cascode example, `ws_root` is an absolute machine-specific path. It should be adapted to your local checkout before running the example.

## Reference Example

The main example lives under [`examples/OTA/cascode`](examples/OTA/cascode):

- `xschem/` contains the source schematics and symbols.
- `spice/` contains exported DUT and testbench netlists used for simulation.
- `sizing/` contains the YAML setup plus runner scripts and notebooks.

For the clearest end-to-end script entry point, see [`examples/OTA/cascode/ihp-sg13g2/sizing/nevergrad_single_obj_opt.py`](examples/OTA/cascode/ihp-sg13g2/sizing/nevergrad_single_obj_opt.py). That script loads `project_setup.yaml`, instantiates the orchestrator, creates a Nevergrad optimizer, runs the optimization loop, and saves visualization outputs.

You can run the example from the managed environment with:

```bash
uv run python examples/OTA/cascode/ihp-sg13g2/sizing/nevergrad_single_obj_opt.py
```
Typical outputs from that workflow include autosaved JSON checkpoints, timestamped log files, and Plotly HTML reports for scores and metric traces.

## Current Focus

The most mature documented path in this repository is the YAML + NGSpice + Nevergrad workflow described above. Ax- and RL-related modules are present under `src/spicexplorer/optimization`, but they should be treated as additional or evolving backends rather than the primary path documented here.

This README intentionally focuses on the current orchestration flow around NGSpice. It does not assume a polished CLI or a broad multi-simulator workflow beyond what is clearly wired into the present codebase.
