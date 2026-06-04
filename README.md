# SpiceXplorer

**A Pythonic toolkit for analog circuit sizing and SPICE-based optimization research.**

SpiceXplorer turns a single declarative project file into a complete, reproducible design-exploration run. You describe the circuit, the technology constraints, the testbenches, and the performance targets once in YAML; SpiceXplorer parses it into typed Python datamodels, drives a SPICE-in-the-loop optimization, scores each candidate against your specs, and emits checkpoints, logs, and interactive Plotly reports.

It is built for research with a long-term horizon — not for any one circuit, PDK, or optimizer:

- **Backend-agnostic optimization.** A common orchestrator abstracts the optimizer behind `Optimizer_Type_Enum`. Nevergrad is the mature, recommended backend; an Ax (Bayesian) backend is available as an optional extra, and an RL backend is evolving.
- **Spec-driven, not score-hardcoded.** Targets, tolerances, weights, error modes, and reward shaping are all data in the YAML, so the same machinery applies to any block you can netlist and measure.
- **Reusable examples, not a fixed demo.** The bundled OTA examples (including the cascode OTA used for the NEWCAS case study) are *reference projects* — starting points to copy and adapt, not the boundary of what the toolkit does.

> **Two ways to use it.** SpiceXplorer is a **Python library** you can script directly, *and* it ships a **web UI** (a VS Code-style "Studio" workspace) for guided setup, score shaping, live runs, and run comparison. Pick whichever fits — they share the same engine and the same `project_setup.yaml`.

```
project_setup.yaml
   └─ Project_Setup.from_yaml(...)            # typed datamodels (core/domains.py)
        └─ Circuit_Optimizer_Orchestrator_with_SPICE
             └─ NGSpice testbench execution    # spice_engine + spicelib
                  └─ spec scoring               # core/utils.py
                       └─ checkpoints · logs · Plotly reports
```

---

## Choosing a front-end

| | **Library / scripts** | **Web UI (Studio)** |
|---|---|---|
| Best for | Batch runs, automation, custom loops, notebooks, CI | Interactive setup, score tuning, watching a run converge, comparing runs |
| Entry point | `spicexplorer.optimization` + a `project_setup.yaml` | `./scripts/run_newcas_ui.sh` → http://localhost:4000 |
| Needs | `uv sync`; ngspice + PDK for real sims | `uv sync --extra ui` + Node/npm; ngspice + PDK for *live* runs (otherwise replay-only) |
| Output | JSON checkpoints, log files, Plotly HTML | Live charts, run history, checkpoint replay/compare, schematic browser |

Both read the **same** YAML and call the **same** orchestrator/optimizer code.

---

## Installation

SpiceXplorer is managed with [`uv`](https://docs.astral.sh/uv/) (environment, locking, and command execution); the build backend is `hatchling`. Python **3.10+** is required (the Ax extra needs 3.11+).

```bash
uv sync                 # default dev environment (library + tests)
uv sync --extra ui      # add FastAPI/uvicorn for the web UI backend
uv sync --extra ax      # add the optional Ax (Bayesian) optimizer backend  (Python 3.11+)
```

`uv sync` creates a project-local `.venv/`, installs the package in editable mode, and pins the interpreter from `.python-version`.

**External prerequisites (for real simulation):**

- **`ngspice`** on your `PATH` — required to run any actual SPICE simulation.
- **A PDK** matching your project's netlists — the bundled examples target the open-source **IHP `ihp-sg13g2`** process. Without the PDK, the library raises at simulation time and the UI degrades to **replay-only** (see below).
- For the **UI frontend** only: **Node.js + npm**.

The RL backend is intentionally not wired in this layout because it still depends on an external `rl_framework` package that is not part of this repository.

---

## Quick start — Python library

Run the bundled reference optimization end-to-end:

```bash
uv run python examples/OTA/cascode/ihp-sg13g2/sizing/nevergrad_single_obj_opt.py
```

This loads `project_setup.yaml`, builds the orchestrator, runs a Nevergrad search across the enabled testbenches, and writes autosaved checkpoints plus Plotly HTML reports.

A minimal script of your own looks like this:

```python
from pathlib import Path
from spicexplorer.optimization import (
    Circuit_Optimizer_Orchestrator_with_SPICE,
    Optimizer_Type_Enum,
)

orchestrator = Circuit_Optimizer_Orchestrator_with_SPICE(
    project_setup_path=Path("examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml"),
    optimizer_type=Optimizer_Type_Enum.NEVERGRAD_SINGLE,
    auto_load=False,
)
orchestrator.initialize()                 # builds one SPICE wrapper per enabled testbench

optimizer = orchestrator.get_optimizer()
optimizer.parameterize()                  # maps dut_params → optimizer search space
optimizer.optimize(keep_history=False)    # the SPICE-in-the-loop search; autosaves checkpoints

# Persist interactive reports
save_dir = optimizer.autosave_checkpoint_dir
optimizer.plot_score(show=False, save_path=save_dir / "score.html")
optimizer.plot_optimization_trace(
    metric_x="ugf", metric_y="dcgain", show=False, save_path=save_dir / "ugf_vs_dcgain.html",
)
```

To inspect or validate a project **without** running SPICE (no ngspice/PDK needed):

```python
from spicexplorer.core.domains import Project_Setup

project = Project_Setup.from_yaml("examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml")
print(project.name, "→", project.ws_root)
print("specs:", [s.name for s in project.optimizer_config.target_specs.targets])
```

> **Gotcha:** `project.optimizer_config.target_specs` is a `ListTargetSpec` wrapper, not a plain list — use `.targets` to reach the underlying list.

Available optimizer types (`Optimizer_Type_Enum`): `NEVERGRAD_SINGLE` (recommended, mature), `NEVERGRAD_CONSTRAINT`, `NEVERGRAD_BODE`, and `AX_SINGLE` / `AX_CONSTRAINT` (require `--extra ax`). `optimizer_type` is a required argument — there is no implicit default.

---

## Quick start — Web UI (Studio)

```bash
uv sync --extra ui
./scripts/run_newcas_ui.sh        # starts FastAPI backend (:8000) + Next.js frontend (:4000)
```

Open **http://localhost:4000**. (The frontend runs on **4000**, not 3000, because VS Code Remote SSH occupies 3000 on the dev server.)

```bash
LOG_LEVEL=DEBUG ./scripts/run_newcas_ui.sh     # verbose library logging
```

The UI is a persistent **Studio workspace** — an activity bar, contextual rails, tabbed center views, an always-on live-run rail, and a ⌘K command palette. Its views:

- **Setup** — load a project YAML (example dropdown / file upload) or build one with the 7-step wizard; edit in Monaco, validate, apply.
- **Score Shaping** — drag a slider per spec to compare linear vs. sigmoid penalty curves.
- **Optimize** — pick algorithm/budget/seed, start a live SPICE run or replay a checkpoint; watch score/metric convergence stream in.
- **Explore** — load two checkpoints and overlay convergence, metric scatter, and the performance envelope.
- **Schematic** — browse the project's Xschem hierarchy and run finite-difference sensitivity on a device.
- **Pipeline** — a read-only DAG of Optimizer → DUT params → Testbenches → Specs.
- **Health** — on-demand sanity check (one sim per testbench + a trial optimizer step).

**PDK-aware degradation:** `GET /api/env` detects ngspice and the PDK. When the PDK is absent the status bar shows a **"PDK missing — replay only"** pill and live Start is disabled (steered to checkpoint **Replay**); score shaping, replay/compare, the wizard, and the pipeline view all still work fully.

For UI architecture, the full API table, and troubleshooting, see **[ui/README.md](ui/README.md)**.

---

## Run in Docker (portable, bundled PDK)

For collaborators who **don't** have ngspice or the IHP PDK installed, the repo
ships a self-contained two-container stack. The backend image **compiles ngspice
from source** and bundles the IHP `ihp-sg13g2` ngspice models + OSDI compact
models (vendored under [`docker/pdk/`](docker/pdk/), Apache-2.0), so **live SPICE
works out of the box** — no host install, no multi-GB EDA image. torch resolves to
CPU-only wheels (no CUDA). **x86-64 only** (the bundled OSDI are x86-64; see
[docker/pdk/README.md](docker/pdk/README.md) to regenerate for arm64).

```bash
cp .env.example .env          # optional — defaults work as-is
docker compose up --build     # builds both images, starts the stack
# open http://localhost:4000
```

Everything is configurable through `.env` (loaded automatically by Compose):

| Variable | Default | Purpose |
|---|---|---|
| `WORKDIR` | `./work` | Host dir bind-mounted at `/work` (your projects + outputs). Set to any drive, e.g. `/mnt/data/spicex`. |
| `UID` / `GID` | `1000` | On **Linux**, set to `$(id -u)`/`$(id -g)` so files written to `WORKDIR` are owned by you, not root. |
| `FRONTEND_PORT` / `BACKEND_PORT` | `4000` / `8000` | Host ports. |
| `LOG_LEVEL` | `INFO` | `spicexplorer` console log level. |
| `INSTALL_AGENTS` | `false` | Install the `agents` dependency extra into the backend image (for the future LLM-agent layer; rebuild required). |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, … | *(unset)* | Passed into the backend at **runtime only** — never baked into the image. For the agent layer. |

To persist a run's outputs to the host, put your `project_setup.yaml` + netlists
under `WORKDIR` and point the run's `ws_root` at `/work`.

This is **complementary** to [`scripts/run_newcas_ui.sh`](scripts/run_newcas_ui.sh):
keep the script for fast native iteration on a machine that already has ngspice +
the PDK (e.g. the research server); use the container for portability and as a
reproducible artifact. See [doc/PLAN_STUDIO_INTEGRATION.md](doc/PLAN_STUDIO_INTEGRATION.md)
or the file headers in [docker/](docker/) for build details.

---

## The YAML project DSL

A run is defined by one `project_setup.yaml`, loaded through `Project_Setup.from_yaml(...)` into the typed dataclasses in [`src/spicexplorer/core/domains.py`](src/spicexplorer/core/domains.py). Abbreviated:

```yaml
project:
  name: CASCODE-OTA
  simulator: ngspice
  ws_root: ..                          # workspace root (see "Portable paths" below)
  netlist: spice/ota-improved.spice    # paths below are relative to ws_root
  outdir:  spice/temp_spice_out

  tech_spec:
    name: ihp-sg13g2
    constraints:                       # named limits, referenceable elsewhere in the file
      min_nfet_l: 0.18u
      max_nfet_l: 10u
      max_nfet_w: 200u

  pvt_corners:                         # PVT operating point(s) for the run
    - temp: 25
      corner: tt
      supply: 1.5

  dut_params:                          # design variables to size
    - name: X_DUT_M1M2_L
      min_val: min_nfet_l              # resolved from tech_spec.constraints
      max_val: max_nfet_l
    - name: X_DUT_M5_NG
      min_val: 1
      max_val: 5
      is_integer: true

  testbenches:                         # one or more enabled SPICE testbenches
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
    target_specs:                      # what gets measured and scored
      - name: ugf
        testbench: tb_ac
        sim_type: ac                   # dc | ac | tran | noise | op | noise_spectrum
        goal: exceed                   # exact | exceed | minimize
        target: 200e6
        tolerance: 10e6
        weight: 100.0
```

Top-level sections: **`project`** (metadata + workspace-relative paths), **`tech_spec`** (`name`, named technology `constraints`, and the planned **`pvt_map`** — see [Roadmap](#roadmap--direction)), **`pvt_corners`** (PVT operating points), **`dut_params`** (design variables), **`testbenches`** (enabled SPICE decks + param overrides), and **`optimizer_config`** (optimizer settings + `target_specs`).

Behaviors supported by the current loader/scorer:

- Engineering-string parsing — `0.18u`, `50f`, `1e6`, `100e9`.
- Resolving `dut_params` bounds from named constraints (`min_nfet_l`, `max_nfet_w`, …).
- Integer variables (`is_integer: true`) and log-scaled variables/specs (`log_scale: true`).
- Enabling/disabling individual testbenches (`enable: true|false`).
- Spec goals `exact` / `exceed` / `minimize`, with `tolerance`, `weight`, `error_type`, and `reward_type` shaping the fitness.

### Portable paths (`ws_root`)

`ws_root` is the workspace root that the `netlist`, `outdir`, testbench, and `schematic` paths are resolved against. `Project_Setup.from_yaml()` resolves it so projects are portable across machines:

- a **relative** path (the examples ship `ws_root: ..`) is resolved against the YAML file's own directory;
- an **absolute** path is used as-is (point it at a workspace outside the repo);
- an **omitted/empty** value defaults to the YAML's own directory; a leading `~` is expanded.

Because the example netlists are committed alongside their YAML, a fresh clone runs the examples **without editing any paths**.

---

## Reference examples

The examples live under [`examples/OTA/`](examples/OTA/), each organized as `xschem/` (schematics/symbols), `spice/` (exported DUT + testbench netlists), and `sizing/` (the YAML setup + runner scripts):

| Example | Path | Notes |
|---|---|---|
| **Cascode (telescopic) OTA** | [`examples/OTA/cascode`](examples/OTA/cascode) | The main, end-to-end reference (NEWCAS case study). |
| **5-transistor OTA** | [`examples/OTA/5t-ota`](examples/OTA/5t-ota) | Smaller, single-stage example. |
| **Folded-cascode OTA** | [`examples/OTA/folded_cascode`](examples/OTA/folded_cascode) | Additional topology; its multi-PVT `pvt_map` block is a work in progress (see Roadmap). |

The clearest end-to-end entry point is [`examples/OTA/cascode/ihp-sg13g2/sizing/nevergrad_single_obj_opt.py`](examples/OTA/cascode/ihp-sg13g2/sizing/nevergrad_single_obj_opt.py). Typical outputs are autosaved JSON checkpoints, timestamped log files (`logs/SpiceXplorer_<timestamp>.log`), and Plotly HTML reports for scores and metric traces.

---

## Project layout

```
src/spicexplorer/
  core/          # typed YAML schema (domains.py), engineering-value parsing, scoring (utils.py)
  spice_engine/  # NGSpice + spicelib wrappers: param injection, run management, raw/log extraction
  optimization/  # Circuit_Optimizer_Orchestrator_with_SPICE + stochastic (Nevergrad) / Ax / RL backends
  viz/           # Optimization_Log_Visualizer — Plotly HTML reports from checkpoints
  logging/       # setup_loggers(...) — named loggers; files always capture DEBUG
  demo/          # data bridge for the bundled case-study traces
ui/
  backend/       # thin FastAPI adapter over the library (no business logic)
  src/           # Next.js 15 Studio frontend (TypeScript, App Router)
examples/OTA/    # reference projects (cascode, 5t-ota, folded_cascode)
tests/           # fast smoke tests (+ slow tests that run real ngspice)
```

---

## Testing

```bash
uv run pytest                 # fast smoke tests, no SPICE simulation (~6 s)
uv run pytest -m slow         # include real ngspice simulation tests
uv run pytest tests/test_smoke_optimization.py -v
```

Tests marked `slow` (and several others) need `ngspice` on `PATH`; they are auto-skipped when it is absent. Tests that require the PDK fail by design on a PDK-less machine.

---

## Roadmap & direction

SpiceXplorer is actively evolving. Current directions:

- **Multi-PVT corner evaluation.** Today `pvt_corners` defines the operating point for a run. The next step is sweeping every candidate across a *set* of PVT corners and aggregating (e.g. worst-case / robust) scores, so designs are optimized to hold across process, voltage, and temperature rather than at a single nominal point.

  A planned **`pvt_map`** field under `tech_spec` will bind each corner label (the `corner:` referenced by `pvt_corners`) to its SPICE process-corner model and the `.lib` file to include, so the optimizer can re-simulate every candidate across all listed corners:

  ```yaml
  tech_spec:
    name: ihp-sg13g2
    constraints: { ... }
    pvt_map:                        # planned — not yet consumed by the loader
      tt:
        - name: tt
          spice_corner: mos_tt
          lib_name: cornerMOSlv.lib
      ss:
        - name: ss
          spice_corner: mos_ss
          lib_name: cornerMOSlv.lib
      ff:
        - name: ff
          spice_corner: mos_ff
          lib_name: cornerMOSlv.lib
  ```

  The `pvt_map` block in the folded-cascode example is the seed of this work-in-progress feature.
- **More optimizer backends.** Nevergrad is mature; the Ax (Bayesian) backend is available behind `--extra ax`, and the RL backend is being brought in (pending the external `rl_framework` dependency).
- **Simulator generality.** The `simulator` field and the SPICE-engine abstraction are designed to host more than NGSpice; NGSpice is the wired path today.

---

## Further reading

- **[ui/README.md](ui/README.md)** — UI architecture, full REST/SSE API, view-by-view guide, and troubleshooting.
- **[CLAUDE.md](CLAUDE.md)** — repository conventions and non-obvious constraints for contributors.
- **[doc/](doc/)** — design notes and the Studio integration plan.
