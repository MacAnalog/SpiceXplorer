# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Python (managed with `uv`)

```bash
uv sync                        # install default dev environment
uv sync --extra ui             # include FastAPI/uvicorn for the web UI
uv sync --extra ax             # include Ax optimizer backend (Python 3.11+ only)

uv run pytest                  # fast tests only (no SPICE simulation, ~6 s)
uv run pytest -m slow          # include real ngspice simulation tests
uv run pytest tests/test_smoke_optimization.py -v   # single file
uv run python examples/OTA/cascode/ihp-sg13g2/sizing/nevergrad_single_obj_opt.py
```

Tests marked `slow` require `ngspice` in PATH and run a real simulation. Tests are auto-skipped if `ngspice` is absent.

### UI (Next.js + FastAPI)

```bash
./scripts/run_newcas_ui.sh              # start both processes (backend :8000, frontend :4000)
LOG_LEVEL=DEBUG ./scripts/run_newcas_ui.sh   # verbose library logging

# Manual startup
LOG_LEVEL=INFO uv run --extra ui uvicorn ui.backend.main:app --reload --port 8000
cd ui && npm run dev -- -p 4000

# Frontend tooling
cd ui && npm run typecheck      # tsc --noEmit
cd ui && npm run lint           # eslint, zero warnings allowed
cd ui && npm run build          # type-checks and builds; delete ui/.next before restarting dev
```

Frontend runs on port 4000 (not 3000) because VS Code Remote SSH occupies 3000 on the server.

## Architecture

### Python library (`src/spicexplorer/`)

The central workflow is: YAML file → `Project_Setup.from_yaml()` → `Circuit_Optimizer_Orchestrator_with_SPICE` → NGSpice testbench execution → scorer → checkpoints + Plotly reports.

| Module | Role |
|---|---|
| `core/domains.py` | Typed dataclasses for the full YAML DSL (`Project_Setup`, `DUT_Param`, `Target_Spec`, etc.) |
| `core/utils.py` | Engineering-string parsing (`0.18u`, `50f`), `compute_relative_absolute_error`, `compute_relative_sigmoid_error` |
| `spice_engine/` | NGSpice + spicelib wrappers: param injection, subprocess management, raw/log extraction |
| `optimization/orchestrator.py` | `Circuit_Optimizer_Orchestrator_with_SPICE` — drives the optimization loop, calls testbenches, scores results |
| `optimization/stochastic/` | Nevergrad backend (`Nevergrad_Spice_Single_Objective` and subclasses) |
| `optimization/rl/` | RL backend (present but not the primary documented path) |
| `viz/` | `Optimization_Log_Visualizer` — reads JSON checkpoints for Plotly HTML reports |
| `logging/` | `setup_loggers(console_level=...)` — configures named loggers; log files always capture DEBUG |

`project.optimizer_config.target_specs` returns a `ListTargetSpec` object, not a plain list — use `.targets` to access the underlying list.

### Web UI

```
Browser (localhost:4000)
  └─ Next.js 15 (TypeScript, App Router, ui/src/)
        │ REST + SSE
  FastAPI (localhost:8000)  ←  ui/backend/
        │ Python imports
  spicexplorer.core + .optimization + .spice_engine
        │ subprocess
  ngspice
```

**Backend** (`ui/backend/`): thin adapter — no business logic. Routes mirror the API table in `ui/README.md`. `optimizer_runner.py` runs the optimizer in a background thread and pushes events to an `asyncio.Queue` via `run_coroutine_threadsafe`; the SSE endpoint drains it. Replay mode drip-feeds CSV rows at 50 ms intervals.

**Frontend** (`ui/src/`): three Zustand stores (`projectStore`, `runStore`, `explorerStore`) hold all cross-tab state. `lib/api.ts` is the single typed fetch client — all backend calls go through it. `types/api.ts` mirrors every FastAPI response shape in TypeScript.

Charts (`components/charts/`) all use `react-plotly.js` via a shared `PlotlyChart.tsx` base with `dynamic(..., { ssr: false })` — same pattern for the Monaco editor. Both must stay SSR-disabled.

## Non-Obvious Constraints

**CSV column names with dots**: Pandas `.itertuples()` silently sanitizes column names like `point.score` into inaccessible attributes. `checkpoint_reader.py` uses `.iterrows()` so `row.get("point.score")` works. Keep this when touching that file.

**Plotly axis titles**: must be `{ title: { text: "..." } }` not a bare string — the TypeScript types require `Partial<DataTitle>`.

**CORS**: The backend uses `allow_origin_regex` matching any `localhost:<port>`. Do not replace it with a static origin list.

**Stale `.next` cache**: Running `npm run build` leaves production chunks that break the dev server. Delete `ui/.next` before restarting after a build.

**`ws_root` in YAML**: The cascode example uses an absolute machine-specific path. It must be updated to the local checkout before running the example.

## Key Files for Common Tasks

| Task | Where to look |
|---|---|
| Change YAML DSL schema | `src/spicexplorer/core/domains.py` |
| Add a new API endpoint | `ui/backend/routes/` + `ui/src/lib/api.ts` + `ui/src/types/api.ts` |
| Change scoring math | `src/spicexplorer/core/utils.py` + `ui/backend/services/score_service.py` |
| Add a new chart | `ui/src/components/charts/` — extend `PlotlyChart.tsx` |
| Demo preset checkpoints | `ui/app_config.json` (repo-root-relative paths) |
| Reference optimization run | `examples/OTA/cascode/ihp-sg13g2/sizing/nevergrad_single_obj_opt.py` |
