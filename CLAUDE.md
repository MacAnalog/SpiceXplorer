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

### Docker (portable, self-contained stack)

```bash
cp .env.example .env            # optional; defaults work
docker compose up --build       # backend (:8000) + frontend (:4000)
```

`compose.yaml` builds two images from [`docker/Dockerfile.backend`](docker/Dockerfile.backend) and [`ui/Dockerfile`](ui/Dockerfile). The backend **compiles ngspice from source** (`--enable-osdi`) and copies the vendored PDK subset in [`docker/pdk/`](docker/pdk/) — so live SPICE works with no host install. Non-obvious points:

- **Native multi-arch via `OSDI_MODE`** (build arg, default `compile`): `compile` builds openvaf (Rust + LLVM-18) and compiles the OSDI from the vendored Verilog-A (`docker/pdk/.../verilog-a/`) for the build's own arch — native on x86-64 **and** arm64, no emulation. `vendor` reuses the committed prebuilt **x86-64** `osdi/*.osdi` (fast, skips openvaf; x86-64/emulation only). The Dockerfile selects via `FROM osdi-${OSDI_MODE}`.
- **CPU-only torch**: `pyproject.toml` pins torch to the PyTorch CPU index on Linux (`[tool.uv.sources]` + `[[tool.uv.index]]`) — no CUDA. Re-run `uv lock` after touching torch deps.
- **The `agents` extra + API keys** are provisioned for a future LLM-agent layer: `INSTALL_AGENTS=true` build arg installs the extra; `ANTHROPIC_API_KEY` etc. are passed at runtime via `.env` → backend `environment:`, never baked into the image.
- **UID/GID + entrypoint**: [`docker/entrypoint-backend.sh`](docker/entrypoint-backend.sh) aligns the runtime user to host `UID`/`GID` (from `.env`) and `gosu`-drops privileges so `/work` bind-mount files aren't root-owned (Linux concern).
- Complementary to `run_newcas_ui.sh` (kept for native dev on a PDK-equipped machine); the container is the portable artifact.

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

**Backend** (`ui/backend/`): thin adapter — no business logic. Routes mirror the API table in `ui/README.md`. `optimizer_runner.py` runs the optimizer in a background thread and pushes events to an `asyncio.Queue` via `run_coroutine_threadsafe`; the SSE endpoint drains it. Replay mode drip-feeds CSV rows at 50 ms intervals. `services/env_probe.py` cheaply detects ngspice + the IHP PDK (no simulation) so the UI can degrade to replay when the PDK is absent.

**Frontend** (`ui/src/`): the app is a **Studio shell** — a persistent VS Code-style workspace, not a tab bar. `app/page.tsx` redirects to `/setup`; the real views live under the `app/(studio)/` route group, where `(studio)/layout.tsx` renders `StudioShell` (activity bar + left rail + tab strip + right rail + bottom panel + status bar + overlays) and each `<view>/page.tsx` segment renders one center view. Because a layout doesn't remount across sibling segments, the rails, status bar, and the live SSE stream survive navigation between views.

Five Zustand stores hold cross-view state: `projectStore` (loaded/applied project), `runStore` (the active run **and** the SSE `EventSource`, hoisted out of components so a run keeps streaming across views, plus `history` persisted to localStorage), `explorerStore` (compare A/B), `uiStore` (navigation selection: `selectedSpec`/`selectedRunId`, panel toggles, `commandOpen`/`wizardOpen`), and `wizardStore` (the new-project wizard form). `lib/api.ts` is the single typed fetch client; `types/api.ts` mirrors every FastAPI response shape.

`components/shell/` holds the shell pieces and `nav.ts` (the single source of truth for the 7 views — adding/renaming a view happens there only). `components/overlays/` holds `CommandPalette` (⌘K) and `WizardOverlay`. Charts (`components/charts/`) all use `react-plotly.js` via a shared `PlotlyChart.tsx` base with `dynamic(..., { ssr: false })` — same pattern for the Monaco editor. Both must stay SSR-disabled.

## Non-Obvious Constraints

**CSV column names with dots**: Pandas `.itertuples()` silently sanitizes column names like `point.score` into inaccessible attributes. `checkpoint_reader.py` uses `.iterrows()` so `row.get("point.score")` works. Keep this when touching that file.

**Plotly axis titles**: must be `{ title: { text: "..." } }` not a bare string — the TypeScript types require `Partial<DataTitle>`.

**CORS**: The backend uses `allow_origin_regex` matching any `localhost:<port>`. Do not replace it with a static origin list.

**Frontend `/api/*` must proxy to the backend — don't shadow it**: `next.config.mjs` rewrites `/api/:path*` to the backend, but the rewrite sits in `afterFiles`, so any local `ui/src/app/api/**/route.ts` handler takes precedence for that path. Such a handler runs in the **frontend** container, which — unlike native dev's shared filesystem — does **not** have the backend's `examples/`, checkpoints, etc., so it returns 404 under Docker (this caused the empty Setup editor). Add endpoints in `ui/backend/routes/`, not as Next route handlers; keep all `/api/*` flowing to the backend.

**Stale `.next` cache**: Running `npm run build` leaves production chunks that break the dev server. Delete `ui/.next` before restarting after a build.

**`ws_root` in YAML**: `Project_Setup.from_yaml()` resolves `ws_root` so the committed examples are portable across machines. A **relative** path (the examples ship `ws_root: ..`) is resolved against the YAML file's own directory; an **absolute** path is used as-is (point it at an out-of-repo workspace); an **omitted/empty** value defaults to the YAML's directory. A leading `~` is expanded. The example netlists are committed inside the repo alongside the YAML, so a fresh clone runs without editing any paths. This is **independent of PDK availability**: live SPICE runs and the sanity check only work where ngspice **and** the IHP `ihp-sg13g2` PDK are installed (the server); on a PDK-less machine they fail by design and the UI shows "PDK missing — replay only".

**PDK-aware degradation**: `GET /api/env` reports `{ngspice_ok, pdk_ok, live_runs_enabled, ...}`. When `pdk_ok` is false the status bar shows the replay-only pill and OptimizeTab disables live Start (steering to Replay). Score Shaping, Compare/Explore on cached checkpoints, the wizard, and the Pipeline view all work without the PDK.

**Run-config overrides are ephemeral**: OptimizeTab's algorithm/budget/seed are sent to `POST /api/optimize/start` and applied in-memory to the loaded `Project_Setup` (`optimizer_runner._apply_overrides`) — the YAML on disk is never rewritten.

**`dut_param.freeze` defaults to `False`**: an omitted `freeze` key means "optimize this param" — matching the wizard default and the historical sweep-everything behavior. Set `freeze: true` to exclude a param from the search space (`NevergradMixin.parameterize` skips it and injects its `val`/`init`, if given). `Project_Setup.from_yaml()` now also **rejects duplicate `dut_param` names** (they previously collapsed silently to one search dimension). The `freeze_to: <value>` YAML key is still parsed-but-ignored — wiring it up is deferred with the PVT / multi-corner work.

**App-Router typed routes**: `next.config.mjs` sets `typedRoutes: true`, so `router.push("/foo")` needs `"/foo" as Route` (import `type { Route } from "next"`).

## Key Files for Common Tasks

| Task | Where to look |
|---|---|
| Change YAML DSL schema | `src/spicexplorer/core/domains.py` |
| Add a new API endpoint | `ui/backend/routes/` + `ui/src/lib/api.ts` + `ui/src/types/api.ts` |
| Change scoring math | `src/spicexplorer/core/utils.py` + `ui/backend/services/score_service.py` |
| Add a new chart | `ui/src/components/charts/` — extend `PlotlyChart.tsx` |
| Add / rename a Studio view | `ui/src/components/shell/nav.ts` + new `ui/src/app/(studio)/<view>/page.tsx` |
| Change the shell layout | `ui/src/components/shell/StudioShell.tsx` (+ ActivityBar/TabStrip/RightRail/BottomPanel/StatusBar) |
| Command palette / ⌘K | `ui/src/components/overlays/CommandPalette.tsx` |
| Live-run SSE / run history | `ui/src/stores/runStore.ts` |
| Demo preset checkpoints | `ui/app_config.json` (repo-root-relative paths) |
| Reference optimization run | `examples/OTA/cascode/ihp-sg13g2/sizing/nevergrad_single_obj_opt.py` |
| Full Studio migration plan | `doc/PLAN_STUDIO_INTEGRATION.md` |
