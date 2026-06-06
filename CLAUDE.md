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
| `core/domains.py` | Typed dataclasses for the full YAML DSL (`Project_Setup`, `DUT_Param`, `Target_Spec`, etc.); engineering-string parsing (`parse_value`: `0.18u`, `50f`) |
| `core/utils.py` | `compute_relative_absolute_error`, `compute_relative_sigmoid_error` (eng-string parsing is `parse_value` in `core/domains.py`) |
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

**Frontend** (`ui/src/`): the app is a **Studio shell** — a persistent VS Code-style workspace, not a tab bar. `app/page.tsx` redirects to `/setup`; the real views live under the `app/(studio)/` route group, where `(studio)/layout.tsx` renders `StudioShell` (activity bar + left rail + center view + right rail + bottom panel + status bar + overlays) and each `<view>/page.tsx` segment renders one center view. Because a layout doesn't remount across sibling segments, the rails, status bar, and the live SSE stream survive navigation between views. The **vertical ActivityBar is the sole top-level nav** (the redundant horizontal top tab strip was removed); each view renders its own in-view sub-tabs via `SubTabStrip` (e.g. Setup's Load/Wizard) — a view with one section renders no bar.

Five Zustand stores hold cross-view state: `projectStore` (loaded/applied project), `runStore` (the active run **and** the SSE `EventSource`, hoisted out of components so a run keeps streaming across views, plus `history` persisted to localStorage), `explorerStore` (compare A/B), `uiStore` (navigation selection: `selectedSpec`/`selectedRunId`, panel toggles, `commandOpen`/`wizardOpen`), and `wizardStore` (the new-project wizard form). `lib/api.ts` is the single typed fetch client; `types/api.ts` mirrors every FastAPI response shape.

`components/shell/` holds the shell pieces and `nav.ts` (the single source of truth for the 8 views — `PRIMARY_VIEWS` top group, `UTILITY_VIEWS` bottom group = Pipeline + Manual Sim, `SETTINGS_VIEW` = Health gear; adding/renaming a view happens there + a new `(studio)/<view>/page.tsx`). `nav.ts` shortcuts must stay 1..8 unique and `CommandPalette` keys off `ALL_VIEWS` with a `/^[1-8]$/` ⌘-number regex — widen both together if you add a view. `components/overlays/` holds `CommandPalette` (⌘K) and `WizardOverlay`. Charts (`components/charts/`) all use `react-plotly.js` via a shared `PlotlyChart.tsx` base with `dynamic(..., { ssr: false })` — same pattern for the Monaco editor. Both must stay SSR-disabled.

## Non-Obvious Constraints

**CSV column names with dots**: Pandas `.itertuples()` silently sanitizes column names like `point.score` into inaccessible attributes. `checkpoint_reader.py` uses `.iterrows()` so `row.get("point.score")` works. Keep this when touching that file.

**Plotly axis titles**: must be `{ title: { text: "..." } }` not a bare string — the TypeScript types require `Partial<DataTitle>`.

**CORS**: The backend uses `allow_origin_regex` matching any `localhost:<port>`. Do not replace it with a static origin list.

**Frontend `/api/*` must proxy to the backend — don't shadow it**: `next.config.mjs` rewrites `/api/:path*` to the backend, but the rewrite sits in `afterFiles`, so any local `ui/src/app/api/**/route.ts` handler takes precedence for that path. Such a handler runs in the **frontend** container, which — unlike native dev's shared filesystem — does **not** have the backend's `examples/`, checkpoints, etc., so it returns 404 under Docker (this caused the empty Setup editor). Add endpoints in `ui/backend/routes/`, not as Next route handlers; keep all `/api/*` flowing to the backend.

**Stale `.next` cache**: Running `npm run build` leaves production chunks that break the dev server. Delete `ui/.next` before restarting after a build.

**`ws_root` in YAML**: `Project_Setup.from_yaml()` resolves `ws_root` so the committed examples are portable across machines. A **relative** path (the examples ship `ws_root: ..`) is resolved against the YAML file's own directory; an **absolute** path is used as-is (point it at an out-of-repo workspace); an **omitted/empty** value defaults to the YAML's directory. A leading `~` is expanded. The example netlists are committed inside the repo alongside the YAML, so a fresh clone runs without editing any paths. This is **independent of PDK availability**: live SPICE runs and the sanity check only work where ngspice **and** the IHP `ihp-sg13g2` PDK are installed (the server); on a PDK-less machine they fail by design and the UI shows "PDK missing — replay only".

**PDK-aware degradation**: `GET /api/env` reports `{ngspice_ok, pdk_ok, live_runs_enabled, ...}`. When `pdk_ok` is false the status bar shows the replay-only pill and OptimizeTab disables live Start (steering to Replay). Score Shaping, Compare/Explore on cached checkpoints, the wizard, and the Pipeline view all work without the PDK.

**Where live SPICE actually runs (verification)**: the **native host (this Mac) has ngspice but NOT the IHP PDK** — so native runs and `test_ngspice_sanity_check` fail by design (`pdk_ok: false`). The **Docker backend container HAS both** (the Dockerfile compiles ngspice and vendors the PDK): `pdk_ok: true`, `live_runs_enabled: true`, ngspice at `/opt/ngspice/bin/ngspice`, `PDK_ROOT=/opt/pdk` (`ihp-sg13g2`). So to verify anything needing a real simulation (PVT corners driving the sim, `/api/simulate/once`, sanity, a live run), use the **running container**: its API is on `localhost:8000` when `docker compose up` is running, or `docker compose exec backend <cmd>` (e.g. `pytest -m slow`). **Never bind a Docker-mapped port from the host** (`:8000` backend / `:4000` frontend) — a native uvicorn on `:8000` collides with the container and crash-loops it. The committed image can be stale; `docker compose up --build` to test current-branch code.

**Run-config overrides are ephemeral**: OptimizeTab's algorithm/budget/seed (and the PVT `active_corner`) are sent to `POST /api/optimize/start` and applied in-memory to the loaded `Project_Setup` (`optimizer_runner._apply_overrides`) — the YAML on disk is never rewritten.

**PVT corners (Phase 1) actually drive the sim**: a top-level `pvt:` block (`PVTConfig` → named `Corner`s with `model_includes`/`temp`/`supplies`) makes corners first-class. `Project_Setup.from_yaml()` calls `_normalize_pvt_block` to desugar `process_bundles`, a singular `supply: {node,value}`, and eng-strings **before** dacite. The optimizer applies `pvt.get_active()` **once** in `Spice_Base_Optimizer.__post_init__` via `NGSpice_Wrapper.apply_corner()` — the **only** ngspice-specific corner seam (strips the hardcoded `.lib`, injects the corner's ordered cross-family includes, sets `.options temp=`, overrides supply `.param`s; idempotent). The optimize loop/scorer/`simulate_circuit` are untouched, so a single-corner run is a strict superset of the legacy "netlist hardcodes the corner" behavior (`pvt: None`). The legacy `tech_spec.pvt_map` / flat `pvt_corners` stay **display-only**. Phase 2 (multi-corner `{tb × corner}` aggregation) is deferred — see `PVT_plan.md`. The UI surfaces corners via `CornerSelect` (Run popover, Optimize toolbar, Health check) and the wizard's PVTStep (emits inline `model_includes`, round-tripped through `yaml_generator._pvt_block_to_form`).

**Manual single sim (`POST /api/simulate/once`)**: evaluate ONE chosen design point through the optimizer's `evaluate(params, append_to_log=False)` primitive (same scoring as a real trial). Mode B = explicit param vector; Mode A = a checkpoint point (best by argmax, or a given index). Mode B `params` values may be **engineering strings** (`"250u"`, `"0.18u"`) or numbers — the route parses them **server-side** via `core/domains.parse_value` (the request type is `dict[str, str | float]`; a malformed value returns a per-field error). PDK-gated; isolated output subfolder (`outdir/manual_sim`) so it can't clobber a live run. UI: its **own `/manual` view** (`tabs/ManualSimTab` wrapping `pvt/ManualSimPanel`), in the ActivityBar utility group — no longer embedded in OptimizeTab. Do **not** route this through `parameterize()`/`ask()` (that sims a *random* point — the `sanity.py` gap).

**`dut_param.freeze` defaults to `False`**: an omitted `freeze` key means "optimize this param" — matching the wizard default and the historical sweep-everything behavior. Set `freeze: true` to exclude a param from the search space (`NevergradMixin.parameterize` skips it and injects its `val`/`init`, if given). `Project_Setup.from_yaml()` now also **rejects duplicate `dut_param` names** (they previously collapsed silently to one search dimension). The `freeze_to: <value>` YAML key is still parsed-but-ignored — wiring it up is deferred with the Phase 2 multi-corner work (PVT Phase 1 single-corner has landed; see the PVT note above).

**App-Router typed routes**: `next.config.mjs` sets `typedRoutes: true`, so `router.push("/foo")` needs `"/foo" as Route` (import `type { Route } from "next"`).

## Key Files for Common Tasks

| Task | Where to look |
|---|---|
| Change YAML DSL schema | `src/spicexplorer/core/domains.py` |
| Add a new API endpoint | `ui/backend/routes/` + `ui/src/lib/api.ts` + `ui/src/types/api.ts` |
| Change scoring math | `src/spicexplorer/core/utils.py` + `ui/backend/services/score_service.py` |
| Add a new chart | `ui/src/components/charts/` — extend `PlotlyChart.tsx` |
| Add / rename a Studio view | `ui/src/components/shell/nav.ts` (`PRIMARY_VIEWS`/`UTILITY_VIEWS`; keep ⌘ shortcuts 1..8 + the `CommandPalette` `/^[1-8]$/` regex in sync) + new `ui/src/app/(studio)/<view>/page.tsx` |
| Add in-view sub-tabs | `ui/src/components/shell/SubTabStrip.tsx` (generic, local `useState` per view, mounted in the view's own toolbar; e.g. `tabs/SetupTab.tsx`) |
| Change the shell layout | `ui/src/components/shell/StudioShell.tsx` (+ ActivityBar/RightRail/BottomPanel/StatusBar) |
| Command palette / ⌘K | `ui/src/components/overlays/CommandPalette.tsx` |
| Edit target specs (ephemeral) | `ui/src/components/tabs/ScoreShapingTab.tsx` (`SpecEditor`) → `spec_overrides` on `POST /api/score` → `score_service.apply_spec_overrides` (mutates the freshly-loaded project in-memory; never writes YAML, not threaded into live runs) |
| Live-run SSE / run history | `ui/src/stores/runStore.ts` |
| Demo preset checkpoints | `ui/app_config.json` (repo-root-relative paths) |
| Reference optimization run | `examples/OTA/cascode/ihp-sg13g2/sizing/nevergrad_single_obj_opt.py` |
| PVT corner schema / apply | `src/spicexplorer/core/domains.py` (`PVTConfig` — incl. duplicate-corner-name rejection in `__post_init__` —, `_normalize_pvt_block`) + `spice_engine/spicelib.py` (`apply_corner`) |
| PVT corner UI | `ui/src/components/pvt/` (`CornerSelect`, `ManualSimPanel`) + `lib/pvt.ts` |
| Manual single sim | `ui/backend/routes/simulate.py` + `ui/src/app/(studio)/manual/page.tsx` → `ui/src/components/tabs/ManualSimTab.tsx` → `ui/src/components/pvt/ManualSimPanel.tsx` |
| PVT design + Phase 2 plan | `PVT_plan.md` |
| Full Studio migration plan | `doc/PLAN_STUDIO_INTEGRATION.md` |
