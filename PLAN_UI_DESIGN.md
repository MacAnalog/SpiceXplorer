# SpiceXplorer UI Design Plan

> **STATUS — the original 4-tab demo plan has been superseded by the Studio shell.**
> The UI shipped a full IDE-style **Studio workspace** (7 views, activity bar, rails,
> right rail, bottom panel, status bar, ⌘K palette, wizard overlay) on branch `dev/ui`.
> This document is now the *design intent + as-built map*. The canonical, always-current
> references are [CLAUDE.md](CLAUDE.md) (architecture + non-obvious constraints), `ui/README.md`
> (API table + run instructions), and [doc/PLAN_STUDIO_INTEGRATION.md](doc/PLAN_STUDIO_INTEGRATION.md)
> (the historical phased migration). Remaining work lives in [TODO.md](TODO.md).

## Goals

The UI is a **conference demo** for NEWCAS 2026 showcasing:
1. A guided project setup wizard that generates the YAML DSL from user input (netlist upload → YAML)
2. The paper's core contribution: **score shaping** (sigmoid vs. linear normalization), API-driven
3. A live optimization run on the cascode OTA case study, with real-time metric streaming
4. Post-run **topology performance exploration**: convergence, metric trade-off scatter, achievable limits

Constraints: easy local setup (two commands), looks polished in a presentation, no external hosting,
and **graceful degradation when the IHP PDK is absent** (live runs fall back to replay; everything else works).

---

## Implementation Status

### ✅ Completed (branch: `dev/ui`)

The UI evolved from the original 4-tab shell into a **persistent Studio workspace**. The center
views from the original plan (Setup / Score Shaping / Optimize / Explorer) survive, but they now
live inside a VS Code-style shell with an activity bar, per-activity left rails, an always-on right
rail, a bottom panel, a status bar, a ⌘K command palette, and a wizard overlay. Run history is a
first-class, comparable unit.

#### Infrastructure
- `pyproject.toml` — `[project.optional-dependencies] ui` group (`fastapi`, `uvicorn`, `python-multipart`); also `ax` extra for the Ax optimizer backend
- `ui/app_config.json` — demo config (default YAML + 4 preset CSV checkpoints + schematic SVG), repo-root-relative paths *(renamed from the planned `demo_config.json`)*
- `ui/.env.local` — `NEXT_PUBLIC_API_URL` (empty = same-origin; Next.js rewrites proxy `/api/*` to the backend, which is what makes VS Code Remote SSH work)
- `scripts/run_newcas_ui.sh` — starts FastAPI (**:8000**) + Next.js (**:4000**) concurrently, with stale-port cleanup and scoped `--reload`. **Port 4000, not 3000** — VS Code Remote SSH occupies :3000 on the server.

#### FastAPI Backend (`ui/backend/`) — thin adapter, no business logic
| Route module | Endpoints |
|---|---|
| `routes/config.py` | `GET /api/config` — demo config + inferred `score_fn` for each preset |
| `routes/project.py` | `POST /api/project/load`, `/validate`, `/generate` (wizard → YAML), `/parse-to-form` (YAML → wizard form) |
| `routes/netlist.py` | `POST /api/netlist/parse` — extract `.param name=val` from an uploaded netlist |
| `routes/score.py` | `POST /api/score` — sigmoid + linear penalties + sweep curve |
| `routes/optimize.py` | `POST /api/optimize/start`, `/stop/{id}`, `GET /api/optimize/stream/{id}` (SSE) |
| `routes/checkpoint.py` | `GET /api/checkpoint` (list), `/{id}`, `/envelope`, `/scatter` — resolves live autosaves too |
| `routes/schematic.py` | `GET /api/schematic` — serves SVG via FileResponse |
| `routes/xschem.py` | `GET /api/xschem/file`, `/resolve`, `/list`, `/project` — Xschem schematic browsing |
| `routes/sensitivity.py` | `GET /api/spec/{name}/sensitivity` — finite-difference metric sensitivity to DUT params |
| `routes/sanity.py` | `POST /api/sanity-check` — single-point simulation smoke test |
| `routes/env.py` | `GET /api/env` — `{ngspice_ok, pdk_ok, live_runs_enabled, ...}` for PDK-aware degradation |

Services: `score_service.py`, `checkpoint_reader.py` (dispatches `.json` via `Optimization_Log_Visualizer` and `.csv` from `NEWCAS_SUBMISSION_APPENDIX`; uses `.iterrows()` for dotted column names like `point.score`), `optimizer_runner.py` (background-thread optimizer → `asyncio.Queue` via `run_coroutine_threadsafe`; `_run_replay()` drips CSV rows at 50 ms; honors algorithm/budget/seed overrides; autosaves cumulative checkpoints for resume), `netlist_parser.py`, `yaml_generator.py` (form → YAML and `project_dict_to_form()` round-trip), `env_probe.py` (cheap ngspice + IHP PDK detection, no simulation).

#### Next.js Frontend (`ui/src/`) — the Studio shell
- `app/page.tsx` redirects to `/setup`; real views live under the `app/(studio)/` route group. `(studio)/layout.tsx` renders `StudioShell` once and never remounts across sibling segments, so the rails, status bar, and the live SSE stream survive navigation between views.
- **7 views** (single source of truth: `components/shell/nav.ts`):
  | View | Path | Gated on project? | Left rail |
  |---|---|---|---|
  | Setup | `/setup` | no | outline |
  | Score Shaping | `/scoring` | yes | specs |
  | Optimize | `/optimize` | yes | runs |
  | Explore | `/compare` | no | runs |
  | Schematic | `/schematic` | no | outline |
  | Pipeline | `/pipeline` | yes | specs |
  | Health | `/health` | no (gear) | outline |
- **Shell** (`components/shell/`): `StudioShell`, `ActivityBar`, `StudioTitleBar`, `TabStrip`, `StudioLeftRail` (+ `rails/`), `RightRail`, `BottomPanel`, `StatusBar`, `RunControl`, `Toolbar`, `nav.ts`.
- **Overlays** (`components/overlays/`): `CommandPalette` (⌘K), `WizardOverlay`.
- **Center views** (`components/tabs/`): `SetupTab`, `ScoreShapingTab`, `OptimizeTab`, `ExplorerTab`, `SchematicTab`, `PipelineView`, `HealthTab`.
- **Wizard** (`components/wizard/`): `WizardShell` + 7 steps (`BasicInfoStep`, `PDKRulesStep`, `DutParamsStep`, `PVTStep`, `TestbenchesStep`, `TargetSpecsStep`, `OptimizerStep`), `optimizer-registry.ts`, `wizard-controls.tsx`.
- **Schematic** (`components/schematic/`): `SchematicViewer`, `DeviceInspector` (with finite-difference `SensitivityChart`).
- **Charts** (`components/charts/`) — all via `PlotlyChart.tsx` base (dynamic import, SSR-disabled): `ScoreConvergenceChart`, `MetricConvergenceChart`, `PenaltyCurveChart`, `MetricScatterChart`, `MetricHistogramChart`, `SensitivityChart`.
- **5 Zustand stores** (`stores/`): `projectStore` (loaded/applied project), `runStore` (active run **+** the SSE `EventSource`, hoisted out of components; `history` persisted to localStorage), `explorerStore` (compare A/B), `uiStore` (navigation selection, panel toggles, `commandOpen`/`wizardOpen`), `wizardStore` (new-project form).
- `lib/api.ts` (single typed fetch client), `lib/launchRun.ts` (shared live-run launcher applying run-config overrides), `lib/utils.ts` (`cn`, `formatNumber`, `formatEng`), `types/api.ts` (mirrors every FastAPI response).
- **UI primitives** (`components/ui/`): `badge`, `button`, `panel`, `select`, `slider`, `separator`, `segmented`, `spec-chip`, `sparkline`, `stat`, `table`, `empty-state`.

#### Create Wizard — ✅ built (was the original plan's top open item)
The 7-step wizard generates a valid `project_setup.yaml` from scratch (netlist upload auto-detects
`.param` rows) and round-trips an existing YAML via "Edit in Wizard" (`POST /api/project/parse-to-form`).
It is launched from the shell as the `WizardOverlay` (⌘K → New Project, or the title bar).

#### PDK-aware degradation — ✅ built
`GET /api/env` drives a replay-only pill in the status bar. When `pdk_ok` is false, OptimizeTab
disables live Start and steers to Replay. Score Shaping, Compare/Explore on cached checkpoints, the
wizard, the Schematic SVG, and the Pipeline view all work without the PDK.

#### Demo data — `ui/app_config.json`
Four preset CSV traces in `examples/OTA/cascode/NEWCAS_SUBMISSION_APPENDIX/`:
`lhsde_sigmoid`, `lhssearch_sigmoid`, `lhsde_relabs`, `logbfgscmaplus_relabs`.
Default YAML: `examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml`.
Schematic SVG: `examples/OTA/cascode/ihp-sg13g2/xschem/ota-improved.svg`.

---

### ❌ Remaining work

The original plan's two big gaps (the Create Wizard and the netlist/generate backend routes) are
**done**. What remains is incremental polish and deeper exploration features, tracked in
[TODO.md](TODO.md). The genuinely-open highlights:

- **OptimizeTab score-function toggle**: the sigmoid/linear `Segmented` control exists in the UI but is **not yet wired to the backend** — `launchLiveRun()` doesn't send the choice, so the score function is still fixed by the YAML. (Algorithm/budget/seed overrides *are* wired.)
- **SetupTab "Apply" still re-reads from disk** (`api.loadProject(yamlPath)`), so unsaved Monaco edits aren't reflected on Apply. Should POST editor content (or save first).
- **Score Shaping → multi-metric explorer**: simultaneous multi-spec editing, equi-score contour overlay, contribution waterfall, worst-case-corner mode (TODO §3).
- **KPI / stat-card rows** per view (TODO §7) — `stat.tsx` exists but is currently only used in the RightRail.
- **Deeper Explore viz**: parallel coordinates, Pareto overlay, brushing & linking, design-point inspector for scatter points (TODO §8). *(Note: a device-parameter `SensitivityChart` already ships inside the Schematic `DeviceInspector` — distinct from the per-spec run-contribution view TODO §8 asks for.)*
- **Plot interactivity** (`displayModeBar: false` strips zoom/pan/download — TODO §6), **density toggles / collapsible panels** (TODO §9), and **misc polish** (report export, inline Monaco markers — TODO §10; the recent-runs rail already shipped).

---

## Architecture (as built)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Next.js 15 Frontend  (localhost:4000)                                     │
│  app/(studio)/layout.tsx → StudioShell  (persists across view navigation)  │
│  ┌──────────┬───────────────────────────────────────────────┬──────────┐  │
│  │ Activity │  TabStrip + swappable center view             │  Right   │  │
│  │   Bar    │  Setup·Scoring·Optimize·Explore·Schematic·    │  Rail    │  │
│  │  + rail  │  Pipeline·Health                              │ (specs/  │  │
│  │          │                                               │  run/    │  │
│  │          ├───────────────────────────────────────────────┤  params) │  │
│  │          │  Bottom panel (log / problems / diff)         │          │  │
│  └──────────┴───────────────────────────────────────────────┴──────────┘  │
│  Status bar · ⌘K CommandPalette · WizardOverlay                            │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ REST + SSE  (Next rewrites /api/* → backend)
┌───────────────────────────────────▼────────────────────────────────────────┐
│  FastAPI Backend  (localhost:8000)                                          │
│  /api/config ✅   /api/env ✅                                               │
│  /api/project/{load,validate,generate,parse-to-form} ✅                     │
│  /api/netlist/parse ✅                                                       │
│  /api/score ✅                                                               │
│  /api/optimize/{start,stop,stream} ✅ (SSE)                                 │
│  /api/checkpoint{,/{id},/envelope,/scatter} ✅                              │
│  /api/schematic ✅   /api/xschem/{file,resolve,list,project} ✅            │
│  /api/spec/{name}/sensitivity ✅   /api/sanity-check ✅                     │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ Python imports
           spicexplorer.core + .optimization + .spice_engine + .viz
                                    │ subprocess
                                  ngspice
```

---

## Directory Structure (as built)

```
ui/
  app_config.json              ← demo paths (default YAML + presets + schematic SVG)
  .env.local                   ← NEXT_PUBLIC_API_URL (empty = same-origin)
  backend/
    main.py  app_config.py
    routes/    config·project·netlist·score·optimize·checkpoint·
               schematic·xschem·sensitivity·sanity·env
    services/  score_service·checkpoint_reader·optimizer_runner·
               netlist_parser·yaml_generator·env_probe
  src/
    app/
      page.tsx                 ← redirects to /setup
      (studio)/
        layout.tsx             ← renders StudioShell (persists across views)
        setup/ scoring/ optimize/ compare/ schematic/ pipeline/ health/  (page.tsx each)
      api/yaml-text/route.ts   ← serve raw YAML for Monaco
    components/
      shell/   StudioShell·ActivityBar·StudioTitleBar·TabStrip·StudioLeftRail·
               RightRail·BottomPanel·StatusBar·RunControl·Toolbar·nav.ts·rails/
      overlays/ CommandPalette·WizardOverlay
      tabs/    SetupTab·ScoreShapingTab·OptimizeTab·ExplorerTab·
               SchematicTab·PipelineView·HealthTab
      wizard/  WizardShell·optimizer-registry·wizard-controls·steps/(7)
      schematic/ SchematicViewer·DeviceInspector
      charts/  PlotlyChart + Score/Metric Convergence·Penalty·Scatter·Histogram·Sensitivity
      ui/      badge·button·panel·select·slider·separator·segmented·
               spec-chip·sparkline·stat·table·empty-state
    lib/       api.ts·launchRun.ts·utils.ts·params.ts
    stores/    projectStore·runStore·explorerStore·uiStore·wizardStore
    types/     api.ts
```

---

## Technology Stack

| Layer | Choice |
|---|---|
| Frontend | Next.js 15 (React 19, TypeScript, App Router, `ui/src/`, `typedRoutes: true`) |
| Styling | TailwindCSS + custom shadcn-style primitives (zinc/indigo theming) |
| Icons | Lucide |
| State | Zustand 5 (**5 stores**) |
| YAML editor | `@monaco-editor/react` (dynamic import, SSR disabled) |
| Backend | FastAPI (Python) — thin adapter over `spicexplorer` |
| Live updates | SSE (`EventSource` on frontend, `StreamingResponse` on backend) |
| Charts | `react-plotly.js` (dynamic import, SSR disabled) |

**Start command:**
```bash
./scripts/run_newcas_ui.sh
# FastAPI → http://localhost:8000
# Next.js  → http://localhost:4000   (NOT 3000 — VS Code Remote SSH owns :3000)
```

---

## Key Implementation Decisions (for future reference)

1. **Shell persists, views swap**: an App-Router layout doesn't remount across sibling segments, so `StudioShell` (rails, status bar, **and the live SSE `EventSource` hoisted into `runStore`**) survives navigation between the 7 views. Adding/renaming a view happens only in `components/shell/nav.ts`.

2. **CSV vs JSON checkpoints**: `checkpoint_reader.py` handles both — `.csv` from `NEWCAS_SUBMISSION_APPENDIX` for demo replay, `.json` from new live runs (and live autosaves) via `Optimization_Log_Visualizer`. Detection is by file extension.

3. **CSV column dotted names**: Pandas `.itertuples()` sanitizes `point.score` into inaccessible attributes — the reader uses `.iterrows()` so `row.get("point.score")` works.

4. **SSE + threading**: the optimizer runs in a background thread; events push to an `asyncio.Queue` via `run_coroutine_threadsafe`. The SSE endpoint drains it. Replay drips CSV rows at 50 ms.

5. **Run-config overrides are ephemeral**: OptimizeTab's algorithm/budget/seed go to `POST /api/optimize/start` and are applied in-memory (`optimizer_runner._apply_overrides`) — the YAML on disk is never rewritten.

6. **PDK-aware degradation is first-class**: `GET /api/env` reports `pdk_ok`; the UI degrades to replay when the PDK is absent rather than erroring.

7. **Monaco / Plotly SSR**: both use `dynamic(() => import(...), { ssr: false })` to avoid `window is not defined`. Plotly axis titles must be `{ title: { text: "..." } }`, not a bare string.

---

## Demo Script (NEWCAS Presentation Flow)

1. **Setup** (1 min): ⌘K or activity bar → Setup. "Load demo…" → "OTA Cascode (default)". **Apply**. Walk the parsed summary (DUT params, testbenches, specs) in the right rail. *(Optionally: New Project wizard to show YAML generation from a netlist upload.)*

2. **Score Shaping** (2 min): Pick the `ugf` spec. Drag the value slider to ~187 MHz (just below the 200 MHz target). Point to linear P̂ (≈0.87) vs sigmoid P̂ (≈0.18). Show how the linear aggregate is dominated by current, masking the UGF miss. "This is the paper's core claim."

3. **Optimize** (2 min): Select a sigmoid preset from the Demo Replay dropdown → **Replay**. Watch score convergence and per-metric best-so-far animate via SSE in the center; spec chips in the right rail turn green as specs are met. *(On the PDK-equipped server: live Start instead of Replay.)*

4. **Explore** (2 min): Load Run A = `lhsde_sigmoid`, Run B = `lhsde_relabs`. Overlay convergence traces. Open the Metric Scatter (UGF vs. Current) to show the feasible cloud. Show the Performance Envelope table — the topology ceiling per spec.

5. **Schematic / Pipeline** (optional): show the Xschem SVG + Device Inspector sensitivity, and the read-only Pipeline DAG linking specs to testbenches.
