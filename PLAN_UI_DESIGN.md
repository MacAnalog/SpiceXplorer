# SpiceXplorer UI Design Plan

## Goals

The UI is a **conference demo** for NEWCAS 2026 showcasing:
1. A guided project setup wizard that generates the YAML DSL from user input (netlist upload → YAML)
2. The paper's core contribution: **score shaping** (sigmoid vs. linear normalization), API-driven
3. A live optimization run on the cascode OTA case study, with real-time metric streaming
4. Post-run **topology performance exploration**: convergence, metric trade-off scatter, achievable limits

Constraints: easy local setup (two commands), looks polished in a presentation, no external hosting.

---

## Implementation Status

### ✅ Completed (branch: `dev/ui`, single commit `2e82124`)

#### Infrastructure
- `pyproject.toml` — added `[project.optional-dependencies] ui` group (`fastapi`, `uvicorn`, `python-multipart`)
- `uv.lock` — updated
- `.gitignore` — added `ui/node_modules/`, `ui/.next/`, exclusions for package.json
- `ui/demo_config.json` — configurable demo paths (repo-root-relative)
- `ui/.env.local` — `NEXT_PUBLIC_API_URL=http://localhost:8000`
- `scripts/run_newcas_ui.sh` — starts both FastAPI (:8000) and Next.js (:3000) concurrently

#### FastAPI Backend (`ui/backend/`)
| File | Description |
|---|---|
| `main.py` | App entry point, CORS for localhost:3000, all routers at `/api` prefix |
| `app_config.py` | Reads `ui/demo_config.json`, resolves repo-root-relative paths |
| `routes/config.py` | `GET /api/config` — returns demo config to hydrate frontend dropdowns |
| `routes/project.py` | `POST /api/project/load`, `POST /api/project/validate` |
| `routes/score.py` | `POST /api/score` — sigmoid + linear penalties |
| `routes/optimize.py` | `POST /api/optimize/start`, `POST /api/optimize/stop/{id}`, `GET /api/optimize/stream/{id}` (SSE) |
| `routes/checkpoint.py` | `GET /api/checkpoint`, `GET /api/checkpoint/{id}`, `/envelope`, `/scatter` |
| `routes/schematic.py` | `GET /api/schematic` — serves SVG via FileResponse |
| `services/score_service.py` | `compute_score()` — calls `compute_relative_absolute_error` and `compute_relative_sigmoid_error` from `spicexplorer.core.utils`, builds penalty curve for PenaltyCurveChart |
| `services/checkpoint_reader.py` | `read_checkpoint()` — unified dispatcher for `.json` (via `Optimization_Log_Visualizer`) and `.csv` (NEWCAS_SUBMISSION_APPENDIX traces). Fixed: uses `df.iterrows()` not `itertuples()` to handle dotted column names like `point.score`. Also computes envelope and scatter. |
| `services/optimizer_runner.py` | `_StreamingOpt` subclass of `Nevergrad_Spice_Single_Objective` that pushes events to an `asyncio.Queue` via `run_coroutine_threadsafe`. `_run_replay()` drip-feeds CSV rows at 50 ms intervals. `_runs: dict[str, RunState]` module-level registry. |

#### Next.js Frontend (`ui/src/`)
| File | Description |
|---|---|
| `app/page.tsx` | 4-tab shell: Setup, Score Shaping, Optimize, Explorer. Indigo underline nav, running status pill. Loads `api.config()` on mount. Score Shaping and Optimize tabs disabled until project is applied. |
| `app/api/yaml-text/route.ts` | `GET /api/yaml-text?path=...` — serves raw YAML file content for Monaco editor |
| `lib/api.ts` | Typed fetch client; `BASE = NEXT_PUBLIC_API_URL ?? "http://localhost:8000"`. All backend calls typed. |
| `lib/utils.ts` | `cn()`, `formatNumber()`, `statusForGoal()` (kept from Codex era) + new `formatEng()` for µ/n/p/k/M/G formatting |
| `types/api.ts` | Full TypeScript mirror of FastAPI response shapes: `ProjectSummary`, `ScoreResponse`, `ScoreCurve`, `SSEEvent`, `CheckpointData`, `CheckpointMeta`, `EnvelopeEntry`, `ScatterPoint`, `DemoConfig` |
| `stores/projectStore.ts` | yaml, yamlPath, summary, validationErrors, isApplied; `apply()` unlocks other tabs |
| `stores/runStore.ts` | runId, isRunning, isReplay, budget, events, bestMetrics, bestParams, currentIter |
| `stores/explorerStore.ts` | availableCheckpoints, runA/B, envelopeA, scatterMetricX/Y, selectedMetric |

**UI primitives** (`components/ui/`): `badge.tsx` (pass/fail/neutral/warning/indigo variants), `select.tsx`, `slider.tsx`, `separator.tsx`, `button.tsx`, `panel.tsx`

**Chart components** (`components/charts/`) — all use `react-plotly.js` via `PlotlyChart.tsx` base (dynamic import, `displayModeBar: false`, zinc/indigo/emerald theming):
- `ScoreConvergenceChart.tsx` — raw + best-so-far score lines, emerald zero line
- `MetricConvergenceChart.tsx` — best-so-far per metric, amber dashed target line
- `PenaltyCurveChart.tsx` — sigmoid vs linear penalty curves, vertical markers for target and current value
- `MetricScatterChart.tsx` — feasibility-colored scatter (zinc = infeasible, indigo = feasible), dashed target lines
- `MetricHistogramChart.tsx` — overlaid histograms, amber target line

**Tab components** (`components/tabs/`):
- `SetupTab.tsx` — Monaco YAML editor (600 ms debounced validation), Load Demo dropdown, Upload, Validate, Apply. Right panel: project meta grid, testbenches list, DUT params table (`formatEng()`), target specs table.
- `ScoreShapingTab.tsx` — Spec selector + value slider (range = target ± 3×range). Debounced 150 ms `POST /api/score`. PenaltyCurveChart updates live. Per-spec breakdown table with sigmoid/linear P̂ columns, aggregate F(x) footer row. Auto callout identifying highest-penalty spec.
- `OptimizeTab.tsx` — Algorithm dropdown (LhsDE, LHSSearch, LogBFGSCMAPlus), budget input, Demo Checkpoint replay dropdown. Start/Stop with SSE via `EventSource`. Progress bar. ScoreConvergenceChart + MetricConvergenceChart (metric dropdown). Live spec status chips (emerald/red) from `bestMetrics`.
- `ExplorerTab.tsx` — Run A/B checkpoint selectors + Load buttons. Row 1: ScoreConvergenceChart overlay + MetricConvergenceChart (metric dropdown). Row 2: MetricScatterChart (X/Y dropdowns) + Performance Envelope table. Row 3: MetricHistogramChart + Best Design Params table. Row 4: Spec Summary table (goal, target, Run A best, Run B best, pass/fail per run).

#### Demo Data
`ui/demo_config.json` points to the three CSV traces in `examples/OTA/cascode/NEWCAS_SUBMISSION_APPENDIX/`:
- `sigmoid_de` — `CASCODE-OTA_LhsDE_2026-02-07_10-54-54_sigmoid-loss.csv`
- `linear_de` — `CASCODE-OTA_LhsDE_2026-02-07_14-53-23_relAbs-loss.csv`
- `blind_search` — `CASCODE-OTA_LHSSearch_2026-02-07_10-53-21_blind-search_sigmoid-loss.csv`

Default YAML: `examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml`

---

### ❌ Not Yet Implemented

#### 1. Create Wizard (Tab 1, Mode B) — highest priority for the demo

The Setup tab currently only has Load/Edit mode. The Create Wizard is the step-by-step form that generates a YAML from scratch. The plan calls for:

**Frontend** — new directory `ui/src/components/wizard/`:
- `WizardShell.tsx` — step navigator, progress indicator, back/next buttons, live YAML preview pane on the right
- `steps/BasicInfoStep.tsx` — project name, description, simulator dropdown, workspace root
- `steps/PDKRulesStep.tsx` — technology name, add key/value constraint rows (e.g., `min_nfet_w = 0.18µ`)
- `steps/DutParamsStep.tsx` — upload DUT netlist → calls `POST /api/netlist/parse`, pre-fills param rows. Each row: name, min, max (dropdown from PDK constraints or raw), is_integer, log_scale, freeze toggles.
- `steps/PVTStep.tsx` — add rows: temp (°C), corner string, supply voltage
- `steps/TestbenchesStep.tsx` — add/remove testbench cards: name, netlist upload, parameter rows, enable toggle
- `steps/TargetSpecsStep.tsx` — add/remove spec rows (accordion): name, testbench dropdown, goal, target, tolerance, range, weight, error_type, reward_type, enable
- `steps/OptimizerStep.tsx` — algorithm dropdown, budget, random seed

The wizard needs a "Save YAML" button that writes the generated file to the workspace root, then switches SetupTab to Load/Edit mode with the new file loaded.

**SetupTab.tsx** needs a segmented control at the top to toggle between Load/Edit and Create Wizard modes.

**Backend** — two new routes and two new services:
- `routes/netlist.py` — `POST /api/netlist/parse`: accept an uploaded `.spice` file, extract `.param name=val` lines via regex, return `[{name, default_val}]`
- `routes/project.py` (extend) — `POST /api/project/generate`: accept wizard form data as JSON, return generated YAML string (via `yaml_generator.py`)
- `services/netlist_parser.py` — regex over `.param` lines; no full SPICE parser needed
- `services/yaml_generator.py` — converts wizard form data dict into a valid `project_setup.yaml` string, using PyYAML + the known YAML DSL schema

#### 2. Minor UX gaps in existing tabs

- **SetupTab**: The "Apply" button currently re-calls `api.loadProject()` even if the YAML was just manually edited (it re-reads from disk, not from the editor). Should POST the current editor content to `/api/project/load` with the YAML text, not the path — or save to a temp file first. This means edits in the Monaco editor that aren't saved to disk won't be reflected after Apply.
- **OptimizeTab**: The algorithm selection currently doesn't affect the live run — `api.startRun()` sends `yaml_path` and `budget` but not the chosen algorithm or score function. The backend `optimizer_runner.py` needs to accept and use those.
- **Score function toggle** described in the plan (sigmoid vs linear radio button) is not in the current OptimizeTab — the score function is fixed by what's in the YAML.

#### 3. Demo checkpoint format

`demo_config.json` currently uses CSV paths from `NEWCAS_SUBMISSION_APPENDIX`. The original plan described using JSON checkpoint files (`demo_sigmoid_de.json`, etc.). The CSV reader in `checkpoint_reader.py` handles both formats but JSON checkpoints from actual runs would give richer data. When new runs are recorded with the live optimizer, their JSON checkpoints appear automatically in the Explorer tab's list.

---

## Architecture (as built)

```
┌─────────────────────────────────────────────────────────┐
│  Next.js Frontend  (localhost:3000)                      │
│  ui/src/app/page.tsx  ← 4-tab shell                     │
│  ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌────────────┐  │
│  │  Setup   │ │ Shaping │ │ Optimize │ │  Explorer  │  │
│  │ (Load/   │ │         │ │          │ │            │  │
│  │  Edit)   │ │         │ │          │ │            │  │
│  │ Wizard ❌│ │         │ │          │ │            │  │
│  └──────────┘ └─────────┘ └──────────┘ └────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │ REST + SSE
┌───────────────────────▼─────────────────────────────────┐
│  FastAPI Backend  (localhost:8000)                       │
│  /api/config            ✅                               │
│  /api/project/load      ✅                               │
│  /api/project/validate  ✅                               │
│  /api/project/generate  ❌ (wizard not built)            │
│  /api/netlist/parse     ❌ (wizard not built)            │
│  /api/score             ✅                               │
│  /api/optimize/*        ✅ (start/stop/stream SSE)       │
│  /api/checkpoint/*      ✅ (list/load/envelope/scatter)  │
│  /api/schematic         ✅                               │
└───────────────────────┬─────────────────────────────────┘
                        │ Python imports
           spicexplorer.core + .optimization + .viz
```

---

## Directory Structure (as built)

```
ui/
  demo_config.json              ← configurable demo paths (repo-root-relative)
  .env.local                    ← NEXT_PUBLIC_API_URL=http://localhost:8000
  backend/
    main.py
    app_config.py
    routes/
      config.py
      project.py
      score.py
      optimize.py
      checkpoint.py
      schematic.py
    services/
      score_service.py
      checkpoint_reader.py
      optimizer_runner.py
      [netlist_parser.py]       ← ❌ not yet
      [yaml_generator.py]       ← ❌ not yet
  src/
    app/
      page.tsx                  ← 4-tab shell
      api/yaml-text/route.ts    ← serve raw YAML for Monaco
    components/
      tabs/
        SetupTab.tsx             ← Load/Edit only; wizard toggle ❌
        ScoreShapingTab.tsx
        OptimizeTab.tsx
        ExplorerTab.tsx
      [wizard/]                 ← ❌ not yet
        [WizardShell.tsx]
        [steps/*.tsx]
      charts/
        PlotlyChart.tsx          ← shared base (dynamic import, zinc theming)
        ScoreConvergenceChart.tsx
        MetricConvergenceChart.tsx
        PenaltyCurveChart.tsx
        MetricScatterChart.tsx
        MetricHistogramChart.tsx
      ui/
        badge.tsx / button.tsx / panel.tsx
        select.tsx / slider.tsx / separator.tsx
    lib/
      api.ts                    ← typed fetch client
      utils.ts                  ← cn(), formatNumber(), formatEng()
    stores/
      projectStore.ts
      runStore.ts
      explorerStore.ts
    types/
      api.ts                    ← shared request/response types
```

---

## Technology Stack

| Layer | Choice |
|---|---|
| Frontend | Next.js 14 (TypeScript, App Router, `ui/src/`) |
| Styling | TailwindCSS + custom shadcn-style primitives |
| Icons | Lucide |
| State | Zustand (3 stores) |
| YAML editor | `@monaco-editor/react` (dynamic import, SSR disabled) |
| Backend | FastAPI (Python) — thin adapter over `spicexplorer` |
| Live updates | SSE (`EventSource` on frontend, `StreamingResponse` on backend) |
| Charts | `react-plotly.js` (dynamic import, SSR disabled), consistent zinc/indigo theming |

**Start command:**
```bash
./scripts/run_newcas_ui.sh
# FastAPI → http://localhost:8000
# Next.js  → http://localhost:3000
```

---

## Key Implementation Decisions (for future reference)

1. **CSV vs JSON checkpoints**: `checkpoint_reader.py` handles both formats — `.csv` from `NEWCAS_SUBMISSION_APPENDIX` for demo replay, `.json` from new live runs via `Optimization_Log_Visualizer`. Detection is by file extension.

2. **CSV column dotted names**: Pandas `.itertuples()` sanitizes dots in column names like `point.score` into inaccessible attributes. The reader uses `.iterrows()` instead, which returns a Series where `row.get("point.score")` works correctly.

3. **SSE + threading**: The optimizer runs in a background thread. Events are pushed to `asyncio.Queue` via `asyncio.run_coroutine_threadsafe(queue.put(event), loop)`. The SSE endpoint drains the queue with a 60 s timeout. Replay mode drips CSV rows at 50 ms intervals via an async coroutine.

4. **Monaco / Plotly SSR**: Both use `dynamic(() => import(...), { ssr: false })` to avoid `window is not defined` errors in Next.js server rendering.

5. **Plotly axis titles**: Must be `{ title: { text: "..." } }` not a bare string — the TypeScript types require `Partial<DataTitle>`.

6. **Score computation**: The backend computes a penalty curve by sweeping `target ± 3×range` across 200 points and returning it to the frontend for `PenaltyCurveChart`. Raw directional error is computed first (`_raw_directional_error`), then normalized via both methods.

---

## Demo Script (NEWCAS Presentation Flow)

1. **Tab 1 — Setup** (1 min): Click "Load demo…" → "OTA Cascode (default)". Hit **Apply**. Walk through the parsed summary (DUT params, testbenches, specs).

2. **Tab 2 — Score Shaping** (2 min): Pick `ugf` spec. Drag value slider to 187 MHz (just below 200 MHz target). Point to linear P̂ (0.87) vs sigmoid P̂ (0.18). Show how the linear aggregate is dominated by current, masking the UGF miss. "This is the paper's core claim."

3. **Tab 3 — Optimize** (2 min): Select "sigmoid_de" from Demo Checkpoint dropdown → hit **Replay**. Watch score convergence and per-metric best-so-far animate via SSE. Spec chips turn green as specs are met.

4. **Tab 4 — Explorer** (2 min): Load Run A = `sigmoid_de`, Run B = `linear_de`. Overlay convergence traces. Open Metric Scatter (UGF vs. Current) to show the feasible cloud. Show Performance Envelope table — topology ceiling for each spec.
