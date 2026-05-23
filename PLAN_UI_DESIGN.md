# SpiceXplorer UI Design Plan

## Goals

The UI is a **conference demo** for NEWCAS 2026 showcasing:
1. A guided project setup wizard that generates the YAML DSL from user input (netlist upload → YAML)
2. The paper's core contribution: **score shaping** (sigmoid vs. linear normalization), API-driven
3. A live optimization run on the cascode OTA case study, with real-time metric streaming
4. Post-run **topology performance exploration**: convergence, metric trade-off scatter, achievable limits

Constraints: easy local setup (two commands), looks polished in a presentation, no external hosting.

---

## Technology Stack

Follows `examples/coding_style.xml`:

| Layer | Choice |
|---|---|
| Frontend | Next.js 14 (TypeScript, App Router) |
| Styling | TailwindCSS + shadcn/ui |
| Icons | Lucide |
| State | Zustand |
| YAML editor | Monaco Editor (`@monaco-editor/react`) |
| Backend | FastAPI (Python) — thin adapter over `spicexplorer` |
| Live updates | Server-Sent Events (SSE) for streaming optimization progress |
| Charts | Plotly.js (`react-plotly.js`) — matches existing `spicexplorer/viz` output |

**Setup target**:
```bash
uv run python ui/backend/main.py   # FastAPI on :8000
cd ui/frontend && npm run dev      # Next.js on :3000
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Next.js Frontend  (localhost:3000)                      │
│  ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌────────────┐  │
│  │  Setup   │ │ Shaping │ │ Optimize │ │  Explorer  │  │
│  │  Wizard  │ │         │ │          │ │            │  │
│  └──────────┘ └─────────┘ └──────────┘ └────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │ REST + SSE
┌───────────────────────▼─────────────────────────────────┐
│  FastAPI Backend  (localhost:8000)                       │
│  /api/project/*   — load, validate, generate YAML        │
│  /api/netlist/*   — upload, parse parameter names        │
│  /api/score       — sigmoid vs. linear penalty compute   │
│  /api/optimize/*  — start/stop runs, SSE stream          │
│  /api/checkpoint  — list/load OptimizationLog files      │
└───────────────────────┬─────────────────────────────────┘
                        │ Python imports
           spicexplorer.core + .optimization + .viz
```

The backend is a **thin FastAPI wrapper** — all domain logic stays in `spicexplorer`. No logic is duplicated.

---

## Demo Configuration File

A single file at `ui/demo_config.json` controls all pre-baked paths so the demo can be reconfigured without touching code:

```json
{
  "default_yaml": "examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml",
  "demo_checkpoints": {
    "sigmoid_de": "examples/OTA/cascode/ihp-sg13g2/sizing/checkpoints/demo_sigmoid_de.json",
    "linear_de":  "examples/OTA/cascode/ihp-sg13g2/sizing/checkpoints/demo_linear_de.json"
  },
  "demo_netlists": {
    "dut":       "examples/OTA/cascode/ihp-sg13g2/spice/ota-improved.spice",
    "tb_ac":     "examples/OTA/cascode/ihp-sg13g2/spice/ota-improved_tb-loopgain.spice",
    "tb_noise":  "examples/OTA/cascode/ihp-sg13g2/spice/ota-improved_tb-noise.spice",
    "tb_tran":   "examples/OTA/cascode/ihp-sg13g2/spice/ota-improved_tb-tran.spice"
  }
}
```

All paths are relative to the repo root. The backend reads this file on startup.

---

## Page Layout

Five-tab shell. The status pill in the header reflects a running optimization.

```
┌────────────────────────────────────────────────────────────────┐
│  ⚡ SpiceXplorer  │ Setup │ Score Shaping │ Optimize │ Explorer │  ● Running  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│                       < active tab >                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

Color palette: **zinc** neutral base, **indigo** accent for interactive elements, **emerald/red** for pass/fail, **amber** for warnings.

---

## Tab 1 — Project Setup

This tab has two modes toggled by a segmented control at the top: **Load/Edit** and **Create Wizard**.

### Mode A: Load / Edit

```
  ┌─ Mode ──────────────────────────────────────────────┐
  │  (● Load/Edit)   (○ Create Wizard)                  │
  └─────────────────────────────────────────────────────┘

  [Load Demo ▾]   [Upload YAML]           [Validate] [Apply]
  ┌──────────────────────────┬──────────────────────────┐
  │  Monaco YAML Editor      │  Parsed Summary           │
  │                          │                           │
  │  project:                │  ✓ Valid schema           │
  │    name: CASCODE-OTA     │  Tech: ihp-sg13g2         │
  │    simulator: ngspice    │                           │
  │    ...                   │  DUT Params (14)          │
  │                          │  ┌──────────────┬───┬───┐ │
  │                          │  │ Name         │Min│Max│ │
  │                          │  │ X_DUT_M1M2_W │180n│200u│
  │                          │  │ ...          │   │   │ │
  │  ── schema errors ──     │  └──────────────┴───┴───┘ │
  │  ❌ line 42: unknown key │  PVT Corners (1)          │
  │                          │  tb_ac ✓  tb_noise ✓      │
  └──────────────────────────┴──────────────────────────┘
```

- **Load Demo** dropdown lists entries from `demo_config.json → default_yaml`
- Monaco editor: full edit + live schema validation on keystroke (debounced, calls `POST /api/project/validate`)
- **Validate** button triggers a full parse and highlights schema errors inline
- **Apply** commits the project to Zustand state (makes it available to Optimize tab)
- Engineering-unit formatting in the summary table (e.g., `180 nm`, `200 µm`)
- Testbench enable toggles in the summary panel (write back to YAML automatically)

### Mode B: Create Wizard

A multi-step form that auto-generates the YAML. The live YAML preview updates on every keystroke in a right panel.

```
  ┌─ Mode ──────────────────────────────────────────────┐
  │  (○ Load/Edit)   (● Create Wizard)                  │
  └─────────────────────────────────────────────────────┘

  ┌── Step Navigator ──────────────────────────────────────────────┐
  │  ① Basic Info  ② PDK Rules  ③ DUT Params  ④ PVT  ⑤ Testbenches  ⑥ Target Specs  ⑦ Optimizer  │
  │      ✓              ✓            ●                                                              │
  └────────────────────────────────────────────────────────────────┘

  ┌── Active Step ─────────────────┬── Live YAML Preview ─────────┐
  │                                │                               │
  │  Step 3: DUT Parameters        │  project:                     │
  │                                │    name: MY-OTA               │
  │  Upload DUT netlist:           │    ...                        │
  │  [📁 ota.spice]  ✓ parsed      │    dut_params:                │
  │                                │      - name: X_DUT_M1_W       │
  │  Detected parameters:          │        min_val: min_nfet_w    │
  │  ┌────────────────────────────┐│        max_val: max_nfet_w    │
  │  │ Name       │ Min  │ Max  │ Int │ Log │        │
  │  │ X_DUT_M1_W │ min_nfet_w▾│max_nfet_w▾│ ○ │ ○ │
  │  │ X_DUT_M1_L │ min_nfet_l▾│max_nfet_l▾│ ○ │ ○ │
  │  │ + Add row  │            │           │   │   │
  │  └────────────────────────────┘│                               │
  │                                │                               │
  │         [← Back]  [Next →]     │  [Copy YAML]  [Save YAML]    │
  └────────────────────────────────┴───────────────────────────────┘
```

**Wizard Steps:**

1. **Basic Info** — project name, description, simulator (dropdown: `ngspice`), workspace root path

2. **PDK Rules** — technology name, add key/value constraint rows (e.g., `min_nfet_w = 0.18u`). Constraints defined here appear as autocomplete options in step 3.

3. **DUT Parameters** — upload DUT netlist → `POST /api/netlist/parse` returns detected `.param` names as pre-filled rows. User sets min/max (from PDK constraint dropdown or raw value), flags `is_integer`, `log_scale`, `freeze`.

4. **PVT Corners** — add rows: temp (°C), corner string, supply voltage. At least one required.

5. **Testbenches** — add/remove testbench cards. Per card: name, upload netlist (stored server-side), set parameter name/val/description rows, enable toggle.

6. **Target Specs** — add/remove spec rows. Per spec: name, testbench (dropdown from step 5), sim_type, goal, target, tolerance, range, weight, error_type, reward_type, enable. Layout: one accordion row per spec, expand to edit.

7. **Optimizer Config** — algorithm dropdown (`LogBFGSCMAPlus`, `LhsDE`, etc.), budget, random seed, lin/log variable bounds.

**Netlist Parsing** (`POST /api/netlist/parse`): Backend scans the uploaded `.spice` file for `.param name=val` lines and returns `[{name, default_val}]`. Simple regex, no full SPICE parse needed.

On **Save YAML**: write the generated file to the workspace root, then switch to Load/Edit mode with the new file loaded.

---

## Tab 2 — Score Shaping

**Goal**: Visually demonstrate the paper's contribution — why sigmoid outperforms linear normalization.

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Score Shaping Visualizer                                        │
  │                                                                  │
  │  Spec: [ugf ▾]    Goal: exceed    Target: 200 MHz               │
  │  Current value m: [──────●──────────────────] 187 MHz            │
  │  α (sigmoid rate): [─────●──────────────────] 5.0               │
  │                                                                  │
  │  ┌──────────────────────────────────────────────────────────┐    │
  │  │  Normalized Penalty P̂(m)          ← current m           │    │
  │  │  1.0 ┤               ____________________________  ─ ─   │    │
  │  │      │          ____/              sigmoid              │    │
  │  │  0.5 ┤     ____/         ╷                             │    │
  │  │      │ ___/  linear─ ─ ─ ╷                             │    │
  │  │  0.0 ┤─────────────────── ╷ ─────────────────────────  │    │
  │  │     far below   -tol  target  +tol   above              │    │
  │  └──────────────────────────────────────────────────────────┘    │
  │                                                                  │
  │  Per-spec breakdown (current design point)          API-driven   │
  │  ┌───────────────┬──────┬───────┬──────────┬──────────────────┐  │
  │  │ Spec          │ Val  │Target │ Sigmoid P̂│ Linear P̂        │  │
  │  │ ugf           │187M  │>200M  │  0.18    │  0.87            │  │
  │  │ dcgain        │ 44   │> 40   │  0.00    │  0.00            │  │
  │  │ pm            │ 57°  │60±10° │  0.00    │  0.00            │  │
  │  │ inoise        │1.08m │<1.2m  │  0.00    │  0.00            │  │
  │  │ current       │ 28µ  │< 25µ  │  0.09    │  0.56            │  │
  │  └───────────────┴──────┴───────┴──────────┴──────────────────┘  │
  │  Aggregate F(x):   Sigmoid = -0.27    Linear = -1.43             │
  │                                                                  │
  │  ╰── Note: linear current penalty (0.56) dominates the          │
  │     aggregate, masking ugf's proximity to spec. Sigmoid         │
  │     keeps both visible to the optimizer.                         │
  └──────────────────────────────────────────────────────────────────┘
```

- Sliders drive `POST /api/score` — backend uses the loaded project's `TargetSpec` objects to compute penalties with both methods
- Chart and table update on every slider change (debounced 150 ms)
- "Aggregate F(x)" row shows the full fitness under both modes
- Callout box auto-generated: identifies which spec has the highest linear penalty and quotes why that's a problem

Backend: `POST /api/score` body: `{project_id, metric_values: {ugf: 187e6, ...}}` → response: `{per_spec: {ugf: {sigmoid: 0.18, linear: 0.87}, ...}, aggregate: {sigmoid: -0.27, linear: -1.43}}`. Computed via `TargetSpec.get_simple_penalty()`.

---

## Tab 3 — Optimize

**Goal**: Launch and watch a live optimization run.

```
  ┌── Run Configuration ──────────────────────────────────── [▶ Start Run] ──┐
  │  Project: CASCODE-OTA  ✓ valid                                            │
  │                                                                           │
  │  Optimizer: [LogBFGSCMAPlus ▾]   Budget: [2000]   Seed: [48]            │
  │  Score fn:  (● Sigmoid  ○ Linear)                                        │
  │  Testbenches: [tb_ac ✓]  [tb_noise ✓]  [tb_tran ✓]                      │
  │  [Load Demo Checkpoint ▾]  ← skip live run, replay saved trace           │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌── Live Progress ────────────────────────── iter 342 / 2000  17% ──────────┐
  │  ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                           │
  │                                                                           │
  │  ┌── Score vs. Iteration ──────────────────────────────────────────────┐  │
  │  │ 50 ┤                                      ╭─── score (best-so-far) │  │
  │  │  0 ┤──────────────────────────────────────  feasibility boundary   │  │
  │  │-100┤  ╲_______________                                              │  │
  │  │-250┤────────────────────────────────────────────────────────────── │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
  │                                                                           │
  │  ┌── Per-Metric Best-So-Far ──────────────────────────────────────────┐  │
  │  │   [ugf ▾]                                                          │  │
  │  │  250M ┤                                            ╭─── best ugf  │  │
  │  │  200M ┤ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  target           │  │
  │  │  150M ┤  ╲_______________________________                         │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
  │                                                                           │
  │  Live Status Chips (best design)                          [■ Stop]        │
  │  [ugf 187M ✗]  [dcgain 44dB ✓]  [pm 57° ✓]  [inoise 1.1mV ✓]           │
  │  [current 22µA ✓]  [tsettle 12µs ✓]                                      │
  └───────────────────────────────────────────────────────────────────────────┘
```

- **Load Demo Checkpoint** dropdown (from `demo_config.json`) replays a saved `OptimizationLog` through the SSE stream at 10× speed — safe fallback for a conference where NGSpice might be slow
- **Per-Metric Best-So-Far** chart: second live chart tracking the best observed value of any one metric vs. iteration (dropdown to switch metric). Dashed target line overlaid.
- **Score function toggle** is the key demo moment — start sigmoid, show convergence, then explain why linear would not converge by pointing to Tab 2
- Stop button triggers `POST /api/optimize/stop/{run_id}`, run is auto-checkpointed
- Completed/stopped runs appear immediately in the Explorer tab's checkpoint list

SSE event schema:
```json
{
  "iter": 342,
  "score": -18.4,
  "best_score": -18.4,
  "metrics": { "ugf": 1.87e8, "dcgain": 44.3, "pm": 57.1, "inoise": 1.08e-3, "current": 2.21e-5, "tsettle": 1.18e-5 },
  "best_params": { "X_DUT_M1M2_W": 2.1e-6, "..." : "..." }
}
```

---

## Tab 4 — Trace Explorer

**Goal**: Post-run exploration of what the optimizer found and what the topology can achieve. Answers the question: *what are the performance limits of this topology?*

```
  ┌── Checkpoint Manager ─────────────────────────────────────────────────────┐
  │  Run A: [demo_sigmoid_de ▾]   Run B: [demo_linear_de ▾]  [+ Add Run]      │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌── Panel Row 1 ─────────────────────────────────────────────────────────────┐
  │  ┌──────────────────────────────┐  ┌──────────────────────────────────────┐│
  │  │  Score Convergence           │  │  Per-Spec Convergence                ││
  │  │  Run A ─── (sigmoid)         │  │  Metric: [ugf ▾]                     ││
  │  │  Run B ─ ─ (linear)          │  │  Run A ─── best-so-far ugf           ││
  │  │  Feasibility boundary ─ ─ ─  │  │  Run B ─ ─                           ││
  │  │                              │  │  ─ ─ target                           ││
  │  └──────────────────────────────┘  └──────────────────────────────────────┘│
  └────────────────────────────────────────────────────────────────────────────┘

  ┌── Panel Row 2: Topology Performance Limits ────────────────────────────────┐
  │                                                                            │
  │  ┌── Metric Scatter ────────────────────┐  ┌── Performance Envelope ─────┐│
  │  │  X: [ugf ▾]   Y: [current ▾]         │  │  What can this topology do? ││
  │  │                                      │  │                              ││
  │  │  ·  ·    ·  ·   ← all visited        │  │  Metric  Best Ever  Target  ││
  │  │  ·    ●●●  ·      designs            │  │  ugf     247 MHz   >200✓    ││
  │  │     ●●●●●●        ● = feasible       │  │  dcgain  52.1 dB   >40 ✓    ││
  │  │  ─ ─ ─ ─ ─ ─ target line            │  │  pm      71°       60±10✓   ││
  │  │                                      │  │  inoise  0.87mV    <1.2 ✓   ││
  │  │  [Run A] [Run B] [Both]              │  │  current 14.2µA    <25  ✓   ││
  │  └──────────────────────────────────────┘  │  tsettle 6.8µs     <15  ✓   ││
  │                                            └──────────────────────────────┘│
  └────────────────────────────────────────────────────────────────────────────┘

  ┌── Panel Row 3 ─────────────────────────────────────────────────────────────┐
  │  ┌── Metric Distribution ───────────────┐  ┌── Best Design Params ───────┐│
  │  │  Metric: [ugf ▾]  Bins: [40]         │  │  Run A (sigmoid)            ││
  │  │  ▓▓▓▓                                │  │  X_DUT_M1M2_W    2.1 µm     ││
  │  │  ▓▓▓▓▓▓▓                             │  │  X_DUT_M1M2_L    360 nm     ││
  │  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓                       │  │  X_DUT_M3M4_W    5.4 µm     ││
  │  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓              │  │  V_BIAS1          0.62 V     ││
  │  │          │target                     │  │  ...                        ││
  │  │  ← Run A ─── Run B ─ ─              │  └─────────────────────────────┘│
  │  └──────────────────────────────────────┘                                 │
  │                                                                            │
  │  ┌── Spec Summary (best design) ────────────────────────────────────────┐ │
  │  │  Metric   Goal   Target   Achieved   Unit   Status   (Run A / Run B) │ │
  │  │  ugf      >      200      221 / 198  MHz    ✓  /  ✗                  │ │
  │  │  dcgain   >      40       44 / 41    dB     ✓  /  ✓                  │ │
  │  │  pm       =60    ±10      58 / 55    deg    ✓  /  ✓                  │ │
  │  │  inoise   <      1.2      1.05/1.19  mV     ✓  /  ✓                  │ │
  │  └──────────────────────────────────────────────────────────────────────┘ │
  └────────────────────────────────────────────────────────────────────────────┘
```

**Performance Envelope** panel is the "topology limits" feature. It reads all `OptimizationLogEntry` objects from the checkpoint (not just the best design) and computes, per metric, the single best value ever observed across all sampled designs — regardless of whether that design met other specs. This tells the designer: "your topology's ceiling for UGF is ~247 MHz even if it came at the cost of power." The feasibility tag (✓/✗ relative to target) is computed independently.

**Metric Scatter**: plots all visited designs as points in 2D metric space. Feasible designs highlighted. Target lines drawn. Shows the trade-off region (e.g., UGF vs. current) — the "shape" of what's achievable by this topology.

**Multi-run comparison** is powered by loading two `OptimizationLog` checkpoint files, each produced by `spicexplorer`'s existing checkpoint mechanism. The backend endpoint `GET /api/checkpoint/{name}` deserializes via the existing `Optimization_Log_Visualizer.load_checkpoint()` method.

---

## Backend API Summary

| Endpoint | Method | Description |
|---|---|---|
| `/api/config` | GET | Return `demo_config.json` to populate demo dropdowns |
| `/api/project/load` | POST | Load and parse a YAML path → return `ProjectSummary` |
| `/api/project/validate` | POST | Validate YAML string against schema → errors list |
| `/api/project/generate` | POST | Accept wizard form data → return generated YAML string |
| `/api/netlist/parse` | POST | Upload `.spice` file → return detected `.param` names |
| `/api/score` | POST | Compute sigmoid + linear penalties for all specs |
| `/api/optimize/start` | POST | Start background run → `{run_id}` |
| `/api/optimize/stop/{run_id}` | POST | Gracefully stop, auto-checkpoint |
| `/api/optimize/stream/{run_id}` | GET (SSE) | Stream `{iter, score, metrics, best_params}` |
| `/api/checkpoint` | GET | List available `.json` checkpoint files |
| `/api/checkpoint/{name}` | GET | Load `OptimizationLog` via `Optimization_Log_Visualizer` |
| `/api/checkpoint/{name}/envelope` | GET | Compute per-metric best-ever across all entries |
| `/api/checkpoint/{name}/scatter` | GET | Return all `{metric_x, metric_y, feasible}` points for scatter |

---

## Directory Structure

```
ui/
  demo_config.json               ← configurable demo paths (repo root-relative)
  backend/
    main.py                      ← FastAPI app, route registration
    routes/
      project.py                 ← /api/project/*
      netlist.py                 ← /api/netlist/*
      score.py                   ← /api/score
      optimize.py                ← /api/optimize/*
      checkpoint.py              ← /api/checkpoint/*
    services/
      optimizer_runner.py        ← background task + SSE event queue
      netlist_parser.py          ← .param regex extraction
      score_service.py           ← wraps TargetSpec penalty methods
      yaml_generator.py          ← wizard form data → YAML string
  frontend/
    src/
      app/
        page.tsx                 ← tab shell + global layout
      components/
        tabs/
          SetupTab.tsx
          ScoreShapingTab.tsx
          OptimizeTab.tsx
          ExplorerTab.tsx
        wizard/
          WizardShell.tsx        ← step navigator + progress
          steps/
            BasicInfoStep.tsx
            PDKRulesStep.tsx
            DutParamsStep.tsx
            PVTStep.tsx
            TestbenchesStep.tsx
            TargetSpecsStep.tsx
            OptimizerStep.tsx
        charts/
          ScoreConvergenceChart.tsx
          MetricConvergenceChart.tsx
          PenaltyCurveChart.tsx
          MetricScatterChart.tsx
          MetricHistogramChart.tsx
        ui/                      ← shadcn/ui primitives
      stores/
        projectStore.ts          ← loaded project config + validation state
        runStore.ts              ← active run state, SSE data, live metrics
        explorerStore.ts         ← loaded checkpoints, selected runs A/B
      types/
        api.ts                   ← shared request/response types
        project.ts               ← frontend mirror of domain types
```

---

## Key Demo Script (NEWCAS Presentation Flow)

1. **Tab 1 — Setup** (1.5 min): Switch to Create Wizard. Upload the OTA netlist — params auto-populate. Walk through steps 3 and 6 to show how a sizing problem is described. Switch back to Load/Edit to show the full generated YAML. Hit Validate — green banner.

2. **Tab 2 — Score Shaping** (2 min): Load the project. Drag the UGF slider to just below target (187 MHz). Point to the linear penalty (0.87) vs. sigmoid (0.18). Show that the linear aggregate (-1.43) is dominated by current, masking the UGF miss. "This is the paper's core claim — let's verify it empirically."

3. **Tab 3 — Optimize** (2 min): Start sigmoid run. Watch the score convergence cross zero at ~500 iterations while the per-metric chart shows each spec converging. If NGSpice is running fast, let it go to 1000 iterations live; otherwise use "Load Demo Checkpoint" to replay at 10× speed.

4. **Tab 4 — Explorer** (2 min): Load sigmoid + linear checkpoints. Overlay convergence traces — sigmoid crosses zero, linear stagnates. Open Metric Scatter (UGF vs. Current) to show the feasible cloud. Show Performance Envelope table: "this topology can reach 247 MHz UGF — our optimizer found 221 MHz. The sigmoid score got us to 90% of the topology's ceiling."
