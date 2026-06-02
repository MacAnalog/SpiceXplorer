# SpiceXplorer UI

A web interface for the SpiceXplorer circuit optimization library. It is a **Studio workspace** — a persistent VS Code-style shell (activity bar, contextual left rail, tabbed center views, always-on run rail, command palette) — providing a guided project setup wizard, interactive score shaping, live SPICE-backed optimization runs, run history, and multi-run exploration. Demonstrated on the cascode OTA case study with the IHP SG13G2 PDK.

> **Live SPICE needs the PDK.** Live optimization and the sanity check require both `ngspice` and the IHP `ihp-sg13g2` PDK (present on the research-group server). On a machine without the PDK the app detects this (`GET /api/env`), shows a **"PDK missing — replay only"** status pill, and disables live runs — while score shaping, checkpoint replay/compare, the wizard, and the pipeline view all work fully.

---

## Quick Start

**Prerequisites:** `uv`, `node`, `npm`, `ngspice` (for live runs).

```bash
# From the repo root — starts both backend (:8000) and frontend (:4000)
./scripts/run_newcas_ui.sh

# Verbose logging (DEBUG shows all spicexplorer library events)
LOG_LEVEL=DEBUG ./scripts/run_newcas_ui.sh
```

Open **http://localhost:4000** in your browser.  
VS Code Remote SSH auto-detects the port and offers to forward it — accept.

> **Why port 4000?** VS Code Remote SSH occupies port 3000 on the server side. The script uses 4000 to avoid the conflict.

### Manual startup (two terminals)

```bash
# Terminal 1 — FastAPI backend
LOG_LEVEL=INFO uv run --extra ui uvicorn ui.backend.main:app --reload --port 8000

# Terminal 2 — Next.js frontend
cd ui && npm run dev -- -p 4000
```

---

## Architecture

```
Browser (localhost:4000)
  └─ Next.js 15 (TypeScript, App Router — Studio shell)
        │ REST + SSE
  FastAPI (localhost:8000)
        │ Python imports
  spicexplorer.core + .optimization + .spice_engine
        │ subprocess
  ngspice  (+ IHP sg13g2 PDK, for live runs)
```

### Shell anatomy

The app is one persistent workspace. `app/page.tsx` redirects to `/setup`; all real views live under the `app/(studio)/` route group. `(studio)/layout.tsx` renders the shell once and only the center segment swaps on navigation, so the rails and the live SSE stream persist across views:

```
StudioTitleBar          brand · + New project · ⌘K
ActivityBar | LeftRail | TabStrip + center view        | RightRail
(icons)     | (project,| (Setup/Scoring/Optimize/      | (live run:
            |  runs,   |  Explore/Schematic/Pipeline)  |  iteration,
            |  ckpts)  | BottomPanel (optimizer log)   |  specs, params)
StudioStatusBar         active view · project · panel toggles · PDK/sim pill
Overlays: CommandPalette (⌘K) · WizardOverlay (+ New project)
```

### Key files

| Path | Role |
|---|---|
| `ui/backend/main.py` | FastAPI app entry, CORS, logging setup |
| `ui/backend/routes/` | One file per API group (config, project, score, optimize, checkpoint, schematic, sanity, netlist, env, xschem) |
| `ui/backend/services/optimizer_runner.py` | Background thread + SSE queue for live/replay runs; `_apply_overrides` for ephemeral run config |
| `ui/backend/services/env_probe.py` | Cheap ngspice + IHP PDK detection (no simulation) → `GET /api/env` |
| `ui/backend/services/checkpoint_reader.py` | Unified CSV + JSON checkpoint loader |
| `ui/backend/services/yaml_generator.py` | Wizard form ⇄ project YAML (generate + parse-to-form) |
| `ui/app_config.json` | Preset checkpoint paths + default YAML (repo-root-relative) |
| `ui/src/app/page.tsx` | Redirect → `/setup` |
| `ui/src/app/(studio)/layout.tsx` | Mounts `StudioShell`; persists across view navigation |
| `ui/src/app/(studio)/<view>/page.tsx` | One thin segment per view (setup, scoring, optimize, compare, schematic, pipeline, health) |
| `ui/src/components/shell/` | `StudioShell`, `ActivityBar`, `TabStrip`, `StudioTitleBar`, `StudioLeftRail`, `RightRail`, `BottomPanel`, `StatusBar`, `nav.ts` |
| `ui/src/components/shell/nav.ts` | **Single source of truth** for the 7 views (id, label, route, icon, shortcut, gating) |
| `ui/src/components/overlays/` | `CommandPalette` (⌘K), `WizardOverlay` (+ New project) |
| `ui/src/components/tabs/` | The center views: SetupTab, ScoreShapingTab, OptimizeTab, ExplorerTab, SchematicTab, HealthTab, PipelineView |
| `ui/src/components/wizard/` | `WizardShell` + 7 step components |
| `ui/src/components/charts/` | Plotly-backed chart components (all SSR-disabled) |
| `ui/src/components/ui/` | Shared primitives: Button, Badge, Panel, Select, Table, EmptyState, SpecChip, Stat, Sparkline, … |
| `ui/src/stores/` | Zustand state: `projectStore`, `runStore` (+ SSE + history), `explorerStore`, `uiStore` (nav/selection/overlays), `wizardStore` |
| `ui/src/lib/api.ts` | Typed fetch client — all backend calls go through here |
| `ui/src/types/api.ts` | TypeScript mirrors of all FastAPI response shapes |
| `ui/.env.local` | `NEXT_PUBLIC_API_URL=http://localhost:8000` |

---

## Workflow

Navigate views via the activity-bar icons, the tab strip, or **⌘1–⌘7**. Views that need an applied project (Score Shaping, Optimize, Pipeline) stay disabled until you apply one on Setup.

| View | What it does |
|---|---|
| **Setup** | Load a project YAML (example dropdown or file upload) or **build one from scratch** with the 7-step wizard; edit in Monaco, validate, apply. Shows project metadata, testbenches, DUT params, target specs. |
| **Score Shaping** | Select a spec, drag a slider to explore metric values. Compares linear vs sigmoid penalty curves with a per-spec breakdown. Deep-linkable from the ⌘K palette and the Pipeline view. |
| **Optimize** | Select algorithm/budget, then start a live SPICE run or replay a preset checkpoint. Streams score + metric convergence; live run progress, spec status, and best params appear in the always-on right rail. Live Start is disabled (steered to Replay) when the PDK is absent. |
| **Explore** | Load two checkpoints (Run A / Run B), overlay convergence, plot metric scatter, inspect the performance envelope and best design params. |
| **Schematic** | Browse the project's Xschem `.sch` hierarchy with symbol resolution. |
| **Pipeline** | Read-only DAG of the problem: Optimizer → DUT params → Testbenches → Target specs. Clicking a spec node deep-links into Score Shaping; spec nodes tint pass/fail live during a run. |
| **Health** (gear) | On-demand sanity check — runs one simulation per testbench + a trial optimizer step, reporting ngspice path, PDK verdict, and per-testbench log tails. |

### Shell features

- **Run history** — every finished run (live or replay) is recorded in the left rail with a score sparkline, persisted to localStorage (metadata only). Click a replay run to re-run it.
- **Command palette (⌘K)** — search to switch views, jump to a spec (→ Score Shaping), jump to a run (→ Optimize), start the new-project wizard, or stop a run.
- **Right rail / bottom panel** — toggle from the status bar; both stay live during a run regardless of the active view (the SSE stream lives in `runStore`).

---

## Feature Summary

### Implemented ✅

- **Studio shell** — App-Router route group with a persistent layout: activity bar, contextual left rail, tab strip, always-on right rail, collapsible bottom panel, status bar. Views are deep-linkable (`/setup`, `/scoring`, …) and switchable via ⌘1–⌘7.
- **Setup view** — Monaco YAML editor (debounced validation), example dropdown, Upload, Validate, Apply, plus the **Create Wizard** toggle.
- **New-project wizard** — 7-step form (Basic Info → PDK Rules → DUT Params w/ netlist upload → PVT → Testbenches → Target Specs → Optimizer) with a live YAML preview; generates + applies a `project_setup.yaml`. Launchable from Setup, the title-bar **+ New project**, or the ⌘K palette. Backed by `POST /api/project/generate`, `POST /api/project/parse-to-form`, `POST /api/netlist/parse`.
- **Score Shaping view** — Spec selector + slider (range = target ± 3×range), live penalty curve, per-spec breakdown (linear/sigmoid), highest-penalty callout. Honors deep-linked spec selection.
- **Optimize view** — Algorithm dropdown, budget input, preset checkpoint replay, Start/Stop with SSE streaming, score + metric convergence charts. **Algorithm/budget/seed are honored on live runs** (applied in-memory; YAML not rewritten). Live Start disables + steers to Replay when the PDK is absent.
- **Right rail + bottom panel** — Live run progress, spec status chips, best params, and the optimizer log; keep updating across view changes (SSE hoisted into `runStore`).
- **Run history** — Persisted run list with score sparklines; click a replay run to re-run.
- **Command palette (⌘K)** — Switch view · jump to spec · jump to run · new project · stop run.
- **Explore view** — Run A/B checkpoint selectors, overlaid convergence, metric scatter (X/Y), performance envelope, metric histogram, best design params, spec summary.
- **Schematic view** — Xschem `.sch` hierarchy browser with symbol resolution.
- **Pipeline view** — Read-only DAG (Optimizer → DUT params → Testbenches → Specs) with clickable spec nodes that deep-link to Score Shaping.
- **Health / sanity check** — One sim per testbench + a trial optimizer step; reports ngspice path, PDK verdict, per-testbench log tails.
- **PDK-aware degradation** — `GET /api/env` drives the status-bar sim/PDK pill and gates live runs.
- **UI primitives** — `Button`, `Badge`, `Panel`, `Select` + `selectCn()`, `Table`, `EmptyState`, `SpecChip`, `Stat`, `Sparkline`, `Segmented`, `Slider`.
- **Logging** — `setup_loggers(console_level=...)`; backend reads `LOG_LEVEL`. Files in `logs/SpiceXplorer_<timestamp>.log`.
- **CORS** — Allows any `localhost:<port>`.

### Not Yet Implemented ❌

- **Run ▾ overrides popover** — the backend honors algorithm/budget/seed overrides, but the title-bar Run popover UI to set them globally isn't built; overrides currently come from the Optimize toolbar only.
- **Per-activity left rails** — the left rail is one always-on panel (project + runs + checkpoints); the spec'd per-activity rail variants (file tree, spec list, compare setup, …) are not split out yet.
- **Schematic device inspector + sensitivity** — W/L sliders and a `GET /api/spec/{name}/sensitivity` endpoint are deferred: they need real finite-difference simulation data, which requires the PDK. Best built on the server.
- **Apply from editor content** — "Apply" re-reads from disk; unsaved Monaco edits are lost on Apply.
- **Score function toggle for live runs** — sigmoid vs linear is fixed by the YAML; no runtime switch.

---

## Logging

### Library log level (console)

Pass `LOG_LEVEL` before the run script:

```bash
LOG_LEVEL=DEBUG   ./scripts/run_newcas_ui.sh   # everything — all optimizer steps
LOG_LEVEL=INFO    ./scripts/run_newcas_ui.sh   # default — startup + milestones
LOG_LEVEL=WARNING ./scripts/run_newcas_ui.sh   # quiet — warnings and errors only
```

The log **file** (in `logs/`) always captures `DEBUG` regardless of the console level. Log files are named `SpiceXplorer_<YYYY-MM-DD_HH-MM-SS>.log`.

### What the logs show

| Logger | What it covers |
|---|---|
| `spicexplorer.optimization.orchestrator` | Orchestrator lifecycle |
| `spicexplorer.spice_engine.spicelib` | ngspice wrapper events, param updates |
| `spicexplorer.optimization.*` | Optimizer steps, best-score updates |
| `ui.backend.services.optimizer_runner` | Live run thread progress + errors |

---

## Smoke Tests

Run from the **repo root**:

```bash
# Default — fast smoke tests (no real SPICE simulation, ~6 s)
uv run pytest

# Include real SPICE simulation tests (slow — one ngspice call per test)
uv run pytest -m slow

# Specific file
uv run pytest tests/test_smoke_spice_engine.py -v
uv run pytest tests/test_smoke_optimization.py -v
```

### Test inventory

| Test | Layer | ngspice required | Slow |
|---|---|---|---|
| `test_ngspice_wrapper_imports` | spice_engine | yes | no |
| `test_ngspice_wrapper_instantiation` | spice_engine | yes | no |
| `test_ngspice_sanity_check` | spice_engine | yes | no |
| `test_ngspice_update_params` | spice_engine | yes | no |
| `test_ngspice_run_and_wait` | spice_engine | yes | **yes** |
| `test_project_setup_loads` | optimization | no | no |
| `test_project_setup_param_bounds` | optimization | no | no |
| `test_orchestrator_no_autoload` | optimization | no | no |
| `test_orchestrator_initialize_creates_wrappers` | optimization | yes | no |
| `test_optimizer_parameterize` | optimization | yes | no |
| `test_one_optimization_step` | optimization | yes | **yes** |

Tests are skipped automatically if `ngspice` is not in `PATH`.

---

## Common Bugs & Debugging

### "Load example…" dropdown missing

**Cause:** The dropdown renders only when the backend responds to `GET /api/config`. If the backend was not yet running when the page loaded, `appConfig` stays `null`.

**Fix:** Make sure both processes are running, then **hard-refresh** the page (`Ctrl+Shift+R`).

### App loads but backend calls fail (CORS error in browser console)

**Cause:** Next.js landed on a different port than the backend's CORS allowlist.

**Fix:** The CORS config uses `allow_origin_regex` matching any `localhost:<port>`. If you still see this after the fix, restart the backend.

### Port 4000 already in use / Next.js falls back to 4001

**Cause:** A previous dev session left Next.js running.

**Fix:**
```bash
lsof -ti tcp:4000 | xargs kill   # kill whatever is on 4000
./scripts/run_newcas_ui.sh
```
> Do **not** kill port 3000 blindly — VS Code Remote SSH may be using it.

### "Start Live Run" button stops immediately with no events

**Cause:** The optimizer thread threw an exception (SPICE binary not found, bad netlist path, missing PDK models, etc.).

**Fix:** Check the error message shown below the Start button. Also check the backend log — the full traceback is printed at `ERROR` level with `[run <id>]` prefix.

**Common root causes:**
- `ngspice` not in PATH → run `which ngspice`; add its directory to PATH or pass `path_to_simulator` in the YAML
- PDK model files not found → check `ws_root` and testbench netlist `.include` paths in the YAML
- YAML `simulator:` field points to wrong binary name

### Stale `.next` build cache after `npm run build`

**Cause:** Running `npm run build` for type-checking leaves production chunks that the dev server can't use.

**Fix:**
```bash
rm -rf ui/.next
# then restart the dev server
```

### Monaco editor shows blank / "Loading editor…" forever

**Cause:** Monaco is SSR-disabled via `dynamic(..., { ssr: false })`; it needs the browser. Usually resolves on its own after hydration. If persistent, check the browser console for chunk load errors.

**Fix:** Hard-refresh. If chunks are missing, delete `ui/.next` and restart.

### `TypeError: object of type 'ListTargetSpec' has no len()`

**Cause:** `project.optimizer_config.target_specs` is a custom `ListTargetSpec` object, not a plain list.

**Fix:** Use `.targets` to access the underlying list: `setup.optimizer_config.target_specs.targets`.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/config` | App config (preset checkpoints, default YAML path) |
| GET | `/api/env` | ngspice + IHP PDK probe → `{ngspice_ok, pdk_ok, live_runs_enabled, pdk_detail, …}` |
| POST | `/api/project/load` | Load + parse a YAML file by path |
| POST | `/api/project/validate` | Validate YAML text without applying |
| POST | `/api/project/generate` | Wizard form → validated YAML (optionally save to disk) |
| POST | `/api/project/parse-to-form` | YAML → wizard form (round-trip for "Edit in wizard") |
| POST | `/api/netlist/parse` | Extract `.param` rows from an uploaded `.spice` netlist |
| POST | `/api/score` | Compute sigmoid + linear penalties for given metric values |
| POST | `/api/optimize/start` | Start live run or replay; accepts `algorithm`/`budget`/`seed` overrides; returns `run_id` |
| POST | `/api/optimize/stop/{run_id}` | Signal the run to stop |
| GET | `/api/optimize/stream/{run_id}` | SSE stream of optimization events |
| GET | `/api/checkpoint` | List all available checkpoints |
| GET | `/api/checkpoint/{id}` | Load checkpoint data (scores, metrics, params) |
| GET | `/api/checkpoint/{id}/envelope` | Best-ever per metric with pass/fail |
| GET | `/api/checkpoint/{id}/scatter` | X/Y scatter points with feasibility |
| DELETE | `/api/checkpoint/{id}` | Delete an autosaved checkpoint (presets are read-only) |
| POST | `/api/sanity-check` | Health check: one sim per testbench + trial step; includes `pdk_ok`/`pdk_detail` |
| GET | `/api/schematic` | Serve circuit SVG |
| GET | `/api/xschem/{file,list,project,resolve}` | Xschem hierarchy browsing for the Schematic view |

SSE events (`/api/optimize/stream/{id}`):

```json
{ "iter": 42, "score": 0.31, "best_score": 0.18, "metrics": {"ugf": 1.9e8}, "best_params": {"X_DUT_M1M2_W": 2e-6} }
{ "heartbeat": true }
{ "done": true }
{ "error": "ngspice exited with code 1: ..." }
```
