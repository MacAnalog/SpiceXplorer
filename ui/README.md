# SpiceXplorer NEWCAS 2026 — UI

Conference demo for NEWCAS 2026 showcasing score shaping, live circuit optimization, and topology performance exploration for the cascode OTA case study.

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
  └─ Next.js 15 (TypeScript, App Router)
        │ REST + SSE
  FastAPI (localhost:8000)
        │ Python imports
  spicexplorer.core + .optimization + .spice_engine
        │ subprocess
  ngspice
```

### Key files

| Path | Role |
|---|---|
| `ui/backend/main.py` | FastAPI app entry, CORS, logging setup |
| `ui/backend/routes/` | One file per API group (config, project, score, optimize, checkpoint, schematic) |
| `ui/backend/services/optimizer_runner.py` | Background thread + SSE queue for live/replay runs |
| `ui/backend/services/checkpoint_reader.py` | Unified CSV + JSON checkpoint loader |
| `ui/src/app/page.tsx` | 4-tab shell, tab navigation, demo config fetch |
| `ui/src/components/tabs/` | SetupTab, ScoreShapingTab, OptimizeTab, ExplorerTab |
| `ui/src/components/charts/` | Plotly-backed chart components (all SSR-disabled) |
| `ui/src/components/ui/` | Shared primitives: Button, Badge, Panel, Select, Table, EmptyState |
| `ui/src/stores/` | Zustand state: projectStore, runStore, explorerStore |
| `ui/src/lib/api.ts` | Typed fetch client — all backend calls go through here |
| `ui/src/types/api.ts` | TypeScript mirrors of all FastAPI response shapes |
| `ui/demo_config.json` | Demo checkpoint paths + default YAML (repo-root-relative) |
| `ui/.env.local` | `NEXT_PUBLIC_API_URL=http://localhost:8000` |

---

## Demo Script (NEWCAS Presentation Flow)

| Tab | Duration | What to show |
|---|---|---|
| **Setup** | 1 min | Load demo → "OTA Cascode (default)" → Apply. Walk through DUT params, testbenches, specs. |
| **Score Shaping** | 2 min | Select `ugf`, drag slider to 187 MHz. Compare linear P̂ (0.87) vs sigmoid P̂ (0.18). Show how linear is dominated by a severe violation. |
| **Optimize** | 2 min | Select `sigmoid_de` from Demo Checkpoint → Replay. Watch score convergence animate and spec chips turn green. |
| **Explorer** | 2 min | Load Run A = `sigmoid_de`, Run B = `linear_de`. Overlay convergence. Open Metric Scatter (UGF vs Current). Show Performance Envelope table. |

---

## Feature Summary

### Implemented ✅

- **Setup tab** — Monaco YAML editor (600 ms debounced validation), Load Demo dropdown, Upload, Validate, Apply. Right panel: project meta grid, testbenches, DUT params, target specs.
- **Score Shaping tab** — Spec selector + slider (range = target ± 3×range), live penalty curve chart, per-spec breakdown table (linear/sigmoid), highest-penalty callout.
- **Optimize tab** — Algorithm dropdown, budget input, Demo Checkpoint replay, Start/Stop with SSE streaming, progress bar, score + metric convergence charts, live spec status chips.
- **Explorer tab** — Run A/B checkpoint selectors, overlaid convergence charts, metric scatter (X/Y selectors), performance envelope table, metric histogram, best design params, spec summary.
- **UI primitives** — `Button`, `Badge` (5 variants), `Panel`/`PanelHeader`/`PanelBody`, `Select` + `selectCn()`, `Thead`/`Th`/`Tr`/`Td`, `EmptyState`.
- **Logging** — `setup_loggers(console_level=...)` in the library; backend reads `LOG_LEVEL` env var. Log files written to `logs/SpiceXplorer_<timestamp>.log`.
- **CORS** — Allows any `localhost:<port>` so the app works regardless of which port Next.js lands on.

### Not Yet Implemented ❌

- **Create Wizard** (highest priority for demo) — Step-by-step form to generate a YAML from scratch: BasicInfo → PDKRules → DUT Params (with netlist upload) → PVT → Testbenches → Target Specs → Optimizer. Requires `POST /api/netlist/parse` and `POST /api/project/generate` backend routes, plus `ui/src/components/wizard/` frontend directory.
- **Apply from editor content** — Currently "Apply" re-reads from disk; edits made in the Monaco editor that aren't saved to disk are lost on Apply.
- **Algorithm selection wired to live run** — The algorithm dropdown in OptimizeTab is UI-only; the backend always uses the algorithm from the YAML.
- **Score function toggle for live runs** — Sigmoid vs linear choice is fixed by YAML; no runtime switch exposed yet.

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

### "Load demo…" dropdown missing

**Cause:** The dropdown is conditional on `demoConfig` — it only renders when the FastAPI backend responds to `GET /api/config`. If the backend is not running or the browser made the request before it was up, `demoConfig` stays `null`.

**Fix:** Make sure both processes are running, then **hard-refresh** the page (`Ctrl+Shift+R`).

### App loads but backend calls fail (CORS error in browser console)

**Cause:** Next.js landed on a different port (e.g., 3001) than the backend's CORS allowlist.

**Fix:** The CORS config now uses `allow_origin_regex` matching any `localhost:<port>`. If you still see this after the fix, restart the backend.

### Port 4000 already in use / Next.js falls back to 4001

**Cause:** A previous dev session left Next.js running.

**Fix:**
```bash
lsof -ti tcp:4000 | xargs kill   # kill whatever is on 4000
./scripts/run_newcas_ui.sh
```
> Do **not** kill port 3000 blindly — VS Code Remote SSH may be using it.

### "Start Live Run" button stops immediately with no events

**Cause:** The optimizer thread threw an exception (SPICE binary not found, bad netlist path, missing PDK models, etc.). The error was previously swallowed silently.

**Fix (already patched):** The error is now shown in the UI below the Start button. Also check the backend log — the full traceback is printed at `ERROR` level with `[run <id>]` prefix.

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
| GET | `/api/config` | Demo config (checkpoints, default YAML path) |
| POST | `/api/project/load` | Load + parse a YAML file by path |
| POST | `/api/project/validate` | Validate YAML text without applying |
| POST | `/api/score` | Compute sigmoid + linear penalties for given metric values |
| POST | `/api/optimize/start` | Start live run or replay; returns `run_id` |
| POST | `/api/optimize/stop/{run_id}` | Signal the run to stop |
| GET | `/api/optimize/stream/{run_id}` | SSE stream of optimization events |
| GET | `/api/checkpoint` | List all available checkpoints |
| GET | `/api/checkpoint/{id}` | Load checkpoint data (scores, metrics, params) |
| GET | `/api/checkpoint/{id}/envelope` | Best-ever per metric with pass/fail |
| GET | `/api/checkpoint/{id}/scatter` | X/Y scatter points with feasibility |
| GET | `/api/schematic` | Serve circuit SVG |

SSE events (`/api/optimize/stream/{id}`):

```json
{ "iter": 42, "score": 0.31, "best_score": 0.18, "metrics": {"ugf": 1.9e8}, "best_params": {"X_DUT_M1M2_W": 2e-6} }
{ "heartbeat": true }
{ "done": true }
{ "error": "ngspice exited with code 1: ..." }
```
