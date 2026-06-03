# TODO

## 1. Create Wizard ✅

Adds a step-by-step YAML generator to the Setup tab. Full spec in `PLAN_UI_DESIGN.md`.

### Backend ✅
- [x] `ui/backend/routes/netlist.py` — `POST /api/netlist/parse`: accept `.spice` upload, extract `.param name=val` lines via regex, return `[{name, default_val}]`
- [x] `ui/backend/services/netlist_parser.py` — regex over `.param` lines (no full SPICE parser needed)
- [x] `ui/backend/routes/project.py` — `POST /api/project/generate`: accept wizard form data as JSON, return generated YAML string; `POST /api/project/parse-to-form`: inverse, load YAML → wizard form dict
- [x] `ui/backend/services/yaml_generator.py` — convert wizard form dict → valid `project_setup.yaml` string via PyYAML; `project_dict_to_form()` for round-trip

### Frontend ✅
- [x] `ui/src/components/wizard/WizardShell.tsx` — step navigator, progress bar, back/next buttons, live YAML preview pane
- [x] `ui/src/components/wizard/steps/BasicInfoStep.tsx` — project name, description, simulator dropdown, workspace root
- [x] `ui/src/components/wizard/steps/PDKRulesStep.tsx` — tech name + add key/value constraint rows
- [x] `ui/src/components/wizard/steps/DutParamsStep.tsx` — netlist upload → `POST /api/netlist/parse`, pre-fills param rows (name, min, max, is_integer, log_scale, freeze)
- [x] `ui/src/components/wizard/steps/PVTStep.tsx` — add rows: temp, corner, supply
- [x] `ui/src/components/wizard/steps/TestbenchesStep.tsx` — add/remove testbench cards with netlist upload and param rows
- [x] `ui/src/components/wizard/steps/TargetSpecsStep.tsx` — accordion rows: name, testbench, goal, target, tolerance, range, weight, error_type, reward_type, enable
- [x] `ui/src/components/wizard/steps/OptimizerStep.tsx` — algorithm dropdown (full Nevergrad registry + Ax branch), budget, random seed, optimizer_kwargs editor
- [x] `ui/src/components/tabs/SetupTab.tsx` — segmented Load/Edit ↔ Create Wizard toggle; "Edit in Wizard" button hydrates wizard from current YAML; "Save YAML" switches back to Load/Edit

---

## 2. UX fixes in existing tabs

- [ ] **Wizard DutParamsStep — column header misalignment**: The header row labels (MIN, MAX, INIT, INT, LOG, FRZ) don't line up with the input cells below them. The name cell contains a flex sub-row (input + optional Badge) that shifts subsequent columns. Fix by centering each header label over its cell, or switch the name column to a fixed-width so the badge doesn't affect column widths (`DutParamsStep.tsx` line 94–103).
- [x] **OptimizeTab — Sanity Check "NoneType has no attribute 'ask'"**: Fixed in `ui/backend/routes/sanity.py` — `_create_optimizer_obj()` is now called before `optimization_step()` so `self.optimizer` is never None when `.ask()` is invoked.
- [ ] **SetupTab — Apply from editor content**: "Apply" currently re-reads from disk via `api.loadProject()`, discarding unsaved Monaco edits. Fix: POST current editor content directly to `/api/project/load` (as YAML text, not path), or write to a temp file first.
- [ ] **OptimizeTab — wire algorithm selection to live run**: `api.startRun()` sends `yaml_path` and `budget` but not the chosen algorithm. Pass algorithm name to backend; `optimizer_runner.py` needs to accept and use it.
- [ ] **OptimizeTab — score function toggle**: Add sigmoid vs linear radio button; pass choice to `api.startRun()` and honour it in the optimizer runner (currently fixed by YAML).

---

## 3. Score Shaping tab — multi-metric interactive explorer

The current tab lets you tweak one spec value at a time in isolation. For multi-metric optimization the key insight is _how specs trade off against each other_, not just individual penalty curves. Rewrite the tab around simultaneous multi-spec editing and aggregate score visualization. **All score computation must call `spicexplorer.core.utils.compute_relative_absolute_error` / `compute_relative_sigmoid_error` through the existing `/api/score` backend — do not duplicate the loss math on the frontend.**

- [ ] **Multi-spec value panel**: replace the single spec+slider with a table of all specs, each row having an inline value input (or mini-slider showing ± 3× range). All rows editable simultaneously; a single debounced `POST /api/score` fires with the full vector of current values and returns per-spec and aggregate penalties in one shot.
- [ ] **Equi-score contour overlay**: pick any two specs as X/Y axes; render a 2-D grid of aggregate F(x) values (sweeping those two specs while holding the rest at their current values) as a Plotly `contour` trace. Overlay the current operating point as a crosshair. This shows the optimizer's actual objective landscape and which spec is the binding constraint.
- [ ] **Score contribution waterfall / bar chart**: horizontal bar chart where each spec contributes its signed penalty to the aggregate F(x). Bars colored red (fail) / green (pass). Updates live as values are changed. Replaces the static per-spec breakdown table.
- [ ] **Sigmoid vs linear toggle applies globally**: the score-function radio (sigmoid / linear) should recompute the full multi-spec vector via `/api/score` and redraw all visualizations, not just the single-spec penalty curve.
- [ ] **"Worst-case corner" mode**: allow setting each spec value to its worst-case across PVT corners (loaded from the current project) and see the resulting aggregate score — answers "would this design pass across all corners."

---

## 4. Demo checkpoint format

- [ ] Record new live runs and save their JSON checkpoints to replace CSV demo traces in `ui/app_config.json`. JSON gives richer data (full param history). `checkpoint_reader.py` already supports both formats.

---

## 4. Wizard as DSL builder (refined scope)

The wizard's job is to produce a valid `project_setup.yaml` that fully describes a run — the YAML IS the artifact, the checkpoint, and the way runs are transported between machines. Frame every step around "what does this become in YAML."

### Netlist-driven parameter flow
- [x] **DUT Params step — auto-detect from netlist**: on netlist upload, parse `.param` lines and pre-fill param rows (name, default_val from netlist). Each row has name, min, max, is_integer, log_scale, freeze.
- [ ] **DUT Params step — optional `params.yaml` enrichment**: allow uploading a `params.yaml` that pre-fills bounds, types, and freeze state for matching param names. Auto-merge: netlist provides names + defaults, params.yaml overlays bounds/options. Show a merge preview before commit.
- [ ] **DUT Params step — manual add**: an "Add custom param" button for params not in the netlist (rare but needed for sweeps over derived quantities).
- [x] **Testbenches step — share param auto-detection**: each testbench card has its own netlist upload → same `.param` extraction → param row prefill.

### Target specs auto-config
- [ ] **Target Specs step — auto-discover candidates**: after testbench netlists are uploaded, scan for typical measurement output names (e.g., `.meas` lines in the netlist, or known patterns like `ugf`, `gain_db`, `pm`, `cmrr`). Present them as a checklist of candidate specs the user can enable, with sensible default goal/target/tolerance.
- [ ] **Target Specs step — manual add**: keep the "Add custom spec" path for anything not auto-detected.
- [ ] **Spec library**: ship a small `examples/spec_library.yaml` with standard analog specs (UGF, PM, GBW, slew rate, CMRR, PSRR, etc.) the wizard can offer as one-click adds.

### Wizard plumbing
- [x] **Round-trip with existing YAML**: wizard is openable from an existing YAML via "Edit in Wizard" — `POST /api/project/parse-to-form` parses it and populates every step's form state. Re-emit via "Save YAML".
- [ ] **Live YAML preview**: right pane shows the generated YAML diffing against the previous step's version (highlight changed lines). User can copy or download at any step.
- [ ] **DSL validation surfaced step-by-step**: each step's "Next" button runs schema validation on its slice of the YAML before allowing progress. Errors point to the exact field.

---

## 5. Plot interactivity & customization

The current charts are read-only — `displayModeBar: false` in `PlotlyChart.tsx:38` strips zoom, pan, hover detail, and download. Restore Plotly's full interactivity selectively.

- [ ] **Enable modebar with curated buttons**: in `PlotlyChart.tsx`, switch to `displayModeBar: 'hover'` and configure `modeBarButtonsToRemove` (drop lasso/select; keep zoom, pan, autoscale, download PNG, hover toggles). Optionally add `modeBarButtonsToAdd` for "toggle log scale" on convergence plots.
- [ ] **Per-chart settings popover**: add a small gear icon in each chart panel header that opens a popover with: log/linear axis toggle, marker size, line width, color scheme (zinc/indigo default vs. high-contrast), trace visibility checkboxes. Persist per-chart settings in a Zustand `chartPrefsStore`.
- [ ] **Download data alongside PNG**: extend the Plotly modebar with a custom "Download CSV" button that exports the underlying trace data, not just the rendered image.
- [ ] **Crosshair / synced hover**: across the convergence charts on the same tab, link hover so hovering on iteration N highlights iteration N in all charts simultaneously. Plotly supports this via `hovermode: 'x unified'` + manual event wiring.
- [ ] **Annotations**: let users click-and-drag to annotate a region (e.g., "this is where DE found feasibility") and persist these notes per checkpoint.

---

## 6. KPI / data cards

The current tabs jump straight into charts and tables — no top-line summary. Add a row of cards at the top of each tab.

- [ ] **Card primitive**: `ui/src/components/ui/stat-card.tsx` — props: label, value, unit, delta (optional), trend icon, status color. Reuse the `Panel` aesthetic.
- [ ] **Optimize tab cards**: Best Score, Iterations Run, Specs Passing (e.g. "4/6 ✅"), Elapsed Time, Est. Remaining.
- [ ] **Score Shaping tab cards**: Aggregate F(x), Highest Penalty Spec, Active Spec Count.
- [ ] **Explorer tab cards** (per loaded run): Final Score, Pareto Frontier Size, Spec Pass Rate, Worst Spec. Side-by-side for Run A and Run B with a delta column.
- [ ] **Setup tab cards**: # DUT Params, # Testbenches, # Target Specs, PVT Corners — quick sanity check that the loaded project is what the user expected.

---

## 7. Better exploration visualization

The Explorer tab today is scatter + envelope table. For real design-space exploration, designers need more.

- [ ] **Parallel coordinates plot**: one axis per spec + per design parameter, one line per evaluated point, colored by feasibility or score. Classic multidimensional optimization viz; Plotly supports it natively (`parcoords`).
- [ ] **Pareto front overlay on scatter**: highlight non-dominated points in the metric scatter with a distinct color and connecting line. Compute via simple O(n²) sweep — n is small (typically <2000 points).
- [ ] **Brushing & linking**: select a region in the scatter → highlight the same points in the convergence chart and parallel coordinates → filter the Best Designs table to just those points. Foundational interactive-viz pattern.
- [ ] **Design point inspector**: click any scatter point → side drawer opens with full param values, all metric values, iteration number, and a "Re-simulate this point" action that drops the params into a new run.
- [ ] **Convergence comparison improvements**: add ribbon/std-band for multiple replays of the same algorithm, and let the user overlay 3+ runs (not just A vs B).
- [ ] **Spec sensitivity view**: small-multiples bar chart showing each spec's contribution to the final score across the run — answers "which spec was hardest."

---

## 8. UI density & editing affordances

Things get crowded once a project has 10+ params and 6+ specs. Reduce clutter without hiding data.

- [ ] **Expandable rows in tables**: DUT params, target specs, and testbenches tables should have a chevron column that expands the row to show full configuration inline (bounds, log_scale, freeze, etc.) instead of cramming everything into columns.
- [ ] **Hover popover for inline edits**: hovering on a param value or spec target shows an "edit" pencil; clicking opens a popover (not a modal) with the relevant fields, an Apply, and a Cancel. Keeps the user in context. Pick a small popover lib (e.g., Radix Popover) or build one on top of the existing primitives.
- [ ] **Collapsible panels**: each panel (`PanelHeader`) should be collapsible with a chevron. Save collapse state to localStorage per tab.
- [ ] **Compact / comfortable density toggle**: a global toggle in the top nav that switches between dense (8 px padding, 11 px font) and comfortable (12 px / 13 px) modes. Useful for screen-share vs. local work.
- [ ] **Sticky tab actions**: Apply, Start Run, etc. should stick to the top of the tab when scrolling so they remain reachable on long pages.
- [ ] **Keyboard shortcuts**: `g s` → Setup, `g o` → Optimize, `g e` → Explorer, `?` → show shortcut sheet. Small win for demo polish.

---

## 9. Misc polish

- [ ] **Export entire run report**: a "Download report" button on the Optimize and Explorer tabs that packages the YAML + JSON checkpoint + all chart PNGs + a summary markdown into a single zip. Useful for sharing post-run.
- [ ] **Recent runs sidebar**: a left rail listing the last N runs (replays + live) with timestamps, status, and a click-to-load action that drops the run into the Explorer tab.
- [ ] **Validation error inline highlighting in Monaco**: the YAML editor already validates with a 600 ms debounce; surface errors as Monaco markers (red squiggles) with hover tooltips, not just a panel below.
