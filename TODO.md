# TODO

> **Context:** since this list was first written the UI migrated from the 4-tab shell to the
> **Studio workspace** (see [PLAN_UI_DESIGN.md](PLAN_UI_DESIGN.md) + [CLAUDE.md](CLAUDE.md)). Several
> large pieces landed that were never tracked here as line items: the persistent Studio shell (7
> views, activity bar, per-activity left rails, always-on right rail, bottom panel, status bar), the
> **⌘K command palette + wizard overlay**, **run history** (`runStore` + localStorage), the
> **Schematic** view (Xschem viewer + `DeviceInspector` + sensitivity endpoint), the read-only
> **Pipeline** DAG view, the **Health** view + **PDK-aware degradation** (`GET /api/env`), the
> **sanity-check** endpoint, and **long-run checkpointing** (autosave + resume). The items below are
> what remains.

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

- [x] **Wizard DutParamsStep — column header misalignment**: Fixed — the header row (`DutParamsStep.tsx:94`) and the value rows (`:105`) now share the same `grid-cols-[1.6fr_1.1fr_1.1fr_0.8fr_0.5fr_0.5fr_0.5fr_auto]` template, so labels line up with their cells.
- [x] **OptimizeTab — Sanity Check "NoneType has no attribute 'ask'"**: Fixed in `ui/backend/routes/sanity.py` — `_create_optimizer_obj()` is now called before `optimization_step()` so `self.optimizer` is never None when `.ask()` is invoked.
- [ ] **SetupTab — Apply from editor content**: "Apply" still re-reads from disk via `api.loadProject(yamlPath)` (`SetupTab.tsx:104`), discarding unsaved Monaco edits. Fix: POST current editor content directly to `/api/project/load` (as YAML text, not path), or write to a temp file first.
- [x] **OptimizeTab — wire algorithm selection to live run**: Done (commit `4ad528d`). The chosen algorithm/budget/seed flow through `lib/launchRun.ts` → `POST /api/optimize/start` and are applied in-memory by `optimizer_runner._apply_overrides`.
- [x] **OptimizeTab — score function toggle**: *Resolved (BUG-10)* — the sigmoid/linear `Segmented` control was a **dead control** (its value was never read or sent to the backend), so it was **removed** to stop misleading the user. The score/error type is configured per-spec via the YAML's `error_type`. Re-add later only if it is properly wired through `launchRun.ts` → `optimizer_runner._apply_overrides`.

---

## 3. Score Shaping tab — multi-metric interactive explorer

The current tab lets you tweak one spec value at a time in isolation. For multi-metric optimization the key insight is _how specs trade off against each other_, not just individual penalty curves. Rewrite the tab around simultaneous multi-spec editing and aggregate score visualization. **All score computation must call `spicexplorer.core.utils.compute_relative_absolute_error` / `compute_relative_sigmoid_error` through the existing `/api/score` backend — do not duplicate the loss math on the frontend.**

- [x] **Multi-spec value panel** *(core done — BUG-02)*: the tab now keeps a per-spec value map and sends the **full vector** of current values on each debounced `POST /api/score`, so the per-spec breakdown table and the aggregate F(x) reflect **all** specs simultaneously (previously only the selected spec). Still open: the richer per-row inline-input/mini-slider UI (today there is one shared slider for the selected spec).
- [ ] **Equi-score contour overlay**: pick any two specs as X/Y axes; render a 2-D grid of aggregate F(x) values (sweeping those two specs while holding the rest at their current values) as a Plotly `contour` trace. Overlay the current operating point as a crosshair. This shows the optimizer's actual objective landscape and which spec is the binding constraint.
- [ ] **Score contribution waterfall / bar chart**: horizontal bar chart where each spec contributes its signed penalty to the aggregate F(x). Bars colored red (fail) / green (pass). Updates live as values are changed. Replaces the static per-spec breakdown table.
- [ ] **Sigmoid vs linear toggle applies globally**: the score-function radio (sigmoid / linear) should recompute the full multi-spec vector via `/api/score` and redraw all visualizations, not just the single-spec penalty curve.
- [ ] **"Worst-case corner" mode**: allow setting each spec value to its worst-case across PVT corners (loaded from the current project) and see the resulting aggregate score — answers "would this design pass across all corners."

---

## 4. Demo checkpoint format

- [ ] Record new live runs and save their JSON checkpoints to replace CSV demo traces in `ui/app_config.json`. JSON gives richer data (full param history). `checkpoint_reader.py` already supports both formats.

---

## 5. Wizard as DSL builder (refined scope)

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

## 6. Plot interactivity & customization

The current charts are read-only — `displayModeBar: false` in `PlotlyChart.tsx:38` strips zoom, pan, hover detail, and download. Restore Plotly's full interactivity selectively.

- [ ] **Enable modebar with curated buttons**: in `PlotlyChart.tsx`, switch to `displayModeBar: 'hover'` and configure `modeBarButtonsToRemove` (drop lasso/select; keep zoom, pan, autoscale, download PNG, hover toggles). Optionally add `modeBarButtonsToAdd` for "toggle log scale" on convergence plots.
- [ ] **Per-chart settings popover**: add a small gear icon in each chart panel header that opens a popover with: log/linear axis toggle, marker size, line width, color scheme (zinc/indigo default vs. high-contrast), trace visibility checkboxes. Persist per-chart settings in a Zustand `chartPrefsStore`.
- [ ] **Download data alongside PNG**: extend the Plotly modebar with a custom "Download CSV" button that exports the underlying trace data, not just the rendered image.
- [ ] **Crosshair / synced hover**: across the convergence charts on the same tab, link hover so hovering on iteration N highlights iteration N in all charts simultaneously. Plotly supports this via `hovermode: 'x unified'` + manual event wiring.
- [ ] **Annotations**: let users click-and-drag to annotate a region (e.g., "this is where DE found feasibility") and persist these notes per checkpoint.

---

## 7. KPI / data cards

The current tabs jump straight into charts and tables — no top-line summary. Add a row of cards at the top of each tab.

- [ ] **Card primitive**: `ui/src/components/ui/stat-card.tsx` — props: label, value, unit, delta (optional), trend icon, status color. Reuse the `Panel` aesthetic.
- [ ] **Optimize tab cards**: Best Score, Iterations Run, Specs Passing (e.g. "4/6 ✅"), Elapsed Time, Est. Remaining.
- [ ] **Score Shaping tab cards**: Aggregate F(x), Highest Penalty Spec, Active Spec Count.
- [ ] **Explorer tab cards** (per loaded run): Final Score, Pareto Frontier Size, Spec Pass Rate, Worst Spec. Side-by-side for Run A and Run B with a delta column.
- [ ] **Setup tab cards**: # DUT Params, # Testbenches, # Target Specs, PVT Corners — quick sanity check that the loaded project is what the user expected.

---

## 8. Better exploration visualization

The Explorer tab today is scatter + envelope table. For real design-space exploration, designers need more.

- [ ] **Parallel coordinates plot**: one axis per spec + per design parameter, one line per evaluated point, colored by feasibility or score. Classic multidimensional optimization viz; Plotly supports it natively (`parcoords`).
- [ ] **Pareto front overlay on scatter**: highlight non-dominated points in the metric scatter with a distinct color and connecting line. Compute via simple O(n²) sweep — n is small (typically <2000 points).
- [ ] **Brushing & linking**: select a region in the scatter → highlight the same points in the convergence chart and parallel coordinates → filter the Best Designs table to just those points. Foundational interactive-viz pattern.
- [ ] **Design point inspector**: click any scatter point → side drawer opens with full param values, all metric values, iteration number, and a "Re-simulate this point" action that drops the params into a new run.
- [ ] **Convergence comparison improvements**: add ribbon/std-band for multiple replays of the same algorithm, and let the user overlay 3+ runs (not just A vs B).
- [ ] **Spec sensitivity view**: small-multiples bar chart showing each spec's contribution to the final score across the run — answers "which spec was hardest." *(Distinct from the already-shipped device-parameter sensitivity in the Schematic `DeviceInspector` (`SensitivityChart` + `GET /api/spec/{name}/sensitivity`), which is finite-difference of metrics vs. DUT params, not per-spec score contribution across a run.)*

---

## 9. UI density & editing affordances

Things get crowded once a project has 10+ params and 6+ specs. Reduce clutter without hiding data.

- [ ] **Expandable rows in tables**: DUT params, target specs, and testbenches tables should have a chevron column that expands the row to show full configuration inline (bounds, log_scale, freeze, etc.) instead of cramming everything into columns.
- [ ] **Hover popover for inline edits**: hovering on a param value or spec target shows an "edit" pencil; clicking opens a popover (not a modal) with the relevant fields, an Apply, and a Cancel. Keeps the user in context. Pick a small popover lib (e.g., Radix Popover) or build one on top of the existing primitives.
- [ ] **Collapsible panels**: each panel (`PanelHeader`) should be collapsible with a chevron. Save collapse state to localStorage per tab.
- [ ] **Compact / comfortable density toggle**: a global toggle in the top nav that switches between dense (8 px padding, 11 px font) and comfortable (12 px / 13 px) modes. Useful for screen-share vs. local work.
- [ ] **Sticky tab actions**: Apply, Start Run, etc. should stick to the top of the tab when scrolling so they remain reachable on long pages.
- [~] **Keyboard shortcuts**: *Partially shipped.* The ⌘K command palette and `⌘1`..`⌘7` / bare-key view switching are wired (`CommandPalette.tsx` global keydown, shortcuts in `nav.ts`). Still open: the `g s`/`g o` chord style and a `?` shortcut-help sheet.

---

## 10. Misc polish

- [ ] **Export entire run report**: a "Download report" button on the Optimize and Explorer tabs that packages the YAML + JSON checkpoint + all chart PNGs + a summary markdown into a single zip. Useful for sharing post-run.
- [x] **Recent runs sidebar**: Done — the run-centric left rail (`shell/rails/RunsRail.tsx`) lists `runStore.history` (replays + live) with score sparklines and a click-to-load (`rerun`) action, plus checkpoint resume/delete. Backed by `runStore.history` persisted to localStorage.
- [ ] **Validation error inline highlighting in Monaco**: the YAML editor already validates with a 600 ms debounce; surface errors as Monaco markers (red squiggles) with hover tooltips, not just a panel below.

---

## 11. Bug fixes — functional audit (2026-06)

Actionable list from the audit. Full per-bug location / root cause / fix direction in [bug_report.md](bug_report.md) (IDs match). 39 confirmed (17 major, 22 minor); the two pre-flagged regressions lead.

> **Status (2026-06): 37 of 39 fixed in this branch** — verified by `pytest` (+7 new regression tests), `ruff`, `tsc`, `eslint` (0 warnings) and `next build`. Exceptions: **BUG-01** has the graceful-degradation half (a "symbols unavailable" banner); full symbol rendering needs the PDK xschem symbol library vendored into the backend image (infra). **BUG-35** keeps its `pvt_map`/`pvt_corners` parsing **deferred with the PVT work** (its `freeze`/default half is done).

### Flagged regressions
- [x] **BUG-01** *(partial — graceful "symbols unavailable" banner; full vendoring deferred)* · `ui/backend/routes/xschem.py` — Schematic symbols all render as red "?" placeholders under Docker: PDK xschem libs and xschem binary absent, so…
- [x] **BUG-02** · `ui/src/components/tabs/ScoreShapingTab.tsx` — Score Shaping aggregate F(x) reflects only the selected spec, not all configured specs

### Major
- [x] **BUG-03** · `ui/src/components/tabs/SetupTab.tsx` — Applying uploaded/edited YAML leaves yamlPath empty: Optimize and Sensitivity silently target the DEFAULT cascode project…
- [x] **BUG-07** · `ui/backend/services/yaml_generator.py` — Wizard target spec with blank Range silently coerces to nan, poisoning optimization penalty/reward normalization
- [x] **BUG-09** · `ui/src/stores/runStore.ts` — Live-run SSE error events are silently dropped — user never sees why a run died
- [x] **BUG-12** · `ui/src/stores/runStore.ts` — EventSource onerror unconditionally finishes the run and cancels reconnect, orphaning the backend optimizer thread
- [x] **BUG-17** · `ui/src/components/tabs/ExplorerTab.tsx` — Deselecting run A (or B) and clicking "Load both" leaves the stale run rendering (no clear path in loadBoth)
- [x] **BUG-18** · `ui/src/components/tabs/ExplorerTab.tsx` — Explorer scatter X/Y and selectedMetric default to hardcoded cascode metric names and are never reconciled against the loaded…
- [x] **BUG-22** · `ui/src/components/schematic/SchematicViewer.tsx` — Symbol text labels mis-anchored on rotated instances (transformPoint rotates opposite to instanceTransform)
- [x] **BUG-26** · `ui/src/components/tabs/ScoreShapingTab.tsx` — Score Shaping deep-link permanently locks the spec dropdown — every dropdown pick snaps back to the deep-linked spec
- [x] **BUG-27** · `ui/src/components/shell/RightRail.tsx` — Right rail "best score" uses Math.min over a max-tracking best_score series, showing the worst (first) best-so-far
- [x] **BUG-32** · `src/spicexplorer/core/domains.py` — OptimizationLog mutable default list is shared across all default-constructed instances, leaking trials between…
- [x] **BUG-33** · `ui/backend/routes/score.py` — score.py _project_cache never invalidated — /api/score returns stale spec values after the on-disk YAML changes
- [ ] **BUG-35** *(deferred — PVT)* · `src/spicexplorer/core/domains.py` — from_yaml silently drops unrecognized YAML keys (dacite strict=False): freeze_to, pvt_corner name/enable, and tech_spec.pvt_map… *(the `freeze` half is fixed; the pvt_map/pvt_corners parsing is deferred with the PVT work)*
- [x] **BUG-36** · `ui/backend/services/optimizer_runner.py` — Live-run SSE best_params are optimizer-space (normalized) values, not physical — inconsistent with checkpoints/resume/replay…
- [x] **BUG-37** · `src/spicexplorer/optimization/base.py` — Omitted target_spec `range` becomes NaN normalizing_coeff and poisons every score in the library evaluate/sanity path…
- [x] **BUG-39** · `src/spicexplorer/optimization/stochastic/nevergrad.py` — Nevergrad parameterize() ignores Param.freeze — frozen params are still swept as free optimization variables

### Minor
- [x] **BUG-04** · `ui/backend/services/yaml_generator.py` — Wizard parse-to-form defaults omitted `freeze` to False, contradicting the library's `freeze=True` dataclass default
- [x] **BUG-05** · `ui/src/components/wizard/optimizer-registry.ts` — Nevergrad "configurable family" optimizers (SamplingSearch, DifferentialEvolution) are unselectable in the wizard dropdown though…
- [x] **BUG-06** · `ui/src/components/wizard/steps/DutParamsStep.tsx` — Duplicate DUT param names silently collapse to one search dimension (no uniqueness validation in load/wizard/parameterize)
- [x] **BUG-08** · `ui/backend/services/netlist_parser.py` — Netlist .param parser does not strip ngspice `$` inline comments (but `;` IS handled, contrary to the claim)
- [x] **BUG-10** · `ui/src/components/tabs/OptimizeTab.tsx` — Score function toggle (sigmoid/linear) in Optimize toolbar is a dead control
- [x] **BUG-11** · `ui/backend/services/optimizer_runner.py` — _run_replay has no try/finally — a corrupt/unparseable checkpoint never sends the done sentinel, leaving the UI stuck "running"
- [x] **BUG-13** · `ui/backend/services/optimizer_runner.py` — RunState entries accumulate forever in the module-level _runs registry (unbounded growth)
- [x] **BUG-14** · `ui/src/components/shell/rails/RunsRail.tsx` — History replay click in RunsRail orphans the active run (no isRunning guard, no stopRun on prior run)
- [x] **BUG-15** · `ui/src/components/tabs/OptimizeTab.tsx` — Replay progress shows the live-run budget (default 200) instead of the checkpoint length
- [x] **BUG-16** · `ui/src/components/tabs/ExplorerTab.tsx` — Metric-scatter colors runs by array position, so run B renders in run A's indigo when A has no scatter points
- [x] **BUG-19** · `ui/backend/services/checkpoint_reader.py` — read_json_checkpoint assumes every fit_summary value is a {curr_val} dict; bare-float fit_summary from the Bode SPICE optimizer…
- [x] **BUG-20** · `ui/src/components/tabs/ExplorerTab.tsx` — Unbounded full-resolution checkpoint/scatter payloads spread into Math.min(...)/Math.max(...) and Plotly — RangeError + memory…
- [x] **BUG-21** · `ui/src/components/tabs/ExplorerTab.tsx` — Envelope 'winner' awards ties to B and labels a winner when only one run is loaded
- [x] **BUG-23** · `ui/src/lib/xschem/parser.ts` — parseAttrs does not skip whitespace around '=', mis-parsing spaced key/value attrs like {dash = 4}
- [x] **BUG-24** · `ui/src/components/tabs/SchematicTab.tsx` — Stale-response race: rapid schematic navigation via the "Open" dropdown can let an earlier load clobber a later one
- [x] **BUG-25** · `ui/src/components/schematic/DeviceInspector.tsx` — Inspector slider 'nominal' default (range midpoint) can disagree with the backend's simulated baseline (_nominal prefers val/init)
- [x] **BUG-28** · `ui/src/components/tabs/HealthTab.tsx` — Health check drops the backend's "PDK missing" verdict
- [x] **BUG-29** · `ui/backend/routes/optimize.py` — Backend /optimize/start has no server-side PDK/live-runs guard; gating is client-only
- [x] **BUG-30** · `ui/src/components/tabs/ScoreShapingTab.tsx` — Deep-linking a disabled (enable:false) spec into Score Shaping silently no-ops
- [x] **BUG-31** · `ui/src/components/overlays/CommandPalette.tsx` — ⌘K "Jump to run" only highlights the rail row; no center view consumes selectedRunId, so the deep-link loads nothing
- [x] **BUG-34** · `src/spicexplorer/core/domains.py` — from_yaml silently accepts duplicate dut_param names (no uniqueness validation; example YAML defines x_dut_Vb1 twice)
- [x] **BUG-38** · `src/spicexplorer/core/domains.py` — DutParams.get_frozen_params does float(p.init) with no None-guard; crashes on default params (dead code)

## 12. Manual simulation feature

Run all enabled testbenches once for a chosen DUT-param vector, reusing the optimizer's sim infra at run-count 1. Full design in [PVT_plan.md](PVT_plan.md) §"Part B — Manual simulation feature".

- [ ] **Backend route** `POST /api/simulate/once` in `ui/backend/routes/simulate.py` (sibling of `sanity.py`): load project, build wrappers, instantiate `Nevergrad_Spice_Single_Objective`, call `evaluate(params, append_to_log=False)` (`optimization/base.py:837`) — do **not** call `parameterize()`/`_create_optimizer_obj()`/`optimization_step()` (those `ask()` a random point).
- [ ] Promote `_build_spicelib_wrappers` to a shared helper (currently mirrored in `optimizer_runner.py` and `sanity.py:110-125`); reuse `probe_pdk` + `_tail_log` + `run_in_executor` from `sanity.py`.
- [ ] Register router in `ui/backend/main.py`; add `simulateOnce()` to `ui/src/lib/api.ts` + `SimulateOnceResponse` to `ui/src/types/api.ts`.
- [ ] **Mode A — load from prior result**: accept `{checkpoint_id, point}`, resolve via `_resolve_checkpoint_path` + `read_checkpoint`; stored params are already engineering-real → feed `evaluate` with no transform. Default to best point (argmax of `scores`).
- [ ] **Mode B — manual values**: accept `{params}`; pre-fill a form from `project.dut_params` using `Param.init` as default + `min_val`/`max_val` hints; partial dicts allowed (unset params keep netlist defaults). Range-check against bounds; respect `is_integer` and the `C*`/`R*` suffix convention.
- [ ] **UI**: collapsible "Manual Sim / Evaluate Point" panel in `OptimizeTab` (PDK-gated like live Start); `Segmented` source toggle (checkpoint | manual); per-spec result table (value vs target, pass/fail, score) + total score + per-testbench log tails; optional "Send to Score Shaping".
- [ ] **Interface gaps to close** (optional but clean): add a `Base_Optimizer.simulate_point(params)` façade; give manual-sim wrappers a distinct output subfolder (`outdir/manual_sim`) so they don't clobber a live run's outputs (`_validate` rmtrees `output_folder`).

## 13. PVT corners — Phase 1 (single chosen corner)

Make corners first-class and actually drive the sim against **one** active corner; the optimizer loop, scorer, and per-trial flow stay untouched. Full design + exact change list in [PVT_plan.md](PVT_plan.md) §"Phase 1". (Note: `tech_spec.pvt_map` and the flat `pvt_corners` are currently **dead config** — silently dropped by non-strict dacite; no `.lib`/temp/supply injection exists today.)

- [ ] **Config schema**: add a top-level `pvt:` block (`active_corner`, reusable `process_bundles`, `corners[]` = process includes + temp + supply + params, `enabled`). Subsumes/replaces `pvt_map` + `pvt_corners`; keep a back-compat shim. See the copy-pasteable YAML in PVT_plan.md.
- [ ] **`core/domains.py`**: add `ModelInclude` / `SupplyOverride` / `Corner` / `PVTConfig` dataclasses; add `pvt: Optional[PVTConfig] = None` to `Project_Setup`; in `from_yaml()` pre-expand each corner's `process:` bundle into `model_includes` before `safe_from_dict`. Leave `TechSpec`/`PVT`/`pvt_corners` as-is (no regression).
- [ ] **`spice_engine/spicelib.py`**: add `NGSpice_Wrapper.apply_corner(corner)` — strip the hardcoded `.lib` line (`remove_Xinstruction`), add the ordered cross-family `.lib <file> <section>` includes (`add_instruction`), emit `.options temp=<t>`, and `set_parameter` for supply/extra params. Isolate all ngspice-specific syntax here (PDK-agnostic seam).
- [ ] **`optimization/base.py`**: in `Spice_Base_Optimizer.__post_init__()` (the existing one-time tb-param setup, ~`:463-475`) apply the active corner once to each enabled testbench wrapper, before the loop. Persists across trials via `SpiceEditor` state.
- [ ] **UI (optional, ephemeral)**: surface `pvt.active_corner` + corner list in `routes/project.py`; let `optimizer_runner._apply_overrides` accept an ephemeral `active_corner` override (in-memory, never rewrite YAML) → gives "switch to a defined corner" for free.
- [ ] **Stays untouched**: nevergrad/Ax optimizer classes, `optimization_step`, `optimize()`, `simulate_circuit`, `evaluate`, `compute_fitness*`, `core/utils` error/reward fns, checkpoint schema, charts.

## 14. PVT corners — Phase 2 (DEFERRED / research)

> **Deferred — not scheduled this round.** Once corners drive a single sim (Phase 1), run the full **testbench × corner** cross-product and collapse **N corner-scores into one scalar** the optimizer consumes. The open research question is the aggregation strategy — candidates: worst-case/min, weighted mean, sum-of-penalties, must-pass-all (constraint), or Pareto/multi-objective (Ax). No strategy is committed. Secondary deferred items: where the corner loop lives, `parallel_sim` fan-out (N× ngspice processes), and `{corner}::{spec}` checkpoint key namespacing (+ the dotted-column `.iterrows()` caveat). See [PVT_plan.md](PVT_plan.md) §"Phase 2".

## 15. UI layout fixes (CSS / overflow / scroll)

From the layout audit — full per-tab findings + global root causes in [ui_layout_report.md](ui_layout_report.md).

- [x] **RC-1 (done) / RC-2 (largely subsumed)**: added `w-full min-w-0` to `inputCn` + `min-w-0` to `Field` (`wizard-controls.tsx`) and `min-w-0` to `selectCn` (`select.tsx`). Deliberately kept `selectCn` off `w-full` so toolbar selects don't stretch to fill the row. Fixes the reported PVT Supply-column overflow and the input-heavy steps.
- [x] **RC-4**: `TabStrip` and `StatusBar` now use `overflow-x-auto whitespace-nowrap [&>*]:shrink-0` (StatusBar footer gets `overflow-hidden`) so labels scroll/truncate instead of wrapping on a narrow center column.
- [x] **RC-3**: added `min-w-0` to truncating spans + `shrink-0` to fixed siblings in `StudioLeftRail` and `StatusBar` so long project names ellipsize.
- [ ] **Sanity Check "clipped-no-scroll"**: confirmed **not** a source CSS defect (the height/scroll chain is intact) — delete `ui/.next` and rebuild if it recurs.

## 16. Redundancy cleanup (low-risk first)

From the redundancy survey — full list + risk ratings in [project_redundancy.md](project_redundancy.md).

- [x] Deleted unused `formatNumber` (`lib/utils.ts`); removed dead `uiStore` fields (`compareRunA`/`compareRunB`/`setCompare`/`setSelectedRunId`); removed the dead `DutParams` class (`domains.py`, BUG-38). *(The one-tab `bottomTab` collapse is deferred — purely cosmetic.)*
- [x] Consolidated `_safe_float` (3 copies) into `ui/backend/services/num.py`; `config.py` now calls `_infer_score_fn` (fixes the `"linear"` divergence vs `checkpoint.py`).
- [x] Extracted `_target_specs_from_yaml` in `checkpoint.py`. *(Deferred — cosmetic/behavior-changing: the `score_service` penalty-helper extraction, the shared `goalSym` glyph helper, and routing all FE pass/fail through `statusForGoal`.)*
- [x] Renamed the duplicate `x_dut_Vb1`→`x_dut_Vb2` in the example YAML (matches the netlist), and `from_yaml` now **rejects** duplicate dut_param names (BUG-34). *(The always-empty `CheckpointMeta.n_iters` branch is deferred.)*
- [ ] **Decide (needs sign-off)**: fate of `demo/newcas_demo_runner.py` and the `optimization/rl/` subtree — both dead relative to the webapp but carry dependent tests.
