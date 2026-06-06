# TODO

> **Context:** since this list was first written the UI migrated from the 4-tab shell to the
> **Studio workspace** (see [PLAN_UI_DESIGN.md](PLAN_UI_DESIGN.md) + [CLAUDE.md](CLAUDE.md)). Several
> large pieces landed that were never tracked here as line items: the persistent Studio shell (7
> views, activity bar, per-activity left rails, always-on right rail, bottom panel, status bar), the
> **⌘K command palette + wizard overlay**, **run history** (`runStore` + localStorage), the
> **Schematic** view (Xschem viewer + `DeviceInspector` + sensitivity endpoint), the read-only
> **Pipeline** DAG view, the **Health** view + **PDK-aware degradation** (`GET /api/env`), the
> **sanity-check** endpoint, **PVT Phase 1** (single chosen corner driving the sim), the
> **manual single-sim** panel, and **long-run checkpointing** (autosave + resume). The items below
> are what remains.
>
> **Audit note (2026-06, static-analysis round):** this refresh is a code-reading pass on the
> server at HEAD — **no app/docker/live-sim was run** (the docker daemon is unreachable this round).
> The new bug items in §17 are confirmed *present in source* by reading each cited file at HEAD;
> items that would need a live UI/sim to confirm are tagged **needs runtime verification (deferred)**.
> The §11 list (prior audit) stays as historical record — most of it shipped; do not re-derive it.

## 0. 2026-06 completion pass (this branch) ✅

A focused pass landed the bulk of the remaining bounded/medium items below. Verified by
`uv run pytest` (70 passed; the 1 failure is the PDK-gated `test_ngspice_sanity_check`, which
only passes in the Docker backend), `tsc --noEmit`, `eslint --max-warnings=0`, and
`next build` (all green), plus +12 backend regression tests in
[tests/test_ui_phase_completion.py](tests/test_ui_phase_completion.py).

**Landed:**
- **§2** Apply-from-editor: edited Monaco buffer is now applied (not a stale disk re-read);
  the backend anchors a relative `ws_root` to the original YAML's dir so resolution is preserved.
- **§3** Score Shaping: per-row inline try-value inputs, a **score-contribution waterfall** chart,
  and a **sigmoid/linear global toggle** (drives the waterfall + F(x) KPI).
- **§5** Wizard DSL: **spec library** (`examples/spec_library.yaml` + `/api/spec-library` + one-click
  add), **`.meas` auto-discovery** (upload → candidate checklist), preview **Copy/Download**. (`§5b`/`§5d`
  manual-add were already done.)
- **§6** Plot interactivity: curated **modebar** (`displayModeBar:'hover'`, no lasso/logo) + a custom
  **Download-CSV** button, centralized in `PlotlyChart`.
- **§7** KPI **stat-card rows** on Setup, Optimize (incl. elapsed/est-remaining via `runStartTs`),
  Score Shaping, and Explorer (A vs B + delta).
- **§8** Explore: **Pareto-front overlay**, **parallel-coordinates** chart, and a **design-point
  inspector** (click a scatter point → params/metrics → **Re-simulate** via `/api/simulate/once`).
- **§9** `g`-chord view nav + **`?` shortcut-help sheet**; sticky tab actions were already satisfied by
  the shell layout.
- **§10** **Monaco inline validation markers**, and a **run-report zip** (`/api/checkpoint/{id}/report`
  → checkpoint + YAML + `summary.md`).
- **§15** all wizard layout/overflow fixes (PVT/Testbenches/PDK/Optimizer/TargetSpecs/BasicInfo grids +
  selects, RightRail `<td>` truncation, wizard wrapper height).
- **§16** deleted the broken orphan `symbolic.py`; populated `n_iters` in the checkpoint listing;
  deduped the `score_service` penalty block; one shared `goalSymbol` helper across 8 sites; routed
  `PipelineView` pass/fail through `statusForGoal`.

**Deferred (with rationale — NOT done this pass):**
- **Needs product sign-off** (irreversible / possibly-intentional): §16 RL backend + demo-runner
  deletion, §16 score-service↔library scorer unification, §16 dual-schematic-path retirement.
- **Blocked**: §3 worst-case-corner (PVT Phase 2), §4 JSON demo checkpoints (needs curated real-run data;
  the reader already supports JSON).
- **Large / lower-ROI**: §3 equi-score contour, §6 per-chart prefs popover + synced-hover + drag-annotations,
  §8 brushing-&-linking + 3+-run ribbon + per-spec score small-multiples, §9 expandable rows + hover-popover
  edit + density toggle (primitive `CollapsiblePanel` added but not retrofitted), §5 params.yaml enrichment +
  per-step validation + live-diff highlighting.

---

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
- [ ] **"Worst-case corner" mode**: allow setting each spec value to its worst-case across PVT corners (loaded from the current project) and see the resulting aggregate score — answers "would this design pass across all corners." *(Blocked on PVT Phase 2 multi-corner aggregation — §16.)*

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

> ⚠️ **Round-trip fidelity gaps** found in the 2026-06 audit — these break the wizard's "the YAML IS the artifact" guarantee and are tracked as bugs in §17 (BUG-A2, BUG-A3, BUG-A6). Fix before adding more wizard scope.

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
- [ ] **Design point inspector**: click any scatter point → side drawer opens with full param values, all metric values, iteration number, and a "Re-simulate this point" action that drops the params into a new run. *(The manual single-sim primitive — §13 — is the natural backend for "Re-simulate this point".)*
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

## 11. Bug fixes — functional audit (2026-06, FIRST round)

Actionable list from the **first** audit. Full per-bug location / root cause / fix direction in [bug_report.md](bug_report.md) (IDs match). 39 confirmed (17 major, 22 minor); the two pre-flagged regressions lead.

> **Status (2026-06): 37 of 39 fixed in this branch** — verified by `pytest` (+7 new regression tests), `ruff`, `tsc`, `eslint` (0 warnings) and `next build`. Exceptions: **BUG-01** had only the graceful-degradation half (a "symbols unavailable" banner) — see §17 BUG-A1 for the now-actionable remainder on the **server** (the PDK IS present here). **BUG-35** keeps its `pvt_map`/`pvt_corners` parsing **deferred** (its `freeze`/default half is done).

### Flagged regressions
- [x] **BUG-01** *(partial — graceful "symbols unavailable" banner; full rendering now actionable on the server, see §17 BUG-A1)* · `ui/backend/routes/xschem.py` — Schematic symbols all render as red "?" placeholders when the backend lacks the PDK xschem libs.
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
- [x] **BUG-25** · `ui/src/components/schematic/DeviceInspector.tsx` — Inspector slider 'nominal' default (range midpoint) can disagree with the backend's simulated baseline (_nominal prefers val/init) *(re-opened in part — see §17 BUG-A2: dut_param string `val` is still never resolved)*
- [x] **BUG-28** · `ui/src/components/tabs/HealthTab.tsx` — Health check drops the backend's "PDK missing" verdict
- [x] **BUG-29** · `ui/backend/routes/optimize.py` — Backend /optimize/start has no server-side PDK/live-runs guard; gating is client-only
- [x] **BUG-30** · `ui/src/components/tabs/ScoreShapingTab.tsx` — Deep-linking a disabled (enable:false) spec into Score Shaping silently no-ops
- [x] **BUG-31** · `ui/src/components/overlays/CommandPalette.tsx` — ⌘K "Jump to run" only highlights the rail row; no center view consumes selectedRunId, so the deep-link loads nothing
- [x] **BUG-34** · `src/spicexplorer/core/domains.py` — from_yaml silently accepts duplicate dut_param names (no uniqueness validation; example YAML defines x_dut_Vb1 twice)
- [x] **BUG-38** · `src/spicexplorer/core/domains.py` — DutParams.get_frozen_params does float(p.init) with no None-guard; crashes on default params (dead code)

---

## 12. Manual simulation feature

Run all enabled testbenches once for a chosen DUT-param vector, reusing the optimizer's sim infra at run-count 1. Full design in [PVT_plan.md](PVT_plan.md) §"Part B — Manual simulation feature".

> **Status: landed** (commits `37952bf`, `c409031`). `POST /api/simulate/once` is registered in
> `ui/backend/main.py:60`, wired in `ui/src/lib/api.ts:143` (`simulateOnce`) +
> `ui/src/types/api.ts:289` (`SimulateOnceResponse`), and surfaced by
> `ui/src/components/pvt/ManualSimPanel.tsx` in OptimizeTab. It calls
> `evaluate(params, append_to_log=False)` (the correct primitive, not `ask()`), is PDK-gated, and
> isolates outputs into `outdir/manual_sim`.

### Landed ✅
- [x] **Backend route** `POST /api/simulate/once` in `ui/backend/routes/simulate.py` — loads the project, builds wrappers, instantiates `Nevergrad_Spice_Single_Objective`, calls `evaluate(params, append_to_log=False)`. Does **not** call `parameterize()`/`_create_optimizer_obj()`/`optimization_step()`.
- [x] `_build_spicelib_wrappers` shared in `ui/backend/services/optimizer_runner.py:53` and reused by simulate.py (with `output_subdir`).
- [x] Router registered in `ui/backend/main.py`; `simulateOnce()` in `ui/src/lib/api.ts`; `SimulateOnceResponse` in `ui/src/types/api.ts`.
- [x] **Mode A — load from prior result**: accepts a checkpoint point; stored params are engineering-real and feed `evaluate` directly. Default to best point (argmax of scores).
- [x] **Mode B — manual values**: accepts an explicit engineering-real `{params}` vector; soft range-check via `_validate_params`.
- [x] **UI**: collapsible Manual-Sim panel in `OptimizeTab` (PDK-gated), `Segmented` source toggle, per-spec result table, total score, active-corner display.
- [x] **Isolated output subfolder** (`outdir/manual_sim`) so a manual sim doesn't clobber a live run's outputs.

### Remaining
- [x] **Two-directional output isolation (BUG-A8 — fixed)**: isolation is currently **one-directional**. Starting a *live* run while a manual sim is in flight rmtrees `ws_root/outdir` (which **contains** `manual_sim`) — `_build_spicelib_wrappers(project)` at `optimizer_runner.py:257` passes **no** `output_subdir`, and `NGSpice_Wrapper._validate` rmtrees its `output_folder` (`spicelib.py:236-238`). Give the live run its own subfolder, or guard against a concurrent manual sim.
- [ ] **`Base_Optimizer.simulate_point(params)` façade** (optional but clean): a named one-shot wrapper around `evaluate(params, append_to_log=False)` so callers (manual sim, future "Re-simulate this point" in Explorer §8) don't reach into `evaluate` directly.
- [ ] **Mode B pre-fill from `project.dut_params`** (optional): seed the manual-values form from `Param.init` (with `min_val`/`max_val` hints), respect `is_integer` and the `C*`/`R*` suffix convention. *(Blocked on BUG-A2 §17 — the dut_param string `val`/`init` must resolve to a number first or the seeds are null.)*
- [ ] **"Send to Score Shaping"** action from the result table (optional).

## 13. PVT corners — Phase 1 (single chosen corner)

Make corners first-class and actually drive the sim against **one** active corner; the optimizer loop, scorer, and per-trial flow stay untouched. Full design + exact change list in [PVT_plan.md](PVT_plan.md) §"Phase 1".

> **Status: landed** (commits `ebc8e9d`, `2375f45`, `37952bf`, `c409031`, `a15b420`; example block
> `dc8b6f5`). Corners are first-class and drive the sim against one active corner. The legacy
> `tech_spec.pvt_map` / flat `pvt_corners` stay **display-only** (BUG-35 parsing still deferred).

### Landed ✅
- [x] **Config schema**: top-level `pvt:` block (`active_corner`, `process_bundles` sugar, `corners[]` with `model_includes`/`temp`/`supplies`, `enabled`, `model_lib_root`). `_normalize_pvt_block` (`core/domains.py:131`) desugars `process_bundles`, singular `supply`, and eng-strings before dacite.
- [x] **`core/domains.py`**: `ModelInclude` / `SupplyOverride` / `Corner` (`:249`) / `PVTConfig` (`:264`, with `model_lib_root` at `:272`) dataclasses; `pvt: Optional[PVTConfig]` on `Project_Setup`; `PVTConfig.get_active()` (`:280`). Legacy `TechSpec`/`PVT`/`pvt_corners` left as-is.
- [x] **`spice_engine/spicelib.py`**: `NGSpice_Wrapper.apply_corner(corner, model_lib_root)` (`:337`) — strips the hardcoded `.lib`, injects ordered cross-family includes, emits `.options temp=`, overrides supply `.param`s. The only ngspice-specific corner seam; idempotent.
- [x] **`optimization/base.py`**: `Spice_Base_Optimizer.__post_init__` applies `pvt.get_active()` once per enabled testbench wrapper before the loop, persisting across trials.
- [x] **UI (ephemeral)**: `pvt.active_corner` + corner list surfaced via `ui/src/components/pvt/CornerSelect.tsx` (Run popover, Optimize toolbar, Health check); `optimizer_runner._apply_overrides` accepts an ephemeral `active_corner` override (in-memory, never rewrites YAML).
- [x] **Wizard PVTStep**: emits inline `model_includes`, round-tripped through `yaml_generator._pvt_block_to_form`.

### Remaining (Phase 1 hardening — see §17 for the bug detail)
- [x] **Wizard `pvt.model_lib_root` now round-trips (BUG-A3 — fixed)**: `_build_pvt_block` (`yaml_generator.py:174-206`) never emits it and `_pvt_block_to_form` (`:253-293`) never reads `block.get("model_lib_root")`, so a project that relies on it loses lib-path resolution after a wizard Save. Needs a form field (`ui/src/types/api.ts:398` `WizardPVTConfig`, `ui/src/stores/wizardStore.ts:46` default, `PVTStep.tsx`) + emit/parse in the generator.
- [x] **Wizard preserves multi-rail corners (BUG-A6 — fixed)**: `_pvt_block_to_form` (`:276-283`) keeps only `supplies[0]` and `_build_pvt_block` (`:197-199`) emits a singular `supply`, so corners with rails 2..N lose them through the wizard. Either model multiple rails in `WizardPVTCorner` (`ui/src/types/api.ts:389-396` + `PVTStep.tsx:121-124`) or warn that multi-rail YAML must be edited in the raw editor.

## 14. PVT corners — Phase 2 (DEFERRED / research)

> **Deferred — not scheduled this round.** Once corners drive a single sim (Phase 1, §13 — landed), run the full **testbench × corner** cross-product and collapse **N corner-scores into one scalar** the optimizer consumes. The open research question is the aggregation strategy — candidates: worst-case/min, weighted mean, sum-of-penalties, must-pass-all (constraint), or Pareto/multi-objective (Ax). No strategy is committed. Secondary deferred items: where the corner loop lives, `parallel_sim` fan-out (N× ngspice processes), `{corner}::{spec}` checkpoint key namespacing (+ the dotted-column `.iterrows()` caveat), and completing **BUG-35** (`pvt_map`/`pvt_corners` parsing) alongside it. The Score-Shaping "Worst-case corner" mode (§3) and Explorer worst-case views depend on this. See [PVT_plan.md](PVT_plan.md) §"Phase 2".

## 15. UI layout fixes (CSS / overflow / scroll)

From the layout audit — full per-tab findings + global root causes in [ui_layout_report.md](ui_layout_report.md).

### Landed ✅
- [x] **RC-1 / RC-2 (largely subsumed)**: added `w-full min-w-0` to `inputCn` + `min-w-0` to `Field` (`wizard-controls.tsx`) and `min-w-0` to `selectCn` (`select.tsx`). Deliberately kept `selectCn` off `w-full` so toolbar selects don't stretch. Fixes the reported PVT Supply-column overflow and the input-heavy steps.
- [x] **RC-4**: `TabStrip` and `StatusBar` now use `overflow-x-auto whitespace-nowrap [&>*]:shrink-0` (StatusBar footer gets `overflow-hidden`) so labels scroll/truncate instead of wrapping on a narrow center column.
- [x] **RC-3**: added `min-w-0` to truncating spans + `shrink-0` to fixed siblings in `StudioLeftRail` and `StatusBar` so long project names ellipsize.
- [x] **Sanity Check / Optimize (Manual-Sim logs) / Compare clipped-no-scroll**: Fixed — the scroll-container clip that prevented reaching tall panels (e.g. Manual-Sim logs) was resolved (commit `42dd636`). These three previously-reported clips are confirmed **gone** at HEAD.

### Remaining (2026-06 re-audit — still present at HEAD, mostly `min-w-0` gaps in the wizard)
- [ ] **Wizard PVT Corners step — horizontal overflow** (`PVTStep.tsx`): the includes/supply rows still overflow on a narrow column. Apply the RC-1 `min-w-0`/`w-full` treatment to the remaining grid cells.
- [ ] **Wizard Testbenches step — horizontal overflow** (`TestbenchesStep.tsx`): param-row cards overflow; add `min-w-0` to the row grid.
- [ ] **Wizard PDK Rules step — horizontal overflow** (`PDKRulesStep.tsx`): the key/value constraint rows overflow.
- [ ] **Wizard Optimizer step — mis-sizing** (`OptimizerStep.tsx`): the algorithm select / kwargs editor mis-size on a narrow column.
- [ ] **Wizard select-bearing steps — mis-sizing** (`TargetSpecsStep.tsx` / `OptimizerStep.tsx` / `BasicInfoStep.tsx`): selects don't size consistently with their row cells (the deliberate `selectCn`-off-`w-full` choice needs a per-step `w-full` opt-in where the select owns its row).
- [ ] **Studio shell Right rail — horizontal overflow** (`RightRail.tsx`): spec-status rows overflow on long metric names; add `min-w-0` + truncation.
- [ ] **Setup (Create Wizard mode) — mis-sizing** (`SetupTab.tsx` → `WizardShell.tsx`): *needs runtime verification (deferred — no live UI on server this round)*. The static chain looks intact; confirm against a running UI before changing CSS.
- Verified **no-issue** at HEAD (do not touch): Setup Load/Edit summary panels, ScoreShapingTab, PipelineView, SchematicTab + DeviceInspector, BottomPanel, RightRail (global), StudioLeftRail.

## 16. Redundancy cleanup (low-risk first)

From the redundancy survey — full list + risk ratings in [project_redundancy.md](project_redundancy.md).

### Landed ✅
- [x] Deleted unused `formatNumber` (`lib/utils.ts`); removed dead `uiStore` fields (`compareRunA`/`compareRunB`/`setCompare`/`setSelectedRunId`); removed the dead `DutParams` class (`domains.py`, BUG-38). *(The one-tab `bottomTab` collapse is deferred — purely cosmetic.)*
- [x] Consolidated `_safe_float` (3 copies) into `ui/backend/services/num.py`; `config.py` now calls `_infer_score_fn` (fixes the `"linear"` divergence vs `checkpoint.py`).
- [x] Extracted `_target_specs_from_yaml` in `checkpoint.py`.
- [x] Renamed the duplicate `x_dut_Vb1`→`x_dut_Vb2` in the example YAML (matches the netlist), and `from_yaml` now **rejects** duplicate dut_param names (BUG-34).

### Remaining (2026-06 re-survey — confirmed present at HEAD)
- [ ] **[deadCode/high] RL optimizer backend** — `src/spicexplorer/optimization/rl/` (`rl_optimizer.py`, `rl_factory.py`, `agent_trainer.py`, `circuit_env.py`, …) is a dormant subtree unreferenced by the webapp backend (`ui/backend/` does not import it). **Decide its fate (needs sign-off)** — keep as research, or remove with its dependent tests.
- [ ] **[deadCode/medium] Orphaned demo runner** — `src/spicexplorer/demo/newcas_demo_runner.py` is a parallel reimplementation of the backend data flow; `tests/test_newcas_demo_runner.py` is its only consumer. **Decide its fate (needs sign-off)** alongside the RL subtree.
- [ ] **[deadCode/medium] Orphaned symbolic Bode fitter** — `src/spicexplorer/optimization/stochastic/symbolic.py` is unreferenced by the webapp (not imported anywhere under `ui/backend/`). Confirm it isn't used by any shipped path, then quarantine or remove.
- [ ] **[deadCode/low] `CheckpointMeta.n_iters`** — rendered in `RunsRail.tsx:154-155` but the list endpoint never populates it (`ui/backend/routes/checkpoint.py:72` sets `"n_iters": None`). Either populate it in the list endpoint or drop the render branch.
- [ ] **[duplicated/high] Constraint-satisfaction (goal+tolerance pass/fail)** — the same "does value meet goal within tolerance" rule is reimplemented in 4+ places (backend `compute_envelope`, `score_service`, FE `statusForGoal`, FE `RightRail`). Route everything through one helper. *(Several of the §17 exact-goal / tolerance bugs are symptoms of this duplication.)*
- [ ] **[duplicated/medium] Spec pass/fail helper duplicated across FE** — two inline copies (notably `RightRail.tsx:40-46`) ignore tolerance, diverging from `HealthTab`/`statusForGoal`. Collapse onto `statusForGoal` (this also fixes BUG-A12 in §17).
- [ ] **[duplicated/low] `score_service` penalty block** — duplicated verbatim between the per-spec loop and the curve loop; factor into one helper.
- [ ] **[duplicated/low] `goalSym` (goal → comparison glyph)** — reimplemented in ~8 components with two **divergent** glyph sets (`>`/`<`/`≈` vs others). Ship one shared helper.
- [ ] **[convoluted/high] `score_service.compute_score`** — a parallel scorer that re-derives the optimizer's penalty instead of calling the library's `evaluate`/`compute_fitness`. Reconcile so the UI and the optimizer never disagree on a point's score.
- [ ] **[convoluted/medium] Two parallel schematic-rendering paths** — legacy pre-rendered SVG `<img>` vs the interactive xschem viewer. Pick one (the interactive viewer) and retire the other once BUG-A1 (§17) lands.
- [ ] **[convoluted/low] `bottomTab`** — a full store field + setter + active-tab compare modeling exactly one tab. Collapse to a boolean or drop. *(Purely cosmetic; deferred.)*

---

## 17. Bug fixes — functional audit (2026-06, SECOND round / static-analysis)

Actionable list from the **second** audit (static analysis on the server — **no app/sim run**). Every item below was confirmed **present in source at HEAD** by reading the cited file. Grouped by severity. IDs are `BUG-A*` to avoid colliding with §11. Anything that would need a live UI/sim to confirm end-to-end is tagged **needs runtime verification (deferred)**.

> **Server grounding:** unlike the first audit (Docker/PDK-less context), the PDK **is present** on this
> server — ngspice/xschem/openvaf installed natively, PDK at `/home/noorizad/local/pdks`,
> `PDK=ihp-sg13g2`, `PDK_ROOT=/home/noorizad/local/pdks`, and the 40 `.sym` files exist under
> `…/libs.tech/xschem/sg13g2_pr/`. So BUG-A1 is a real, fixable env-contract gap here, not an infra blocker.

> **Status (fixed 2026-06, this round):** **ALL of BUG-A1..A16 are fixed**, plus five follow-ups found by
> an adversarial diff review (delete_checkpoint multi-root, resume `best_metrics` seeding, sensitivity/sanity
> output isolation, exact-goal winner tie) — re-verified clean by a second review pass. Regression tests in
> [tests/test_audit_redo_backend.py](tests/test_audit_redo_backend.py) + [tests/test_pvt_wizard_roundtrip.py](tests/test_pvt_wizard_roundtrip.py)
> + [tests/test_audit_fixes.py](tests/test_audit_fixes.py); full `pytest` (53, incl. real-SPICE slow) +
> `tsc --noEmit` + `eslint --max-warnings=0` all green. No commit yet — staged in the worktree.

### Major
- [x] **BUG-A1** *(fixed)* · [`ui/backend/routes/xschem.py:47-54`](ui/backend/routes/xschem.py#L47-L54) (`_pdk_xschem_dir`, `@lru_cache`) + [`:57-66`](ui/backend/routes/xschem.py#L57-L66) (`_search_roots`) — **Device symbols render as "missing symbol" placeholders when the backend has `PDK_ROOT` but not `PDK`.** Env-contract divergence: `_pdk_xschem_dir` requires **both** `PDK_ROOT` *and* `PDK` set, returning None otherwise, whereas the sim/PDK probe ([`env_probe.py:25`](ui/backend/services/env_probe.py#L25) `_PDK_ENV_VARS`, [`:42-58`](ui/backend/services/env_probe.py#L42-L58) `_candidate_pdk_roots`, [`:61-76`](ui/backend/services/env_probe.py#L61-L76) `_find_model_lib`) accepts any of `PDK_ROOT`/`PDK`/`IHP_PDK_ROOT` and falls back to a bounded `rglob`. Fix: make the xschem resolver use the **same** PDK-root discovery as the probe (or default `PDK` to `ihp-sg13g2` / derive it from the resolved root). FE manifestation: [`SchematicTab.tsx:63-69,89-99,385-388`](ui/src/components/tabs/SchematicTab.tsx#L63-L69) + [`SchematicViewer.tsx:182`](ui/src/components/schematic/SchematicViewer.tsx#L182). *(needs runtime verification (deferred) for the end-to-end render; the env-contract gap itself is confirmed in source.)*
- [x] **BUG-A2** *(fixed)* · [`yaml_generator.py:46-47`](ui/backend/services/yaml_generator.py#L46-L47) (`_build_dut_param`) — **`freeze: true` is silently dropped on YAML generation**; `_build_dut_param` only emits `freeze` when the form value is `False` (`if row.get("freeze") is False`), so a frozen DUT param is written without `freeze` and is then **swept as a free optimization dimension**. Fix: emit `freeze: true` when set (or always emit the explicit boolean).
- [x] **BUG-A3** *(fixed)* · [`yaml_generator.py:174-206`](ui/backend/services/yaml_generator.py#L174-L206) (`_build_pvt_block`) + [`:253-293`](ui/backend/services/yaml_generator.py#L253-L293) (`_pvt_block_to_form`, returns only `{active_corner, corners}` at `:290-293`) — **`pvt.model_lib_root` is dropped on the YAML→form→YAML round-trip**, breaking lib-file path resolution for projects that set it (`PVTConfig.model_lib_root` exists at [`domains.py:272`](src/spicexplorer/core/domains.py#L272) and is consumed by `apply_corner`). Form-type gap: [`api.ts:398-401`](ui/src/types/api.ts#L398-L401) (`WizardPVTConfig`), [`wizardStore.ts:46`](ui/src/stores/wizardStore.ts#L46) default, `PVTStep.tsx`. Caller path is `ui/src/components/tabs/SetupTab.tsx`. *(Also tracked under §13 Remaining.)*
- [x] **BUG-A4** *(fixed)* · [`yaml_generator.py:46-47`](ui/backend/services/yaml_generator.py#L46-L47) string `val` resolution — *Library half:* **`dut_param` string `val` is never resolved.** `resolve_all_parameter_ranges` ([`domains.py:832-846`](src/spicexplorer/core/domains.py#L832-L846)) calls `ressolve_val` **only** for testbench params (`:844`), never in the dut_params loop (`:835-839`), and `resolve_min_max` ([`:313-321`](src/spicexplorer/core/domains.py#L313-L321)) resolves `init` but not `val`. So a numeric-but-engineering-string operating point (e.g. `val: 0.9`/`val: 1.2u`) serializes to `null` in the project summary ([`project.py:63`](ui/backend/routes/project.py#L63) `isinstance(p.val, (int, float))`) and the sensitivity `_nominal` can't parse it ([`sensitivity.py:84-99`](ui/backend/routes/sensitivity.py#L84-L99) via [`num.safe_float`](ui/backend/services/num.py#L8-L14), which can't parse eng-strings), so the Device Inspector slider nominal silently drops to range-center ([`DeviceInspector.tsx:19-27`](ui/src/components/schematic/DeviceInspector.tsx#L19-L27)). Fix: call `ressolve_val` in the dut_params loop too (mirror the tb-param path). *(Re-opens part of §11 BUG-25.)*
- [x] **BUG-A5** *(fixed)* · [`ExplorerTab.tsx:26-31`](ui/src/components/tabs/ExplorerTab.tsx#L26-L31) (`bestOf`), used at [`:191-192`](ui/src/components/tabs/ExplorerTab.tsx#L191-L192) + [`:198`](ui/src/components/tabs/ExplorerTab.tsx#L198) (envelope/winner) and [`:495-498`](ui/src/components/tabs/ExplorerTab.tsx#L495-L498) (spec-summary pass/fail) — **`exact`-goal specs get the wrong "best" value (max instead of closest-to-target)** in the performance-envelope and spec-summary tables. `bestOf` only special-cases `minimize`; everything else (incl. `exact`) reduces to max. Fix: for `exact`, choose the sample with the smallest `|v − target|`. Corroborating server-side copy in BUG-A11.
- [x] **BUG-A11** *(fixed)* · [`checkpoint_reader.py:131-144`](ui/backend/services/checkpoint_reader.py#L131-L144) (`compute_envelope`) — **server-side `compute_envelope` returns max (not closest-to-target) and a wrong `passes` for `exact`-goal specs**: `goal == "minimize"` → `min`, `else` → `max`, so `exact` takes the max branch; `passes` then compares that wrong `best_ever` against target±tol. Reached via [`checkpoint.py:143-151`](ui/backend/routes/checkpoint.py#L143-L151) (`GET /checkpoint/{id}/envelope`), consumed by [`ExplorerTab.tsx:541-575`](ui/src/components/tabs/ExplorerTab.tsx#L541-L575) via [`api.ts:121-124`](ui/src/lib/api.ts#L121-L124). Fix together with BUG-A5 (one shared closest-to-target rule).

### Minor
- [x] **BUG-A6** *(fixed)* · [`yaml_generator.py:276-283`](ui/backend/services/yaml_generator.py#L276-L283) (`_pvt_block_to_form`, keeps `supplies[0]` only) + [`:197-199`](ui/backend/services/yaml_generator.py#L197-L199) (`_build_pvt_block`, emits singular `supply`) — **multi-rail PVT corners lose supply rails 2..N through the wizard.** Single-rail FE model: [`api.ts:389-396`](ui/src/types/api.ts#L389-L396) (`WizardPVTCorner`), [`PVTStep.tsx:121-124`](ui/src/components/wizard/steps/PVTStep.tsx#L121-L124). *(Also tracked under §13 Remaining.)*
- [x] **BUG-A7** *(fixed)* · [`project.py:47`](ui/backend/routes/project.py#L47) (`_summarise`) — **`target_spec` weight of 0 is misreported as 1.0** in the parsed project summary (`float(s.weight) if s.weight else 1.0` — `0.0` is falsy). Consumed at [`SetupTab.tsx:373`](ui/src/components/tabs/SetupTab.tsx#L373) (weight Σ) and [`:414`](ui/src/components/tabs/SetupTab.tsx#L414) (per-spec column). Fix: `float(s.weight) if s.weight is not None else 1.0`.
- [x] **BUG-A8** *(fixed — live/manual/sensitivity/sanity each isolated)* · [`optimizer_runner.py:53-77`](ui/backend/services/optimizer_runner.py#L53-L77) (`_build_spicelib_wrappers`; live run calls it with **no** `output_subdir` at [`:257`](ui/backend/services/optimizer_runner.py#L257)) + [`simulate.py:177`](ui/backend/routes/simulate.py#L177) (manual sim uses `output_subdir="manual_sim"`) + [`spicelib.py:234-242`](src/spicexplorer/spice_engine/spicelib.py#L234-L242) (`_validate` rmtrees `output_folder` at `:238`) — **starting a live run while a manual sim is in flight rmtrees the manual sim's working directory** (isolation is one-directional). *(Also tracked under §12 Remaining.)*
- [x] **BUG-A9** *(fixed)* · [`base.py:63`](src/spicexplorer/optimization/base.py#L63) (write — CWD-relative `./auto_save`) and [`:449-450`](src/spicexplorer/optimization/base.py#L449-L450) (`get_auto_save_name` builds under `autosave_checkpoint_dir`) vs [`checkpoint.py:53`](ui/backend/routes/checkpoint.py#L53) + [`:63`](ui/backend/routes/checkpoint.py#L63) (read under `REPO_ROOT/'auto_save'`) — **autosave checkpoints are written CWD-relative but discovered under `REPO_ROOT/auto_save`**, so Resume ([`optimize.py:70-76`](ui/backend/routes/optimize.py#L70-L76) 404 path) and live checkpoint listing silently break when the backend CWD ≠ repo root. `run_newcas_ui.sh:71-73` starts uvicorn before any `cd` (the `cd` at `:78` is for the frontend), so today they coincide — but it's fragile. Fix: anchor the autosave dir to a known root (or have the reader honor CWD).
- [x] **BUG-A10** *(fixed)* · [`MetricConvergenceChart.tsx:19-31`](ui/src/components/charts/MetricConvergenceChart.tsx#L19-L31) (`bestSoFar`) — **best-so-far curve treats `exact` goals as maximize** (`minimize` → `Math.min`, `else` → `Math.max`). Call sites: [`ExplorerTab.tsx:305-310`](ui/src/components/tabs/ExplorerTab.tsx#L305-L310) and [`OptimizeTab.tsx:266-271`](ui/src/components/tabs/OptimizeTab.tsx#L266-L271). Fix: for `exact`, track the value closest to target.
- [x] **BUG-A12** *(fixed)* · [`RightRail.tsx:40-46`](ui/src/components/shell/RightRail.tsx#L40-L46) — **RightRail spec pass/fail ignores tolerance for `exceed`/`minimize`**, contradicting `HealthTab`/`statusForGoal` (only the `exact` branch reads `spec.tolerance`). Fix: route through `statusForGoal` (ties into §16 duplicated/medium).
- [x] **BUG-A13** *(fixed — incl. resume seeding)* · Backend: [`optimizer_runner.py:152-162`](ui/backend/services/optimizer_runner.py#L152-L162) (`_StreamingOpt` `_emit` — `best_score`/`best_params` are the instance-tracked running best, set on improvement at [`:142-151`](ui/backend/services/optimizer_runner.py#L142-L151), while `metrics` is the **current** step's `fit` `curr_val` at `:156-160`). Store: [`runStore.ts:193-209`](ui/src/stores/runStore.ts#L193-L209) (`pushEvent` merges `bestMetrics` from `e.metrics`). Consumers: [`RightRail.tsx:39`](ui/src/components/shell/RightRail.tsx#L39) and [`PipelineView.tsx:111,195`](ui/src/components/tabs/PipelineView.tsx#L111). — **Right-rail "Spec status" / Pipeline pass-fail use the latest-seen metrics, not the best-scoring trial's**, so they're inconsistent with the displayed best params/score. Fix: emit the best trial's metrics alongside `best_params`, and merge those (not the running `metrics`) into `bestMetrics`. *(needs runtime verification (deferred) for the visible inconsistency; the data-flow mismatch is confirmed in source.)*
- [x] **BUG-A14** *(fixed)* · [`score_service.py:103`](ui/backend/services/score_service.py#L103) (`return … "linear": -total_linear, "sigmoid": -total_sigmoid`) vs [`ScoreShapingTab.tsx:228`](ui/src/components/tabs/ScoreShapingTab.tsx#L228) (header defines `F(x) = Σ wᵢ · P̂ᵢ`, a sum of non-negative penalties) and [`:292-296`](ui/src/components/tabs/ScoreShapingTab.tsx#L292-L296) (footer renders `aggregate.sigmoid`/`aggregate.linear` raw) — **the "F(x) aggregate" footer shows a negated value under a header that defines F(x) as a sum of non-negative penalties.** Fix: either return the unnegated penalty sum or relabel the footer (and keep the optimizer-facing negation separate).
- [x] **BUG-A15** *(fixed)* · [`env_probe.py:28-33`](ui/backend/services/env_probe.py#L28-L33) (`_PDK_LIB_SUBPATHS`) + [`:61-76`](ui/backend/services/env_probe.py#L61-L76) (`_find_model_lib`) — **PDK fast-path subpaths miss the *tech-prefixed* `…/libs.tech/ngspice/models/` layout the real install uses**, forcing a full-tree `rglob` on every probe. The list has `{tech}/libs.tech/ngspice/{lib}` (no `models/`) and `libs.tech/ngspice/models/{lib}` (no tech prefix), but **not** `{tech}/libs.tech/ngspice/models/{lib}`. The server PDK is `PDK_ROOT=/home/noorizad/local/pdks` with the lib at `ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib`, so the candidate root + needed subpath is exactly the missing tech-prefixed `models/` combo → falls through to `root.rglob` (walks the whole PDK tree, no caching) on every `/api/env`, `/api/sanity-check`, `/api/optimize/start`, `/api/simulate/once`. Confirmed against the on-disk layout (filesystem read, **no runtime needed**). Fix: add `f"{_PDK_TECH}/libs.tech/ngspice/models/{_PDK_MODEL_LIB}"` to `_PDK_LIB_SUBPATHS`. (Matches bug_report.md **ENV-1**.)
- [x] **BUG-A16** *(fixed)* · [`checkpoint_reader.py:15`](ui/backend/services/checkpoint_reader.py#L15) (`read_json_checkpoint`, no `limit` param) + [`:104-105`](ui/backend/services/checkpoint_reader.py#L104-L105) (`read_checkpoint` JSON branch drops `limit`, CSV branch forwards it); route at [`checkpoint.py:96-101`](ui/backend/routes/checkpoint.py#L96-L101) (passes `limit`) — **the JSON checkpoint reader ignores the `limit` parameter** that the CSV reader and the route honor, so a large JSON checkpoint is always returned full-resolution. Fix: add `limit` to `read_json_checkpoint` and truncate consistently with the CSV path.

---

## 18. Project encapsulation & run isolation (report.md) — LANDED ✅

The full encapsulation epic from [report.md](report.md) shipped on `feat/pvt` (commits `ae4f9bd` →
`7a228d8`, P0→P4). Each project is now an encapsulated directory under `WORK_ROOT` (`/work` in Docker,
`<repo>/work` native) and each optimization **run** is an isolated, self-contained folder. This
**fixes the Docker checkpoint data-loss bug** (autosave was written CWD-relative to `/app`, inside the
image layer, gone on `docker compose down`). **None of this code was covered by the §11 or §17 audits**
— it is the primary target of the §19 third round below.

- [x] **P0 — pin the contract** (`ae4f9bd`): `tests/test_ws_root_contract.py` asserts all three
  `from_yaml` `ws_root` branches + `~` expansion + the "resolved output/autosave under the project root,
  `REPO_ROOT not in parents`" guard (the test that would have caught the `yaml_path=""` regression).
- [x] **P1 — `WORK_ROOT` + deterministic autosave** (`bd255a2`): `app_config.work_root()` /
  `auto_save_root()` as the single source of truth; `Base_Optimizer.__init__` takes an optional
  `output_root` kwarg (default unchanged → CLI/example scripts byte-identical); `optimizer_runner`
  passes the per-run dir; `WORK_ROOT=/work` in compose; `auto_save/` added to `.gitignore`.
- [x] **P2 — per-run isolation leaf** (`f2335a3`, `cb48b64`, `d8ce8a7`): each run writes
  `runs/<ts>_<algo>_<runid8>/` with `checkpoints/`, `run.log`, `events.ndjson`, `config_snapshot.yaml`,
  `run.json`, and `sim/`; `project_service.py` owns all `/work` bookkeeping (registry scan, scaffold,
  copy-example, per-run dirs, `reconcile_stale_runs()` startup reconciler flipping crashed
  `running`→`error`); the wrapper's `rmtree` is scoped to this run's `sim/`.
- [x] **P3 — project registry + UI switcher** (`59cc957`, `bd58fcd`, `d94ec8a`, `c922765`, `97f867f`):
  `routes/projects.py` (`GET/POST /api/projects`, `from-example`, `{id}/runs`); `resolve_project(project_id,
  yaml_path)` resolver with `yaml_path` back-compat; ⌘P `ProjectsOverlay` + title-bar switcher;
  `projectStore` gains `{id, name, projects[], switchProject}`; the Runs rail is rescoped to
  `GET /api/projects/{id}/runs` (server-persisted, replacing localStorage history); checkpoint catalog +
  preset checkpoints scoped to the active project.
- [x] **P4 — lifecycle niceties** (`7a228d8`): `rename_project` (manifest-only), `fork_project`
  (`copytree`), `soft_delete_project` → `.trash` + `restore_project` + `list_trash`; `rename_run` /
  `delete_run`. (report.md §10 had deferred these; they shipped early.)
- [x] **Build/caching** (`4b18284`): Docker layer-caching + build-efficiency improvements.

> **Audit status:** the §19 third round (below) is a static-analysis pass over exactly this new surface
> (`project_service.py`, `routes/projects.py`, the `optimizer_runner` per-run isolation, `base.py`
> `output_root`, the rescoped `checkpoint.py`, and the P3/P4 UI) **plus** the NEWCAS-critical core/PVT
> paths the user called out. The `faef65a` `update_params` change (absolute-SI values + skip undeclared
> params) touches the NEWCAS sim path and is in scope.

## 19. Bug fixes — functional audit (2026-06, THIRD round / encapsulation + NEWCAS + PVT)

Actionable list from the **third** audit — a multi-agent static-analysis pass on `feat/pvt` at HEAD
(15 subsystem finders; **every** finding re-checked by an independent adversarial verifier that re-read
the cited code). Full per-bug location / scenario / fix / verifier note in [bug_report_r3.md](bug_report_r3.md)
(IDs match). **73 raised → 73 confirmed REAL (deduped to 52 distinct bugs); 17 refuted** (recorded at the end
of the report so they aren't re-raised). No app/sim was run — items needing a live trial to *see* the
symptom (B4, B13, B31) are confirmed at the source/data-flow level. **None of this surface was covered by
§11 or §17.** Reachability: 🟢 shipped cascode/default flow · 🟡 valid user config / non-default toggle ·
⚪ latent.

> **Status (2026-06-06): ALL FIXED except one deferred (B42).** Landed on `feat/pvt` in 6 commits —
> `750ba6c` (Tier 0), `932ac80` (Tier 1 NEWCAS-core), `d816fba` (Tier 2 PVT), `09a211c` (Tier 3),
> `e5c4c43` (Tier 4 backend), `47409ad` (Tier 4 frontend + B4-residual). Verified by `uv run pytest`
> (167 pass; the lone failure is the host-PDK-gated `test_ngspice_sanity_check`, which **passes in the
> Docker container**), `tsc --noEmit`, `eslint --max-warnings=0`, **+~75 new regression tests**, an
> adversarial bypass review per tier, and the real-SPICE suites (PDK sanity + slow optimization + a
> corner-applied sim + B5 autosave-reset run) re-run **against real ngspice + the IHP PDK in the
> container**. The one deferred item is **B42** (run stuck `running` on a hard-kill until the next
> startup reconcile — needs a heartbeat/PID liveness signal; normal restart self-heals).
>
> Severities below are the verifier-corrected ones (several finder "major"s were downgraded where the
> trigger is a non-default opt-in or the shipped example dodges it).

### Tier 0 — Security & data-integrity (fix first) ✅ FIXED 2026-06-06

> Landed (uncommitted in the worktree). Verified by `uv run pytest` (121 pass + the documented
> PDK-gated `test_ngspice_sanity_check` fail), `tsc --noEmit`, `eslint --max-warnings=0`, +21 new
> regression tests in [tests/test_audit_r3_tier0.py](tests/test_audit_r3_tier0.py), and a 5-agent
> adversarial bypass review (B1/B2 SOLID; B3 cross-run residual + B4 TOCTOU surfaced and handled below).

- [x] **BUG-B1** *(fixed)* · `viz/plotting.py` — replaced `eval(entry["log_file"])` with `ast.literal_eval` + drop-to-`None` fallback (`import ast`); `log_file` is unused downstream (popped before plotting, ignored by the web readers). Reviewer-confirmed: no other `eval`/`exec`/unsafe-`yaml`/untrusted-`pickle` on the checkpoint read path; DoS-safe (SyntaxError/Memory/Recursion caught). Also guarded a non-list `optimization_log`.
- [x] **BUG-B2** *(fixed)* · `checkpoint.py` — new `_validated_yaml_path` (`.yaml/.yml` + `resolve()` under **narrowed** roots: `REPO_ROOT/examples`, `work_root()`, and temp **only for `spx_uploaded_*`**); NUL-byte → `None` (no 500); report 400s on a bad non-empty path; envelope/scatter route through it. **Sibling LFI also fixed:** `project.py /yaml-text` now uses the same validator (was suffix+existence only → arbitrary `.yaml` read). *(new: tracked as BUG-B53.)*
- [x] **BUG-B3** *(fixed, incl. cross-run residual)* · `checkpoint.py` + `api.ts`/`RunsRail.tsx` — unscoped multi-match → 409; **and** because the catalog dedups same-stemmed checkpoints from different *runs* of one project to a single row (reviewer finding), the delete now takes a precise `?path=` (UI passes `c.path`) and removes exactly that file; bare-stem multi-match 409s in the scoped case too; candidate match restricted to `.json/.csv`; preset + under-autosave containment enforced for the path case.
- [x] **BUG-B4** *(fixed; one residual deferred)* · `optimizer_runner.stop_runs_for` returns `(attempted, still_alive)`; `delete_project`/`delete_run` **409 instead of moving** the dir while a worker is alive (reviewer-confirmed `is_alive()==False` is a sound quiescence signal — all run-dir writes finish before the thread exits). **Residual (deferred):** a *start-after-stop* TOCTOU — a run that STARTS in the gap between the join and the `shutil.move` isn't seen, so it can still resurrect the dir. Needs a per-project "deleting" tombstone that `start_run`/`run_dir` honor; tracked below.
- [x] **BUG-B53** *(new, fixed)* · `project.py:208-221` `/yaml-text` — was an arbitrary-file read of any `.yaml/.yml` on the host (suffix+existence only, no containment); now gated through `_validated_yaml_path`. Surfaced by the Tier-0 bypass review.
- [ ] **B4-residual** *(deferred follow-up)* · start-after-stop TOCTOU on project/run delete (see B4). Single-user/localhost + microsecond window → low risk; close with a delete tombstone honored by `start_run`. Same writer-resurrection family also touches `/simulate/once` (no `stop_runs_for`).

### Tier 1 — Major: NEWCAS core library (`examples/OTA/cascode` path) ✅ FIXED 2026-06-06

> Landed (uncommitted in the worktree). Verified by `uv run pytest` (137 pass + the PDK-gated
> sanity-check fail) and +14 regression tests in [tests/test_audit_r3_newcas.py](tests/test_audit_r3_newcas.py).
> All loader-level fixes were also reproduced live against the shipped cascode example.

- [x] **BUG-B5** *(fixed)* · `base.py` `optimize()` — index the best by the just-appended entry (`len-1`), not the absolute `trial`; the autosave log-reset now also resets `global_best_index`; the best index is seeded from any retained history (`keep_history`); the progress read guards the empty-log window. No more **IndexError mid-run** at `budget ≥ 2500`/low cadence (also fixes the `keep_history` mislabel, B25).
- [x] **BUG-B6** *(fixed)* · `base.py` `denormalize_params` — normalize the nevergrad candidate as `(val − bounds.min)/range` (log **and** linear), so a non-zero lower bound (e.g. the example's `log_variable_bounds.min = 1`) no longer pushes the physical value outside `[min_val, max_val]`.
- [x] **BUG-B7** *(fixed)* · `domains.py` `TargetSpec.__post_init__` — `parse_value(self.target)` when it's a str, so an `XeY` YAML target (`200e6`/`25e-6`) is coerced before the `abs(0.05*target)` fallback and before `meets_spec`/`get_simple_penalty` arithmetic.
- [x] **BUG-B8** *(fixed)* · `domains.py` `resolve_all_parameter_ranges` — non-frozen params **always** run `resolve_min_max` (incl. the `min ≥ max` check), so plain-numeric `min: 5, max: 1` (or `min == max`) is rejected at load instead of silently inverting the search range.
- [x] **BUG-B9** *(fixed)* · `domains.py` `TargetSpec.__post_init__` — a `None` / non-finite `weight` (explicit `weight:`/`weight: null`) is coerced to `1.0`, so `np.float64(weight)` can't NaN-poison the aggregate fitness; string weights are `parse_value`'d.
- [x] **BUG-B10** *(fixed)* · `domains.py` `resolve_all_parameter_ranges` — frozen params skip the mandatory-bounds path: `val`/`init` eng-strings are resolved and min/max validated only when both are present, so a frozen constant like `freeze: true, val: "0.18u"` (no bounds) loads instead of raising "missing min or max".

### Tier 2 — Major: PVT ✅ FIXED 2026-06-06

> Landed (uncommitted in the worktree). Verified by `uv run pytest` (142 pass) + 4 regression tests
> in [tests/test_pvt_corner.py](tests/test_pvt_corner.py), and **against real ngspice + the IHP PDK in
> the container**: a `ss_125C_1V62` corner-applied folded-cascode netlist simulated cleanly
> (`run_sanity_check` True; `mos_ss` in / `mos_tt` out / VDD→1.62), and the PVT + slow-optimization
> suites passed (26).

- [x] **BUG-B11** *(fixed)* · `spicelib.py` `apply_corner` — strip each referenced `.lib` basename **exactly once up front**, then add all corner includes, so two `model_includes` sharing one `lib_file` (different sections) no longer collapse to the last; re-apply stays idempotent (the upfront strip clears prior corner includes too).
- [x] **BUG-B12** *(fixed)* · `spicelib.py` `apply_corner` — before applying supply overrides, compute the netlist's declared `.param` names (`ed.get_all_parameter_names()`); a supply `node` that isn't declared (e.g. the source instance `Vdd` instead of the param `VDD`, or an undeclared `VSS`) now logs a loud WARNING (via the `spicexplorer` logger, so it surfaces in the SSE log) that the override adds a dangling `.param` and will NOT change the supply — instead of silently running at the netlist default.

### Tier 3 — Major: backend & UI (encapsulation / lifecycle) ✅ FIXED 2026-06-06

> Landed (uncommitted in the worktree). Verified by `uv run pytest` (151 pass) + 9 regression tests
> in [tests/test_audit_r3_tier3.py](tests/test_audit_r3_tier3.py), `tsc --noEmit`, `eslint
> --max-warnings=0`. B13's thread-scoping assumption was confirmed against real ngspice in the
> container: in a real `parallel_sim=True` run, **all 503 `spicexplorer` log records were emitted on
> the worker thread (0 on sub-threads)**, so the filter can't drop a run's own logs.

- [x] **BUG-B13** *(fixed)* · `optimizer_runner.py` — each run's `_QueueLogHandler` + per-run `FileHandler` now carry a `_RunThreadFilter(threading.get_ident())` so they only pass records from THIS run's worker thread; two concurrent runs no longer cross-contaminate `run.log`/SSE. (spicelib's parallel `RunTask` threads log to the separate `spicelib.*` tree, not captured here, so nothing is lost.)
- [x] **BUG-B14** *(fixed)* · `xschem.py` — `_allowed_roots()` now includes `work_root()`, so an encapsulated project's `xschem/` under `/work` validates instead of 403-ing under Docker. (`_search_roots` already resolves project-local refs via `base_dir`.)
- [x] **BUG-B15** *(fixed)* · `yaml_generator.py` (+ `WizardDutParam.val`) — `_build_dut_param` now emits `val` and `project_dict_to_form` carries it, so a frozen param's pinned operating point round-trips through the wizard instead of falling back to `init`/netlist default.
- [x] **BUG-B16** *(fixed)* · `projectStore.ts` `switchProject` — if a run is live it now `useRunStore.getState().stopRun()`s (server-stop + record to history + close the EventSource) BEFORE rebinding, so the old run's SSE no longer streams into the new project's UI and the stream isn't orphaned. (Lazy `import` avoids a store cycle.)

### Tier 4 — Minor ✅ FIXED 2026-06-06 (one deferred: B42)

> Landed (commits `e5c4c43` backend Groups A/B/C, `47409ad` Group D + B4-residual). Verified by
> `uv run pytest` (167 pass) + 16 regression tests in [tests/test_audit_r3_tier4_backend.py](tests/test_audit_r3_tier4_backend.py),
> `tsc --noEmit`, `eslint --max-warnings=0`, and the real-SPICE suites re-run in the container.

**Core/scoring:** [x] **B17** zero-target tolerance floored to a scale-aware positive · [x] **B18** `parse_value(None)` → clear `ValueError` · [x] **B19** `log_scale` tolerance mapped as a log-space band (`_log_space_band`) · [x] **B20** dead EXCEED-reward `elif` removed · [x] **B21** log-reward operands floored (no `-inf` at exact match) · [x] **B22** exponential-error exponent clamped (no `inf`) · [x] **B23** *(documented)* constraint-first aggregation is intentional · [x] **B24** `compute_log_normalization` base-10 · [x] **B25** *(via B5)* `global_best_index` log-relative + history-seeded.

**Optimizer/lifecycle:** [x] **B26** Bode + Constraint optimizers forward `output_root` · [x] **B27** resume runs only the **remaining** budget · [x] **B28** serial no-RAW sim scores the trial as a failure (no run abort) · [x] **B29** duplicate target-spec names rejected · [x] **B36** replay with no `checkpoint_id` → 400 · [x] **B41** lazy autosave-dir mkdir (no stray `./auto_save`) · [ ] **B42** *(deferred)* `optimizer_runner.py` run stuck `running` on a hard-kill until next startup reconcile — needs a heartbeat/PID liveness signal (disproportionate for a minor edge; normal `docker compose up` self-heals via the startup reconciler) · [x] **B43** `reconcile_stale_runs` also scans `.trash`.

**Routes/scoping:** [x] **B35** target-spec tolerance always emitted (>0; no `None`→500) · [x] **B37** invalid `project_id` → 400 · [x] **B38** exact-stem checkpoint resolution · [x] **B39** envelope/scatter use only enabled specs.

**PVT:** [x] **B30** temp strip preserves combined `.options` lines · [x] **B31** unknown `active_corner` surfaced (live-run warning streamed via the `spicexplorer` logger + `sanity` `warnings[]`) · [x] **B32** corner with both `process` + `model_includes` rejected · [x] **B33** sanity `warnings[]` note that per-tb rows run the netlist corner (the trial step uses the active corner) · [x] **B34** `get_active()` warns when the active corner is `enabled:false`.

**Frontend:** [x] **B40** per-corner PVT `params` round-trip the wizard · [x] **B44** *(with B16)* delete-active-project resets `runStore` · [x] **B45** Pareto/feasible overlay handles `exact` (closest-to-target; half-plane rect skipped) · [x] **B46** `formatEng(0)` → unprefixed `"0.000"` · [x] **B47** superseded in-flight run recorded to history · [x] **B48** Score Shaping parses eng-strings (`parseEng`) · [x] **B49** scatter click resolves run by trace-slot index · [x] **B50** manual sims use a unique output subfolder · [x] **B51** run-rename Escape cancels (no blur-commit) · [x] **B52** ProjectsOverlay surfaces a load/create open failure.

### Cross-cutting root causes ✅ (all resolved by the fixes above)
- [x] `tolerance:None` serialization → fixed at source (B17 + B35). · [x] `output_root` threaded into Bode/Constraint + lazy mkdir for the one-off routes (B26, B41). · [x] checkpoint resolution/delete scoped + exact-stem + precise-`path` delete (B3, B38). · [x] `stop_runs_for` reports liveness; lifecycle routes 409 + a delete tombstone (B4 + residual). · [x] `spicexplorer` logger run-scoped; corner warning re-routed to the streamed logger (B13, B31). · [x] exact-goal closest-to-target applied to the FE overlays (B45). · [x] `projectStore`↔`runStore` teardown on switch/delete (B16, B44).

### B4-residual ✅ FIXED 2026-06-06
- [x] start-after-stop TOCTOU on project delete — a per-project "deleting" tombstone (`optimizer_runner.begin/end_project_delete`); `/optimize/start` refuses (409) a run for a project whose delete is in flight, and `delete_project` holds the tombstone across the whole stop→move window.

### Watch / hardening (refuted-but-noted — not confirmed bugs)
- [ ] `soft_delete_project` trash id is `project_id__{whole-second-ts}` with **no random suffix** (unlike `delete_run`'s uuid suffix) — verifiers judged a same-second re-delete collision not currently reachable; add a suffix defensively.
- [ ] `restore_project` joins `meta["name"]` into the destination path; `_assert_under_work_root` blocks escape *above* WORK_ROOT but not lateral clobber — harden with name validation.
