# Part 4 — Redundancy Survey

A static-analysis audit of dead code, duplicated logic, and over-indirect paths in SpiceXplorer. Every claim below was verified by reading the actual code path; each item cites a `file:line` anchor. No code was modified — this is a survey of what *could* be removed or consolidated, with a per-item risk rating and a suggested cleanup order at the end.

Scope: the Python library (`src/spicexplorer`), the FastAPI backend (`ui/backend`), the Next.js frontend (`ui/src`), and the committed example YAML.

---

> **✅ Status (2026-06).** Cleaned up in branch `dev/ui`: deleted dead `formatNumber` (#3) and the dead `DutParams` class (BUG-38); removed dead `uiStore` fields `compareRunA`/`compareRunB`/`setCompare`/`setSelectedRunId` (#4); consolidated `_safe_float` into `ui/backend/services/num.py` (#6); `config.py` now reuses `_infer_score_fn`, fixing the `"linear"` divergence (#7); extracted `_target_specs_from_yaml` (#11); renamed the duplicate `x_dut_Vb1`→`x_dut_Vb2` + `from_yaml` now rejects duplicates (#13). **Deferred** (cosmetic, behavior-changing, or needs sign-off): `bottomTab` collapse (#14), shared `goalSym` glyph + FE pass/fail dedup (#8/#9/#10), `score_service` penalty-helper (#12), `CheckpointMeta.n_iters` (#5), legacy SVG schematic route (#15), and `newcas_demo_runner.py` / the RL subtree (#1/#2 — product sign-off). Verified by `pytest`/`ruff`/`tsc`/`eslint`/`build`.

## Dead code

### 1. `newcas_demo_runner.py` — fully orphaned parallel implementation of the backend data flow
- **Location:** [src/spicexplorer/demo/newcas_demo_runner.py](src/spicexplorer/demo/newcas_demo_runner.py) (350 lines)
- **Why it is redundant:** This module is a complete re-implementation of the FastAPI backend's job. It re-declares the trace-parsing constants `METRIC_PREFIX`/`PARAM_PREFIX` ([newcas_demo_runner.py:29](src/spicexplorer/demo/newcas_demo_runner.py#L29), [:32](src/spicexplorer/demo/newcas_demo_runner.py#L32)) and the `_metric_names`/`_param_names` helpers ([:187](src/spicexplorer/demo/newcas_demo_runner.py#L187), [:195](src/spicexplorer/demo/newcas_demo_runner.py#L195)) that already exist verbatim in [checkpoint_reader.py:62-80](ui/backend/services/checkpoint_reader.py#L62-L80). It has its own `_safe_float` ([:72](src/spicexplorer/demo/newcas_demo_runner.py#L72)), its own `_infer_trace_kind` ([:147](src/spicexplorer/demo/newcas_demo_runner.py#L147)) duplicating the score-fn inference, a `load_demo_config` that hand-maps the YAML, and a `run_live_demo` streaming loop ([:272](src/spicexplorer/demo/newcas_demo_runner.py#L272)) duplicating `optimizer_runner`. It references a `/api/demo/asset/schematic` endpoint ([:135](src/spicexplorer/demo/newcas_demo_runner.py#L135)) that **does not exist** — grep finds no `demo` route in `ui/backend/routes/` or `main.py`. It is the legacy Next.js-demo CLI bridge, superseded entirely by the FastAPI backend, and is imported **only** by [tests/test_newcas_demo_runner.py:1](tests/test_newcas_demo_runner.py#L1). No webapp, orchestrator, or `run_newcas_ui.sh` path touches it.
- **Removal risk:** **medium** — one test depends on it, so removal means deleting that test too; nothing at runtime is affected.

### 2. RL optimizer backend — unreferenced by the webapp and the documented Nevergrad path
- **Location:** [src/spicexplorer/optimization/rl/](src/spicexplorer/optimization/rl/) (`agent_trainer.py`, `circuit_env.py`, `rl_factory.py`, `rl_optimizer.py`, `custom_agents/{base,ddpg,sac,td3}.py`, `models/{actor,critic,base}.py`, `utils/`)
- **Why it is redundant:** Nothing in `ui/backend` or the orchestrator's documented flow imports `optimization.rl`. The only importers are [tests/test_rl_factory.py:54](tests/test_rl_factory.py#L54) and [tests/test_hyperparameters.py:3](tests/test_hyperparameters.py#L3). The entire webapp drives `Nevergrad_Spice_Single_Objective`, and CLAUDE.md itself describes RL as "present but not the primary documented path." It is a large dormant subtree.
- **Removal risk:** **high** — tests depend on it and it may be a deliberate research WIP; not safe to delete without product sign-off. Flagged as dead *relative to the webapp*, not as a confident deletion.

### 3. `formatNumber` — exported but never imported
- **Location:** [ui/src/lib/utils.ts:5-20](ui/src/lib/utils.ts#L5-L20)
- **Why it is redundant:** Grep across all of `ui/` finds `formatNumber` only at its own definition site — zero call sites. The codebase standardized on `formatEng` ([utils.ts:27](ui/src/lib/utils.ts#L27)), which is used across ~9 files. `formatNumber` is leftover.
- **Removal risk:** **low** — confirmed zero references; safe to delete.

### 4. `uiStore` dead fields: `setSelectedRunId`, `compareRunA`, `compareRunB`, `setCompare`
- **Location:** [ui/src/stores/uiStore.ts:43-44](ui/src/stores/uiStore.ts#L43-L44), [:61](ui/src/stores/uiStore.ts#L61), [:64](ui/src/stores/uiStore.ts#L64), [:81-82](ui/src/stores/uiStore.ts#L81-L82), [:93](ui/src/stores/uiStore.ts#L93), [:95](ui/src/stores/uiStore.ts#L95)
- **Why it is redundant:** These members are declared and initialized but never read or called outside `uiStore.ts`. `selectedRunId` is only ever written through `openRun` ([:94](ui/src/stores/uiStore.ts#L94)); `compareRunA`/`compareRunB`/`setCompare` have no consumer at all — the Explorer's compare A/B state actually lives in `explorerStore`. The store's own comment ([:33](ui/src/stores/uiStore.ts#L33)) describes them as "the seed for ... later phases." (`setSelectedRunId` at [:93](ui/src/stores/uiStore.ts#L93) is redundant given `openRun` is the sole writer.)
- **Removal risk:** **low** — grep confirms no external readers/callers; removal is contained to `uiStore`.

### 5. `CheckpointMeta.n_iters` — rendered but never populated by the backend
- **Location:** backend [checkpoint.py:53](ui/backend/routes/checkpoint.py#L53) (autosave: `n_iters: None`) and [:62-71](ui/backend/routes/checkpoint.py#L62-L71) (preset: field omitted) vs. frontend [RunsRail.tsx:154-155](ui/src/components/shell/rails/RunsRail.tsx#L154-L155)
- **Why it is redundant:** `RunsRail` conditionally renders `c.n_iters != null && (...)`, but `_list_autosave_checkpoints` always sets `n_iters` to `None` and `list_checkpoints` omits it for presets. The value is therefore always `null`/`undefined` and the UI branch never displays. It is dead end-to-end — the data is never computed at list time. (Type allows it: [types/api.ts:173](ui/src/types/api.ts#L173) `n_iters?: number | null`.)
- **Removal risk:** **low** — either drop the field or actually compute it; the current state is a guaranteed-empty UI branch.

---

## Duplicated logic

### 6. `_safe_float` defined identically four times
- **Location:** [checkpoint_reader.py:14](ui/backend/services/checkpoint_reader.py#L14), [optimizer_runner.py:21](ui/backend/services/optimizer_runner.py#L21), [sensitivity.py:65](ui/backend/routes/sensitivity.py#L65), [newcas_demo_runner.py:72](src/spicexplorer/demo/newcas_demo_runner.py#L72)
- **Why it is redundant:** The same float-coercion-with-finite-check helper is copy-pasted across four modules. It should live once (e.g. in `checkpoint_reader` or a small backend `utils`) and be imported.
- **Removal risk:** **low** — identical implementations; consolidating to one import has no behavioral effect. (Note: item 1 removes the demo-runner copy outright.)

### 7. `score_fn` inference duplicated and divergent between `checkpoint.py` and `config.py`
- **Location:** [config.py:13-19](ui/backend/routes/config.py#L13-L19) vs [checkpoint.py:15-21](ui/backend/routes/checkpoint.py#L15-L21)
- **Why it is redundant:** `checkpoint.py` exposes the inference as the helper `_infer_score_fn`; `config.py` inlines the same filename→score_fn rule. They have **already drifted**: `checkpoint.py` maps a filename containing `"linear"` to `"relative-absolute"`, but `config.py`'s inline copy omits the `"linear"` branch, so a `linear`-named preset is reported as `"unknown"` by `/api/config` while `/api/checkpoint` reports `"relative-absolute"`.
- **Removal risk:** **low** — pure read-only inference; collapsing `config.py` to call `_infer_score_fn(path)` is mechanical and (modulo the intended `"linear"` fix) behavior-preserving.

### 8. `goalSym` (goal → comparison symbol) reimplemented in 8+ components with inconsistent glyphs
- **Location:** [SetupTab.tsx:28](ui/src/components/tabs/SetupTab.tsx#L28), [ScoreShapingTab.tsx:17-19](ui/src/components/tabs/ScoreShapingTab.tsx#L17-L19), [ExplorerTab.tsx:22](ui/src/components/tabs/ExplorerTab.tsx#L22), [PipelineView.tsx:21-22](ui/src/components/tabs/PipelineView.tsx#L21-L22), [RightRail.tsx:45](ui/src/components/shell/RightRail.tsx#L45), [SpecsRail.tsx:35](ui/src/components/shell/rails/SpecsRail.tsx#L35), [DeviceInspector.tsx:125-126](ui/src/components/schematic/DeviceInspector.tsx#L125-L126), [TargetSpecsStep.tsx:84](ui/src/components/wizard/steps/TargetSpecsStep.tsx#L84)
- **Why it is redundant:** The same goal→symbol mapping is re-declared in every view, and the **glyphs disagree**: `ScoreShapingTab` / `ExplorerTab` / `SetupTab` / `RightRail` emit `>` / `<`, while `PipelineView` / `SpecsRail` / `DeviceInspector` / `HealthTab` emit `≥` / `≤` for the same goals — a visible inconsistency a single shared helper in `lib/utils.ts` would fix.
- **Removal risk:** **low** — UI-only string; extracting one helper is safe, though it forces a one-time choice of a canonical glyph set.

### 9. Spec pass/fail logic duplicated across FE components and divergent from the tolerance-aware version
- **Location:** [HealthTab.tsx:15](ui/src/components/tabs/HealthTab.tsx#L15) (`passesSpec`), [ExplorerTab.tsx:31](ui/src/components/tabs/ExplorerTab.tsx#L31) (`passesSpec`), [PipelineView.tsx:104-108](ui/src/components/tabs/PipelineView.tsx#L104-L108) (`specPass`), [RightRail.tsx:36-44](ui/src/components/shell/RightRail.tsx#L36-L44) (inline)
- **Why it is redundant:** `HealthTab` and `ExplorerTab` each wrap `lib/utils.statusForGoal` in an identical `passesSpec` helper, while `PipelineView` (`specPass`) and `RightRail` re-derive pass/fail inline. The two inline copies compare against the **bare target** for `exceed`/`minimize` — `PipelineView.tsx:107` does `v >= target` / `v <= target` and `RightRail.tsx:40-42` likewise — *ignoring tolerance*, so they can report a different pass/fail than `statusForGoal` (which applies `target ± tol`) for the same spec. The canonical comparison should be `statusForGoal` everywhere.
- **Removal risk:** **low** — consolidating to `statusForGoal` is low-effort, but note it is a **behavior change** for the two inline copies (they would start honoring tolerance).

### 10. `statusForGoal` (FE) duplicates the backend's directional-error / feasibility comparison
- **Location:** [utils.ts:38-62](ui/src/lib/utils.ts#L38-L62) vs [score_service.py:11-19](ui/backend/services/score_service.py#L11-L19) (`_raw_directional_error`) and `checkpoint_reader.py` `compute_envelope`/`compute_scatter` feasibility
- **Why it is redundant:** The exceed/minimize/exact pass-or-fail comparison (with the default `tolerance = 0.05 * target`, see [utils.ts:48](ui/src/lib/utils.ts#L48) and [score_service.py:42](ui/backend/services/score_service.py#L42)) is implemented independently on the frontend (`statusForGoal`) and on the backend (`_raw_directional_error` plus the envelope/scatter feasibility checks). The library already owns the canonical `compute_*_error` math in `core/utils.py`. That is four-plus copies of the constraint-satisfaction rule that must be kept in lockstep by hand.
- **Removal risk:** **high** — spans the HTTP boundary and the optimizer; truly unifying would mean the FE always asking the backend for status. High blast radius — flagged for awareness rather than a quick fix.

### 11. Two near-identical `yaml_path → target_specs` extraction blocks in `checkpoint.py`
- **Location:** [checkpoint.py:135-145](ui/backend/routes/checkpoint.py#L135-L145) (envelope) and [:166-176](ui/backend/routes/checkpoint.py#L166-L176) (scatter)
- **Why it is redundant:** `checkpoint_envelope` and `checkpoint_scatter` contain the *same* `try/except` that loads `Project_Setup.from_yaml` and maps `target_specs.targets` into the `{name, target, goal, tolerance}` dict (including the duplicated local `from pathlib import Path as _Path` / `from spicexplorer.core.domains import Project_Setup` imports). It should be one shared `_target_specs_from_yaml(path)` helper.
- **Removal risk:** **low** — identical blocks; extraction is mechanical.

### 12. `score_service` per-spec penalty loop and curve loop repeat the same normalized-penalty computation
- **Location:** [score_service.py:54-63](ui/backend/services/score_service.py#L54-L63) (per-spec) vs [:91-97](ui/backend/services/score_service.py#L91-L97) (curve)
- **Why it is redundant:** The raw→linear/sigmoid penalty calculation (`_raw_directional_error` + `compute_relative_absolute_error`/`compute_relative_sigmoid_error`, with the `raw <= 0` short-circuit returning `0.0`) is written twice: once in the per-spec loop and again verbatim inside the curve-building loop. A single inner helper (`raw → (linear, sigmoid)`) removes the second copy.
- **Removal risk:** **low** — same module, identical math; refactor is behavior-preserving.

### 13. Duplicate `dut_param` definition `x_dut_Vb1` in the folded_cascode example YAML
- **Location:** [project_setup.yaml:165-173](examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml#L165-L173)
- **Why it is redundant:** Two `dut_param` blocks are both named `x_dut_Vb1` with identical `min_val: 0.1` / `max_val: max_vdd` (lines 165 and 170). The second is almost certainly meant to be a different bias node (e.g. `x_dut_Vb2`); as written it is a redundant/shadowing parameter that bloats the search space with a duplicate variable name — a latent data bug.
- **Removal risk:** **low** — a pure data fix in an example file, but verify against the testbench netlist's actual bias nodes before deciding to rename (→ `x_dut_Vb2`) vs. delete.

---

## Convoluted / over-indirect paths

### 14. `bottomTab` — an over-engineered single-value tab mechanism
- **Location:** [uiStore.ts:49](ui/src/stores/uiStore.ts#L49), [:67](ui/src/stores/uiStore.ts#L67), [:85](ui/src/stores/uiStore.ts#L85), [:98](ui/src/stores/uiStore.ts#L98) + [BottomPanel.tsx:15-34](ui/src/components/shell/BottomPanel.tsx#L15-L34)
- **Why it is unclear:** `bottomTab` is typed as the literal `"log"` only, `setBottomTab` can only be called with `"log"` ([BottomPanel.tsx:31](ui/src/components/shell/BottomPanel.tsx#L31)), and `BottomPanel`'s `bottomTab === "log"` test ([:34](ui/src/components/shell/BottomPanel.tsx#L34)) is therefore always true. A full store field + setter + active-tab comparison models a tab strip that has exactly one tab.
- **Removal risk:** **low** — can be inlined to a static label until a second bottom tab exists; no behavior change.

### 15. Two parallel schematic-rendering paths: legacy SVG `<img>` vs. the xschem viewer
- **Location:** [SetupTab.tsx:451-452](ui/src/components/tabs/SetupTab.tsx#L451-L452) (`api.schematicUrl()` SVG via `<img>`) + `routes/schematic.py` / `app_config.schematic_svg_path` **vs.** the xschem stack (`routes/xschem.py` + `lib/xschem/*` + `components/schematic/SchematicViewer`)
- **Why it is unclear:** The app ships a complete interactive xschem parser/renderer (`lib/xschem`, `SchematicViewer`, `/api/xschem/*`) **and** still keeps the old pre-rendered-SVG route (`/api/schematic`, `schematic_svg_path` in `app_config.json`), which `SetupTab` embeds via a plain `<img>` hardcoded to a single committed SVG. Two unrelated mechanisms answer "show the schematic."
- **Removal risk:** **medium** — `SetupTab` still *actively* uses the SVG path, so it is not dead; but the two approaches overlap and the SVG route is a candidate to retire once `SetupTab` adopts the xschem viewer.

---

## Suggested cleanup order

Ordered lowest-risk / highest-clarity-gain first; each is independent unless noted.

1. **Delete `formatNumber`** (item 3) — zero references, pure subtraction.
2. **Remove `uiStore` dead fields** `setSelectedRunId`/`compareRunA`/`compareRunB`/`setCompare` (item 4) — contained, no external readers.
3. **Collapse `bottomTab`** to a static label (item 14) — removes a fake one-tab abstraction.
4. **Fix the duplicate `x_dut_Vb1`** in the example YAML (item 13) — one-line data fix after confirming the intended bias node.
5. **Consolidate `_safe_float`** into one shared helper (item 6) and **call `_infer_score_fn` from `config.py`** (item 7) — also fixes the existing `"linear"` divergence.
6. **Extract `_target_specs_from_yaml`** in `checkpoint.py` (item 11) and the **inner penalty helper** in `score_service.py` (item 12) — same-file, behavior-preserving.
7. **Extract one `goalSym` helper** into `lib/utils.ts` (item 8) — pick a canonical glyph set, removes a visible inconsistency.
8. **Route all FE pass/fail through `statusForGoal`** (item 9) — low effort but a deliberate behavior change for the two tolerance-ignoring inline copies; do it consciously.
9. **Either compute or drop `CheckpointMeta.n_iters`** (item 5) — resolve the guaranteed-empty UI branch.
10. **Retire the legacy SVG schematic route** once `SetupTab` adopts the xschem viewer (item 15) — medium, requires a UI migration first.
11. **Decide the fate of `newcas_demo_runner.py`** (item 1, medium) and the **RL subtree** (item 2, high) — both need product sign-off (each carries a dependent test); they are dead relative to the webapp but may be intentional.
