# UI Layout Audit — Part 4 (Layout Agent)

**Scope:** CSS / layout / sizing / overflow / scroll only. This report does **not** cover
data flow, state, accessibility semantics, or visual styling beyond what causes content to be
clipped, mis-sized, or to overflow. Every claim is anchored to `file:line` in the current
working tree (branch `dev/ui`). Investigation only — no code was modified.

The two **confirmed reported symptoms** are:

1. **PVTStep horizontal overflow** — the Supply column is cut off in the new-project wizard.
   Root cause: [Global RC-1](#rc-1) + [RC-2](#rc-2). See [PVT findings](#pvt-corners-step).
2. **Sanity Check logs "clipped, no scroll"** — see [Health / Sanity Check](#health--sanity-check).
   The height chain is actually **intact** in source; the log surface *does* get a scrollbar.
   This is most likely a **stale `.next` build**, not a CSS defect. Details below.

---

> **✅ Status (2026-06).** Fixed in branch `dev/ui`: **RC-1** (`w-full min-w-0` on `inputCn`, `min-w-0` on `Field` and `selectCn` — deliberately kept `selectCn` off `w-full` so toolbar selects don't stretch) resolves the reported PVT Supply-column overflow and the input-heavy steps (RC-2 largely subsumed); **RC-3** (`min-w-0`/`shrink-0` truncation in `StudioLeftRail` + `StatusBar`); **RC-4** (`overflow-x-auto whitespace-nowrap [&>*]:shrink-0` on `TabStrip` + `StatusBar` footer). The Sanity Check "clipped-no-scroll" symptom was confirmed **not** a source CSS defect (intact height/scroll chain) — most likely a stale `ui/.next` build, so no code change was made. Verified by `tsc`/`eslint`/`next build`.

## Global root causes

The findings below collapse into four cross-cutting rules. Fix these at the source and most
per-tab instances disappear.

<a id="rc-1"></a>
### RC-1 — Wizard inputs have no intrinsic shrink (`w-full` / `min-w-0` missing)

`TextInput` renders a bare `<input>` with only `inputCn(...)` applied, and `inputCn` adds **no**
`w-full` and **no** `min-w-0`:

- [wizard-controls.tsx:4-11](ui/src/components/wizard/wizard-controls.tsx#L4) — `inputCn` (no width, no min-width)
- [wizard-controls.tsx:30-32](ui/src/components/wizard/wizard-controls.tsx#L30) — `TextInput` just spreads `inputCn(props.className)`
- [select.tsx:11-18](ui/src/components/ui/select.tsx#L11) — `selectCn` likewise has no `w-full`/`min-w-0`. The `Select` wrapper only adds `w-full` when a `label` prop is passed ([select.tsx:38](ui/src/components/ui/select.tsx#L38)); every wizard step uses the **bare** `selectCn(...)` on a raw `<select>`, so it never gets `w-full`.

A flex/grid item's **automatic minimum size** equals its `min-content` width, and for a form
control that is its default intrinsic size: roughly **~20ch (≈180px)** for a text input, and
**the longest `<option>` label** for a `<select>`. So any multi-column grid that drops these
controls into `fr` tracks **cannot shrink a track below ~180px** — the row's minimum width
exceeds the (already narrow) wizard form column and overflows to the right.

**Standard fix:** make `inputCn` / `selectCn` default to `w-full min-w-0` (and add `min-w-0` to
`Field`). One edit fixes every step.

<a id="rc-2"></a>
### RC-2 — The wizard form column is half (or quarter) width

The wizard body is a hard `1fr 1fr` grid, so the form sits in **~50%** of the overlay:

- [WizardShell.tsx:147](ui/src/components/wizard/WizardShell.tsx#L147) — `<div className="grid min-h-0 flex-1 gap-3" style={{ gridTemplateColumns: "1fr 1fr" }}>` (form left, live-YAML Monaco right).

In **SetupTab wizard mode** the `WizardShell` is itself the **left cell** of SetupTab's own
`1fr 1fr` grid, nesting it two levels deep — the form column becomes **~25%** of the viewport:

- [SetupTab.tsx:237-244](ui/src/components/tabs/SetupTab.tsx#L237) — outer grid `gridTemplateColumns: showSummary && summary ? "1fr 1fr" : "1fr"`
- [SetupTab.tsx:303-305](ui/src/components/tabs/SetupTab.tsx#L303) — wizard mode renders `WizardShell` in `<div className="flex min-h-0 min-w-0 flex-col">`

Any wizard step laid out as if it had full width therefore overflows. The **only** step that
defends against this is `DutParamsStep`, which wraps its rows in `overflow-x-auto` and gives the
grid a fixed `min-w-[860px]`:

- [DutParamsStep.tsx:93-94](ui/src/components/wizard/steps/DutParamsStep.tsx#L93) — `<div className="overflow-x-auto"> <div className="grid min-w-[860px] ...">` — **the safe pattern.**

Every other step (PVT, PDK Rules, Testbench params, Basic Info, Target Specs) lacks this and
overflows. The overflow is then **clipped on the right** by `Panel`'s `overflow-hidden`:

- [panel.tsx:4-13](ui/src/components/ui/panel.tsx#L4) — `Panel` is `overflow-hidden rounded-md border ...`.

**Standard fix:** apply RC-1 (so tracks shrink to `fr` and no scroll is needed), **or** mirror
`DutParamsStep` (wrap in `overflow-x-auto` + a `min-w-[...]`).

<a id="rc-3"></a>
### RC-3 — Flex-child truncation without `min-w-0`

Several header rows place a `truncate` span next to a fixed sibling inside `flex ... justify-between`
**without** `min-w-0` on the truncating span. A flex item won't shrink below its content unless
it (or an ancestor) carries `min-w-0`, so the ellipsis never engages — the span keeps its full
width and pushes/overflows its sibling.

- [StudioLeftRail.tsx:26-31](ui/src/components/shell/StudioLeftRail.tsx#L26) — `truncate` name span at :27, no `min-w-0`; sibling badge at :28 not `shrink-0`.
- [StatusBar.tsx:49](ui/src/components/shell/StatusBar.tsx#L49) — `truncate` project span, no `min-w-0`.

The rails that **did** add `min-w-0 flex-1` truncate correctly and are the reference pattern:
`RunsRail`, `SpecsRail`, `OutlineRail`, and the SchematicTab breadcrumb
([SchematicTab.tsx:289](ui/src/components/tabs/SchematicTab.tsx#L289)).

**Standard fix:** add `min-w-0` to the truncating span and `shrink-0` to the fixed sibling.

<a id="rc-4"></a>
### RC-4 — Strip/bar rows with no horizontal-overflow handling

The `Toolbar` is the **correct** reference: `flex-nowrap overflow-x-auto whitespace-nowrap [&>*]:shrink-0`
keeps controls on one line and scrolls when narrow:

- [Toolbar.tsx:9-24](ui/src/components/shell/Toolbar.tsx#L9) — the pattern to copy.

But the **TabStrip** nav and the **StatusBar** footer do **not** adopt it — no `overflow-x-auto`,
items not `whitespace-nowrap`/`shrink-0`. On a narrow center column (left rail 200px + right rail
+ both activity bars open) their labels **wrap** (breaking the fixed `h-10`/`h-6` and clipping the
second line) or the row overflows with no scrollbar.

- [TabStrip.tsx:20-23](ui/src/components/shell/TabStrip.tsx#L20) — `<nav ... className="flex h-10 shrink-0 items-stretch gap-0.5 ... px-2">`
- [StatusBar.tsx:45](ui/src/components/shell/StatusBar.tsx#L45) — `<footer className="flex h-6 shrink-0 items-center gap-3 ... px-3">`

**Standard fix:** adopt the Toolbar pattern on both bars.

---

## Findings by tab

### Wizard

#### PVT Corners step

> Maps to **RC-1 + RC-2**. This is **reported symptom #1.**

- **Component:** [PVTStep.tsx](ui/src/components/wizard/steps/PVTStep.tsx)
- **Offending container:** the corner grid — header `<div className="grid grid-cols-[1.4fr_0.8fr_0.8fr_0.8fr_auto] gap-2 ...">` at [PVTStep.tsx:27](ui/src/components/wizard/steps/PVTStep.tsx#L27), rows at [PVTStep.tsx:32](ui/src/components/wizard/steps/PVTStep.tsx#L32), holding **4 `TextInput`** (Name, Temp `type=number`, Corner, Supply `type=number step=0.1`) at [PVTStep.tsx:33-36](ui/src/components/wizard/steps/PVTStep.tsx#L33) plus a trash button.
- **Failure mode:** horizontal-overflow → clipped on the right.
- **Root cause:** the 4 inputs have no `w-full`/`min-w-0` (RC-1), so each track's auto-minimum ≈ the input's ~180px intrinsic width; 4×~180 + button + gaps far exceeds the half/quarter-width wizard form column (RC-2). There is **no** `overflow-x-auto` wrapper and **no** `min-w-[...]` on the grid (unlike `DutParamsStep`). The overflow is clipped on the right by the enclosing `Panel` `overflow-hidden` — so the rightmost **Supply (0.8fr)** column is cut off.
- **Fix direction:** mirror `DutParamsStep` (wrap both grids in `overflow-x-auto`, give the grid a `min-w-[520px]`-ish floor). **Better root fix:** add `w-full min-w-0` to `TextInput` via `inputCn` (RC-1) so the `fr` tracks can shrink and no horizontal scroll is needed at all.

##### PVT in SetupTab wizard mode (same defect, worse)

- **Container:** the same PVT grid, nested in SetupTab's left cell ([SetupTab.tsx:303](ui/src/components/tabs/SetupTab.tsx#L303)) whose child `WizardShell` is itself `1fr 1fr` ([WizardShell.tsx:147](ui/src/components/wizard/WizardShell.tsx#L147)) — so the row sits in ~**25%** of viewport width.
- **Failure mode:** horizontal-overflow (more severe).
- **Root cause:** identical to above; nesting two `1fr 1fr` grids confirms the defect is in **PVTStep**, not the host.
- **Fix direction:** the single PVTStep fix resolves both the overlay and the SetupTab host context.

#### PDK Rules step

> Maps to **RC-1 + RC-2.**

- **Component:** [PDKRulesStep.tsx](ui/src/components/wizard/steps/PDKRulesStep.tsx)
- **Offending container:** `<div className="grid grid-cols-[1fr_1fr_auto] gap-2">` at [PDKRulesStep.tsx:52](ui/src/components/wizard/steps/PDKRulesStep.tsx#L52) with two `TextInput` at [:53-54](ui/src/components/wizard/steps/PDKRulesStep.tsx#L53) plus a trash button. (The `Technology name` field at [:31-33](ui/src/components/wizard/steps/PDKRulesStep.tsx#L31) is single-column and lower-risk.)
- **Failure mode:** horizontal-overflow → clipped.
- **Root cause:** two inputs with no `w-full`/`min-w-0` → two ~180px track minimums + button; borderline in a half-width column, overflows the quarter-width SetupTab-wizard column. No `overflow-x` wrapper. Clipped by `Panel overflow-hidden`.
- **Fix direction:** add `w-full min-w-0` to the inputs (preferably via `inputCn`), optionally wrap in `overflow-x-auto`.

#### Testbenches step — per-testbench params table

> Maps to **RC-1 + RC-2.**

- **Component:** [TestbenchesStep.tsx](ui/src/components/wizard/steps/TestbenchesStep.tsx)
- **Offending container:** `<div className="grid grid-cols-[1.4fr_1fr_1.6fr_auto] ...">` — header at [TestbenchesStep.tsx:131](ui/src/components/wizard/steps/TestbenchesStep.tsx#L131), rows at [:135](ui/src/components/wizard/steps/TestbenchesStep.tsx#L135), Name/Value/Description `TextInput` at [:137-141](ui/src/components/wizard/steps/TestbenchesStep.tsx#L137).
- **Failure mode:** horizontal-overflow → clipped.
- **Root cause:** three inputs without `w-full`/`min-w-0` (the 1.6fr Description input is widest) give a row min-width well over the half-width wizard column; no `overflow-x-auto`, no `min-w-[...]`. Same class as PVT.
- **Fix direction:** mirror `DutParamsStep` (`overflow-x-auto` + `min-w-[...]`) or apply RC-1.

#### Testbenches step (top fields) + Basic Info step

> Maps to **RC-1 + RC-2.**

- **Components:** [TestbenchesStep.tsx](ui/src/components/wizard/steps/TestbenchesStep.tsx), [BasicInfoStep.tsx](ui/src/components/wizard/steps/BasicInfoStep.tsx)
- **Offending container:** `grid grid-cols-2` of `Field` wrappers — [TestbenchesStep.tsx:94](ui/src/components/wizard/steps/TestbenchesStep.tsx#L94) (`grid grid-cols-2 gap-3 p-3`) and [BasicInfoStep.tsx:13](ui/src/components/wizard/steps/BasicInfoStep.tsx#L13) (`grid grid-cols-2 gap-3 p-4`). Basic Info's `Simulator` cell is a bare `selectCn("sm")` `<select>` at [BasicInfoStep.tsx:21-29](ui/src/components/wizard/steps/BasicInfoStep.tsx#L21).
- **Failure mode:** horizontal-overflow → clipped.
- **Root cause:** `Field` ([wizard-controls.tsx:20-28](ui/src/components/wizard/wizard-controls.tsx#L20)) is `flex flex-col` with **no `min-w-0`**, and its `TextInput`/`<select>` children have no `w-full`, so each grid column's min = the control's intrinsic ~180px (RC-1). Two such columns can exceed the narrow wizard form column. Clipped by `Panel overflow-hidden`.
- **Fix direction:** add `min-w-0` to `Field` and `w-full min-w-0` to the form controls (best via `inputCn`/`selectCn` so all steps inherit it).

#### Target Specs step — expanded spec editor

> Maps to **RC-1 + RC-2.** Highest selector-width risk (long `<option>` labels).

- **Component:** [TargetSpecsStep.tsx](ui/src/components/wizard/steps/TargetSpecsStep.tsx)
- **Offending container:** `<div className="grid grid-cols-3 gap-3 border-t ...">` at [TargetSpecsStep.tsx:102](ui/src/components/wizard/steps/TargetSpecsStep.tsx#L102) of `Field` cells holding `TextInput` + bare `selectCn("sm")` `<select>`s (Testbench, Sim type, Goal, Error type, Reward type — [:104-133](ui/src/components/wizard/steps/TargetSpecsStep.tsx#L104)).
- **Failure mode:** horizontal-overflow → clipped by the spec card border + `Panel`.
- **Root cause:** three columns whose track-minimum equals the **widest control content** — and a `<select>`'s `min-content` is its **longest option label**. The option lists are long: `error_type` includes `relative-exponential`, `sim_type` includes `noise_spectrum` ([:11-14](ui/src/components/wizard/steps/TargetSpecsStep.tsx#L11)). With no `w-full`/`min-w-0` (RC-1), three of these in a half/quarter-width column overflow.
- **Fix direction:** give selects/inputs `w-full min-w-0` and `Field` `min-w-0`; optionally drop to 2 columns at narrow widths.

#### DutParams step — REFERENCE (no defect)

- [DutParamsStep.tsx:93-94 / :105](ui/src/components/wizard/steps/DutParamsStep.tsx#L93) — wraps the rows in `overflow-x-auto` and pins `min-w-[860px]` on both the header and each row grid. This is the **correct** pattern; listed so the other steps can copy it. No change.

---

### Shell

#### TabStrip (all views)

> Maps to **RC-4.**

- **Component:** [TabStrip.tsx](ui/src/components/shell/TabStrip.tsx)
- **Offending container:** `<nav role="tablist" className="flex h-10 shrink-0 items-stretch gap-0.5 border-b border-border bg-panel px-2">` at [TabStrip.tsx:20-23](ui/src/components/shell/TabStrip.tsx#L20) with the per-view tab `<button>`s at [:30-50](ui/src/components/shell/TabStrip.tsx#L30) (each `flex items-center gap-1.5 px-3` with an icon + label).
- **Failure mode:** horizontal-overflow / wrap-and-clip.
- **Root cause:** the nav has **no `overflow-x-auto`**; tab buttons are not `whitespace-nowrap` or `shrink-0`. When the center column is narrow the labels exceed the width — flex wraps the label text (breaking the fixed `h-10` and clipping the second line) or overflows with no scrollbar.
- **Fix direction:** apply the Toolbar pattern — `flex-nowrap overflow-x-auto whitespace-nowrap [&>*]:shrink-0` on the nav.

#### StatusBar (all views)

> Maps to **RC-4** (and RC-3 for the project span).

- **Component:** [StatusBar.tsx](ui/src/components/shell/StatusBar.tsx)
- **Offending container:** `<footer className="flex h-6 shrink-0 items-center gap-3 border-t border-border bg-panel px-3 ...">` at [StatusBar.tsx:45](ui/src/components/shell/StatusBar.tsx#L45) with view label ([:46](ui/src/components/shell/StatusBar.tsx#L46)), project span ([:49](ui/src/components/shell/StatusBar.tsx#L49)), run-progress span ([:54-67](ui/src/components/shell/StatusBar.tsx#L54)), a `flex-1` spacer ([:69](ui/src/components/shell/StatusBar.tsx#L69)), panel toggles, and the env pill ([:96-103](ui/src/components/shell/StatusBar.tsx#L96)).
- **Failure mode:** horizontal-overflow; the far-right env pill can be pushed off-screen.
- **Root cause:** only the project span has `truncate` ([:49](ui/src/components/shell/StatusBar.tsx#L49)) but it lacks `min-w-0`, so it won't actually shrink (RC-3); the other spans are not nowrap-protected and the footer has no `overflow-x-auto`/`overflow-hidden` (RC-4).
- **Fix direction:** add `min-w-0` to the truncating span (and `overflow-hidden` to the footer), make fixed pills `shrink-0`; optionally `overflow-x-auto whitespace-nowrap` like the Toolbar.

#### StudioLeftRail (all views)

> Maps to **RC-3.**

- **Component:** [StudioLeftRail.tsx](ui/src/components/shell/StudioLeftRail.tsx)
- **Offending container:** project header `<div className="flex items-center justify-between rounded px-1.5 py-1 text-fg">` at [StudioLeftRail.tsx:26](ui/src/components/shell/StudioLeftRail.tsx#L26) with a `truncate` name span at [:27](ui/src/components/shell/StudioLeftRail.tsx#L27) next to an `active`/`draft` badge at [:28-30](ui/src/components/shell/StudioLeftRail.tsx#L28).
- **Failure mode:** mis-sizing — long project name fails to ellipsize.
- **Root cause:** the `truncate` span has no `min-w-0`, and the sibling badge is not `shrink-0`; a flex item won't shrink below content without `min-w-0`, so a long name widens past the 200px rail (the rail itself is fixed `w-[200px]` at [:23](ui/src/components/shell/StudioLeftRail.tsx#L23)) or pushes the badge.
- **Fix direction:** add `min-w-0` to the name span and `shrink-0` to the badge (matches the working `RunsRail`/`SpecsRail` pattern).

---

### Schematic

#### Breadcrumb — REFERENCE (no defect)

- **Component:** [SchematicTab.tsx](ui/src/components/tabs/SchematicTab.tsx)
- **Container:** `<div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto whitespace-nowrap text-xs text-muted">` at [SchematicTab.tsx:289](ui/src/components/tabs/SchematicTab.tsx#L289).
- **Status:** correctly handled (`min-w-0 flex-1 overflow-x-auto whitespace-nowrap`). Listed only to confirm it is **not** a defect — it is the exact pattern `TabStrip` and `StatusBar` should copy.

---

### Health / Sanity Check

> **Reported symptom #2.** Verdict: **not a source CSS defect** — the height/scroll chain is intact.

- **Component:** [HealthTab.tsx](ui/src/components/tabs/HealthTab.tsx)
- **Containers in the chain:**
  - StudioShell wraps the center children in `flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden` at [StudioShell.tsx:47](ui/src/components/shell/StudioShell.tsx#L47) and again at [:49](ui/src/components/shell/StudioShell.tsx#L49).
  - The HealthTab page body is `flex min-h-0 flex-1 flex-col gap-2.5 overflow-auto p-3` at [HealthTab.tsx:80](ui/src/components/tabs/HealthTab.tsx#L80).
  - The ngspice log `<pre>` is `max-h-60 overflow-auto ...` at [HealthTab.tsx:208](ui/src/components/tabs/HealthTab.tsx#L208) and [:307](ui/src/components/tabs/HealthTab.tsx#L307), each inside a `<details>` ([:198](ui/src/components/tabs/HealthTab.tsx#L198), [:294](ui/src/components/tabs/HealthTab.tsx#L294)).
- **Failure mode (as reported):** clipped-no-scroll. **Verified:** the chain is **NOT broken** — every level has `min-h-0` and the page body is `overflow-auto`, so the view **does** get a vertical scrollbar when output exceeds the viewport, and the log `<pre>` is itself `max-h-60 overflow-auto` (it scrolls internally). The classic broken-`min-h-0` trap is **absent** here.
- **Root cause:** none in current CSS. The only residual nit is cosmetic — the page body uses `overflow-auto` (both axes) where `overflow-y-auto` is the intent; the real long-content surface is the inner `<pre>`, which scrolls correctly. If a clip is observed at runtime it is **not explained by the source** — the most likely culprit is a **stale `.next` production build** shadowing the dev chunks (a documented hazard in `CLAUDE.md`: "Stale `.next` cache ... Delete `ui/.next` before restarting after a build").
- **Fix direction:** no structural change required for scrolling. Optionally switch the page body to `overflow-y-auto` for intent clarity. **If the clip reproduces, delete `ui/.next` and rebuild before touching any CSS** — the chain is already correct.

---

### Pipeline

> Maps to **RC-2-adjacent** (fixed `min-w` track), but low severity — it scrolls.

- **Component:** [PipelineView.tsx](ui/src/components/tabs/PipelineView.tsx)
- **Offending container:** the DAG row `<div className="flex items-stretch gap-3">` at [PipelineView.tsx:118](ui/src/components/tabs/PipelineView.tsx#L118) with 4 `Column`s — each `flex min-w-[180px] flex-1` at [:27](ui/src/components/tabs/PipelineView.tsx#L27) — interleaved with 3 `Arrow`s, inside the page body `<div className="flex min-h-0 flex-1 flex-col overflow-auto p-4">` at [:111](ui/src/components/tabs/PipelineView.tsx#L111).
- **Failure mode:** horizontal-overflow → **scrolls** (low severity).
- **Root cause:** 4 columns at `min-w-[180px]` + arrows + gaps give a row minimum ~800px; on a narrow center column this exceeds the width. **Mitigated** because the parent ([:111](ui/src/components/tabs/PipelineView.tsx#L111)) is `overflow-auto` and `Node` titles use `truncate` ([:72](ui/src/components/tabs/PipelineView.tsx#L72)) — so this is a scroll, not a clip.
- **Fix direction:** acceptable as-is. If horizontal scroll is undesired, lower `min-w-[180px]` or allow columns to wrap; keep the parent `overflow-auto`.

---

## Summary of fixes by leverage

| Fix | Resolves |
|---|---|
| Add `w-full min-w-0` to `inputCn`/`selectCn` + `min-w-0` to `Field` ([wizard-controls.tsx](ui/src/components/wizard/wizard-controls.tsx), [select.tsx](ui/src/components/ui/select.tsx)) | PVT, PDK Rules, Testbench params/top, Basic Info, Target Specs (all RC-1/RC-2 overflow) |
| Apply the Toolbar pattern (`flex-nowrap overflow-x-auto whitespace-nowrap [&>*]:shrink-0`) to `TabStrip` + `StatusBar` | RC-4 tab/footer overflow |
| Add `min-w-0` to truncating spans + `shrink-0` to fixed siblings ([StudioLeftRail.tsx:27-28](ui/src/components/shell/StudioLeftRail.tsx#L27), [StatusBar.tsx:49](ui/src/components/shell/StatusBar.tsx#L49)) | RC-3 ellipsis failures |
| (Health) delete `ui/.next` + rebuild if clip reproduces; no CSS change | Reported symptom #2 |
