# UI Layout Audit — Part 4 (Layout Agent)

> **STALE NOTE (2026-06):** this audit predates the nav restructure. The horizontal top `TabStrip`
> was removed (the vertical ActivityBar is now the sole top-level nav; in-view sub-tabs use the new
> `SubTabStrip`), Manual Sim moved to its own `/manual` view, and the Optimize toolbar lost its
> algorithm dropdown. Line anchors referencing `TabStrip.tsx` or OptimizeTab's algorithm `<select>`
> no longer resolve. See `CLAUDE.md` for the current shell layout.

**Scope:** CSS / layout / sizing / overflow / scroll only. This report does **not** cover data
flow, state, accessibility semantics, or visual styling beyond what causes content to be clipped,
mis-sized, or to overflow. Every claim is anchored to `file:line` in the **current working tree at
HEAD** (branch `feat/pvt`).

**Round constraints.** This is a **static-analysis** planning round — no app/docker/live-sim was
run on the server. The report says what *should* be done; **no application code was modified.** A
prior audit on the older `dev/ui` branch already fixed several of the issues it found; this pass
re-derives everything against HEAD and only reports what is **still present**. Anything that would
require a running browser to confirm is marked *needs runtime verification (deferred — no live UI on
server this round)*.

**What changed since the prior audit (verified at HEAD).** Three of the prior audit's global fixes
have landed and are confirmed in source, so they are **not** re-reported as active:

- `inputCn` now ships `w-full min-w-0` ([wizard-controls.tsx:4-13](ui/src/components/wizard/wizard-controls.tsx#L4)) and `Field` now ships `min-w-0` ([wizard-controls.tsx:24](ui/src/components/wizard/wizard-controls.tsx#L24)) — so **text** inputs shrink with their track. This is why the input-only wizard grids no longer overflow as badly as the prior report claimed.
- Commit **42dd636** added `[&>*]:shrink-0` to the three flex-column scroll tabs — confirmed present at [OptimizeTab.tsx:205](ui/src/components/tabs/OptimizeTab.tsx#L205), [HealthTab.tsx:101](ui/src/components/tabs/HealthTab.tsx#L101), [ExplorerTab.tsx:253](ui/src/components/tabs/ExplorerTab.tsx#L253). The prior "Sanity Check logs clipped, no scroll" symptom is **resolved by source** (see [Health](#health--sanity-check)).
- `selectCn` carries `min-w-0` ([select.tsx:11-21](ui/src/components/ui/select.tsx#L11)).

The residual issues below are the ones that **survive** those fixes — chiefly the wizard grids that
still use bare `fr` tracks and mix `<input type="number">`, plus a `<select>` mis-sizing and one
rail-table clip.

---

## Global root causes

Fix these at the source and most per-tab instances disappear. They are ordered by leverage.

<a id="grc-1"></a>
### GRC-1 — Explicit grid tracks default to a `min-content` floor; nothing uses `minmax(0,…fr)`

Every explicit-track grid in the wizard declares bare `fr`/`auto` columns (e.g.
[PVTStep.tsx:113](ui/src/components/wizard/steps/PVTStep.tsx#L113) `grid-cols-[1.4fr_0.7fr_0.8fr_0.8fr]`).
CSS treats a bare `Nfr` track as `minmax(auto, Nfr)`, and the `auto` floor pins each track to its
content's **min-content** width. A track can therefore grow *wider* than its `fr` share, and the
row's total can exceed the container — the classic "grid child overflows parent." A grep across
`ui/src` finds **zero** uses of `minmax(0,…)`, so **no** grid opts out of this floor. The robust fix
the whole wizard is missing is `minmax(0,Nfr)` on the `fr` tracks (and/or `min-w-0` on the grid
container itself).

*Affects:* PVTStep, TestbenchesStep params, OptimizerStep `optimizer_kwargs`, PDKRulesStep.

<a id="grc-2"></a>
### GRC-2 — `<input type="number">` keeps UA spinner chrome that `min-w-0` on the input does not strip from track sizing

`inputCn` correctly sets `w-full min-w-0` on every `TextInput`, which tames **text** inputs in a
shrinking track. But a `<input type="number">` keeps a **UA-imposed intrinsic minimum** (the
up/down spinner) that inflates the **grid track's** `min-content` floor even when the input element
itself carries `min-w-0`. Combined with GRC-1, this is what actually pushes the rightmost number
column past the card edge. The grids that overflow worst are precisely the ones mixing number inputs
into narrow `fr` tracks — PVT Temp/Supply-V ([PVTStep.tsx:118](ui/src/components/wizard/steps/PVTStep.tsx#L118), [:124](ui/src/components/wizard/steps/PVTStep.tsx#L124)) and OptimizerStep budget/seed ([OptimizerStep.tsx:114](ui/src/components/wizard/steps/OptimizerStep.tsx#L114), [:117](ui/src/components/wizard/steps/OptimizerStep.tsx#L117)).

<a id="grc-3"></a>
### GRC-3 — Only `DutParamsStep` adopted the `overflow-x-auto` + `min-w-[Npx]` escape hatch; sibling steps did not

`DutParamsStep` wraps its 8-column grid in `overflow-x-auto` and gives the grid a hard
`min-w-[860px]` ([DutParamsStep.tsx:93-94](ui/src/components/wizard/steps/DutParamsStep.tsx#L93), repeated on each row at [:105](ui/src/components/wizard/steps/DutParamsStep.tsx#L105)), so when the half-width wizard form column is too narrow the grid **scrolls horizontally** instead of clipping. **PVTStep** (the newest step), **TestbenchesStep**, **OptimizerStep**, and **PDKRulesStep** never received this treatment, so their multi-column grids have **no horizontal-overflow handling at all**. This inconsistency is exactly why DUT params are fine but PVT corners clip — the reference pattern to copy already exists in the tree.

<a id="grc-4"></a>
### GRC-4 — `selectCn()` intentionally omits `w-full`, so `<select>` fields don't fill their grid track

[select.tsx:11-21](ui/src/components/ui/select.tsx#L11) sets `min-w-0` but **deliberately not**
`w-full` (comment: would stretch toolbar selects). In the wizard step forms (TargetSpecsStep,
OptimizerStep, BasicInfoStep) the `<select>`s are placed in grid `<Field>` cells alongside `w-full`
`TextInput`s but are **not** given an explicit `w-full`, so they render narrower than sibling inputs
and shrink to content width — a **mis-sizing / visual-misalignment**, not overflow. Where `w-full`
*is* needed it is added ad hoc (`RunControl`, `CornerSelect`, and the `Select` wrapper's labeled
branch at [select.tsx:41](ui/src/components/ui/select.tsx#L41)), which is easy to forget.

<a id="grc-5"></a>
### GRC-5 — `Panel` always sets `overflow-hidden` (flex auto-min-size 0) — already mitigated where it bit, by 42dd636

[panel.tsx:8](ui/src/components/ui/panel.tsx#L8) hard-codes `overflow-hidden` on every `<Panel>`.
In a `flex flex-col` scroll container an `overflow-hidden` child gets an automatic flex min-size of
0, so the column algorithm crushes the Panel to fit the viewport and the Panel clips its own
spillover (log tails, tables) instead of letting the container scroll. This was the single mechanism
behind the entire **clipped-no-scroll** family (Health trial logs, Optimize Manual Sim logs,
Explorer envelope/spec tables). The fix idiom is `[&>*]:shrink-0` on the flex-col scroll container,
which forces children to natural height so the **container** scrolls. **Commit 42dd636 applied it to
all three affected flex tabs and it is present and correct at HEAD** — see the [Health](#health--sanity-check),
[Optimize](#optimize), and [Compare/Explore](#compare--explore) sections. **No active
clipped-no-scroll regression remains in those tabs.** The grid-based tabs (Setup, Score Shaping) are
immune because grid items default to `min-height:auto`, not 0, so they were correctly left out of
that change.

---

## Findings by tab

### New-project Wizard

#### PVT Corners step — main corner grid (reported repro)

> Maps to **GRC-1 + GRC-2 + GRC-3.** This is the reported "Supply column cut off" symptom.

- **Component:** [PVTStep.tsx](ui/src/components/wizard/steps/PVTStep.tsx)
- **Offending container:** [PVTStep.tsx:113](ui/src/components/wizard/steps/PVTStep.tsx#L113) — `<div className="grid grid-cols-[1.4fr_0.7fr_0.8fr_0.8fr] gap-2">` holding Name ([:115](ui/src/components/wizard/steps/PVTStep.tsx#L115)), **Temp `type=number`** ([:118](ui/src/components/wizard/steps/PVTStep.tsx#L118)), Supply node ([:121](ui/src/components/wizard/steps/PVTStep.tsx#L121)), **Supply (V) `type=number`** ([:124](ui/src/components/wizard/steps/PVTStep.tsx#L124)), inside the `p-3` corner card ([:98](ui/src/components/wizard/steps/PVTStep.tsx#L98)).
- **Failure mode:** horizontal-overflow → clipped on the right.
- **Root cause:** the 4-column grid uses bare `fr` tracks (no `minmax(0,…)`, GRC-1), the grid container has no `min-w-0`, and there is **no** `overflow-x-auto` wrapper (unlike DutParamsStep, GRC-3). Two of the four cells are `<input type="number">` whose UA spinner inflates each track's `min-content` floor (GRC-2) even though `inputCn` already sets `min-w-0`. Inside the **half-width** wizard form column (`WizardShell` body is `1fr 1fr` at [WizardShell.tsx:147](ui/src/components/wizard/WizardShell.tsx#L147)) the four track minimums sum past the corner card and the rightmost **Supply (V)** column is pushed off the right edge; the spillover is clipped by `Panel`'s `overflow-hidden` ([panel.tsx:8](ui/src/components/ui/panel.tsx#L8), GRC-5). Confirmed present at HEAD (the step was added in a15b420). It is **worse in SetupTab "Create Wizard" mode**, where `WizardShell` is itself the left cell of SetupTab's own `1fr 1fr` grid ([SetupTab.tsx:243](ui/src/components/tabs/SetupTab.tsx#L243)), nesting the form into ~25% of viewport width.
- **Fix direction (highest leverage in this tab):** lowest-risk is to **mirror DutParamsStep** — wrap the grid in `overflow-x-auto` and give it a hard `min-w-[Npx]` (GRC-3). Alternatively, switch the tracks to `minmax(0,1.4fr) minmax(0,0.7fr) minmax(0,0.8fr) minmax(0,0.8fr)` and add `min-w-0` to the grid (GRC-1); since two cells are number inputs, also consider letting the row wrap on narrow widths (e.g. drop to `grid-cols-2`) so the spinner floors stop competing for one line.

#### PVT Corners step — model-includes grid

> Maps to **GRC-1.** Lower severity (text-only).

- **Component:** [PVTStep.tsx](ui/src/components/wizard/steps/PVTStep.tsx)
- **Offending container:** [PVTStep.tsx:153](ui/src/components/wizard/steps/PVTStep.tsx#L153) — `<div className="grid grid-cols-[1fr_1fr_auto] gap-2">` (lib_file / section / remove), nested inside the already-narrow `bg-zinc-50 p-2` includes box ([:134](ui/src/components/wizard/steps/PVTStep.tsx#L134)) inside the corner card.
- **Failure mode:** horizontal-overflow (low severity).
- **Root cause:** bare `fr` tracks (GRC-1) with no `minmax(0,…)`. Both data cells are `TextInput` (so `inputCn`'s `min-w-0` mostly saves it), but the `auto` floor plus the fixed `auto` trash-button column leave no slack at minimum width — same class as the main grid, lower severity.
- **Fix direction:** `minmax(0,1fr) minmax(0,1fr) auto`, or rely on the same `overflow-x-auto` wrapper if the whole step is wrapped (consistent with the GRC-1/GRC-3 fix).

#### Testbenches step — per-testbench params grid

> Maps to **GRC-1 + GRC-3.** Medium-low severity (text-only).

- **Component:** [TestbenchesStep.tsx](ui/src/components/wizard/steps/TestbenchesStep.tsx)
- **Offending container:** the params grid `grid-cols-[1.4fr_1fr_1.6fr_auto]` — header at [TestbenchesStep.tsx:131](ui/src/components/wizard/steps/TestbenchesStep.tsx#L131), rows at [:135](ui/src/components/wizard/steps/TestbenchesStep.tsx#L135), Name/Value/Description `TextInput`s at [:137-141](ui/src/components/wizard/steps/TestbenchesStep.tsx#L137).
- **Failure mode:** horizontal-overflow.
- **Root cause:** 4-column `fr`/`auto` grid with no `minmax(0,…)`, no `min-w-0` on the grid, and no overflow-x wrapper (GRC-1, GRC-3). All-text inputs shrink acceptably via `inputCn`, but a long Description value (the 1.6fr track) plus the fixed `auto` remove column can still drive the row past the half-width form column.
- **Fix direction:** apply `minmax(0,…fr)` to the three `fr` tracks, or wrap in `overflow-x-auto` with a `min-w-[Npx]` like DutParamsStep.
- *(The testbench top fields at [TestbenchesStep.tsx:94](ui/src/components/wizard/steps/TestbenchesStep.tsx#L94) `grid grid-cols-2` are text-only with `col-span-2` on the long fields — low risk now that `inputCn`/`Field` carry `min-w-0`.)*

#### Optimizer step — `optimizer_kwargs` grid + number-input rows

> Maps to **GRC-1 + GRC-2.** Lower severity than PVT (fewer columns).

- **Component:** [OptimizerStep.tsx](ui/src/components/wizard/steps/OptimizerStep.tsx)
- **Offending containers:** the `optimizer_kwargs` grid `grid-cols-[1.2fr_1.6fr_auto]` — header at [OptimizerStep.tsx:162](ui/src/components/wizard/steps/OptimizerStep.tsx#L162), rows at [:166](ui/src/components/wizard/steps/OptimizerStep.tsx#L166); and the page-level `grid grid-cols-2` ([:66](ui/src/components/wizard/steps/OptimizerStep.tsx#L66)) hosting six `<input type="number">` (budget/seed at [:114](ui/src/components/wizard/steps/OptimizerStep.tsx#L114), [:117](ui/src/components/wizard/steps/OptimizerStep.tsx#L117); lin/log bounds at [:123-124](ui/src/components/wizard/steps/OptimizerStep.tsx#L123) and [:129-130](ui/src/components/wizard/steps/OptimizerStep.tsx#L129)).
- **Failure mode:** mis-sizing / crowding (the kwargs row can also overflow).
- **Root cause:** the kwargs grid uses bare `fr` (GRC-1). On the 2-column page grid the number-input spinner floor (GRC-2) crowds the two number columns toward the edge in the narrow form column. Lower severity than PVT because it is only 2 columns and the kwargs rows are text-only.
- **Fix direction:** use `minmax(0,…fr)` on the kwargs grid; the page `grid-cols-2` is fine once GRC-2 is mitigated (consider `appearance-none` to drop the number spinners, or accept the existing `min-w-0`).

#### PDK Rules step — constraints grid

> Maps to **GRC-1.** Lowest severity of the step grids.

- **Component:** [PDKRulesStep.tsx](ui/src/components/wizard/steps/PDKRulesStep.tsx)
- **Offending container:** [PDKRulesStep.tsx:52](ui/src/components/wizard/steps/PDKRulesStep.tsx#L52) — `<div className="grid grid-cols-[1fr_1fr_auto] gap-2">` with two `TextInput`s ([:53-54](ui/src/components/wizard/steps/PDKRulesStep.tsx#L53)) plus a fixed remove column.
- **Failure mode:** horizontal-overflow (borderline).
- **Root cause:** bare `fr` (GRC-1), no `min-w-0` on the grid, no overflow-x wrapper. Two text inputs + the `auto` remove column; `inputCn`'s `min-w-0` keeps it mostly in bounds — same root pattern as the others, lowest severity.
- **Fix direction:** switch to `minmax(0,1fr) minmax(0,1fr) auto` for consistency with the global fix.

#### All select-bearing steps — `<select>` does not fill its `<Field>` cell

> Maps to **GRC-4.** Mis-sizing, not overflow.

- **Components:** [TargetSpecsStep.tsx](ui/src/components/wizard/steps/TargetSpecsStep.tsx), [OptimizerStep.tsx](ui/src/components/wizard/steps/OptimizerStep.tsx), [BasicInfoStep.tsx](ui/src/components/wizard/steps/BasicInfoStep.tsx)
- **Offending elements:** `<select className={selectCn("sm")}>` inside grid `<Field>` cells — TargetSpecsStep [:105](ui/src/components/wizard/steps/TargetSpecsStep.tsx#L105), [:111](ui/src/components/wizard/steps/TargetSpecsStep.tsx#L111), [:116](ui/src/components/wizard/steps/TargetSpecsStep.tsx#L116), [:125](ui/src/components/wizard/steps/TargetSpecsStep.tsx#L125), [:130](ui/src/components/wizard/steps/TargetSpecsStep.tsx#L130); OptimizerStep [:69](ui/src/components/wizard/steps/OptimizerStep.tsx#L69), [:82](ui/src/components/wizard/steps/OptimizerStep.tsx#L82), [:94](ui/src/components/wizard/steps/OptimizerStep.tsx#L94), [:102](ui/src/components/wizard/steps/OptimizerStep.tsx#L102); BasicInfoStep [:22](ui/src/components/wizard/steps/BasicInfoStep.tsx#L22).
- **Failure mode:** mis-sizing (under-sizing / visual misalignment within the grid).
- **Root cause:** `selectCn` sets `min-w-0` but intentionally not `w-full` ([select.tsx:13-15 comment](ui/src/components/ui/select.tsx#L13), GRC-4). These selects sit in grid cells next to `w-full` `TextInput`s but are not given an explicit `w-full`, so they shrink to content width and render narrower / misaligned against their sibling inputs (visible inside the 3-column TargetSpecs editor at [TargetSpecsStep.tsx:102](ui/src/components/wizard/steps/TargetSpecsStep.tsx#L102), where text fields and selects share a row).
- **Fix direction:** add `w-full` at each wizard `<select>` call site (`selectCn("sm") + " w-full"`, the pattern `RunControl`/`CornerSelect` already use), or add a `w-full` opt-in variant to `selectCn` for in-form fields. *(Visual severity needs runtime verification — deferred, no live UI this round.)*

#### DutParamsStep — REFERENCE (no defect)

- [DutParamsStep.tsx:93-94](ui/src/components/wizard/steps/DutParamsStep.tsx#L93) wraps the 8-column rows in `overflow-x-auto` and pins `min-w-[860px]` on both header and each row grid ([:105](ui/src/components/wizard/steps/DutParamsStep.tsx#L105)). This is the **correct** pattern; listed so the other steps can copy it. No change.

---

### Setup

#### Create-Wizard mode — Monaco height chain may not receive a definite height

> Mis-sizing. **Needs runtime verification (deferred — no live UI on server this round).**

- **Component:** [SetupTab.tsx](ui/src/components/tabs/SetupTab.tsx) → [WizardShell.tsx](ui/src/components/wizard/WizardShell.tsx)
- **Offending container:** when `mode !== "load"` the wizard renders inside `<div className="flex min-h-0 min-w-0 flex-col">` at [SetupTab.tsx:304](ui/src/components/tabs/SetupTab.tsx#L304) — this wrapper has **no `h-full` / `flex-1`**, and it is a **grid item** of the container at [SetupTab.tsx:238-244](ui/src/components/tabs/SetupTab.tsx#L238) (`grid min-h-0 flex-1 overflow-auto`, auto-sized rows). `WizardShell`'s root is `flex h-full min-h-0 flex-col` ([WizardShell.tsx:108](ui/src/components/wizard/WizardShell.tsx#L108)) and its body is `grid min-h-0 flex-1` ([WizardShell.tsx:147](ui/src/components/wizard/WizardShell.tsx#L147)) feeding two Monaco editors with `height="100%"` ([WizardShell.tsx:187](ui/src/components/wizard/WizardShell.tsx#L187), and the load-mode Monaco at [SetupTab.tsx:267](ui/src/components/tabs/SetupTab.tsx#L267)).
- **Failure mode:** mis-sizing — Monaco `height="100%"` needs an unbroken chain of *definite* heights; with no definite height flowing from the auto-sized grid row into the no-`h-full` wrapper, the `100%` editors can collapse to their `loading` min-height instead of filling.
- **Root cause:** a broken definite-height chain at the [SetupTab.tsx:304](ui/src/components/tabs/SetupTab.tsx#L304) wrapper. **Lower confidence:** the grid item *does* stretch via the default `align-items: stretch`, so it may render acceptably; this is a sizing risk, not a clip — confirm visually.
- **Fix direction:** give the wizard a definite height to distribute — add `h-full` (or `flex-1`) to the [SetupTab.tsx:304](ui/src/components/tabs/SetupTab.tsx#L304) wrapper so `WizardShell`'s `h-full` resolves against a stretched grid item; OR make the wizard branch not depend on `h-full` (intrinsic `min-height` on the Monaco columns).

#### Load/Edit mode — REFERENCE (verified safe)

- [SetupTab.tsx:238-244](ui/src/components/tabs/SetupTab.tsx#L238) is a **grid** (`grid min-h-0 flex-1 gap-3 overflow-auto p-3`), not a flex column, so the `Panel` `overflow-hidden` auto-min-size-0 crush that hit the three flex tabs does **not** apply here (grid items default to `min-height:auto` and size the row to content; the container scrolls). The left Monaco `Panel` ([:248](ui/src/components/tabs/SetupTab.tsx#L248)) deliberately uses `flex min-h-0 flex-col` + `height="100%"` ([:267](ui/src/components/tabs/SetupTab.tsx#L267)) to fill the grid-row height. This is correctly left out of the 42dd636 change. No action.

---

### Optimize

> **Previously reported, now fixed (verified at HEAD).** Maps to **GRC-5.**

- **Component:** [OptimizeTab.tsx](ui/src/components/tabs/OptimizeTab.tsx) (Manual Sim region)
- **Container:** [OptimizeTab.tsx:205](ui/src/components/tabs/OptimizeTab.tsx#L205) — `flex min-h-0 flex-1 flex-col gap-2.5 overflow-auto p-3 [&>*]:shrink-0`.
- **Original failure mode:** clipped-no-scroll — `ManualSimPanel`'s per-spec result + ngspice log tails rendered below the fold and were unreachable (Panel-auto-min-size-0 crush, GRC-5).
- **Status:** **resolved.** Commit 42dd636 added `[&>*]:shrink-0` here so each Panel keeps natural height and the container scrolls; the commit message and in-browser note confirm the Manual Sim logs are reachable. The accompanying explanatory comment is at [OptimizeTab.tsx:202-204](ui/src/components/tabs/OptimizeTab.tsx#L202). **No action needed.**

### Health / Sanity Check

> **Previously reported ("logs clipped, no scroll"), now fixed (verified at HEAD).** Maps to **GRC-5.**

- **Component:** [HealthTab.tsx](ui/src/components/tabs/HealthTab.tsx)
- **Container:** [HealthTab.tsx:101](ui/src/components/tabs/HealthTab.tsx#L101) — `flex min-h-0 flex-1 flex-col gap-2.5 overflow-auto p-3 [&>*]:shrink-0` (explanatory comment at [:98](ui/src/components/tabs/HealthTab.tsx#L98)).
- **Original failure mode:** clipped-no-scroll — the children are `Panel`s; `Panel`'s `overflow-hidden` ([panel.tsx:8](ui/src/components/ui/panel.tsx#L8)) gave each a flex auto-min-size of 0, the column crushed it, and the trial-sim-log tails fell below the fold with no scrollbar.
- **Status:** **resolved.** 42dd636 added `[&>*]:shrink-0` to this container too (the fix was **not** limited to Manual Sim — its commit explicitly named HealthTab). Each individual log tail is bounded by an inner `<pre className="max-h-60 overflow-auto …">` at [HealthTab.tsx:238](ui/src/components/tabs/HealthTab.tsx#L238) and [:337](ui/src/components/tabs/HealthTab.tsx#L337); the outer container now scrolls to reach them. The reported repro is resolved by source. **No action needed.**

### Compare / Explore

> **Previously reported, now fixed (verified at HEAD).** Maps to **GRC-5.**

- **Component:** [ExplorerTab.tsx](ui/src/components/tabs/ExplorerTab.tsx)
- **Container:** [ExplorerTab.tsx:253](ui/src/components/tabs/ExplorerTab.tsx#L253) — `flex min-h-0 flex-1 flex-col gap-2.5 overflow-auto p-3 [&>*]:shrink-0` (comment at [:251](ui/src/components/tabs/ExplorerTab.tsx#L251)).
- **Original failure mode:** clipped-no-scroll — the envelope/spec-summary `Panel`s (each with its own inner `min-h-0 flex-1 overflow-auto` table) could be flex-crushed by the same GRC-5 mechanism.
- **Status:** **resolved** by the same 42dd636 `[&>*]:shrink-0`. **No action needed.**

---

### Studio shell (all views)

#### Right rail — Best-params table values cannot wrap/ellipsize

> Horizontal-overflow / silent clip. Lower severity. **Needs runtime verification (deferred — no live UI on server this round)** to confirm a real param name actually clips.

- **Component:** [RightRail.tsx](ui/src/components/shell/RightRail.tsx)
- **Offending container:** the Best-params `<table className="w-full text-xs">` at [RightRail.tsx:128](ui/src/components/shell/RightRail.tsx#L128); the key `<td className="… font-mono …">` at [:132](ui/src/components/shell/RightRail.tsx#L132) and value `<td>` at [:133](ui/src/components/shell/RightRail.tsx#L133) have **no** `truncate` / `break-all` / `max-w-0`. The wrapping div at [:127](ui/src/components/shell/RightRail.tsx#L127) is `overflow-hidden rounded border`, inside a fixed `w-[300px]` aside ([:57](ui/src/components/shell/RightRail.tsx#L57)).
- **Failure mode:** horizontal-overflow → silently clipped (no scroll).
- **Root cause:** a long unbroken param name or eng-string value cannot wrap, so the `overflow-hidden` wrapper at [:127](ui/src/components/shell/RightRail.tsx#L127) clips it rather than scrolling — a wide value is truncated with no horizontal scrollbar. Lower severity because the rail width is generous and values are usually short eng-strings.
- **Fix direction:** add `truncate max-w-0` (with `table-fixed` on the table) or `break-all` to the key/value `<td>`s so long names ellipsize or wrap inside the 300px rail.

#### Right rail body / Left rail / Bottom panel — REFERENCE (verified safe)

- **Right rail body:** [RightRail.tsx:80](ui/src/components/shell/RightRail.tsx#L80) — `min-h-0 flex-1 overflow-y-auto p-3`; the aside ([:57](ui/src/components/shell/RightRail.tsx#L57)) is `flex w-[300px] shrink-0 flex-col` with a `shrink-0` header. Tall spec-status / best-params lists scroll inside the rail. No action.
- **Left rail:** the rail body sits between a `shrink-0` header and footer with `min-h-0 flex-1 overflow-y-auto`; scrolls correctly. No action.
- **Bottom panel:** a fixed `h-48 shrink-0` panel whose log body is `min-h-0 flex-1 overflow-y-auto` (capped at 500 lines); scrolls correctly and does not steal the center scroll region. No action.

---

### Score Shaping — REFERENCE (verified safe)

- [ScoreShapingTab.tsx:156](ui/src/components/tabs/ScoreShapingTab.tsx#L156) is a **grid** (`grid min-h-0 flex-1 gap-3 overflow-auto p-3`), so no Panel flex-crush. The tall per-spec breakdown's inner table region is `min-h-0 flex-1 overflow-auto` ([:237](ui/src/components/tabs/ScoreShapingTab.tsx#L237)), giving it its own scroll. Correctly excluded from 42dd636. No action.

### Pipeline — REFERENCE (verified safe)

- [PipelineView.tsx:115](ui/src/components/tabs/PipelineView.tsx#L115) is `flex min-h-0 flex-1 flex-col overflow-auto p-4`; its direct children are a description div, an optional PVT banner, and the DAG column row — none are `Panel`s with `overflow-hidden`, and the column nodes are not height-constrained, so there is no flex-crush. The container scrolls vertically/horizontally for tall or wide DAGs. No action.

### Schematic — REFERENCE (verified safe)

- The inspector `<aside className="w-[290px] shrink-0 overflow-hidden …">` at [SchematicTab.tsx:403](ui/src/components/tabs/SchematicTab.tsx#L403) clips at its edge, **but** its child `DeviceInspector` root is `flex h-full min-h-0 flex-col overflow-y-auto` ([DeviceInspector.tsx:132](ui/src/components/schematic/DeviceInspector.tsx#L132)), so tall inspector content (selects + sliders + sensitivity result) scrolls **inside** the aside. The toolbar breadcrumb is `min-w-0 flex-1 overflow-x-auto whitespace-nowrap` ([SchematicTab.tsx:300](ui/src/components/tabs/SchematicTab.tsx#L300)) — the correct strip pattern. No action.

---

## Summary of fixes by leverage

| # | Fix | Resolves | Severity |
|---|---|---|---|
| 1 | **Mirror DutParamsStep** (`overflow-x-auto` + `min-w-[Npx]`) **or** switch tracks to `minmax(0,…fr)` + `min-w-0` on the grid, across PVTStep ([:113](ui/src/components/wizard/steps/PVTStep.tsx#L113), [:153](ui/src/components/wizard/steps/PVTStep.tsx#L153)), TestbenchesStep ([:131/:135](ui/src/components/wizard/steps/TestbenchesStep.tsx#L131)), OptimizerStep ([:162/:166](ui/src/components/wizard/steps/OptimizerStep.tsx#L162)), PDKRulesStep ([:52](ui/src/components/wizard/steps/PDKRulesStep.tsx#L52)) | GRC-1 + GRC-3 horizontal-overflow — **the reported PVT Supply-column clip** and its siblings | **High** |
| 2 | On the **number-input** rows, mitigate the spinner floor (`appearance-none`, or let the row wrap on narrow widths) — PVT Temp/Supply-V, Optimizer budget/seed/bounds | GRC-2 (the part the existing `min-w-0` does not fix) | High (couples with #1) |
| 3 | Add `w-full` to each wizard `<select>` (or a `w-full` opt-in to `selectCn`) | GRC-4 select mis-sizing in TargetSpecs/Optimizer/BasicInfo | Medium (cosmetic) |
| 4 | Add `truncate max-w-0`+`table-fixed` (or `break-all`) to the Best-params `<td>`s ([RightRail.tsx:132-133](ui/src/components/shell/RightRail.tsx#L132)) | Right-rail value clip | Low (needs runtime verification) |
| 5 | Give the Create-Wizard wrapper a definite height (`h-full`/`flex-1` at [SetupTab.tsx:304](ui/src/components/tabs/SetupTab.tsx#L304)) | Monaco height-chain collapse risk in wizard mode | Low (needs runtime verification) |
| — | **Already fixed by 42dd636** (`[&>*]:shrink-0` on Optimize/Health/Explorer scroll containers) — verified present and **complete** (covers all three tabs the GRC-5 crush affected, not just Manual Sim) | clipped-no-scroll family | **Done** |

**Net:** the single highest-leverage action is **#1** — bringing the four wizard step grids in line
with the `overflow-x-auto` + `min-w` (or `minmax(0,…fr)`) pattern that DutParamsStep already proves
works — because it directly resolves the reported PVT overflow and removes the same latent overflow
from three sibling steps in one consistent edit. Everything in the clipped-no-scroll family is
already resolved at HEAD by commit 42dd636, which is **complete** for the tabs it targeted.
