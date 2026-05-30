# SpiceXplorer UI — ★ Studio Integration Plan

> Target: take the SpiceXplorer web UI from its current (already partially refactored) 4‑tab
> layout to the **★ Studio · interactive** workspace described in the design handoff.
> This is a **delta plan**, not a rebuild — most of the shell, all six feature surfaces, the
> wizard, and most backend routes already exist. The work is *re‑composition* (a persistent
> activity bar + left rail + tabbed center + always‑on right rail + bottom panel + ⌘K palette),
> plus a handful of genuinely‑new pieces (uiStore, RightRail, BottomPanel, PipelineView,
> schematic inspector, sensitivity endpoint, run history).

---

## 1. Executive Summary

The Studio experience is a single IDE‑style workspace: a **title bar** (project + ⌘K + Run ▾ +
Fork + status pill), a **vertical activity bar** (6 icons + gear) that drives a **context‑dependent
left rail**, a **tab strip** over a **swappable center view**, an **always‑on right rail** (live
specs + run progress + best params + Pin/Export), a **collapsible bottom panel** (Terminal /
Optimizer log / Problems / Diff), a **status bar**, and two overlays (**⌘K command palette**,
**7‑step New‑Project wizard**). Run history becomes a first‑class, comparable unit.

The crucial reality: **much of this already exists.** The four center surfaces
(`SetupTab`, `OptimizeTab`, `ExplorerTab`, `ScoreShapingTab`), a richer‑than‑spec Xschem
`SchematicTab`, a fully‑wired 7‑step wizard, the four Zustand stores, the Plotly chart suite, the
UI primitives, the indigo/stone CSS palette, and most backend routes (`/api/project/generate`,
`/api/netlist/parse`, `/api/optimize/start` with algo+budget+replay overrides) are all built and
working. What is genuinely missing is the **shell composition layer** (`uiStore`, `StudioShell`,
`ActivityBar`, `TabStrip`, `StatusBar`, `RightRail`, `BottomPanel`, the 6 per‑activity rails),
**run history**, the **⌘K palette + keyboard map**, the **Run ▾ overrides popover**, the
**PipelineView DAG**, the **schematic inspector + sensitivity**, and a couple of small backend
extensions.

Headline constraint: **the IHP PDK (`ihp-sg13g2`) is absent on this Mac.** `ngspice` is present,
but live SPICE optimization and sanity simulation will fail on unresolved `.lib` model directives.
Therefore live Optimize must **gracefully degrade to Replay**, while everything else — YAML
validation, Score Shaping, Compare/Explore on cached CSV checkpoints, Schematic SVG, the wizard —
works fully. The plan treats PDK‑absent as a **first‑class supported mode**, not an error.

---

## 2. Current State vs Target

The shell *components* exist but the *shell architecture* does not. Honest accounting, drawn from
the reconciliation:

### Already done (reuse verbatim or near‑verbatim)
- **Center surfaces (feature‑complete):** `ui/src/components/tabs/SetupTab.tsx` (Monaco + Load
  demo/Validate/Apply + summary panels + wizard launch), `OptimizeTab.tsx` (algo/budget/replay
  selectors, Start/Stop, SSE `EventSource`, 2 convergence charts, live spec badges, best‑params
  table), `ExplorerTab.tsx` (A/B load + 4 comparison charts + envelope), `ScoreShapingTab.tsx`
  (spec selector, try‑value slider, dual penalty curve, breakdown table), `SchematicTab.tsx`
  (Xschem file picker + hierarchy nav + symbol resolution — richer than the spec's static SVG
  assumption), `HealthTab.tsx` (manual sanity check).
- **Stores:** `ui/src/stores/{projectStore,runStore,explorerStore,wizardStore}.ts` — minimal and
  well‑factored.
- **Wizard:** `ui/src/components/wizard/WizardShell.tsx` + 7 steps
  (`steps/{BasicInfoStep,PDKRulesStep,DutParamsStep,PVTStep,TestbenchesStep,TargetSpecsStep,OptimizerStep}.tsx`)
  + `wizardStore` + debounced live YAML preview + Generate&save→onSaved→apply. **Note: the spec's
  numbered filenames (`01-BasicInfo.tsx`…) are STALE — do not rebuild.**
- **UI primitives:** `ui/src/components/ui/` already has `button, badge, panel, slider, select,
  stat, spec-chip, table, separator, empty-state, tabs, segmented, PlotlyChart`. **The spec's
  `SegmentedControl.tsx` (new) is STALE — reuse `segmented.tsx`.**
- **Charts:** `ScoreConvergenceChart, MetricConvergenceChart, MetricScatterChart,
  MetricHistogramChart, PenaltyCurveChart` + `PlotlyChart` wrapper.
- **Design tokens:** indigo/stone palette already in `ui/src/app/globals.css` `@theme` vars.
- **Backend:** `POST /api/project/generate`, `POST /api/project/parse-to-form` (round‑trip, beyond
  spec), `POST /api/netlist/parse` (params only), `POST /api/optimize/start` (the `StartRequest`
  fields are `yaml_path`, `replay` (bool), `checkpoint_id`, `budget` — one endpoint serves both live
  and replay), full checkpoint/score/schematic/xschem/sanity routes.
- **HealthTab + `GET /health`** — functional sanity check + liveness probe (see §3).

### Partially done (needs surgery, not a build)
- **Convergence:** `OptimizeTab.tsx` bundles charts WITH the run‑config form and Stop/Pin/Export
  controls that must move out (Run ▾ + RightRail). SSE lifecycle is component‑local; must be hoisted.
- **Compare:** `ExplorerTab.tsx` charts done; A/B selection is local `useState` in a `Toolbar`,
  must move to a left rail + store.
- **Shaping:** `ScoreShapingTab.tsx` slider/curve/breakdown done; `selectedSpec` is local
  `useState` (~line 23), must become store‑driven for deep‑linking.
- **Setup:** just needs rename + drop tab context; wizard launch lifts to title bar.
- **`/api/optimize/start`:** `yaml_path`+`budget`+`replay`+`checkpoint_id` wired
  ([optimize.py:18-22](ui/backend/routes/optimize.py#L18-L22)); **`algorithm` + `loss_shape` + `seed`
  overrides all missing** — the Run ▾ popover needs all three added to `StartRequest` and threaded
  into `optimizer_runner.start_run`.
- **`/api/netlist/parse`:** returns `{params}` only; spec wants `{params, transistors}`.
- **Tokens:** palette present; **JetBrains Mono font + typed `tokens.ts` missing.**

### Not started (genuinely new)
`ui/src/stores/uiStore.ts`; `StudioShell.tsx`, `ActivityBar.tsx`, `TabStrip.tsx`,
`StatusBar.tsx`, `RightRail.tsx`, `BottomPanel.tsx`; the six per‑activity rails
(`RunHistoryRail, FileTreeRail, PipelineOutlineRail, SchematicListRail, SpecListRail,
CompareSetupRail`); **run history** (`runStore` holds one active run only — no `history`, no
localStorage, no sparklines); `openSpec()/openRun()` deep‑link actions; `PipelineView.tsx`;
schematic device‑group inspector + W/L sliders + sensitivity bars; `CommandPalette.tsx` + ⌘K /
⌘1..6 / ⌘Enter / ⌘. map; **Run ▾** overrides popover; **Fork**; backend
`GET /api/spec/{name}/sensitivity` (+ sensitivity service); `Sparkline.tsx`, `Kbd.tsx`;
`tokens.ts`; `views/` and `overlays/` dirs.

> **Spec‑accuracy note:** the handoff's `file_refactor_map` frames the shell refactor as "begun."
> At the shell level it has **not** — `page.tsx` is still a plain 4‑tab switch with no
> StudioShell/ActivityBar/RightRail/BottomPanel. What is "new vs the old 4‑tab assumption" is the
> wizard, the Xschem viewer, HealthTab, and the netlist/generate/parse‑to‑form backend.

---

## 3. The Health Feature (preserve and carry forward)

### How it works today
- **`GET /health`** (`ui/backend/main.py`) returns `{"status":"ok"}` — a zero‑dependency liveness
  probe. It does not check anything; the backend is "healthy" if it answers.
- **`HealthTab.tsx`** is a manual, on‑demand **sanity check**. The user clicks *Run health check*
  (disabled until `isApplied`), which calls `api.sanityCheck(yamlPath)` → `POST /api/sanity-check`.
  The backend runs **1 SPICE simulation per enabled testbench + 1 trial optimizer iteration** and
  returns a `SanityCheckResponse`: overall `ok` flag, `elapsed_ms_total`, YAML‑load + optimizer‑init
  timers, `ngspice` path, per‑testbench `{status, elapsed, log_size, log_tail}`, and a trial
  `{score, per‑metric pass/fail, sim log_tails}`. Reachable today as a secondary **LeftRail button**
  (not a top tab — so the spec's "demote from primary nav" is *smaller* than implied).

### Why it's valuable
It is the single fastest way to answer "is this project actually runnable end‑to‑end on this
machine?" — it exercises the YAML load, the simulator binary, the PDK model resolution, and one
optimizer step, surfacing log tails for debugging. For a conference demo on a PDK‑less laptop it is
also the **primary signal that live simulation is unavailable** (the testbench sims will fail on
missing `.lib` model libraries).

### Carry into Studio (explicitly preserved)
- **Keep `HealthTab.tsx` logic unchanged.** Re‑host it behind the **gear icon** at the bottom of
  `ActivityBar.tsx` as a **Settings/Diagnostics panel** (left rail content when `activeActivity ===
  "settings"`, center shows the existing HealthTab body). No internal rewrite.
- **Surface health/PDK state in two always‑visible places:**
  1. **Status bar** (`StatusBar.tsx`): a small pill — `● sim ready` (green) when the last sanity
     check passed, `● PDK missing — replay only` (amber) when sims fail or PDK undetected, `● sim
     untested` (grey) before first check.
  2. **Right rail / Run ▾:** when PDK is missing, the live‑Run affordance is disabled and steers to
     Replay (see §7).
- **Extend the sanity result to explicitly report the PDK condition.** Add a `pdk_ok: bool` +
  `pdk_detail: str` to `SanityCheckResponse` (and a lightweight `GET /api/env` probe — see §6/§7)
  so the UI distinguishes "simulator binary present but PDK models unresolved" from "ngspice
  missing." Settings panel shows: ngspice path, PDK_ROOT (if any), and a one‑line verdict.
- **Future Settings home (open question, §10):** log level, simulator path, parallel‑sim toggle —
  all natural additions to this panel, none required for v1.

---

## 4. Target Architecture

### Shell anatomy → existing components
```
StudioTitleBar      (rename of shell/Topbar.tsx; strip tabs, add ⌘K + Run ▾ + Fork + status pill)
 ┌───┬────────────┬──────────────────────────────────────────────┬──────────────┐
 │ A │  Left rail │  TabStrip (6 tabs)                            │  RightRail   │
 │ c │ (per       │  ┌──────────────────────────────────────────┐│ (always-on)  │
 │ t │  activity) │  │  Center view (one of 6)                  ││ progress     │
 │ B │            │  │  Pipeline|Convergence|Yaml|Schematic|    ││ spec chips   │
 │ a │            │  │  Shaping|Compare                         ││ best params  │
 │ r │            │  └──────────────────────────────────────────┘│ Pin/Export   │
 │   │            │  BottomPanel (Terminal/Log/Problems/Diff)    │              │
 └───┴────────────┴──────────────────────────────────────────────┴──────────────┘
StudioStatusBar     (new; indigo strip)
Overlays: CommandPalette (⌘K), WizardShell (existing)
```

| Studio piece | Existing file | Action |
|---|---|---|
| `StudioTitleBar` | `shell/Topbar.tsx` | rename; strip tab nav; add ⌘K trigger + Run ▾ popover + Fork + status pill |
| `ActivityBar` | — | **new** (~60 lines), lucide icons, gear→Settings/Health |
| Left rails ×6 | `shell/LeftRail.tsx` | split; salvage checkpoint list into `RunHistoryRail` |
| `TabStrip` | — | **new** (~70 lines), reuse `ui/segmented.tsx` |
| Center views | `tabs/*` | rename→`views/*`, read selection from `uiStore` |
| `RightRail` | — | **new** (~180); hoist spec‑status + best‑params out of `OptimizeTab` |
| `BottomPanel` | — | **new** (~140); Optimizer‑log tab off `runStore.events` |
| `StatusBar` | — | **new** (~40); reads `projectStore` + `runStore` |
| `StudioShell` | — | **new** (~80); composition root |
| `shell/Toolbar.tsx` | (used by 3 tabs) | **retire deferred** — delete only after consumers migrate |

### New store: `ui/src/stores/uiStore.ts` (the linchpin)
Holds **navigation + selection** only (mock state shape from the prototype):
```ts
// state
activeActivity: 'runs'|'setup'|'pipeline'|'schematic'|'specs'|'compare'|'settings'
activeTab:      'pipeline'|'convergence'|'yaml'|'schematic'|'shaping'|'compare'
selectedRunId: string | null
selectedSpec:  string | null      // replaces ScoreShapingTab local state
selectedDevice: string | null     // schematic inspector
compareRunA: string | null
compareRunB: string | null        // (may delegate to explorerStore — see below)
rightOpen: boolean
bottomOpen: boolean
bottomTab: 'terminal'|'log'|'problems'|'diff'
commandOpen: boolean
wizardOpen: boolean               // launch existing WizardShell
// actions
setActivity(a)            // + sensible default tab per activity
setTab(t)
openSpec(name)            // setActivity('specs') + setTab('shaping') + selectedSpec=name  → deep-link
openRun(id)              // selectedRunId=id; activity stays
setCompare(a,b)
toggleRight() / toggleBottom() / setBottomTab(t)
openCommand()/closeCommand()
openWizard()/closeWizard()
```
**North‑star mapping:** principle 4 ("score shaping lives next to its spec") = `openSpec()`;
principle 3 ("activity bar ≠ tabs") = `activeActivity` and `activeTab` are independent fields.

### Edits to existing stores (minimal)
- **`runStore`** — Phase 3: add `history: RunRecord[]` (metadata only — id, label, algo, budget,
  bestScore, sparkline points, specPass counts, checkpointId, ephemeral overrides) + localStorage
  persistence; add an `appendRunToHistory()` action invoked on run completion. **Do not persist
  full convergence arrays** (5 MB localStorage cap — load on demand via `/api/checkpoint/{id}`).
  Hoist the SSE `EventSource` handler out of `OptimizeTab` into a store‑level subscription so the
  bottom panel + right rail update regardless of active view.
- **`explorerStore`** — keep as the Compare data engine; `uiStore.compareRunA/B` are the
  *selection* (drive the left‑rail dropdowns), which call `explorerStore.setRunA/setRunB` to load.
  (Single source of truth for selection = uiStore; data stays in explorerStore.)
- **`projectStore`** — no shape change; `PipelineView` + `RightRail` read `summary.target_specs`,
  testbenches, DUT params from it.
- **`wizardStore`** — no change.

### Navigation model — **decision: App‑Router routes per screen** (chosen by maintainer)
**Use Next.js App‑Router routes for the six center views**, so every screen is deep‑linkable
(`/optimize`, `/scoring`, `/compare`, `/pipeline`, `/schematic`, `/setup`). The earlier
single‑page proposal was rejected in favor of real URLs for shareable demo links and browser
back/forward. The one real objection to routes — "the right rail, bottom panel, and live SSE
stream would remount on navigation" — is **fully solved by App‑Router's persistent layouts plus
Zustand‑hosted state**, so we adopt routes without losing persistence:

- **Persistent shell via a route‑group layout.** Put all six screens under a route group
  `ui/src/app/(studio)/` with a shared `(studio)/layout.tsx` that renders `StudioShell`
  (`ActivityBar` + the per‑activity left rail + `TabStrip` + `RightRail` + `BottomPanel` +
  `StatusBar` + overlays). In the App Router a layout **does not unmount when navigating between
  its child segments** — only the `page.tsx` (the center view) swaps. So the right rail, bottom
  panel, and status bar stay mounted across `/optimize → /scoring → /compare`.
- **Live‑run + SSE survive navigation because they live in Zustand, not the React tree.** Phase 2
  hoists the `EventSource` handler out of `OptimizeTab` into `runStore` (a module‑level store
  outside React's lifecycle). The stream and its event buffer are unaffected by which route is
  mounted — this is required regardless of nav model, and it's what makes routes safe here.
- **`uiStore` still owns *selection*, the URL owns *which view*.** `activeTab` is derived from the
  pathname (via `usePathname()`); `setTab(t)` becomes `router.push('/'+t)`. `uiStore` keeps the
  orthogonal **selection** state (`activeActivity`, `selectedRunId`, `selectedSpec`,
  `compareRunA/B`, overlay flags). Deep‑links combine both: `openSpec(name)` =
  `selectedSpec=name` **+** `router.push('/scoring')`; `openRun(id)` = `selectedRunId=id` (+ push
  to `/optimize` or `/compare` as appropriate).
- **Selection in the URL too (the deep‑link payoff).** Carry selection in query params so a link
  fully restores state: `/scoring?spec=ugf`, `/optimize?run=r12`, `/compare?a=sigmoid_de&b=linear_de`.
  Each view reads `useSearchParams()` on mount to hydrate `uiStore`; `openSpec/openRun/setCompare`
  write them via `router.push`. This makes ⌘K "Jump to spec/run" produce shareable URLs.
- **⌘1..6 keyboard tabs** become `router.push` to the six segments (still a one‑liner in
  `useGlobalShortcuts`).
- **Monaco/Plotly** stay `dynamic(..., {ssr:false})`; because the shell layout persists and only the
  page swaps, the YAML editor (on `/setup`) and charts mount per route as today — no regression.
- **Root redirect:** `ui/src/app/page.tsx` becomes a redirect to the default screen (e.g.
  `/setup`, or `/optimize` once a project is applied). The old monolithic `page.tsx` tab switch is
  removed.

**Net:** routes give deep‑linking and back/forward for free; the persistent `(studio)/layout.tsx`
plus the Zustand‑hosted SSE give us the same "nothing remounts, the live run keeps streaming"
behavior the single‑page approach promised. The cost is modest: a route group + six thin
`page.tsx` segments that each render the corresponding view component and hydrate `uiStore` from
search params.

---

## 5. Feature‑by‑Feature Build‑out

### (a) Tabs + Activity bar
- **Current:** `page.tsx` holds `activeTab` `useState` (TabId `setup|schematic|optimize|explorer|
  shaping|health`), keyboard 1‑6 with `isApplied` gating; tabs live in `Topbar` (4) + `LeftRail`
  buttons (schematic/health). No activity‑bar / left‑rail‑context concept.
- **Target:** activity bar = 6 vertical icons (runs, setup, pipeline, schematic, specs, compare) +
  gear, driving `activeActivity` → swaps left‑rail content. Tab strip = 6 tabs (pipeline,
  convergence, yaml, schematic, shaping, compare) driving `activeTab` → swaps center. Selected
  icon/tab highlighted; activity badge counts (runs=N from history, specs=N from summary).
- **Add:** `uiStore.ts`, `shell/StudioShell.tsx`, `shell/ActivityBar.tsx`, `shell/TabStrip.tsx`,
  `shell/StatusBar.tsx`.
- **Modify:** `app/page.tsx` (drop the activeTab switch + Topbar/LeftRail composition → render
  `<StudioShell/>`; **keep** the `useEffect` that calls `api.config()` + `listCheckpoints()`, or move
  it into StudioShell; pass `appConfig` via store/context since views currently take it as a prop).
  Rename `shell/Topbar.tsx`→`StudioTitleBar.tsx` (strip tabs). Relocate the `TabId/TAB_META` types
  that `LeftRail` imports from `Topbar` into `uiStore` or a shared const to avoid breakage.
- **Backend:** none.
- **Acceptance:** app looks like Studio; every existing flow still works end‑to‑end; activity icon
  changes left rail, tab changes center, independently; ⌘1..6 selects tabs.

### (b) Run history
- **Current:** `runStore` holds **one active run only** (`runId, isRunning, isReplay, budget,
  events, bestMetrics, bestParams, currentIter`). No history, no persistence, no sparklines.
- **Target:** every result‑producing action appends an immutable row (north‑star 1). `RunHistoryRail`
  = scrollable clickable cards with `Sparkline` of score‑convergence; click → `openRun(id)` sets
  `selectedRunId`, updating right rail + center views. Replay traces and completed live runs both
  land here. Salvage the existing checkpoint‑list logic (cap 12, deletable autosaves, refresh) from
  `LeftRail.tsx`.
- **Add:** `shell/rails/RunHistoryRail.tsx`, `ui/Sparkline.tsx`; `runStore.history` +
  `appendRunToHistory()` + localStorage (metadata only).
- **Modify:** `runStore.ts` (hoist SSE handler; append on completion), `LeftRail.tsx` (split).
- **Backend:** reuse `GET /api/checkpoint` (list) + `GET /api/checkpoint/{id}` (lazy convergence
  load). No new route.
- **Acceptance:** runs are first‑class; clicking a run re‑renders right rail + convergence/compare;
  history survives reload (metadata); convergence arrays loaded on demand.

### (c) Score‑shaping slider
- **Current:** `ScoreShapingTab.tsx` already has spec selector, try‑value drag slider, live dual
  `PenaltyCurveChart` (sigmoid vs linear), per‑spec breakdown table with penalty bars, dominant‑spec
  callout, backed by `POST /api/score`. **`selectedSpec` is local `useState` (~line 23).**
- **Target (prototype):** dragging updates a live preview; spec dropdown resets try‑value to target;
  shape pills (sigmoid/linear/none) update legend + aggregate highlight; breakdown rows clickable →
  switch spec + reset try‑value. **Selection must be store‑driven so spec nodes/chips deep‑link in.**
- **Add:** nothing new.
- **Modify:** rename `tabs/ScoreShapingTab.tsx`→`views/ShapingView.tsx`; swap local `selectedSpec`
  for `uiStore.selectedSpec`; ensure `openSpec(name)` lands here with that spec pre‑selected.
- **Backend:** none (`/api/score` unchanged, `score.py` keep).
- **Acceptance:** clicking a spec chip in the right rail or a spec node in PipelineView opens
  Shaping with that spec selected; slider/curve/aggregate update live.

### (d) Compare A/B
- **Current:** `ExplorerTab.tsx` loads Run A & B from checkpoints via `explorerStore.setRunA/B`,
  redraws 4 charts (convergence overlay, metric overlay, scatter, histogram) + envelope/best‑params/
  spec‑summary tables. A/B chosen via **local `pickA/pickB` `useState`** in a `Toolbar`.
- **Target:** when Compare activity active, left rail (`CompareSetupRail`) shows Run A & Run B
  dropdowns (only non‑live runs); changing either updates `uiStore.compareRunA/B`→`explorerStore`
  and redraws all four charts. Body unchanged. Scatter X/Y selectors stay in the view.
- **Add:** `shell/rails/CompareSetupRail.tsx`.
- **Modify:** rename `tabs/ExplorerTab.tsx`→`views/CompareView.tsx`; replace `pickA/pickB` local
  state with store selection; chart body verbatim.
- **Backend:** reuse `GET /api/checkpoint/{id}/envelope` + `.../scatter`. None new.
- **Acceptance:** selecting A/B in the left rail redraws all 4 charts; live run excluded from
  selectors; Diff bottom‑panel tab can compare A↔B YAML.

### (e) ⌘K Command palette
- **Current:** none. Only keyboard handling is plain 1‑6 tab switch in `page.tsx`.
- **Target (prototype):** modal (w≈620) on Cmd+K with search + 4 groups — **Switch view** (6 tabs),
  **Jump to run** (`runStore.history`), **Jump to spec** (`summary.target_specs`), **Actions**
  (Start run, Stop run, Fork, Open wizard, Toggle right/bottom). Typed search filters; clicking an
  item dispatches the `go` action and closes; ESC closes; items without an action are disabled.
- **Add:** `overlays/CommandPalette.tsx` (~150), `ui/Kbd.tsx`; a global keyboard hook
  (`useGlobalShortcuts`) for **⌘K** (palette), **⌘1..6** (tabs), **⌘Enter** (start run / steer to
  replay if PDK missing), **⌘.** (stop run).
- **Modify:** mount palette + hook in `StudioShell`; ⌘K trigger button in `StudioTitleBar`.
- **Backend:** none.
- **Acceptance:** ⌘K opens; typing filters; selecting "Switch view → Convergence" sets the tab;
  "Jump to run r12" sets `selectedRunId`; "Jump to spec ugf" calls `openSpec('ugf')`.

### (f) + New project wizard
- **Current:** **fully built.** `WizardShell.tsx` (7 steps, side‑stepper, left form + right
  debounced live YAML preview via `api.generateProject`, Generate&save→`onSaved`→`apply`) + 7 step
  components + `wizardStore`. Backend `POST /api/project/generate` + `POST /api/project/parse-to-form`
  done. **The spec's numbered step files and "new WizardShell" are STALE.**
- **Target:** reachable from the new shell. Launch from `StudioTitleBar` ("+ New project") and/or
  the **Setup** activity / Command‑palette action; on completion set `wizardOpen=false`,
  `activeActivity='setup'`, `activeTab='yaml'` (matches prototype). Confirm wizard's loss/seed
  fields match the Run‑▾ override expectations (§6).
- **Add:** nothing.
- **Modify:** wire `uiStore.openWizard()/closeWizard()` to existing `WizardShell` (replace
  `SetupTab`'s `showWizard` local state); keep `onSaved`→`setYamlPath`→`loadProject`→`apply`.
- **Backend:** `POST /api/netlist/parse` extend to also return `transistors` if step 3 needs them
  (otherwise no change — params path already works).
- **Acceptance:** wizard launches from title bar/palette; final step writes YAML, applies it, and
  lands on the YAML view with the project live.

---

## 6. Backend Delta

Base URL `/api`. ✅ = exists, ⚠️ = extend, 🆕 = new.

| Route | Status | In → Out | Reuses |
|---|---|---|---|
| `POST /project/generate` | ✅ done | wizard form → `{yaml_text, path, validation}` (writes file) | `services/yaml_generator.generate_yaml` |
| `POST /project/parse-to-form` | ✅ done (beyond spec) | YAML → wizard form (round‑trip) | `yaml_generator.project_dict_to_form` |
| `POST /netlist/parse` | ⚠️ extend | `.spice` file → currently `{params}`; add `{params, transistors}` | `services/netlist_parser.parse_params` (+ new `parse_transistors`) |
| `POST /optimize/start` | ⚠️ extend | already accepts `yaml_path, replay, checkpoint_id, budget`; **add `algorithm`, `loss_shape`, `seed`** | `services/optimizer_runner.start_run` (thread the three new args) |
| `GET /spec/{name}/sensitivity` | 🆕 (Phase 5) | spec name (+ optional `run_id`) → per‑device sensitivities `[{device, spec, sensitivity}]` | new `routes/sensitivity.py` + new sensitivity service (finite‑difference over best params, or static from checkpoint) |
| `GET /env` (recommended) | 🆕 (Phase 0) | — → `{ngspice_path, ngspice_ok, pdk_root, pdk_ok, pdk_detail}` | `shutil.which('ngspice')` + PDK probe (see §7) |
| `GET /config` | ✅ keep | — | — |
| `POST /score` | ✅ keep | metric values + spec → penalty breakdown/curve | `services/score_service.compute_score` |
| `GET /schematic` | ✅ keep | — → SVG | `app_config.schematic_svg` |
| `GET /checkpoint`, `/{id}`, `/{id}/envelope`, `/{id}/scatter`, `DELETE /{id}` | ✅ keep | run‑history + compare data | `services/checkpoint_reader` |
| `POST /sanity-check` | ⚠️ extend | add `pdk_ok`+`pdk_detail` to response | `NGSpice_Wrapper.run_sanity_check` |
| `GET /xschem/*` | ✅ keep | schematic viewer | xschem service |

**Detail — `POST /optimize/start` overrides:** add `algorithm: Optional[str]`,
`loss_shape: Optional[str]`, and `seed: Optional[int]` to `StartRequest` in
`ui/backend/routes/optimize.py` (today it is only `yaml_path`/`replay`/`checkpoint_id`/`budget`);
thread them into `optimizer_runner.start_run`. Per the design risk decision: **runtime overrides are
ephemeral** — they attach to the run record only and do **not** rewrite YAML on disk. `budget` +
`replay` + `checkpoint_id` are already wired (one endpoint serves both live and replay), so the
spec's "extend /optimize/start" is partly satisfied — but **algorithm is NOT yet honored**, contrary
to the spec's assumption.

**Detail — sensitivity:** v1 may compute a cheap static proxy (per‑device W/L gradient of each spec
from the best checkpoint's local neighborhood) rather than re‑simulating, since live sims need the
PDK. Document the proxy clearly. Ship cascode‑only (schematic is hard‑coded for cascode anyway).

---

## 7. Environment & PDK Degradation Strategy *(prominent)*

### Machine‑migration fixes (Phase 0 — do first)
Absolute server paths baked into example YAML (confirmed by grep on this checkout):
- `examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml:9` — `ws_root:
  /home/noorizad/code/TCAD/modules/SpiceXplorer/examples/OTA/cascode/ihp-sg13g2/` →
  `/Users/danialnoorizadeh/Code/SpiceXplorer/examples/OTA/cascode/ihp-sg13g2/`
- `examples/OTA/5t-ota/ihp-sg13g2/sizing/project_setup.yaml:8` — same prefix swap for `5t-ota`.
- `examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml:8` — same prefix swap for
  `folded_cascode`.
- `ui/app_config.json` — **no change needed.** `default_yaml` points at the cascode example (OK);
  `schematic_svg` → `examples/OTA/cascode/ihp-sg13g2/xschem/ota-improved.svg` **exists** (709 KB,
  verified) and is the only cascode SVG; the 4 preset CSV checkpoints resolve repo‑relative via
  `ui/backend/app_config.py`. These paths are repo‑relative (not machine‑absolute), so the migration
  did not break them — only the YAML `ws_root` values below are absolute and stale.
- `ngspice`: **present in PATH** on this Mac (`command -v ngspice` succeeds) — no fix.
- **PDK:** `ihp-sg13g2` **not installed.** Live sims fail on `.lib cornerMOSlv.lib mos_tt`.

### Graceful degradation (first‑class supported mode)
**Detection (backend):** add `GET /api/env` that returns:
```jsonc
{ "ngspice_path": "/opt/homebrew/bin/ngspice", "ngspice_ok": true,
  "pdk_root": null, "pdk_ok": false,
  "pdk_detail": "IHP ihp-sg13g2 models not found (PDK_ROOT unset; .lib cornerMOSlv.lib unresolved)" }
```
Probe logic: `ngspice_ok = shutil.which('ngspice') is not None`; `pdk_ok` = `PDK_ROOT`/`PDK` env
set **and** the expected model libs resolvable on disk (cheap filesystem check, no simulation). Also
fold `pdk_ok`/`pdk_detail` into `POST /api/sanity-check` so a real sanity run confirms the static
probe.

**UX behavior when `pdk_ok === false`:**
- **Works fully (no degradation):** YAML load/validate (Setup), Score Shaping (`/api/score`),
  Compare/Explore on the 4 preset CSV traces, Schematic SVG/Xschem viewer, run **history of replay
  runs**, the wizard, PipelineView, the command palette.
- **Live Optimize disabled, steered to Replay:** the **Run ▾** popover's *Start (live)* button is
  disabled with a tooltip; the primary CTA becomes **Replay a checkpoint**. `⌘Enter` triggers Replay
  (not live) when PDK missing. `_run_replay()` drip‑feeds cached CSV/JSON at 50 ms — no simulator
  needed — so the demo runs identically.
- **Sanity check** still runnable but expected to report `ok=false` with the PDK detail in log tails.

**Exact UX copy:**
- Status bar pill: `● PDK missing — replay only`
- Run ▾ disabled tooltip: `Live optimization needs the IHP sg13g2 PDK, which isn't installed on this
  machine. Use Replay to drive the demo from cached runs.`
- Settings/Diagnostics verdict line: `ngspice: ready · PDK (ihp-sg13g2): not found → live sims
  unavailable, replay enabled.`
- Replay CTA: `Replay cached run ▾` (lists the 4 preset checkpoints).

This makes the conference demo fully realistic without the PDK: every screen lights up, and the only
substituted path (live sim → replay) is visually indistinguishable in the convergence view.

---

## 8. Phased Roadmap

Ordered so the app + replay demo stay runnable at every step. Each phase ends green.

### Phase 0 — Environment fixes (~0.5 day)
- Fix the three `ws_root` paths (§7) and verify `ui/app_config.json` resolution.
- Add `GET /api/env` PDK/ngspice probe; add `pdk_ok`/`pdk_detail` to `SanityCheckResponse`.
- **DoD:** `./scripts/run_newcas_ui.sh` starts both processes; all 4 preset replays load; `/api/env`
  reports `pdk_ok=false` cleanly.

### Phase 1 — Stand up the shell, no functional change (~3 days)
- Add `stores/uiStore.ts`. Create the route group `ui/src/app/(studio)/layout.tsx` rendering
  `StudioShell` (`ActivityBar` + per‑activity left rail + `TabStrip` + `RightRail` + `BottomPanel` +
  `StatusBar` + overlays), and six thin segment pages `(studio)/{setup,scoring,optimize,compare,
  pipeline,schematic}/page.tsx`, each rendering the matching view component and hydrating `uiStore`
  from `useSearchParams()` (wrap in `<Suspense>`). Build `StudioShell`, `ActivityBar`, `TabStrip`,
  `StatusBar`. Rename `Topbar`→`StudioTitleBar` (strip tabs; relocate `TabId/TAB_META` into `uiStore`
  or a shared const). Move existing tab components `tabs/*`→`views/*` with **zero internal changes**.
  Turn `app/page.tsx` into a redirect to the default screen. `TabStrip` uses `usePathname()` for the
  active tab and `router.push('/'+tab)` to switch. Add JetBrains Mono to `app/layout.tsx`.
- **DoD:** app looks like Studio; each view has its own URL; back/forward works; every existing flow
  works end‑to‑end; navigating between views does **not** remount the shell.
- *Already satisfied:* center surfaces, primitives, palette CSS vars.

### Phase 2 — Right rail + bottom panel, always‑on (~2 days)
- Build `RightRail` (hoist spec status + best‑params out of `OptimizeTab`; Stop/Pin/Export here) and
  `BottomPanel` (Optimizer‑log tab functional day 1 off `runStore.events`). **Hoist the SSE
  `EventSource` handler into `runStore`** so the rail + log update on any view. Delete the moved
  buttons from `OptimizeTab`.
- **DoD:** spec status + run progress visible on every view; live updates during a replay run; log
  streams in the bottom panel.

### Phase 3 — Run history left rail + deep linking (~3 days)
- Extend `runStore` with `history` (metadata only) + localStorage. Build `RunHistoryRail` +
  `ui/Sparkline.tsx`. Add `openSpec()/openRun()`. Build the other rails (`FileTreeRail`,
  `SpecListRail`, `CompareSetupRail`, `SchematicListRail`, `PipelineOutlineRail`). Replace tab‑local
  selection: `ShapingView`→`uiStore.selectedSpec`; `CompareView` A/B→left rail + store.
- **DoD:** run history first‑class; tabs deep‑link to selected specs/runs; all 6 left‑rail contexts
  work.
- *Already satisfied:* the checkpoint‑list logic to salvage into `RunHistoryRail`.

### Phase 4 — Wizard launch + Command palette + Run ▾ (~4 days; backend mostly done)
- **Backend:** add `algorithm`+`loss_shape`+`seed` to `POST /optimize/start` (only `budget`/`replay`/
  `checkpoint_id` exist today); (optional) add `transistors` to `POST /netlist/parse`. *Generate +
  wizard backend already done.*
- **Frontend:** wire the existing `WizardShell` to `uiStore.openWizard` + title‑bar "+ New project".
  Build `CommandPalette` + `useGlobalShortcuts` (⌘K, ⌘1..6, ⌘Enter→start‑or‑replay, ⌘.→stop). Build
  the **Run ▾** popover in `StudioTitleBar`; move algorithm/budget/loss/seed/replay out of
  `OptimizeTab`; pipe overrides to `/optimize/start`; disable live + steer to Replay when PDK missing.
- **DoD:** wizard reachable + functional; Run ▾ overrides work (ephemeral, no YAML rewrite); ⌘K
  palette works.
- *Already satisfied:* whole wizard, `/generate`, `/parse-to-form`, `/netlist/parse` (params),
  budget + replay/checkpoint overrides. *Not yet:* algorithm/loss_shape/seed overrides.

### Phase 5 — Pipeline view + Schematic inspector (~4 days)
- Build `views/PipelineView.tsx` (read‑only DAG, divs + SVG overlay, from `projectStore.summary`);
  click spec node→`openSpec()`, click optimizer node→Run ▾. Build the schematic device‑group
  inspector on top of the existing Xschem viewer (`SchematicTab`→`views/SchematicView.tsx`): W/L
  sliders + sensitivity bars. **Backend:** `GET /api/spec/{name}/sensitivity` + sensitivity service
  (proxy v1, cascode‑only).
- **DoD:** visual DAG ships; click‑the‑circuit affordances work; sensitivity bars render. *Stretch:*
  Pipeline Fork branch; URL query sync; hover‑highlight upstream/downstream.
- *Note:* the spec's "wrap a static `/api/schematic` SVG" is stale — build the inspector on the
  richer Xschem viewer; keep the SVG as a fallback render.

**Cleanup (rolling):** retire `shell/Toolbar.tsx` only after `OptimizeTab/ExplorerTab/SetupTab`
toolbars have migrated (Phases 2–4) — not safe to delete in Phase 1.

---

## 9. Design Tokens & Conventions

- **Palette (already in `globals.css` `@theme`):** bg `#fafaf9`, panel `#ffffff`, panelAlt
  `#f5f5f4`, border `#e7e5e4`, borderDk `#d6d3d1`, ink `#1c1917`, text `#292524`, textMute
  `#57534e`, textDim `#a8a29e`; accent `#4f46e5` / hover `#4338ca` / soft `#eef2ff` / mid `#c7d2fe`;
  success `#16a34a`, warn `#d97706`, error `#dc2626`. **No re‑add needed** (spec's "add CSS vars" is
  stale).
- **Fonts:** UI `'Inter', system-ui, sans-serif`; mono `'JetBrains Mono', ui-monospace, monospace`.
  **Add the JetBrains Mono font link/`@font-face` in `app/layout.tsx`** (currently missing) without
  making Monaco eager.
- **Radii:** sm 4 / md 6 / lg 8 / xl 10 px. **Spacing:** 1=4, 2=8, 3=12, 4=16, 5=20, 6=24, 8=32 px.
- **Codify:** keep the CSS `@theme` vars as the runtime source; add a typed
  `ui/src/lib/tokens.ts` object (colors/radii/spacing/fonts) for TS‑side consumers (charts, SVG
  overlays, inline styles) so Plotly/`PipelineView`/`Sparkline` read the same constants. Mirror, do
  not fork, the Tailwind theme.
- **Conventions to preserve (from CLAUDE.md):** Monaco + Plotly stay `dynamic(..., {ssr:false})`;
  Plotly axis titles must be `{title:{text:"..."}}`; `checkpoint_reader.py` keeps `.iterrows()` (dot
  column names); CORS stays `allow_origin_regex` localhost; delete `ui/.next` before restarting dev
  after a build; `ListTargetSpec` → use `.targets`.

---

## 10. Risks & Open Questions

**From the design handoff:**
- **SSE event ordering across runs** — runStore holds one active run. Recommendation: **serial‑only
  queue** for Phase 3; parallel streams Phase 6+.
- **Algorithm override ↔ YAML round‑trip** — Run ▾ overrides are **ephemeral** (run record only), do
  not rewrite YAML. (Adopted.)
- **localStorage size** — persist run **metadata only**; load convergence on demand from
  `/api/checkpoint/{id}`.
- **Monaco bundle** — keep `dynamic ssr:false`; do not let the Studio shell eager‑load it.
- **PipelineView hover** — highlight upstream/downstream on spec hover is a stretch (extra code).
- **Run forking semantics** — does Fork copy YAML or just runtime overrides? Affects backend; v1
  Fork = clone overrides + same YAML (no new file).
- **Max parallel runs** — backend worker‑pool sizing is out of scope for v1.
- **Schematic for non‑OTA** — current SVG/Xschem is cascode‑specific; ship cascode‑only; generic
  rendering is a separate effort.
- **Settings surface** — what lives in HealthTab's future home? (log level, simulator path,
  parallel‑sim toggle — none required for v1.)

**Added by this plan:**
- **PDK‑absent demo realism** — the only substituted path is live‑sim→replay; visually identical in
  convergence. Risk: a reviewer asks to start a *live* run — mitigate with the disabled Run ▾ tooltip
  + Settings verdict making the limitation explicit and intentional.
- **App‑Router vs single‑page** — **App‑Router routes chosen** (§4) for deep‑linkable per‑view URLs
  + browser back/forward. Persistence is preserved by a route‑group `(studio)/layout.tsx` (layouts
  don't remount across sibling segments) plus hoisting the SSE handler into `runStore` (Phase 2).
  Tradeoff vs single‑page: slightly more wiring (route group + 6 thin `page.tsx` segments +
  `usePathname`/`useSearchParams` hydration) and the SSE‑in‑store hoist becomes a hard prerequisite
  rather than a nicety. Watch for `useSearchParams()` needing a `<Suspense>` boundary in the App
  Router.
- **Command‑palette scope** — v1 = static groups (Switch view / Jump to run / Jump to spec /
  Actions) with typed filtering, matching the prototype; fuzzy ranking + recents are stretch.
- **Sensitivity fidelity** — v1 is a static proxy (no live sim, since PDK absent); label it as such
  so it isn't read as a true finite‑difference gradient.

**Effort for a conference demo:** Phase 0–3 (env fix + shell + right rail + bottom panel + run
history + deep links) ≈ **8.5 days** delivers a fully demo‑able Studio on replay. Adding Phase 4
(palette + Run ▾ + wizard launch) ≈ **+4 days** for the keyboard‑first story. Phase 5 (Pipeline +
schematic inspector + sensitivity) ≈ **+4 days** for the "click‑the‑circuit" wow factor. **Minimum
viable demo = Phases 0–4 (~12.5 days); full ★ Studio = ~16.5 days.**
