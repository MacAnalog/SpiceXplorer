# Handoff: SpiceXplorer UI — Observatory direction

## Overview

This is a high-fidelity visual redesign of the SpiceXplorer NEWCAS 2026 demo UI. The "Observatory" direction is a light, dense, modern-engineering aesthetic — restrained indigo accent, geometric sans + monospace pairing, generous data density without crowding. Think Linear/Vercel applied to circuit optimization.

Four tabs are designed (matching your existing tab structure):

1. **Setup** — load/edit YAML *or* run a step-by-step wizard that generates YAML from a netlist
2. **Score Shaping** — sigmoid vs. linear penalty comparison (the paper's headline)
3. **Optimize** — live convergence with SSE, progress, per-spec status chips, best-design table
4. **Explorer** — two-run comparison: convergence overlay, scatter, envelope table, histograms, pass/fail summary

The redesign also introduces a persistent left rail (project list + checkpoint list + system status) and a top status bar — both new to the existing app.

## About the design files

The HTML/JSX files in this bundle are **design references, not production code**. They use:
- Vanilla React via Babel-in-browser (your codebase uses Next.js 14 + TypeScript)
- Hand-drawn SVG charts (your codebase uses `react-plotly.js`)
- Inline `<style>` strings (your codebase uses TailwindCSS + shadcn-style primitives)
- Hardcoded mock data (your codebase has typed fetch in `lib/api.ts` + Zustand stores)

**The task is to recreate this visual design inside the existing `ui/` Next.js project**, preserving its tech stack (Next.js App Router, TypeScript, Tailwind, Zustand, Monaco, react-plotly.js, Lucide icons) and editing the existing files listed under *Mapping to your codebase* below. Do not copy the HTML into the repo.

## Fidelity

**High-fidelity.** Exact hex values, font sizes, line heights, paddings, and component specs are documented below. The dev should match them pixel-perfectly using Tailwind utilities and the existing UI primitives.

---

## How to view the reference

Open `observatory_reference.html` in a browser. The top bar lets you jump between the four tabs. All four screens are clickable — change the spec dropdown on Score Shaping, change the metric dropdown on Optimize, etc.

Source files:
- `observatory.jsx` — all four tabs + topbar + left rail (one file, scroll to find tab components)
- `charts.jsx` — themed SVG chart primitives (PenaltyCurveChart, ScoreConvergenceChart, MetricConvergenceChart, ScatterChart, HistogramChart, SchematicPanel, Sparkline)
- `data.js` — mock data shapes (good cue for what backend payloads should look like)

---

## Mapping to your codebase

The existing `ui/src/` file tree maps to this design as follows. Edit these files in place — do not create parallel files.

| Existing file | What it becomes |
|---|---|
| `ui/src/app/page.tsx` | New shell: top bar (logo + tabs + status pills) + left rail + main content area. Move the tab nav from inline into a `Topbar` component. |
| `ui/src/app/layout.tsx` | Import Geist + JetBrains Mono from `next/font/google`. Replace existing body font. |
| `tailwind.config.ts` | Extend `theme.colors` with the design tokens listed below. Extend `fontFamily.sans` / `mono`. |
| `ui/src/components/ui/button.tsx` | Restyle to match: 5px radius, 12px font, three variants (default/primary/danger), `sm` size. |
| `ui/src/components/ui/badge.tsx` | Add `live` variant (indigo pulse-dot), keep existing `ok/warn/fail`. Pill shape (999px radius), 11px font. |
| `ui/src/components/ui/panel.tsx` | Card with 6px radius, 1px zinc-200 border, panel header with title + optional right-side mute text. |
| `ui/src/components/ui/select.tsx` | Native `<select>` styled to match toolbar (4px radius, 12px font, zinc border). |
| `ui/src/components/ui/slider.tsx` | New track style: 2px rail, 12px circular thumb, optional ticks + labels. See *Score Shaping > slider* spec. |
| `ui/src/components/ui/separator.tsx` | 1px zinc-200, vertical default, 18px height. |
| `ui/src/components/tabs/SetupTab.tsx` | **Two modes** (segmented toggle in toolbar): "Load / Edit YAML" (existing Monaco editor) and "Create Wizard" (new — see *Wizard* below). Right pane is a stacked summary: project meta, target specs table, schematic figure. |
| `ui/src/components/tabs/ScoreShapingTab.tsx` | Left column: PenaltyCurveChart + slider + callout. Right column: per-spec breakdown table with inline bar widgets + F(x) footer row. |
| `ui/src/components/tabs/OptimizeTab.tsx` | Toolbar with algo/budget/score-fn/replay selectors. Stat strip (3 cards). 2×2 grid: F(x) convergence, metric convergence, live spec chips + event log, best design table. |
| `ui/src/components/tabs/ExplorerTab.tsx` | Toolbar with Run A/B selectors. 2×3 grid: F(x) overlay, metric overlay, scatter, envelope table, histogram, spec summary table. |
| `ui/src/components/charts/PlotlyChart.tsx` | Update theme defaults — see *Plotly theming* below. |
| `ui/src/components/charts/PenaltyCurveChart.tsx` | Indigo solid (sigmoid) + cyan dashed (linear). Inline legend top-right with current P̂ values. Vertical markers: emerald dashed = target, orange solid = current. See `charts.jsx > PenaltyCurveChart`. |
| `ui/src/components/charts/ScoreConvergenceChart.tsx` | Light raw line + bold indigo best-so-far + optional cyan overlay for second run + emerald dashed zero line. |
| `ui/src/components/charts/MetricConvergenceChart.tsx` | Bold indigo trace + red dashed target line with "target Xunit" label at right. |
| `ui/src/components/charts/MetricScatterChart.tsx` | Indigo dots = run A feasible, cyan dots = run B feasible, zinc dots = infeasible. Emerald-shaded feasible region (5% opacity). Dashed emerald target lines. |
| `ui/src/components/charts/MetricHistogramChart.tsx` | Overlaid bars: indigo 70% opacity (A) over cyan 55% (B). Red dashed target line. |
| `ui/src/components/wizard/` | **New folder.** See *Wizard* section below. |
| `ui/src/stores/projectStore.ts` | Add `mode: "load" | "wizard"`, `wizardStep`, `wizardData` (the form payload built up step-by-step). |
| `ui/src/types/api.ts` | Add types for `POST /api/netlist/parse`, `POST /api/project/generate` (matches the backend `Not Yet Implemented` section of your plan). |

---

## Design tokens

Add to `tailwind.config.ts`:

```ts
extend: {
  colors: {
    // Surface
    bg:      "#fafafa",   // page background
    panel:   "#ffffff",   // cards, dropdowns
    hairline:"#f4f4f5",   // table row striping, hover bg
    border:  "#e4e4e7",   // 1px borders
    // Ink
    fg:      "#0a0a0a",   // primary text
    muted:   "#71717a",   // secondary text, axis labels
    faint:   "#a1a1aa",   // tertiary text, disabled
    // Brand
    primary:    "#4f46e5",  // indigo-600 — selection, best-so-far, primary buttons
    "primary-soft": "#eef2ff",
    secondary:  "#0891b2",  // cyan-600 — comparison/second series, linear-curve
    tertiary:   "#ea580c",  // orange-600 — current-value markers
    // Status
    ok:       "#059669",  // emerald-600 — pass, target met, zero line
    "ok-soft":"#d1fae5",
    danger:   "#dc2626",
    "warn-soft": "#fef3c7",
  },
  fontFamily: {
    sans: ["Geist", "Inter Tight", "system-ui", "sans-serif"],
    mono: ["JetBrains Mono", "IBM Plex Mono", "ui-monospace", "monospace"],
  },
}
```

In `ui/src/app/layout.tsx`:
```tsx
import { Geist, JetBrains_Mono } from "next/font/google";
const geist = Geist({ subsets: ["latin"], variable: "--font-sans" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });
```

### Typography

| Use | Family | Size | Weight | Notes |
|---|---|---|---|---|
| Brand wordmark | Geist | 14px | 600 | letter-spacing: -0.01em |
| Tab label | Geist | 13px | 400 (500 active) | active = primary border-bottom 1.5px |
| Panel title | Geist | 11px | 500 | + `.mute` span at 400/muted color for subtitle |
| Section eyebrow | Geist | 10px | 500 | UPPERCASE, letter-spacing: 0.06em, color: muted |
| Table th | Geist | 10px | 500 | UPPERCASE, letter-spacing: 0.06em, color: muted, bg hairline |
| Body text | Geist | 12px | 400 | line-height 1.4 |
| Stat value | JetBrains Mono | 18px | 400 | for "147 / 1000" etc. |
| Stat eyebrow | Geist | 10px | 500 | UPPERCASE |
| Code / data | JetBrains Mono | 11–11.5px | 400 | font-variant-numeric: tabular-nums |
| YAML / log | JetBrains Mono | 11.5px | 400 | line-height 1.55, dark bg `#1c1c1f` |

`font-variant-numeric: tabular-nums` should be on the root container so all numbers line up in tables.

### Spacing scale

The design uses Tailwind's default scale. Common rhythms:
- Toolbar padding: `px-4 py-2` (16/8)
- Panel header padding: `px-3 py-[7px]`
- Panel body padding: `px-3 py-2.5` (12/10)
- Table cell padding: `px-2.5 py-[5px]` (10/5)
- Grid gaps: `gap-2.5` (10px) between cards, `gap-3` (12px) for section padding

### Radii & borders

- Cards/panels: `rounded-md` (6px)
- Pills/badges: `rounded-full`
- Buttons: `rounded` (5px)
- Inputs/selects: `rounded` (4–5px)
- All borders: `1px solid #e4e4e7` (border-zinc-200)
- Hairline dividers (inside cards): `1px solid #f4f4f5`

### Shadows

The design is intentionally shadow-free except for:
- Segmented toggle active state: `0 1px 2px rgba(0,0,0,0.06)` — soft elevation cue

Avoid stronger shadows; the design relies on borders, not depth.

---

## Plotly theming

Override defaults in `ui/src/components/charts/PlotlyChart.tsx`:

```ts
const baseLayout: Partial<Layout> = {
  font: { family: "JetBrains Mono, ui-monospace", size: 10, color: "#71717a" },
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  margin: { l: 44, r: 14, t: 14, b: 28 },
  xaxis: {
    gridcolor: "rgba(0,0,0,0.06)",
    linecolor: "#a1a1aa",
    tickfont: { color: "#71717a", size: 10 },
    title: { font: { color: "#71717a", size: 10 } },
    zeroline: false,
  },
  yaxis: { /* same as xaxis */ },
  showlegend: false, // we render legends inline in HTML for tighter layout
};
```

Series colors (use Plotly trace `line.color` / `marker.color`):

| Role | Color | Use |
|---|---|---|
| Primary series | `#4f46e5` | best-so-far, sigmoid curve, run A |
| Secondary series | `#0891b2` | comparison overlay, linear curve, run B |
| Tertiary marker | `#ea580c` | current-value vertical marker |
| Reference target | `#059669` (dashed) | spec target line on penalty curves |
| Reference target | `#dc2626` (dashed) | spec target line on metric convergence |
| Raw points / muted | `#71717a` at 0.55 opacity | raw per-iteration score line |

Stroke widths: `1.6` for primary traces, `0.8` for raw points, `1` for axes/markers.

---

## Screens

### 0. Shell (used on every tab)

`ui/src/app/page.tsx`

**Layout**: vertical flex, top to bottom:
- `Topbar` — 44px tall, white bg, 1px bottom border
- `Content` — flex row, fills remaining height
  - `LeftRail` — 200px wide, white bg, 1px right border, scrollable
  - `Main` — flex column, scrollable

**Topbar children** (left to right):
1. Brand: 18px square indigo checkmark icon + "SpiceXplorer" 14px/600 + version pill "0.4.2 · NEWCAS demo" 11px mono/muted
2. Tabs row (flex-1): four tabs each `px-3` flex row with `label` 13px + small `kbd` (1px border, 10px mono, "1"/"2"/"3"/"4"). Active tab gets primary-color and 1.5px bottom border (overlapping the bar's bottom border by -1px).
3. Status pills (flex right, gap 10px):
   - `<Badge variant="ok">● project applied</Badge>` (emerald soft bg)
   - `<Badge variant="live">● replay · iter 147/1000</Badge>` (indigo soft bg, pulsing 6px dot)
   - Project name "cascode-ota-sg13g2" 12px mono muted

The "replay" pill is only shown when `runStore.isReplay && runStore.isRunning`.

**LeftRail children** (top to bottom, each section preceded by uppercase eyebrow):
- **PROJECT** — 3 rows: active project (with "v3" mono badge), 2 muted alternates
- **CHECKPOINTS** — list of `runStore`-known checkpoints, each row 2 lines: label + best-score badge on top, then `${iters} iters · ${timestamp}` 9px faint
- **RESOURCES** — `ngspice ▸ 41`, `workers 8/8`, `cache hit 62%` — each 11px mono muted with right-aligned mono badge

Rail items use `px-1.5 py-1` rounded-sm, hover bg `hairline`.

---

### 1. Setup tab (the wizard is the hero)

`ui/src/components/tabs/SetupTab.tsx` — see `observatory.jsx > ObsSetupTab`.

**Layout**:
- Toolbar row (above the two-column body):
  - Segmented mode toggle: `[ Load / Edit YAML ] [ Create Wizard ]` (active = white bg with soft shadow, inactive = muted text on hairline bg)
  - Separator
  - In "Load" mode: demo selector + "Upload .yaml" btn
  - Right side: "Validate" + "Apply →" (primary)
- **Wizard stepper bar** (only in wizard mode): horizontal pills `1. Project › 2. PDK rules › 3. DUT params › 4. PVT corners › 5. Testbenches › 6. Target specs › 7. Optimizer`. Done steps get emerald checkmark in the numbered circle; active step gets indigo bg + filled indigo circle.
- **Two-column body** (1fr 1fr, gap 12px, padding 12px, scrollable):
  - **Left**: either Monaco editor OR the wizard step form
  - **Right**: stacked panels: Project summary (definition list 100px / 1fr grid), Target specs (compact table), DUT schematic figure

#### Wizard step form (left column when mode="wizard")

A `<Panel>` with:
- Header: `step 3 · DUT parameters` + sub `· 11 params · 1 frozen`; right side has "Upload netlist" + "+ Add param" buttons
- Body:
  - Mini status row: "cascode_ota.spice · parsed 11 .param declarations · auto-filled" with `ok` pill
  - The DUT params table (see column spec below)
  - Bottom row: "← PDK rules" (left-aligned) and "PVT corners →" (right, primary)

**DUT params table columns**: param (mono) | min (mono) | max (mono) | init (mono) | log (●/○ indigo/faint) | int (●/○) | freeze (● orange "frozen" / ○)
Frozen rows render at 0.55 opacity.

**Each wizard step** (`ui/src/components/wizard/steps/*.tsx`) follows the same Panel shell:
- BasicInfoStep: project name, description, simulator dropdown (ngspice/Xyce/...), workspace path text input
- PDKRulesStep: tech name + add-row table of key/value constraints
- DutParamsStep: as above
- PVTStep: add-row table of `{temp, corner, supply}` — same visual rhythm
- TestbenchesStep: list of testbench cards with name, netlist upload, params subtable, enable toggle
- TargetSpecsStep: accordion list — each row collapses to `name | goal | target | tolerance | weight` chips, expands to full form
- OptimizerStep: algo dropdown + budget input + seed input

A shared `WizardShell.tsx` renders the stepper bar, holds the current step component, and shows the live YAML preview in the right column (replacing the Project Summary panel — wizard mode toggles the right column too).

The "Save YAML" button (in the bottom-right of the stepper bar when on the last step) POSTs the wizard data to `/api/project/generate`, writes the resulting YAML to the workspace root, then flips `mode` to "load" with that file loaded.

#### Load mode (left column when mode="load")

Just the existing Monaco editor — but restyle the host frame:
- Wrap in `<Panel>` with header "project_setup.yaml" + mono mute path
- Header right side: "UTF-8 · 142 lines" mono mute + `ok` pill ("valid")
- Use the Monaco theme `vs-dark` with these token color overrides matching the design's YAML colors:
  - keys: `#c4b5fd` (violet-300)
  - strings: `#86efac` (emerald-300)
  - numbers: `#fda4af` (rose-300)
  - comments: `#71717a` italic

#### Right column

Three panels stacked, all the same `<Panel>` primitive:

1. **Project summary** — definition list `<dl>` with `grid-template-columns: 100px 1fr`, dt = muted Geist, dd = mono. Pulls from `projectStore.summary`.
2. **Target specs** — compact table (mono cells), columns: name | goal | target | tol | weight
3. **Device under test · schematic** — embed `<img src="/api/schematic" />` or inline SVG from `GET /api/schematic`. Wrap in 6px padding inside the panel. The mock uses a stylized cascode OTA SVG (`charts.jsx > SchematicPanel`) — real implementation pulls from the backend.

---

### 2. Score Shaping tab

`ui/src/components/tabs/ScoreShapingTab.tsx` — see `observatory.jsx > ObsScoreTab`.

**Layout**: toolbar + 2-column body (1.5fr 1fr, gap 12px, padding 12px).

**Toolbar**:
- "spec" label + spec dropdown (one entry per `target_specs[]`, label like `ugf · > 200MHz`)
- separator
- "range" label + read-only mono caption `target ± 3 × 50MHz`
- right side: mono muted caption `POST /api/score · 150ms debounce` (purely decorative info — communicates the technical reality)

**Left column** (1.5fr):
1. **Penalty curve panel** (PenaltyCurveChart) — chart at full panel width × 280px height, then a slider underneath showing the swept range with `target` tick mark, current-value thumb, and below-row labels `loEdge | now Xunit | hiEdge`.
2. **Callout** — full-width, primary-soft bg, 2px left border indigo, body text noting which spec dominates F(x) under each shaping. Updates as `bestMetrics` change.

**Right column** (1fr):
- **Per-spec breakdown** table. Columns: spec (mono) | current (mono, green if met else red) | sigmoid P̂ (mono, with inline bar widget) | linear P̂ (mono, inline bar) | w (mono)
- Last row is a sticky `tr.footer` with hairline bg, top border, F(x) aggregate: `colspan=2 "F(x) aggregate" | totalSig (indigo) | totalLin (cyan) | Σ weights`
- The selected spec row gets `bg-primary-soft`.

**Inline bar widget** (used for P̂ cells): 36px wide × 4px tall hairline bg track with primary/secondary fill. Inline-block, vertical-align middle, margin-right 6px. The P̂ number follows in mono.

**Slider widget** (`ui/src/components/ui/slider.tsx`):
- Track: 28px tall container, 2px rail at y=13 in `border` color
- Filled portion (from rail start to thumb): 2px in primary color
- Thumb: 12×12 circle, 2px primary border, white fill, centered with `transform: translateX(-50%)`
- Target marker: 1px vertical line, muted color, with `target` label 9px mono muted below
- Debounce: 150ms before firing `POST /api/score` (preserve existing behavior)

---

### 3. Optimize tab

`ui/src/components/tabs/OptimizeTab.tsx` — see `observatory.jsx > ObsOptimizeTab`.

**Layout**: toolbar + scrollable body with stat strip (3-col) + two 2-col grids.

**Toolbar**:
- "algorithm" label + select (`LhsDE` / `LHSSearch` / `LogBFGSCMAPlus`)
- "budget" label + numeric input (width 72px)
- "score" label + segmented `[sigmoid][linear]`
- separator
- "demo replay" label + select (lists checkpoints from `demoConfig`)
- right side: red "Stop" btn with pulsing dot + outline "+ Save checkpoint"

**Stat strip** (`grid-cols-3 gap-2.5`): three `<Stat>` cards, each:
- `eyebrow` 10px UPPERCASE muted
- `value` 18px mono
- optional inline progress bar OR `delta` text 11px (color: ok/dn for direction)

Stats: iteration/budget (with embedded progress bar), best F(x) (with delta), specs met (with name list).

**Top row (`grid-cols-2`)**:
1. F(x) convergence panel: raw line at 0.55 opacity muted + best-so-far solid indigo. Panel header has mono "● best ● raw" legend on the right.
2. Metric convergence panel: select dropdown in panel header switches between ugf/gain/pm/current. Chart shows best-so-far + red dashed target line.

**Bottom row (`grid-cols-2`)**:
1. Live spec status panel: 6 `<SpecChip>` (one per target spec). Below: 4-line console log in mono 11px showing recent SSE events (timestamp + iter + score + status).
2. Best design panel: table with columns param (mono) | value (mono) | min..max (mono muted) | sparkline. The sparkline column is 60×14px showing the param's trajectory across iterations.

**SpecChip** (`ui/src/components/ui/spec-chip.tsx` — new): rounded-sm border ok/fail variant, dot + mono name + mono value + smaller mono muted `>${target}${unit}` tail.

**Event log** — render `runStore.events.slice(-6)`, monospace, color status keywords:
- `eval ✓` — fg
- `eval ✗` — danger
- `★ best` — primary

---

### 4. Explorer tab

`ui/src/components/tabs/ExplorerTab.tsx` — see `observatory.jsx > ObsExplorerTab`.

**Layout**: toolbar + 3 rows of 2-column grids.

**Toolbar**:
- "run A" select + "run B" select (from checkpoint list)
- "Load both" btn
- separator + mono muted summary "2 runs · 400 evals · 21.4s sim time"
- right side: "Export CSV" + "Compare report" (primary)

**Row 1**: F(x) overlay (A primary + B secondary, raw hidden) | UGF best-so-far overlay (A indigo + B cyan + target red dashed)

**Row 2**: Metric scatter (UGF × current, feasibility coloring, target lines, shaded feasible region) | Performance envelope table

**Row 3**: UGF histogram (overlaid A primary + B secondary, target red dashed) | Spec summary pass/fail table

**Performance envelope table** columns: spec | target (mono muted) | run A best (colored indigo if winner) | run B best (colored cyan if winner) | winner pill ("A" indigo-soft / "B" cyan-soft).

**Spec summary table** columns: spec | goal | A pill (ok/fail) | B pill (ok/fail)

---

## Interactions

### Tab navigation
- Click any tab to switch
- Keyboard shortcuts `1`/`2`/`3`/`4` for tabs (the `kbd` chip on each tab is a hint, not decorative — wire it up to `useEffect` keydown listener on `document`)
- Score Shaping and Optimize tabs are disabled until `projectStore.isApplied === true` (apply Tailwind `opacity-40 cursor-not-allowed`, don't fire click handler)

### Mode toggle (Setup)
- Switching between "Load" and "Wizard" preserves form state for each (don't reset)
- "Apply →" works in both modes:
  - Load: POST current Monaco buffer to `/api/project/load` (fix existing bug where Apply re-reads from disk)
  - Wizard: prompt to save YAML first, then load

### Score shaping slider
- Drag updates value in real time visually
- 150ms debounce before `POST /api/score`
- Re-paints PenaltyCurveChart with new P̂ values inline
- Updates the per-spec table row for the active spec
- Updates the callout text by recomputing `argmax` across `sigP * w`

### Optimize
- Start with selected algorithm + budget + score fn + replay
- Open `EventSource` to `GET /api/optimize/stream/{id}`
- For each SSE event: append to `runStore.events`, update best-so-far line if score improved, update progress bar, update best-design table, update spec chips
- Stop closes the EventSource and POSTs to `/api/optimize/stop/{id}`

### Explorer
- "Load both" fetches `GET /api/checkpoint/{A}` and `GET /api/checkpoint/{B}` in parallel
- All chart panels are reactive to the selectors — no separate "refresh" button per panel
- "Compare report" generates a PDF/HTML diff (placeholder — punt to existing feature backlog)

---

## State management

Extend the existing Zustand stores rather than introducing new ones.

```ts
// projectStore.ts — add
mode: "load" | "wizard"
wizardStep: WizardStepId        // "basic" | "pdk" | "dut" | "pvt" | "tb" | "specs" | "opt"
wizardData: WizardFormData      // grows as steps complete
setMode(m): void
setWizardStep(s): void
patchWizardData(partial): void
saveWizardYaml(): Promise<void> // POSTs to /api/project/generate, then loadProject(path)
```

```ts
// runStore.ts — existing keys ok; ensure events keeps last 200 only (window for live log)
// Also: bestParams update only when a new best F(x) lands.
```

```ts
// explorerStore.ts — existing keys ok; "runA" / "runB" hold whole CheckpointData payloads.
```

---

## New backend routes (already in your plan, surfaced here for clarity)

- `POST /api/netlist/parse` → `{ params: Array<{name: string, default_val: number}> }`
- `POST /api/project/generate` → `{ yaml: string, yaml_path: string }` (writes file to workspace root)

These power the wizard's "Upload netlist" auto-fill and "Save YAML" finalize. Both are documented as `❌ Not Yet Implemented` in your existing plan — this redesign requires them.

---

## Iconography

Use `lucide-react` (already a dependency).

- Brand mark: hand-drawn rounded square with indigo fill + white checkmark (see `observatory.jsx > ObsTopbar`). Render as inline SVG, do not import an icon library glyph.
- Tab keyboard hints (`1`/`2`/`3`/`4`): plain text in a small border-radius chip — not an icon.
- No other icons in the design. Specifically: no icons inside buttons, no icons inside table headers, no icons next to status pills (use a colored dot instead — `<span className="dot" />`).

This is intentional — the design relies on type and color hierarchy. Resist adding icons.

---

## Out of scope (do not implement)

- Light/dark theme toggle — the design is light-only
- Mobile responsive — desktop-only, min-width 1280
- Animation/transitions beyond opacity + the pulse animation on live dots
- Drag-and-drop — table rows are not reorderable
- The dark "Bench" or paper "Manuscript" alternative directions explored alongside this one

---

## Files in this bundle

- `README.md` — this document
- `observatory_reference.html` — open in a browser to see the full reference at fidelity. Tab switcher at top.
- `observatory.jsx` — React source for all 4 tabs + topbar + left rail. Read this for component structure and exact CSS values. The inline `<style>` block at the top (`obsCSS`) is the authoritative source for every spacing/color/font value.
- `charts.jsx` — SVG chart primitives. Read for chart layout (axis positions, marker placement, legend treatment) — replicate this layout in your `react-plotly.js` chart components.
- `data.js` — mock data shapes. Useful as a spec for what `lib/api.ts` responses should look like.

---

## Suggested implementation order

1. Tokens first — add Tailwind colors + fonts. Test by restyling the existing `Button` and `Badge`.
2. Shell — rebuild `app/page.tsx` with the new Topbar + LeftRail layout. Existing tabs keep working but now sit inside the new shell.
3. Chart theming — apply Plotly defaults in `PlotlyChart.tsx`. All existing chart components inherit.
4. Score Shaping — smallest tab; verifies the slider + table + callout vocabulary.
5. Optimize — biggest behavioral surface (SSE, live updates). Stat strip + spec chips are new primitives.
6. Explorer — pure composition over existing chart components; should be quick once charts are themed.
7. Setup (Wizard) — largest scope. Build `WizardShell` + the 7 step components. Wire up `POST /api/netlist/parse` and `POST /api/project/generate`.

Stop after each step and verify visual fidelity against the reference HTML before moving on.
