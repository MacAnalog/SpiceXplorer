# Handoff: SpiceXplorer Studio refactor

## What this is

A complete design handoff for refactoring the existing **SpiceXplorer UI** (`MacAnalog/SpiceXplorer` on branch `dev/ui`) from a 4-tab interface to a unified Studio workspace.

The work is **frontend-heavy**, **incremental** (5 phases, each leaving `main` shippable), and **preserves the existing backend** with only 3 small additions.

## About the design files

The files in this bundle — `SpiceXplorer UX directions.html`, `Studio refactor handoff.html`, and the entire `wireframes/` folder — are **design references** built as React-in-HTML prototypes (Babel-transpiled JSX, served from a single HTML). They demonstrate intended layout, behavior, and interaction flow. **They are not production code to copy directly.**

The real implementation lives in the user's existing **Next.js 15 + TypeScript + FastAPI + Zustand** stack at `ui/` in the repo. Recreate the prototype's IA, components, and interactions there — using the project's established patterns (TypeScript types in `src/types/api.ts`, the `Button`/`Badge`/`Panel`/`Select`/`Table` primitives in `src/components/ui/`, Plotly charts in `src/components/charts/`, Zustand stores in `src/stores/`, FastAPI routes in `ui/backend/routes/`).

**Critical:** do not replace existing chart components, primitives, or the FastAPI backend. Wrap and compose what's already there.

## Fidelity

**High-fidelity** for layout, typography, spacing, color, and interaction behavior. Pixel-target the prototype using the codebase's existing libraries — Plotly for charts (not the inline SVGs from the prototype), Monaco for the YAML editor (already in `SetupTab.tsx`), shadcn/Radix-style primitives in `src/components/ui/`.

## The plan (read this first)

Open **`Studio refactor handoff.html`** in a browser — it is the authoritative spec. It contains:

1. **Executive summary** — what's changing and why
2. **North-star principles** — 6 design decisions that drive everything
3. **Studio shell anatomy** — the 6 regions and how they compose
4. **File-by-file refactor map** — every existing file with a status (`keep` / `modify` / `new` / `retire`)
5. **State model** — new `uiStore.ts` shape + small edits to existing stores
6. **Backend API delta** — 3 new endpoints, everything else stable
7. **New components** — 8 pieces to build, with rough line counts
8. **5-phase migration plan** — start at Phase 1 and don't skip ahead
9. **Patterns & conventions** — design tokens, naming, where state lives
10. **Risks & open questions** — flag these before Phase 1

**Start at Phase 1** (stand up the shell with no functional change). The plan is structured so the app remains shippable between every phase.

## Repo context

- **Repo**: `https://github.com/MacAnalog/SpiceXplorer`
- **Branch**: `dev/ui` (refactor target — create `dev/studio` from here)
- **UI directory**: `ui/`
- **Stack**: Next.js 15 App Router · TypeScript · Zustand · Plotly · Monaco · FastAPI · SSE
- **Read first**: `ui/README.md` in the repo — it documents the existing UI, API, common bugs, and what's "Implemented ✅" vs "Not Yet Implemented ❌". The Studio refactor closes most of the "Not Yet Implemented" items.

## The interactive prototype

Open **`SpiceXplorer UX directions.html`** in a browser, scroll to the **"Higher fidelity"** section, and click the **"★ Studio · interactive"** artboard. This is the source of truth for behavior. Try:

- Clicking the activity-bar icons (left edge) — the left rail content swaps
- Clicking the center tabs — the main view swaps
- Score shaping tab → drag the try-value slider, click different specs in the breakdown table
- Compare tab → change Run A/B dropdowns
- Top-bar **Run ▾** → algorithm/budget/loss/replay popover
- Top-bar **⌘K** search → command palette
- Setup activity → **+ New project** → 7-step wizard with Back/Next

The static "Studio (remix v1)" artboard next to it is a single-frame visual reference.

## Screens / views

The Studio has **one workspace** with 6 swappable views in the center, 6 activity contexts in the left rail, and an always-on right rail. Detailed specs for each view, the shell anatomy, and the component breakdown are in `Studio refactor handoff.html` §3 and §7.

| View | Maps to existing file | Status |
|---|---|---|
| Pipeline | *(new)* — read-only DAG | new component, ~260 lines |
| Convergence | `OptimizeTab.tsx` | refactor — strip run-config form |
| YAML setup | `SetupTab.tsx` | drop-in — rename to a view |
| Schematic | `/api/schematic` SVG + new inspector | extend |
| Score shaping | `ScoreShapingTab.tsx` | drop-in — read selected spec from store |
| Compare | `ExplorerTab.tsx` | rename + move A/B selectors to left rail |

## State management

A new **`uiStore.ts`** (Zustand) owns navigation and selection. The four existing stores (`projectStore`, `runStore`, `explorerStore`, `scoreStore`) keep their domain shapes with small additions.

Key new actions:
- `setActivity(a)` — switches activity AND jumps to its default tab
- `openSpec(name)` — deep-link: opens spec in score-shaping
- `openRun(id)` — deep-link: focuses a run in the convergence view

Full TypeScript shape and store edits in `Studio refactor handoff.html` §5.

## Backend API

**Only 3 new endpoints**, all small:

- `POST /api/project/generate` — takes wizard JSON state, emits validated YAML
- `POST /api/netlist/parse` — wizard's netlist upload step, returns `{ params, transistors }`
- `GET /api/spec/{name}/sensitivity` — per-device sensitivity (Phase 5)

**1 endpoint extended**: `POST /api/optimize/start` now accepts runtime overrides for algorithm / budget / loss_shape / seed.

Everything else (CORS, SSE, checkpoint loading, score computation) stays exactly as-is. Detail in `Studio refactor handoff.html` §6.

## Design tokens

Indigo + stone palette, Inter + JetBrains Mono. The handoff doc §9.1 has the canonical token object — codify it as both CSS variables in `globals.css` and a typed `tokens.ts` constant.

Headline values:
- Accent: `#4f46e5` (indigo-600), hover `#4338ca`, soft `#eef2ff`, mid `#c7d2fe`
- Neutrals (stone scale): bg `#fafaf9`, panel `#ffffff`, border `#e7e5e4`, text `#292524`, mute `#57534e`
- Semantic: success `#16a34a`, warn `#d97706`, error `#dc2626`
- Fonts: `"Inter"` for UI, `"JetBrains Mono"` for code/values
- Radii: 4 / 6 / 8 / 10 px
- Spacing scale: 4 / 8 / 12 / 16 / 20 / 24 / 32 px

## What to build, in order

Phase 1 first. Do not skip ahead.

1. **Phase 1 — Shell scaffold** (~3 days). Add `uiStore`, build `StudioShell` + `ActivityBar` + `TabStrip` + `StatusBar`, mount existing tabs as views with zero internal changes.
2. **Phase 2 — Right rail + bottom panel** (~2 days). Always-on live specs, run progress, log stream. Hoist SSE state to the store.
3. **Phase 3 — Run history + deep links** (~3 days). `runStore.history`, `RunHistoryRail`, `openSpec()` / `openRun()` deep-linking.
4. **Phase 4 — Wizard + ⌘K + Run ▾** (~5 days). Closes the README's highest-priority gap.
5. **Phase 5 — Pipeline view + Schematic inspector** (~4 days). New DAG view + click-the-circuit affordances.

Full per-phase scope and "definition of done" in `Studio refactor handoff.html` §8.

## Out of scope

Explicitly **not** part of this work — see §11:

- Backend rewrite
- Replacing Plotly or the existing UI primitives
- Multi-tenant / auth
- Mobile / tablet
- Drag-and-drop pipeline editor (Pipeline v1 is read-only)
- Generic schematic rendering (ship with cascode OTA only)

## Open questions to resolve before Phase 1

These need a product call. The handoff doc §10 lists them:

1. Should hovering a spec node in Pipeline highlight upstream/downstream nodes?
2. Run forking semantics — does Fork copy YAML or just runtime overrides?
3. Max parallel runs (drives backend worker pool design)?
4. Schematic rendering for projects that aren't the cascode OTA?
5. What lives in Settings (HealthTab's future home)?

## Files in this bundle

```
design_handoff_spicexplorer_studio/
├── README.md                              ← this file
├── Studio refactor handoff.html           ← AUTHORITATIVE SPEC — read in browser
├── SpiceXplorer UX directions.html        ← interactive prototype (open in browser)
└── wireframes/                            ← prototype source (JSX, transpiled in-browser)
    ├── design-canvas.jsx                  ← canvas/artboard host (starter component)
    ├── common.jsx                         ← sketchy lo-fi primitives
    ├── runlog.jsx, notebook.jsx, schematic.jsx, nodes.jsx, ide.jsx   ← lo-fi sketches (5 directions)
    ├── hifi-common.jsx                    ← hi-fi primitives (HFText, HFButton, HFChart, HFIcon, HFSchematic, …)
    ├── hifi-runlog.jsx                    ← hi-fi A (Run Log workspace)
    ├── hifi-nodes.jsx                     ← hi-fi D (Pipeline / node graph)
    ├── hifi-ide.jsx                       ← hi-fi E (IDE three-pane)
    ├── hifi-remix.jsx                     ← static remix v1
    ├── hifi-studio-ctx.jsx                ← Studio context provider + mock data shapes
    ├── hifi-studio-views.jsx              ← 6 interactive center views
    ├── hifi-studio-rails.jsx              ← 6 left-rail variants + always-on right rail + bottom panel
    ├── hifi-studio.jsx                    ← Studio shell, top bar, activity bar, status bar, wizard, ⌘K
    └── app.jsx                            ← canvas wiring
```

**Reading the prototype source:** `hifi-studio-ctx.jsx` declares the mock state shape (a useful starting point for `uiStore.ts`). `hifi-studio-views.jsx` shows the score-shaping math, the penalty curve, and the compare view's chart composition. `hifi-studio.jsx` shows the shell composition and the wizard / palette overlay patterns.

The lo-fi sketches and the static remix are kept for context but are not required to implement Studio. The interactive `★ Studio` artboard is the only authoritative behavioral reference.
