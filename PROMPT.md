# Task: SpiceXplorer Webapp Audit + PVT Architecture Planning

## Scope and constraints
This is an INVESTIGATION AND PLANNING pass. Do NOT modify application code, configs, or tests this round. The only files you create or edit are the four markdown deliverables specified at the end. Read broadly, trace root causes, and write the plans.

Before analysis, build a map of the webapp: entry points, routes/pages, frontend components, backend handlers, and how state flows between UI, the simulation layer, and the spicexplorer library. Reference this map throughout the deliverables.

## Reference material
- @examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml (current project setup, PVT-relevant config)
- @docker/pdk/ihp-sg13g2/libs.tech/ngspice/models/ (available corner model files for ihp-sg13g2)
- @TODO.md (existing task list to extend)

## Part 1: Webapp functionality audit
Do a comprehensive functional review of the webapp and flag potential bugs. For each issue report: file/location, observed symptom, root-cause hypothesis (with the specific code path), severity (blocker / major / minor), and suggested fix direction. Do not implement fixes.

Confirmed issues to start from (verify root cause, do not assume):
1. xschem display is broken and renders "missing symbol". This is a regression (it worked previously). Check symbol library path resolution, recent changes to schematic/symbol handling, and any environment or working-directory assumptions. Inspect git history around the rendering path if it helps localize the regression.
2. Score Shaping tab is non-functional: it only computes one score at a time instead of all configured scores. Trace the score computation flow (UI trigger to backend to spicexplorer) and identify where iteration over the full score set is missing or short-circuited.

Beyond these, actively hunt for additional bugs across all tabs and data flows.

## Part 2: Manual simulation feature (design only, no code)
Document a design for running simulations manually, independent of an optimization run.

Requirements:
- Run across all testbenches on demand.
- Reuse the optimizer's existing simulation infrastructure with run count = 1 (do not build a parallel sim path).
- Support two input modes for each testbench and DUT parameter set:
  a) load values from a prior optimization result, or
  b) accept manual user-supplied values.

Deliver: proposed data flow, the integration points into the existing sim infra, required UI changes, and any interface gaps. No implementation.

## Part 3: PVT corner architecture (design only, no code)
Design a PVT corner / pvt_map system. This is the largest planning item.

Goals:
- Define PVT corners in config (extend or evolve the project_setup.yaml schema). Inspect the available corner files under the ihp-sg13g2 models directory to ground the schema in real corner definitions.
- The design must be PDK-agnostic and built for future PDK expansion. Do not hardcode ihp-sg13g2 assumptions into the core abstraction.
- Support two operating modes: hardcode/enumerate multiple PDK runs, and simply switching a simulation to a different PVT corner.
- Runtime behavior once corners are defined: full cross product of testbench x PVT corner (every testbench runs in every defined corner).

Phasing (important, keep these separate in the plan):
- Phase 1 (target this round's plan): define corners and run optimization against a SINGLE chosen corner. This is the minimal-change path for the optimizer core (the spicexplorer library). The plan should specify exactly what changes and what stays untouched.
- Phase 2 (research, deferred): multi-corner score aggregation in the optimizer. Flag the open question of how scores from multiple corners get aggregated. Outline candidate strategies but mark as future work; do not commit to one.

Deliver a concrete config schema proposal, the abstraction boundary that keeps it PDK-agnostic, and the phase-1 vs phase-2 split.

## Part 4: Frontend rendering and layout audit (Layout Agent)

Assign this to a dedicated Layout Agent that owns frontend rendering only. It runs in parallel with the functional bug-audit agent. Scope is strictly CSS, layout, sizing, overflow, and scroll behavior. Do NOT change component logic, data flow, or business code, and do NOT edit files another agent is touching. No code changes this round; produce findings only.

Symptoms to investigate (verify root cause, do not assume):
- Horizontal overflow: input boxes and rows extend past their parent container. Reproduction is visible in the PVT Corners wizard step: the corner row (NAME / TEMP / CORNER / SUPPLY columns) overflows the card on the right edge and the SUPPLY field is clipped. Check column widths, hardcoded fixed widths, and missing min-width:0 on flex/grid children.
- Clipped content with no scroll: containers cut off their content and expose no scrollbar. Reproduction is the Sanity Check tab: the logs extend below the visible area and cannot be scrolled, and the only way to read them is zooming the browser out. Check container height constraints, overflow settings, and missing overflow-y:auto on log/scroll regions.
- Cross-tab pattern: the same overflow and no-scroll behavior recurs in other tabs. Sweep every tab; do not stop at these two examples.

For each instance report: tab, component, and offending container element; failure mode (horizontal overflow / clipped-no-scroll / mis-sizing); root-cause hypothesis naming the specific style or layout rule responsible; and recommended fix direction (for example: add overflow-y:auto, set min-width:0 on flex children, replace fixed widths with constrained responsive widths, let the container grow or scroll).


## Deliverables (markdown only)
1. bug_report.md: results of the Part 1 audit. Group by area, include the per-bug fields listed above (location, symptom, root cause, severity, fix direction).
2. PVT_plan.md: the Part 3 design. Include the proposed config schema, PDK-agnostic abstraction, cross-product execution model, and the explicit phase-1 (single-corner) vs phase-2 (multi-corner aggregation, deferred) breakdown. Fold the Part 2 manual-simulation design in here or in a clearly labeled section if it shares infra; otherwise note where it lives.
3. TODO.md: update the existing file with an actionable task list covering the bug fixes, the manual simulation feature, and PVT phase 1. Keep phase-2 aggregation listed as deferred/research.
4. project_redundancy.md: summary of dead code, duplicated logic, and convoluted code paths found during the audit. For each item, give location, why it is redundant or unclear, and removal/refactor risk (low / medium / high).
5. ui_layout_report.md (this agent's own file). Group findings by tab and lead with the global root causes that explain multiple instances.