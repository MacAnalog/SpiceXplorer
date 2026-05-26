# Xschem Viewer

An in-browser viewer for xschem `.sch` / `.sym` files, integrated as a tab in
the SpiceXplorer UI. View-only, with hierarchy navigation. No xschem process
is required at runtime — the renderer is a pure-TypeScript reimplementation.

## Why view-only

xschem is a Tcl/Tk app with no web frontend. Three paths were considered:

1. VNC / Xpra — full xschem in the browser, full editing. Heavyweight, per-user
   X processes, sluggish over high-latency links.
2. Pure JS parse + render — what's implemented here. Stable file format, no
   server process per user, scales freely. View-only.
3. Custom React schematic editor that emits xschem files — biggest implementation
   cost, risks netlist divergence from xschem's canonical output.

Editing is left to the native xschem app (or, later, an LLM-driven agent that
rewrites `.sch` text). The web viewer's job is fast browsing of the project's
schematics inline with the sizing / optimization workflow.

## User flow

The **Schematic** tab is pinned to the left rail above **Health** (keyboard `5`).

1. Apply a project YAML in the Setup tab.
   - If the YAML declares `project.schematic`, that file is auto-opened on
     entry to the tab.
   - In any case, the viewer scans the project's `ws_root/xschem/` directory
     and lists every `.sch` file in the **Open** dropdown.
2. Or click **Upload .sch** to view a schematic from anywhere on your machine.
   The file is parsed client-side; symbols are resolved against the PDK and
   xschem libraries on the server. Symbol files that live next to the uploaded
   `.sch` (sibling references) will not resolve unless they're also in a
   known library path.
3. Pan with click-drag, zoom with the wheel (cursor-pinned), **Fit** to reset.
4. Hierarchy:
   - **Up** — return to the parent schematic. Disabled at the top of the stack.
   - **Down ▾** — dropdown listing every subcircuit instance in the current
     view, labeled `<instance-name> — <symbol-stem>`. Pick one to descend.
   - Click a subcircuit instance in the canvas — same as picking it from Down.
     Subcircuits get a dashed yellow halo on hover.
5. Inspection:
   - Click a non-subcircuit instance — opens a floating popover with every
     attribute (`name`, `model`, `w`, `l`, `value`, …). Multi-line attrs (e.g.
     the `value` of a `code_shown.sym` block holding ngspice directives)
     render in a scrollable monospace box.
   - Right-click on any instance — same popover, useful for subcircuits where
     left-click descends.
   - Escape or click background to dismiss.

## Architecture

```
Browser
  └─ Schematic tab — SchematicTab.tsx
       │  ├─ fetches .sch + every referenced .sym
       │  └─ probes sibling .sch for each subcircuit (=> navigable)
       │
       └─ SchematicViewer.tsx
            │  pan/zoom, instance click handling
            │  symbol inlining via SVG transform groups
            │  click-to-inspect popover
            │
            └─ lib/xschem/
                 parser.ts   — text → AST
                 render.ts   — bbox, transforms, layer colors
                 types.ts    — AST shapes

  Browser fetches  /api/xschem/...  (proxied by Next.js to the FastAPI backend)

Backend
  └─ ui/backend/routes/xschem.py
       /xschem/file?path=…     — raw text of an absolute .sch/.sym path
       /xschem/resolve?ref=…   — resolve a relative ref against the search path
       /xschem/list?dir=…      — list .sch files in a directory
       /xschem/project?yaml=…  — find a project's xschem/ dir and list its .sch
```

### Key files

| Path | Role |
|------|------|
| [ui/backend/routes/xschem.py](../ui/backend/routes/xschem.py) | All 4 backend endpoints + path whitelist |
| [ui/src/lib/xschem/types.ts](../ui/src/lib/xschem/types.ts) | Typed AST: `XschemFile`, `Instance`, primitives |
| [ui/src/lib/xschem/parser.ts](../ui/src/lib/xschem/parser.ts) | Tokenizer + record parser |
| [ui/src/lib/xschem/render.ts](../ui/src/lib/xschem/render.ts) | Bbox math, layer→color, instance transforms |
| [ui/src/components/schematic/SchematicViewer.tsx](../ui/src/components/schematic/SchematicViewer.tsx) | SVG renderer + interactions |
| [ui/src/components/tabs/SchematicTab.tsx](../ui/src/components/tabs/SchematicTab.tsx) | Tab shell: history stack, breadcrumb, Up/Down |

## Xschem file format

Records are line-based but `{...}` property blocks may span multiple lines.
The parser supports the full set of geometric and structural primitives plus
the header tags it can safely skip.

| Code | Form | Meaning |
|------|------|---------|
| `v {...}` | header | xschem version line (multi-line content stored verbatim) |
| `G {...}` | header | global attrs |
| `K {...}` | header | "kind" attrs — `type=subcircuit`, `format=…`, `template=…` |
| `V {...}` / `S {...}` / `E {...}` | header | versions / selection / end-of-defs (skipped) |
| `L <layer> x1 y1 x2 y2 {p}` | primitive | line |
| `B <layer> x1 y1 x2 y2 {p}` | primitive | box (layer 5 = pin marker, rendered filled) |
| `P <layer> n x1 y1 … {p}` | primitive | polyline (n points) |
| `A <layer> cx cy r start ext {p}` | primitive | arc |
| `T {text} x y rot flip hs vs {p}` | text | rotate ∈ 0..3, flip ∈ 0,1 |
| `N x1 y1 x2 y2 {p}` | net/wire | always rendered with the layer-2 color |
| `C {symref} x y rot flip {p}` | instance | references a `.sym` file |

### Quirks the parser handles

- Multi-line `{...}` blocks: nested braces are balanced; `\\"` escapes inside
  quoted attribute values are honored.
- `*`-prefixed comment lines at line start (common inside the `v {…}` header
  of PDK `.sym` files).
- Unbalanced stray `"` characters that xschem itself tolerates (e.g.
  `value=5u pwl(...)"}` in `tb-ac.sch`). A `"` only opens a quoted region when
  it immediately follows `=`, matching xschem's `key="value"` lexical rule.

### What the parser intentionally skips

- Tcl-eval'd expressions in `format=` / `value=` are stored as raw text. They
  are not evaluated; only `@<attr>` placeholders inside `T {…}` records get
  substituted (see "Placeholder expansion").
- Per-layer fill / line styles beyond layer→color mapping.
- Slotted symbols and rotation-aware text counter-rotation (the renderer
  rotates text along with its parent instance instead of keeping it upright).

## Symbol resolution

A `C {ref} …` record references `ref` (e.g. `devices/title.sym`,
`sg13g2_pr/sg13_lv_nmos.sym`, or bare `vsource.sym`). The backend resolves
against, in order:

1. The directory of the file doing the reference (`base` query param).
2. `$PDK_ROOT/$PDK/libs.tech/xschem` — the PDK's xschem library.
3. `<xschem_install>/share/xschem/xschem_library/` — xschem's bundled libraries.
4. `<xschem_install>/share/xschem/xschem_library/devices/` — same dir, but as
   a separate root so bare refs like `vsource.sym` resolve.
5. Anything in `$XSCHEM_LIBRARY_PATH` (colon-separated).

All resolved paths are validated against the allowed-root whitelist before
being returned, so a malicious `ref=../../etc/passwd` cannot escape.

A symbol becomes **navigable** (clickable to descend) when:

- its `.sym` declares `K {type=subcircuit …}`, **and**
- a sibling `.sch` exists at the same path with the suffix swapped.

Both checks happen on the frontend when a schematic is loaded; the result is
the `navigableSymrefs` set passed to the viewer.

## Placeholder expansion

Symbol text records like `T {@name}` and `T {@value}` are substituted at
render time from the instance's attribute map. The expansion is generic:
`@<attr>` is replaced with `inst.attrs[<attr>]`, plus one special case
(`@symname` → the symbol filename stem).

This means a symbol-author-defined placeholder (e.g. `T {@spiceprefix}` for a
custom subcircuit prefix) renders correctly without any change to the viewer.

## Rendering details

- **Coordinate system**: xschem stores Y with the same convention as SVG
  (positive Y down). The schematic's overall bbox sets the initial viewBox,
  padded by 40 user-units.
- **Layer colors**: a fixed 16-entry palette in
  [render.ts](../ui/src/lib/xschem/render.ts) tuned to xschem's default dark
  theme. Layer 2 = wires (cyan), layer 4 = symbol outlines (green), layer 5 =
  pin markers (yellow).
- **Text size**: `fontSize ≈ 21.5 * hsize`, matching what xschem's own SVG
  exporter produces. Text scales with the schematic; zoom in for legibility,
  same as the native app.
- **Instance transform**: SVG `translate(x, y) rotate(-rot*90) scale(-1, 1)`.
  xschem's rotate is CW; SVG's is CCW. Flip is applied first (matches xschem
  source).
- **Text orientation**: text rendered with the "readable orientation" rule —
  the effective `(parent_rot + text_rot) % 4` is snapped from 2→0 and 3→1, and
  parent-instance flips do not mirror the glyphs. Labels stay upright (or 90°
  rotated) regardless of how the symbol is rotated or mirrored, matching
  xschem's display behavior.
- **Full-circle arcs**: an `A` record with extent ≥360° (used for voltage and
  current source bodies) is rendered as an SVG `<circle>` rather than a path,
  since SVG's `A` command degenerates to nothing when start and end points
  coincide.
- **Non-scaling strokes**: lines and boxes use `vector-effect="non-scaling-stroke"`
  so wires stay readable at any zoom level.

## Known limitations

- Layer-color palette is an approximation. Specific xschem color overrides via
  `xschemrc` are ignored.
- Pin labels rely on a `T {label}` inside the symbol; symbols that only carry
  a `B {dir=in/out}` for the pin but no explicit text render as bare yellow
  squares.
- Text alignment relative to its anchor is the SVG default (top-left, no
  horizontal centering) — xschem's rotation values encode alignment as well as
  orientation, but only the readable-orientation snap is implemented here, so
  some labels are slightly off-position relative to their pin marker.
- No selection / no measurement / no net highlighting.
- Schematic SVG is not virtualized — very large schematics (thousands of
  primitives) may be sluggish to pan/zoom.

## Adding an editing agent later

The viewer is structured so an agent can drive it indirectly:

- The backend can already serve and resolve files. Add a `POST` endpoint that
  writes a new `.sch` text back to disk (mirror the whitelist), and the agent
  can author edits without touching the frontend.
- The frontend's `parseXschem` round-trips losslessly enough to render any
  generated file; the natural integration point is a "Refresh" action on the
  Schematic tab that re-fetches whatever is currently displayed.
- Editing affordances (move, rotate, change params) would live in a new
  layer over `SchematicViewer` — the renderer already exposes per-instance
  attrs, so wiring a "modified by agent" highlight or a diff overlay is just
  a styling pass.

## YAML field

An optional `project.schematic` field in the project YAML points the viewer
at the main `.sch` for the design. The path is relative to `ws_root`, same
convention as `netlist`. When present, the field is auto-opened on entry to
the Schematic tab:

```yaml
project:
  name: CASCODE-OTA
  ws_root: /path/to/project/
  netlist: spice/ota-improved.spice
  schematic: xschem/ota-improved.sch   # optional; used only by the UI viewer
  outdir: spice/temp_spice_out
```

The field is a `Path | str | None` on `Project_Setup` ([src/spicexplorer/core/domains.py](../src/spicexplorer/core/domains.py)).
It's not consumed by the optimizer or netlist runner — it only affects which
schematic the viewer pre-selects.

## Endpoints (cheat sheet)

```
GET /api/xschem/file?path=<absolute path>
GET /api/xschem/resolve?ref=<relative ref>&base=<absolute parent file>
GET /api/xschem/list?dir=<absolute dir>
GET /api/xschem/project?yaml_path=<absolute YAML path>
```

All return JSON; `file` and `resolve` include both the resolved absolute
`path` and the raw file `content`.
