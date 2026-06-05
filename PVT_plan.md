# PVT Corner System — Design & Implementation (as-landed at HEAD)

> **✅ STATUS — Phase 1 + Manual Sim LANDED on branch `feat/pvt`.** Corners now drive
> the simulation against a single `active_corner`; the manual-sim feature ships. Phase 2
> (multi-corner aggregation) remains deferred research. Landing commits: `ebc8e9d`,
> `2375f45`, `37952bf`, `c409031`, `a15b420`, `dc8b6f5` (current `HEAD` is `e6b68ed`).
>
> **What landed, with current anchors:**
> - **Core** ([core/domains.py](src/spicexplorer/core/domains.py)): `ModelInclude`
>   ([:231-240](src/spicexplorer/core/domains.py#L231-L240)), `SupplyOverride`
>   ([:242-246](src/spicexplorer/core/domains.py#L242-L246)), `Corner`
>   ([:248-261](src/spicexplorer/core/domains.py#L248-L261)), `PVTConfig` + `get_active`/
>   `enabled_corners`/`get` ([:263-291](src/spicexplorer/core/domains.py#L263-L291)),
>   `Project_Setup.pvt` field ([:749](src/spicexplorer/core/domains.py#L749)),
>   `_normalize_pvt_block` desugaring ([:131-186](src/spicexplorer/core/domains.py#L131-L186)),
>   called in `from_yaml` ([:808](src/spicexplorer/core/domains.py#L808)).
> - **Engine** ([spice_engine/spicelib.py](src/spicexplorer/spice_engine/spicelib.py)):
>   `NGSpice_Wrapper.apply_corner()` ([:337-383](src/spicexplorer/spice_engine/spicelib.py#L337-L383))
>   — the only ngspice-specific corner seam.
> - **Optimizer** ([optimization/base.py](src/spicexplorer/optimization/base.py)):
>   `Spice_Base_Optimizer.__post_init__` applies `pvt.get_active()` once per enabled
>   wrapper ([:491-499](src/spicexplorer/optimization/base.py#L491-L499)); the manual-sim
>   primitive `evaluate(params, append_to_log=False)`
>   ([:861-899](src/spicexplorer/optimization/base.py#L861-L899)).
> - **Backend**: `pvt`/`pvt_config` in the project summary
>   ([routes/project.py:84-122](ui/backend/routes/project.py#L84-L122)); ephemeral
>   `active_corner` override on live runs
>   ([optimizer_runner.py:203-240](ui/backend/services/optimizer_runner.py#L203-L240));
>   `POST /api/simulate/once` ([routes/simulate.py:211-215](ui/backend/routes/simulate.py#L211-L215)).
> - **UI**: `CornerSelect` ([ui/src/components/pvt/CornerSelect.tsx](ui/src/components/pvt/CornerSelect.tsx)),
>   `ManualSimPanel` ([ui/src/components/pvt/ManualSimPanel.tsx](ui/src/components/pvt/ManualSimPanel.tsx)),
>   PVT in the wizard's PVTStep, Pipeline DAG, and Setup summary.
> - **Tests**: `tests/test_pvt_corner.py`, `tests/test_pvt_wizard_roundtrip.py`.

**As-landed vs as-designed.** This document was originally the *pre-implementation* design
doc (its anchors came from commit `a11aa05` and had drifted). It is now rewritten as a
**current-HEAD** document: the design sections below describe **what exists** with
accurate `file:line` anchors verified against `HEAD`. Where the landed code differs from
the original sketch, the difference is called out. A new
[Known gaps in the landed Phase 1 / manual sim](#known-gaps-in-the-landed-phase-1--manual-sim)
subsection folds in this round's freshly-found PVT/manual-sim bugs (cross-referenced to
[bug_report.md](bug_report.md)).

**Scope.** Make process/voltage/temperature (PVT) corners first-class so they actually
drive SPICE simulation, superseding the dead `tech_spec.pvt_map` and the display-only flat
`pvt_corners`. Phase 1 (landed) runs the optimizer against **one chosen corner**; Phase 2
(deferred research) would run the full **testbench × corner** cross-product and aggregate.
Part B folds in the **manual-simulation** feature, which shares the same sim primitive.

---

## Part A — PVT corner architecture

### Legacy (display-only) state that PVT replaces

Two legacy config shapes are **parsed-but-never-simulated** and are deliberately kept for
backward-compat / display only:

| Config key (YAML) | Dataclass field | Runtime use |
|---|---|---|
| `tech_spec.pvt_map` | **None** — silently dropped by non-strict dacite | none |
| `pvt_corners` (top-level) | `Project_Setup.pvt_corners: List[PVT]` ([domains.py:733](src/spicexplorer/core/domains.py#L733)) | logged + displayed only |
| `pvt_corners[].enable` | **None** on `PVT` ([domains.py:218-222](src/spicexplorer/core/domains.py#L218-L222)) — dropped | none |

- `TechSpec` has only `name` + `constraints` — **no `pvt_map` field**
  ([domains.py:206-216](src/spicexplorer/core/domains.py#L206-L216)). Parsing uses
  `safe_from_dict(..., config=DECITE_CONFIG)`
  ([domains.py:810](src/spicexplorer/core/domains.py#L810)); `DECITE_CONFIG`
  ([domains.py:913-918](src/spicexplorer/core/domains.py#L913-L918)) does **not** set
  `strict`, so dacite's default (`strict=False`) silently ignores unexpected keys.
  `pvt_map` is therefore parsed into nothing.
- The legacy [`folded_cascode/.../project_setup.yaml`](examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml)
  still carries the dead `tech_spec.pvt_map`
  ([:38-56](examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml#L38-L56))
  **and** the flat `pvt_corners`
  ([:64-69](examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml#L64-L69)) —
  the exact shapes the new `pvt:` block replaces. (It does not yet carry a `pvt:` block;
  the **cascode** example does.)
- The display-only `pvt_corners` is read by `Project_Setup.summary()`
  ([domains.py:892-894](src/spicexplorer/core/domains.py#L892-L894)) and serialized for the
  UI at [routes/project.py:84-86,121](ui/backend/routes/project.py#L84-L86). **No optimizer
  / spice-engine code reads it.**

**A corner = an ordered set of `(lib_file, section)` selections** across device families,
plus environment (temp, supply). Verified against
`docker/pdk/ihp-sg13g2/libs.tech/ngspice/models/` — the **real** `.LIB <section>` tokens:

| File | Real sections (subset) |
|---|---|
| `cornerMOSlv.lib` | `mos_tt`, `mos_ss`, `mos_ff`, `mos_sf`, `mos_fs` (+ `_mismatch`, `_stat`) |
| `cornerMOShv.lib` | HV equivalents: `mos_tt`, `mos_ss`, `mos_ff`, … |
| `cornerRES.lib` | `res_typ`, `res_bcs`, `res_wcs` (+ `_mismatch`, `_stat`) |
| `cornerCAP.lib` | `cap_typ`, `cap_bcs`, `cap_wcs` (+ `_mismatch`, `_stat`) |
| `cornerHBT.lib` | `hbt_typ`, `hbt_bcs`, `hbt_wcs` (+ `_mismatch`, `_stat`) |

One logical corner spans multiple device families: e.g. cascode's `tt` bundle references
`mos_tt`@`cornerMOSlv.lib` **and** `res_typ`@`cornerRES.lib`
([cascode project_setup.yaml:60-63](examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml#L60-L63)).

### Config schema (landed)

A single top-level `pvt:` block. A **corner** is a self-contained bundle: process
selections across device families + environment (temp, supply, extra params). Corners are
**named** and referenced by name. Two operating modes:

- **(i) enumerate** — list many corners, mark which are `enabled` (Phase 2, deferred);
- **(ii) switch** — set a single `active_corner` and optimize against just it (Phase 1).

The **landed** cascode example
([examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml:57-90](examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml#L57-L90)),
grounded in the real corner sections above:

```yaml
pvt:
  # The single corner the optimizer runs against in single-corner mode (Phase 1).
  # Must match one corner's `name` below. Overridable ephemerally by the UI.
  active_corner: tt_27C_1V5

  # Reusable named PROCESS bundles: ordered (file, section) include lists.
  # PDK-agnostic — these are ihp-sg13g2's tokens but core never interprets them.
  process_bundles:
    tt:
      - { lib_file: cornerMOSlv.lib, section: mos_tt }
      - { lib_file: cornerRES.lib,   section: res_typ }
    ss:
      - { lib_file: cornerMOSlv.lib, section: mos_ss }
      - { lib_file: cornerRES.lib,   section: res_wcs }
    ff:
      - { lib_file: cornerMOSlv.lib, section: mos_ff }
      - { lib_file: cornerRES.lib,   section: res_bcs }

  # CORNERS = process bundle + environment. `process:` references a bundle by key,
  # OR inline `model_includes:` for a one-off corner.
  corners:
    - name: tt_27C_1V5            # typical — reproduces the netlist's hardcoded corner
      process: tt
      temp: 27                    # °C → emitted as `.options temp=27`
      supply: { node: VDD, value: 1.5 }   # overrides `.param VDD`
      enabled: true

    - name: ss_125C_1V35          # slow-slow, hot, -10% supply
      process: ss
      temp: 125
      supply: { node: VDD, value: 1.35 }
      enabled: true

    - name: ff_m40C_1V65          # fast-fast, cold, +10% supply
      process: ff
      temp: -40
      supply: { node: VDD, value: 1.65 }
      enabled: false              # defined but excluded from the (deferred) sweep

  # OPTIONAL: directory prepended to each include's lib_file, so corners are portable.
  # null → use the netlist's own `.lib` search path / PDK env (current behavior).
  model_lib_root: null
```

**Desugaring** is done by `_normalize_pvt_block`
([domains.py:131-186](src/spicexplorer/core/domains.py#L131-L186)) **in-place, before
dacite**, called from `from_yaml`
([domains.py:808](src/spicexplorer/core/domains.py#L808)):

- `process_bundles:` (named, reusable) are **inlined** into each corner's `model_includes`
  then dropped — sugar, not a retained field
  ([domains.py:152-172](src/spicexplorer/core/domains.py#L152-L172)). A dangling `process:`
  ref raises `ValueError`.
- a singular `supply: {node, value}` is **widened** to `supplies: [...]`, so the schema
  extends to multi-rail without breaking
  ([domains.py:174-177](src/spicexplorer/core/domains.py#L174-L177)).
- `temp`, supply `value`, and `params` values are coerced through `parse_value`, so
  engineering strings ("900m", "1.2") and ints all work
  ([domains.py:179-186](src/spicexplorer/core/domains.py#L179-L186)).

Schema notes:

- `supplies` is the canonical core shape (a `List[SupplyOverride]`), so a multi-rail design
  (`VDD`, `VDDH`) is representable; the singular `supply` is sugar.
- `params: {}` is the extensibility escape hatch for any other per-corner condition (bias
  currents, body bias, …), applied as plain `.param` overrides.
- This **subsumes** `tech_spec.pvt_map` + the flat `pvt_corners`; the legacy fields stay
  parsed-but-display-only (no regression for existing YAMLs).

### PDK-agnostic abstraction boundary

The seam is a generic **`ModelInclude`** directive + a **`Corner`** environment record.
Core never sees `ihp-sg13g2`, `cornerMOSlv.lib`, `mos_tt`, or any notion of
"MOS / RES / CAP families."

**Lives in CORE (PDK-agnostic), [core/domains.py](src/spicexplorer/core/domains.py):**

```python
@dataclass
class ModelInclude:                  # domains.py:231-240
    lib_file: str                    # opaque token; engine emits `.lib <file> <section>`
    section: str                     # core never enumerates valid sections

@dataclass
class SupplyOverride:                 # domains.py:242-246
    node: str                        # the `.param` name to override (e.g. "VDD")
    value: float

@dataclass
class Corner:                        # domains.py:248-261
    name: str
    model_includes: List[ModelInclude]   # bundle expanded at load time
    temp: float = 27.0
    supplies: List[SupplyOverride] = field(default_factory=list)
    params: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class PVTConfig:                     # domains.py:263-291
    active_corner: str
    corners: List[Corner] = field(default_factory=list)
    model_lib_root: Optional[str] = None
    def get(self, name) -> Optional[Corner]: ...
    def get_active(self) -> Corner: ...        # raises if active_corner is undefined
    def enabled_corners(self) -> List[Corner]: ...
```

**Lives in DATA / CONFIG (PDK-specific):** file names (`cornerMOSlv.lib`), section names
(`mos_tt`, `res_typ`), family groupings, temperatures, and supply rail names — all inside
the YAML `pvt:` block. **Adding a new PDK = writing a new `pvt:` block (data), zero core
change.**

**The render/injection seam** is a single method that turns a `Corner` into SPICE edits,
**isolated in the spice engine** so simulator-specific emission (`.lib` vs Spectre
`include`/`section`) stays there:

```python
# NGSpice_Wrapper.apply_corner — spice_engine/spicelib.py:337-383
def apply_corner(self, corner: "Corner", model_lib_root: str | None = None) -> None:
    # (1) per library file: strip the netlist's prior `.lib <…file> <section>`
    #     (path-agnostic basename match), then add the corner's ordered include.
    for inc in corner.model_includes:
        basename = re.escape(Path(inc.lib_file).name)
        self._strip_matching_instructions(rf"^\s*\.lib\s+\S*{basename}\s+\S+")
        path = inc.lib_file if not model_lib_root else str(Path(model_lib_root) / inc.lib_file)
        ed.add_instruction(f".lib {path} {inc.section}")
    # (2) temperature — authoritative: strip any prior, then `.options temp=`.
    self._strip_matching_instructions(r"^\s*\.options?\s+temp\s*=")
    ed.add_instruction(f".options temp={corner.temp}")
    # (3) supplies then extra params override `.param` defaults.
    for s in corner.supplies:        ed.set_parameter(s.node, s.value)
    for k, v in corner.params.items(): ed.set_parameter(k, v)
```

It is **idempotent** — re-applying replaces rather than accumulates directives (the
`_strip_matching_instructions` helper at
[spicelib.py:322-335](src/spicexplorer/spice_engine/spicelib.py#L322-L335) only strips when
a match exists, so the common first-apply case doesn't emit a misleading spicelib ERROR).
The `.lib` / `.options temp` syntax is the **only** ngspice-specific corner emission; a
future Spectre backend overrides this one method.

> **Landed-vs-designed deltas:** (a) `apply_corner` takes `model_lib_root` as a parameter
> (not a constructor field), passed by `__post_init__`. (b) The strip regex matches on the
> library **basename** so it works whether the netlist's `.lib` used a bare filename or a
> full path. (c) Temperature is set via `.options temp=` (authoritative) rather than a bare
> `.param temp`, which ngspice does **not** treat as the simulation temperature.

### Cross-product execution model (Phase 2 design)

The full evaluation grid would be **{enabled testbenches} × {enabled corners}**. Today
(Phase 1) it is `{enabled testbenches} × {the single active corner}`.

- **Enumeration.** `simulate_circuit()`
  ([base.py:503-541](src/spicexplorer/optimization/base.py#L503-L541)) iterates
  `self.spicelib_wrappers.items()` (one wrapper per enabled testbench, built by
  `_build_spicelib_wrappers` at
  [optimizer_runner.py:53-77](ui/backend/services/optimizer_runner.py#L53-L77)). The
  cross-product nests a corner loop outside it: for each enabled `Corner`,
  `apply_corner(corner)` on each wrapper, run, collect.
- **Run labeling.** `NGSpice_Wrapper._move_to_run_folder(label=…)`
  ([spicelib.py:269-279](src/spicexplorer/spice_engine/spicelib.py#L269-L279)) already names
  run folders by testbench; a composite `f"{tb}__{corner.name}"` would key artifacts by
  (testbench, corner).
- **Result keying.** Metric extraction at
  [base.py:874-877](src/spicexplorer/optimization/base.py#L874-L877)
  (`self.spicelib_wrappers[target.testbench].extract_scalar_variable_from_raw(...)`) would
  re-key from `{spec: value}` to `{(corner, spec): value}`, so each spec yields one value
  **per corner**.
- **Checkpoint shape.** `fit_summary` is keyed by spec name today
  ([base.py:923-926](src/spicexplorer/optimization/base.py#L923-L926)); under multi-corner
  it would become `f"{corner}::{spec}"`. **Caveat:** dotted/`::` keys interact with the
  pandas-column mangling the checkpoint reader already works around with `.iterrows()`
  ([checkpoint_reader.py:77-78](ui/backend/services/checkpoint_reader.py#L77-L78) reads
  `row.get("point.score")`) — keep that pattern when namespacing keys.

**Important:** the cross-product only changes *how many* `(metric, value)` pairs feed the
scorer. How they collapse into one scalar objective is the Phase 2 open question.

### Phasing

#### Phase 1 (LANDED) — single chosen corner

**What it does:** corners become first-class and actually drive the sim, while the
optimization loop, scorer, and per-trial flow are **untouched**. Only the *one-time*
netlist preparation gained a corner-apply step.

The seam is in `Spice_Base_Optimizer.__post_init__`
([base.py:491-499](src/spicexplorer/optimization/base.py#L491-L499)) — the existing one-time
tb-param setup. After applying tb params it applies the active corner to **every enabled
wrapper** (the dict only holds enabled testbenches; disabled ones are skipped by the wrapper
builder and by the guard at
[base.py:472-473](src/spicexplorer/optimization/base.py#L472-L473)):

```python
pvt = getattr(self.setup_obj, "pvt", None)
if pvt is not None:
    corner = pvt.get_active()
    for tb_name, wrapper in self.spicelib_wrappers.items():
        wrapper.apply_corner(corner, model_lib_root=pvt.model_lib_root)
```

Every subsequent trial inherits the corner because `SpiceEditor` netlist state persists
across runs.

**What changed (the entire Phase-1 footprint):**

1. **[core/domains.py](src/spicexplorer/core/domains.py)** — added `ModelInclude` /
   `SupplyOverride` / `Corner` / `PVTConfig`
   ([:231-291](src/spicexplorer/core/domains.py#L231-L291)); the `pvt:` field on
   `Project_Setup` ([:749](src/spicexplorer/core/domains.py#L749)); `_normalize_pvt_block`
   ([:131-186](src/spicexplorer/core/domains.py#L131-L186)) called in `from_yaml`
   ([:808](src/spicexplorer/core/domains.py#L808)); `summary()` now also logs the active
   corner ([:895-904](src/spicexplorer/core/domains.py#L895-L904)). `TechSpec`, `PVT`, and
   `pvt_corners` were left as-is (display-only, no regression).
2. **[spice_engine/spicelib.py](src/spicexplorer/spice_engine/spicelib.py)** — added
   `apply_corner` ([:337-383](src/spicexplorer/spice_engine/spicelib.py#L337-L383)) and the
   `_strip_matching_instructions` helper
   ([:322-335](src/spicexplorer/spice_engine/spicelib.py#L322-L335)). `update_params`,
   `run_and_wait`, run-folder logic untouched.
3. **[optimization/base.py](src/spicexplorer/optimization/base.py)** — the one-time
   corner-apply block ([:491-499](src/spicexplorer/optimization/base.py#L491-L499)).
4. **UI / backend (ephemeral override)** — the project summary surfaces `pvt`/`active_corner`
   + the corner list ([routes/project.py:84-122](ui/backend/routes/project.py#L84-L122));
   `_apply_overrides` accepts an ephemeral `active_corner`
   ([optimizer_runner.py:203-240](ui/backend/services/optimizer_runner.py#L203-L240),
   threaded through `RunState.active_corner` at
   [:31-32](ui/backend/services/optimizer_runner.py#L31-L32)) — same in-memory,
   never-rewrite-YAML pattern as algorithm/budget/seed. `CornerSelect`
   ([ui/src/components/pvt/CornerSelect.tsx](ui/src/components/pvt/CornerSelect.tsx)) is the
   reusable picker (lists every defined corner — Phase 1 runs against a single one, so even
   `enabled: false` corners are selectable; `enabled` only gates the deferred sweep).

**Stays untouched (optimizer core):** the nevergrad / Ax classes; `optimization_step`
([nevergrad.py:186-197](src/spicexplorer/optimization/stochastic/nevergrad.py#L186-L197)),
the `optimize()` loop, `simulate_circuit`
([base.py:503-541](src/spicexplorer/optimization/base.py#L503-L541)), `evaluate`
([base.py:861-899](src/spicexplorer/optimization/base.py#L861-L899)); `compute_fitness` /
`compute_constraint_violation_penalty_for_spec` scoring math
([base.py:901-998](src/spicexplorer/optimization/base.py#L901-L998)); `core/utils.py`;
checkpoint schema; all charts. A single-corner run is a **strict superset** of the legacy
"netlist hardcodes the corner" behavior — with `pvt: None` the apply step is a no-op, and
the cascode `tt_27C_1V5` corner is authored to reproduce the netlist's hardcoded
`.lib cornerMOSlv.lib mos_tt` / `res_typ` / temp 27 / VDD 1.5 exactly.

#### Phase 2 (DEFERRED, research) — multi-corner aggregation

Phase 1 makes one corner drive the sim. Phase 2 runs the **{tb × corner} cross-product** and
must collapse **N corner-scores into one scalar** that the optimizer consumes
(`self.optimizer.tell(candidate, -1 * curr_score)`,
[nevergrad.py:196](src/spicexplorer/optimization/stochastic/nevergrad.py#L196)).

**The open question:** `compute_fitness()`
([base.py:901-937](src/spicexplorer/optimization/base.py#L901-L937)) produces one
`total_score` from one `performance_array`. With multiple corners there are multiple
`performance_array`s. How they combine sets the optimizer's incentive structure. Candidate
strategies (**no commitment to one**):

| Strategy | Behavior | Trade-off |
|---|---|---|
| **Worst-case / min** | `score = min_c score_c` | Robust, classic for PVT sign-off; pessimistic, can stall on one pathological corner. |
| **Weighted mean** | `score = Σ w_c · score_c` | Smooth gradient; can mask a single failing corner. |
| **Sum of penalties** | accumulate per-corner penalties before reward | Aligns with the existing penalty/reward split ([base.py:928-932](src/spicexplorer/optimization/base.py#L928-L932)); double-counts shared violations. |
| **Must-pass-all (constraint)** | any corner missing spec → large penalty (`MAX_PENALTY`, [base.py:43](src/spicexplorer/optimization/base.py#L43)); reward only when all pass | Matches real tape-out sign-off semantics. |
| **Pareto / multi-objective** | keep corners as separate objectives | Natural fit for the Ax backend; changes the optimizer contract from scalar `tell` to multi-objective. |

Secondary Phase 2 items: where the corner loop lives (inside `simulate_circuit` vs a new
outer driver); parallelism — the existing `parallel_sim` path
([base.py:515-541](src/spicexplorer/optimization/base.py#L515-L541)) would fan out tb × corner
(N× more concurrent ngspice processes); and the `fit_summary` / checkpoint key namespacing
(`{corner}::{spec}`) plus the dotted-column caveat. **All out of scope for this round.**
Phase 2 would also be the natural place to wire up `freeze_to: <value>` (currently
parsed-but-ignored) and a per-corner manual sim. See
[Known gaps](#known-gaps-in-the-landed-phase-1--manual-sim) below for the bugs that should be
fixed before Phase 2 builds on this surface.

### Known gaps in the landed Phase 1 / manual sim

These are real, code-level defects in the **landed** implementation (verified statically at
`HEAD`), folded in from [bug_report.md](bug_report.md):

- **WIZ-2 / BUG-A3 — `pvt.model_lib_root` dropped on the YAML→form→YAML wizard round-trip
  (major).** `_build_pvt_block` never emits `model_lib_root`
  ([yaml_generator.py:174-206](ui/backend/services/yaml_generator.py#L174-L206)) and
  `_pvt_block_to_form` never reads it
  ([yaml_generator.py:253-293](ui/backend/services/yaml_generator.py#L253-L293), returns only
  `{active_corner, corners}` at :290-293). The field **is** load-bearing — it's passed to
  `apply_corner` at [base.py:499](src/spicexplorer/optimization/base.py#L499) and prepended to
  each include's path at
  [spicelib.py:368](src/spicexplorer/spice_engine/spicelib.py#L368). A project that sets a
  non-null `model_lib_root`, edited in the wizard and saved, regenerates with bare `.lib`
  paths that no longer resolve. Doesn't bite shipped examples (cascode sets
  `model_lib_root: null`).
- **WIZ-4 / BUG-A6 — multi-rail corners lose supply rails 2..N through the wizard (minor).**
  `_pvt_block_to_form` keeps only `supplies[0]`
  ([yaml_generator.py:276-283](ui/backend/services/yaml_generator.py#L276-L283)) and
  `_build_pvt_block` emits a singular `supply`
  ([yaml_generator.py:197-199](ui/backend/services/yaml_generator.py#L197-L199)). Multi-rail
  **is** representable in core (`SupplyOverride` / `supplies: List`, widened by
  `_normalize_pvt_block`), so the loss is purely in the wizard layer. Documented as intentional
  (docstring at [yaml_generator.py:258-259](ui/backend/services/yaml_generator.py#L258-L259)
  steers multi-rail users to the raw editor) but silent — no warning that Save drops rails.
- **SCH-2 / BUG-A4 — `dut_param` string `val` is never resolved (minor).**
  `resolve_all_parameter_ranges` resolves dut_param `min`/`max`/`init` but calls
  `ressolve_val` **only** in the testbench-param loop
  ([domains.py:832-846](src/spicexplorer/core/domains.py#L832-L846); `ressolve_val` defined at
  [:323-325](src/spicexplorer/core/domains.py#L323-L325), sole call at :844). A
  `dut_param.val` given as an engineering string (e.g. `"0.18u"`) stays a `str`, so
  [project.py:63](ui/backend/routes/project.py#L63) serializes it as `val: null` (it fails the
  `isinstance(p.val, (int, float))` check). This **undercuts** the manual-sim Mode B form
  pre-fill — `ManualSimPanel` seeds each input from `p.init ?? p.val`
  ([ManualSimPanel.tsx:51-57](ui/src/components/pvt/ManualSimPanel.tsx#L51-L57)), and a
  string `val` arrives as `null`. Not triggered by the cascode example (no `dut_param` sets
  `val`).
- **OPT-2 / BUG-A8 — one-directional output isolation (minor).** The manual sim isolates its
  outputs under `ws_root/outdir/manual_sim`
  ([simulate.py:177](ui/backend/routes/simulate.py#L177) →
  `_build_spicelib_wrappers(..., output_subdir="manual_sim")` at
  [optimizer_runner.py:53-77](ui/backend/services/optimizer_runner.py#L53-L77)), but a **live
  run** builds wrappers with no subdir (`output_folder = ws_root/outdir`,
  [optimizer_runner.py:257](ui/backend/services/optimizer_runner.py#L257)) and
  `NGSpice_Wrapper._validate` `rmtree`s that folder on construction
  ([spicelib.py:234-242](src/spicexplorer/spice_engine/spicelib.py#L234-L242)) — recursively
  removing the nested `manual_sim/` subtree. Starting a live run while a manual sim is
  in-flight can delete the manual sim's working dir out from under ngspice. No lock serializes
  the two. (Reproducing the race needs concurrent execution — deferred — but the rmtree-of-
  parent path is confirmed statically.)

---

## Part B — Manual simulation feature (LANDED, shares sim infra)

**Goal:** "evaluate one design point on demand" — run all enabled testbenches once for a
user- or checkpoint-supplied param vector and return metrics + score. No new sim path: the
optimizer already contains the exact primitive.

### The shared primitive (landed)

`Spice_Constraint_Satisfaction.evaluate(parameterization, append_to_log=False)`
([base.py:861-899](src/spicexplorer/optimization/base.py#L861-L899)) does precisely the four
steps a manual sim needs:

1. `simulate_circuit(parameterization)` — injects params into every enabled testbench and
   runs ngspice once each ([base.py:868](src/spicexplorer/optimization/base.py#L868); loop at
   [:509-541](src/spicexplorer/optimization/base.py#L509-L541) — run-count is structurally 1).
2. extracts each enabled `TargetSpec`'s scalar metric from the RAW
   ([base.py:874-877](src/spicexplorer/optimization/base.py#L874-L877)).
3. `compute_fitness(...)` → `(total_score, fit_summary)`
   ([base.py:881](src/spicexplorer/optimization/base.py#L881)).
4. `append_to_log=False` skips polluting `optimization_log`
   ([base.py:884-892](src/spicexplorer/optimization/base.py#L884-L892)); then
   `clean_up(delete_raw_only=True)`.

`Spice_Base_Optimizer.plot_solution` already calls
`self.evaluate(parameterization, append_to_log=False)`
([base.py:639](src/spicexplorer/optimization/base.py#L639)), so one-shot evaluation of a given
vector was an **existing, exercised** path.

**Key subtlety — input domain.** `evaluate` expects **denormalized engineering-real values**
(e.g. `X_DUT_M1M2_W = 72e-6`); `denormalize_params`
([base.py:121-142](src/spicexplorer/optimization/base.py#L121-L142)) exists only to undo the
optimizer's normalized space. Manual sim **bypasses `denormalize_params`** and feeds real
values straight to `evaluate`. The route casts the inbound vector to float
([simulate.py:149-150](ui/backend/routes/simulate.py#L149-L150)). Note `update_params` adds
`k`/`p` suffixes for `C*`/`R*`-prefixed names
([spicelib.py:312-315](src/spicexplorer/spice_engine/spicelib.py#L312-L315)), so manual values
for those must follow the same convention.

### The route (landed): `POST /api/simulate/once`

[routes/simulate.py](ui/backend/routes/simulate.py) — registered in
[main.py](ui/backend/main.py#L23) and mounted under `/api`. It is a **sibling** of
`sanity.py`, but instead of simming a *random* `ask()` point it evaluates a *chosen* vector.
Flow of `_run_single_sim`
([simulate.py:122-208](ui/backend/routes/simulate.py#L122-L208)):

```
ManualSimPanel  ──{ yaml_path, params | checkpoint_id+point, active_corner }──▶
POST /api/simulate/once          (async → loop.run_in_executor, simulate.py:211-215)
  _run_single_sim:
    probe_pdk()                                        # PDK-gated (simulate.py:130)
    Project_Setup.from_yaml(yaml_path)                 # simulate.py:142
    resolve params  (Mode B: req.params  |  Mode A: _resolve_params_from_checkpoint)
    ephemeral active_corner override                   # simulate.py:160-170
    _validate_params(...)  → non-fatal warnings        # simulate.py:172-173
    wrappers = _build_spicelib_wrappers(project, output_subdir="manual_sim")  # isolated
    opt = Nevergrad_Spice_Single_Objective(project, wrappers)   # applies active corner
    score, fit_summary = opt.evaluate(params, append_to_log=False)            # one shot
  → { ok, score, metrics{spec:{curr_val,score}}, params_used, active_corner,
      warnings, log_files, log_tails, elapsed_ms, pdk_ok, … }
```

It does **not** call `parameterize()` / `_create_optimizer_obj()` / `optimize()` — those
build the nevergrad optimizer and pick a **random** candidate via `ask()`
([nevergrad.py:188](src/spicexplorer/optimization/stochastic/nevergrad.py#L188)), which was
the `sanity.py` gap. The optimizer is constructed only to reach `evaluate`; constructing it
applies the active PVT corner for free (Phase 1).

### Two input modes

**Mode A — from a checkpoint** (`checkpoint_id` + optional `point`).
`_resolve_params_from_checkpoint`
([simulate.py:67-102](ui/backend/routes/simulate.py#L67-L102)) resolves the path via
`_resolve_checkpoint_path` ([checkpoint.py:43](ui/backend/routes/checkpoint.py#L43)) +
`read_checkpoint`, then pulls one point's vector. `point` omitted → the **best** iteration
(argmax of finite scores; checkpoints don't persist a best index, so it is recomputed). The
stored vectors are already **engineering-real** (written by `evaluate` as
`params=parameterization`, [base.py:886-888](src/spicexplorer/optimization/base.py#L886-L888))
so they feed `evaluate` with no transform. Re-simulating the best point should reproduce its
stored `curr_val`s — doubling as a result-validation tool.

**Mode B — manual user-supplied values** (`params`). `ManualSimPanel`
([ui/src/components/pvt/ManualSimPanel.tsx](ui/src/components/pvt/ManualSimPanel.tsx)) pre-fills
a numeric form from each `dut_param`'s `init` (fallback `val`)
([ManualSimPanel.tsx:51-57](ui/src/components/pvt/ManualSimPanel.tsx#L51-L57)). A **partial**
dict is valid: `update_params` only sets provided keys; unset params keep their netlist
`.param` defaults. Both modes converge on the identical
`evaluate(params, append_to_log=False)` call, so a manual sim produces the **same
`fit_summary` and `score`** as a real trial — directly comparable to checkpoint rows and to
Score Shaping.

### Required UI (landed)

- **`ManualSimPanel`** in OptimizeTab — design-centric, PDK-gated (disabled when
  `pdk_ok === false`, like live Start). `Segmented` source toggle (Manual entry / From
  checkpoint); Mode A is a checkpoint `<select>` + "best point" default; Mode B is a numeric
  table seeded from `init`/`val`. Per-spec result table (`curr_val` via `formatEng`, target,
  pass/fail via `statusForGoal`) + total score; collapsible per-testbench log tails.
- **`CornerSelect`** in the panel routes the chosen corner to the route's `active_corner`
  (ephemeral override).

### Remaining interface gaps (manual sim)

1. **No `simulate_point(params)` façade.** `evaluate(append_to_log=False)` is only reachable on
   a fully-constructed optimizer; the route constructs `Nevergrad_Spice_Single_Objective`
   directly ([simulate.py:178](ui/backend/routes/simulate.py#L178)). A
   `Base_Optimizer.simulate_point(params) -> (score, fit_summary)` would be the clean library
   seam.
2. **Input-domain validation is advisory only.** `_validate_params`
   ([simulate.py:105-119](ui/backend/routes/simulate.py#L105-L119)) **warns** (does not reject)
   on an unknown param or a value outside `[min_val, max_val]`; nothing enforces `is_integer`.
   `evaluate` trusts the real values.
3. **Mode B pre-fill depends on `dut_param.val` resolution (SCH-2).** Because `ressolve_val` is
   never run on dut_params (gap above), a string `val` surfaces as `null` in the summary and
   the form can't seed from it — only `init` reliably pre-fills.
4. **Output isolation is one-directional (OPT-2).** `manual_sim` is isolated from a live run,
   but a live run's `rmtree` of the shared parent can still clobber `manual_sim`. Needs a
   per-run subfolder for live runs, an rmtree scoped to the run's own subtree, or a mutex.
5. **Per-corner manual sim is single-corner.** The panel applies one `active_corner` (Phase 1).
   Sweeping a chosen point across all enabled corners is the natural follow-on with Phase 2.

---

## Load-bearing findings (summary)

- **Legacy `tech_spec.pvt_map` is dead config** — silently dropped by non-strict dacite
  (`DECITE_CONFIG`, [domains.py:913-918](src/spicexplorer/core/domains.py#L913-L918)); `TechSpec`
  has no such field; zero `src/` consumers. Still present in the folded_cascode example.
- **The flat `pvt_corners` is parsed but never drives a sim** — only logged
  ([domains.py:892-894](src/spicexplorer/core/domains.py#L892-L894)) and displayed
  ([project.py:84-86](ui/backend/routes/project.py#L84-L86)).
- **PVT Phase 1 landed:** the `pvt:` block (`PVTConfig` + `_normalize_pvt_block`) makes corners
  first-class; `NGSpice_Wrapper.apply_corner`
  ([spicelib.py:337-383](src/spicexplorer/spice_engine/spicelib.py#L337-L383)) is the **only**
  ngspice-specific seam (strip+inject `.lib`, `.options temp=`, supply `.param`); it is applied
  **once** in `Spice_Base_Optimizer.__post_init__`
  ([base.py:491-499](src/spicexplorer/optimization/base.py#L491-L499)). The optimize loop,
  scorer, and `simulate_circuit` are untouched — a single-corner run is a strict superset of the
  legacy hardcoded-corner behavior.
- **A corner = an ordered set of `(lib_file, section)`** across MOS/RES/CAP/HBT — verified
  against the real `corner*.lib` `.LIB <section>` tokens.
- **Manual sim landed:** `POST /api/simulate/once`
  ([simulate.py](ui/backend/routes/simulate.py)) reuses
  `evaluate(params, append_to_log=False)` verbatim, with Mode A (checkpoint point) and Mode B
  (manual vector). It deliberately avoids the random `ask()` point that was the `sanity.py` gap.
- **Multi-corner aggregation is the genuine research item** — deferred to Phase 2 (strategy
  table above; commit to none).
- **Known gaps in the landed work (see [bug_report.md](bug_report.md)):** WIZ-2 (`model_lib_root`
  dropped in the wizard round-trip), WIZ-4 (multi-rail supplies dropped), SCH-2 (`dut_param`
  string `val` never resolved — undercuts Mode B pre-fill), OPT-2 (one-directional output
  isolation between live run and manual sim).
