# PVT Corner System — Design & Implementation Plan

> **✅ STATUS — Phase 1 + Manual Sim IMPLEMENTED (branch `feat/pvt`).** Corners now
> drive the simulation against a single `active_corner`; the manual-sim feature ships.
> Phase 2 (multi-corner aggregation) remains deferred research. What landed:
> - **Core:** `ModelInclude`/`SupplyOverride`/`Corner`/`PVTConfig` + `Project_Setup.pvt`;
>   `_normalize_pvt_block` desugars the YAML (`process_bundles`, singular `supply`).
> - **Engine:** `NGSpice_Wrapper.apply_corner()` (strip+inject `.lib`, `.options temp=`,
>   supply `.param`) — the only ngspice-specific seam; applied once in
>   `Spice_Base_Optimizer.__post_init__`.
> - **Backend:** `pvt` in the project summary; ephemeral `active_corner` override on
>   live runs + sanity; `POST /api/simulate/once` (manual evaluate-a-point).
> - **UI:** `CornerSelect` in the Run popover / Optimize toolbar / Health check;
>   `ManualSimPanel` in Optimize; PVT in the Pipeline DAG, Setup summary, and the wizard.
> - **Tests:** `tests/test_pvt_corner.py`, `tests/test_pvt_wizard_roundtrip.py`.
>
> Anchors/line-numbers below are from the original audit (commit `a11aa05`) and may have
> shifted; they remain accurate as *locations*, not exact lines.

**Scope:** Make process/voltage/temperature (PVT) corners first-class so they actually drive SPICE simulation, replacing today's dead `pvt_map` / unused `pvt_corners` config. Phase 1 (this round) runs the optimizer against **one chosen corner**; Phase 2 (deferred research) runs the full **testbench × corner** cross-product and aggregates. A shorter companion section folds in the **manual-simulation** feature, which shares the same sim infrastructure.

All claims are grounded in the code paths read during this audit; every anchor is a clickable `file:line` link relative to the repo root.

> **Repo-state note:** the working-tree edits listed in the audit brief (`ui/backend/routes/project.py`, `SetupTab.tsx`, `api.ts`) have since been committed. `HEAD` is now [`a11aa05` "example PVT support added to yaml file"](#) (newer than the brief's `c937bb3`). The only uncommitted item is an untracked `PROMPT.md`. None of this changes the design below.

---

## Part A — PVT corner architecture

### Current state

**What `domains.py` parses today.**

| Config key (in YAML) | Dataclass field? | Runtime use |
|---|---|---|
| `tech_spec.pvt_map` | **None** — silently discarded | none |
| `pvt_corners` (top-level) | `Project_Setup.pvt_corners: List[PVT]` | logged + displayed only |
| `pvt_corners[].enable` | **None** on `PVT` — silently discarded | none |

- `TechSpec` has only `name` and `constraints` — **no `pvt_map` field** ([core/domains.py:149-159](src/spicexplorer/core/domains.py#L149-L159)). The YAML nests `pvt_map:` inside `tech_spec:` ([project_setup.yaml:38-56](examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml#L38-L56)) with keys `name` / `spice_corner` / `lib_name`, none of which map to a field.
- Parsing goes through `safe_from_dict(cls, proj, logger, config=DECITE_CONFIG)` ([core/domains.py:663](src/spicexplorer/core/domains.py#L663)). `DECITE_CONFIG` ([core/domains.py:757-761](src/spicexplorer/core/domains.py#L757-L761)) sets only a `type_hooks` entry for `ListTargetSpec` and does **not** set `strict`. dacite's `Config.strict` defaults to `False`, so the unexpected-key check never fires. **`pvt_map` is therefore parsed into nothing — never stored, never reachable.** The YAML comment "each of the following line would update the .lib statement" ([project_setup.yaml:39-40](examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml#L39-L40)) is aspirational.
- `pvt_corners` **does** parse into `List[PVT]` where `PVT = {temp, corner, supply}` ([core/domains.py:161-165](src/spicexplorer/core/domains.py#L161-L165), field at [domains.py:607](src/spicexplorer/core/domains.py#L607)). The YAML also supplies `enable: True` ([project_setup.yaml:66](examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml#L66)), which `PVT` does not declare — also silently dropped.

**Is `pvt_corners` used at runtime?** No. Its only consumers are read-only:

| Consumer | What it does |
|---|---|
| [`Project_Setup.summary()` domains.py:745-747](src/spicexplorer/core/domains.py#L745-L747) | logs temp/corner/supply |
| [`routes/project.py:78-80,93`](ui/backend/routes/project.py#L78-L80) | returns `{temp, corner, supply}` for display |
| [`demo/newcas_demo_runner.py:121`](src/spicexplorer/demo/newcas_demo_runner.py#L121) | copies into a demo payload |
| [`services/yaml_generator.py:162,188,231,326`](ui/backend/services/yaml_generator.py#L162) | wizard form ⇄ yaml round-trip |

**No optimizer / spice-engine code reads `pvt_corners`.** The `corner` string (e.g. `"tt"`) never reaches a `.lib` line; `temp` never reaches `.temp`; `supply` never reaches a source.

**How `.lib` / temp / supply injection works today: it doesn't.** The only netlist mutation the engine performs is `set_parameter` per param, via `NGSpice_Wrapper.update_params()` ([spice_engine/spicelib.py:291](src/spicexplorer/spice_engine/spicelib.py#L291)). There is **no** `.lib`, `.temp`, `.options temp`, or supply-source injection anywhere in `src/`. The corner, temperature, and supply are whatever the testbench `.spice` file hardcodes:

- `.lib cornerMOSlv.lib mos_tt` — hardcoded process corner (every tb netlist under `examples/OTA/folded_cascode/ihp-sg13g2/spice/`).
- `.param VDD = 1.8` — supply as a plain param.
- `.param temp = 27` — a **bare `.param`**, not ngspice's `.temp` / `.options temp` directive, so its effect on the actual simulation temperature is not guaranteed by ngspice semantics. A real per-corner temp must be emitted as `.temp <val>` or `.options temp=<val>`.

**A corner = an ordered set of `(lib_file, section)` selections** across device families. Verified against `docker/pdk/ihp-sg13g2/libs.tech/ngspice/models/`:

| File | Sections (subset) |
|---|---|
| `cornerMOSlv.lib` | `mos_tt`, `mos_ss`, `mos_ff`, `mos_sf`, `mos_fs` (+ `_mismatch`, `_stat`) |
| `cornerMOShv.lib` | HV equivalents |
| `cornerRES.lib` | `res_typ`, `res_bcs`, `res_wcs` (+ `_mismatch`, `_stat`) |
| `cornerCAP.lib` | `cap_typ`, `cap_bcs`, `cap_wcs` (+ `_mismatch`) |

The YAML's `pvt_map.tt` already references `mos_tt`@`cornerMOSlv.lib` **and** `res_typ`@`cornerRES.lib` ([project_setup.yaml:41-46](examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml#L41-L46)) — both real sections — confirming one logical corner spans multiple device families.

**The seam already available in spicelib** (vendored `BaseEditor`, the parent of `SpiceEditor`):

| Hook | Location | Use |
|---|---|---|
| `set_parameter(name, value)` | [base_editor.py:492](#) | override `.param VDD`, `.param temp` |
| `add_instruction(instr)` | [base_editor.py:732](#) | append `.lib …`, `.options temp=…` |
| `remove_Xinstruction(regex)` | [base_editor.py:775](#) | strip the hardcoded `.lib …` line before re-adding |
| `.LIB` recognized directive | [base_editor.py:53](#) | — |

The **per-trial** sim entry point is `simulate_circuit()` ([base.py:479-517](src/spicexplorer/optimization/base.py#L479-L517)) → `wrapper.update_params()` → run. The **one-time** setup point is `Spice_Base_Optimizer.__post_init__()` ([base.py:463-475](src/spicexplorer/optimization/base.py#L463-L475)), where testbench params are already applied once before the loop.

### Proposed config schema

Replace the broken `tech_spec.pvt_map` + unused flat `pvt_corners` with a single top-level `pvt:` block. A **corner** is a self-contained bundle: process selections across device families + environment (temp, supply, extra params). Corners are **named** and referenced by name. The schema supports both operating modes:

- **(i) enumerate** — list many corners, mark which are `enabled`, run them all (Phase 2);
- **(ii) switch** — set a single `active_corner` and optimize against just that one (Phase 1).

Copy-pasteable, extending [`project_setup.yaml`](examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml):

```yaml
# ------------------------------------------------------------------
# PVT — process/voltage/temperature corner definitions
# ------------------------------------------------------------------
pvt:
  # Which corner the optimizer runs against in single-corner mode (Phase 1).
  # Must match one corner's `name` below. May be overridden ephemerally by the UI.
  active_corner: tt_25C_1V2

  # Reusable named PROCESS bundles: an ordered list of model-include directives.
  # Each directive is a generic (file, section) pair — PDK-agnostic. The strings
  # are ihp-sg13g2's here, but core never interprets them.
  process_bundles:
    tt:
      - { lib_file: cornerMOSlv.lib, section: mos_tt }
      - { lib_file: cornerRES.lib,   section: res_typ }
      - { lib_file: cornerCAP.lib,   section: cap_typ }
    ss:
      - { lib_file: cornerMOSlv.lib, section: mos_ss }
      - { lib_file: cornerRES.lib,   section: res_wcs }
      - { lib_file: cornerCAP.lib,   section: cap_wcs }
    ff:
      - { lib_file: cornerMOSlv.lib, section: mos_ff }
      - { lib_file: cornerRES.lib,   section: res_bcs }
      - { lib_file: cornerCAP.lib,   section: cap_bcs }

  # CORNERS = process bundle + environment. `process:` references a bundle by key,
  # OR inline `model_includes:` for a one-off corner.
  corners:
    - name: tt_25C_1V2
      process: tt                          # reference into process_bundles
      temp: 25                             # °C  → emitted as `.options temp=25`
      supply: { node: VDD, value: 1.2 }    # overrides `.param VDD`
      params: {}                           # optional extra per-corner .param overrides
      enabled: true

    - name: ss_125C_1V08                   # slow-slow, hot, -10% supply
      process: ss
      temp: 125
      supply: { node: VDD, value: 1.08 }
      enabled: true

    - name: ff_m40C_1V32                   # fast-fast, cold, +10% supply
      process: ff
      temp: -40
      supply: { node: VDD, value: 1.32 }
      enabled: false                       # defined but excluded from sweeps

    - name: tt_mismatch_27C                # inline one-off (no bundle reference)
      model_includes:
        - { lib_file: cornerMOSlv.lib, section: mos_tt_mismatch }
        - { lib_file: cornerRES.lib,   section: res_typ }
      temp: 27
      supply: { node: VDD, value: 1.2 }
      enabled: false

  # OPTIONAL: where the .lib files live, so corners are portable. If omitted,
  # the netlist's own `.lib` search path / PDK env is used (current behavior).
  model_lib_root: null   # e.g. "${PDK_ROOT}/ihp-sg13g2/libs.tech/ngspice/models"
```

Schema notes:

- `supply` is `{node, value}` so a multi-rail design (`VDD`, `VDDH`) can extend to `supplies: [...]` without breaking the schema.
- `params: {}` is the extensibility escape hatch for any other per-corner condition (bias currents, body bias, …).
- `process_bundles` (named, reusable) directly replaces the old `pvt_map` intent but is **actually consumed**, and avoids repeating the cross-family include list in every corner.
- This **subsumes and replaces** `tech_spec.pvt_map` and the flat `pvt_corners`. A backward-compat shim can translate legacy `pvt_corners` + `pvt_map` into a `pvt:` block; new projects use `pvt:`.

### PDK-agnostic abstraction boundary

The seam is a generic **`ModelInclude`** directive + a **`Corner`** environment record. Core never sees `ihp-sg13g2`, `cornerMOSlv.lib`, `mos_tt`, or the notion of "MOS / RES / CAP families."

**Lives in CORE (PDK-agnostic), [`core/domains.py`](src/spicexplorer/core/domains.py):**

```python
@dataclass
class ModelInclude:
    """One generic model-library include: opaque (file, section)."""
    lib_file: str          # opaque token; core emits `.lib <file> <section>`
    section: str           # core never enumerates valid sections

@dataclass
class SupplyOverride:
    node: str              # e.g. "VDD" — the .param name to override
    value: float

@dataclass
class Corner:
    name: str
    model_includes: List[ModelInclude]   # bundle expanded at load time
    temp: float
    supplies: List[SupplyOverride]
    params: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class PVTConfig:
    active_corner: str
    corners: List[Corner]
    model_lib_root: Optional[str] = None
    # process_bundles are expanded at load → not retained as a field
    def get_active(self) -> Corner: ...
    def enabled_corners(self) -> List[Corner]: ...
    def get(self, name: str) -> Corner: ...
```

**Lives in DATA / CONFIG (PDK-specific):** the file names (`cornerMOSlv.lib`), section names (`mos_tt`, `res_typ`), family groupings, temperatures, and supply rail names — all inside the YAML `pvt:` block. **Adding a new PDK = writing a new `pvt:` block (data), zero core change.**

**The render/injection seam** is a single pure method that turns a `Corner` into SPICE edits, in the spice engine, so the simulator-specific emission (`.lib` vs Spectre `include`/`section`) is isolated there:

```python
# new method on NGSpice_Wrapper (spice_engine/spicelib.py)
def apply_corner(self, corner: Corner) -> None:
    ed = self.editor
    # 1. strip any hardcoded corner .lib lines already in the netlist
    ed.remove_Xinstruction(r"^\.lib\s+\S+\.lib\s+\w+")   # matches `.lib <file>.lib <section>`
    # 2. add the chosen process includes (ordered, cross-family)
    for inc in corner.model_includes:
        path = inc.lib_file if not self.model_lib_root else f"{self.model_lib_root}/{inc.lib_file}"
        ed.add_instruction(f".lib {path} {inc.section}")
    # 3. environment
    ed.add_instruction(f".options temp={corner.temp}")
    for s in corner.supplies:
        ed.set_parameter(s.node, s.value)        # override .param VDD
    for k, v in corner.params.items():
        ed.set_parameter(k, v)
```

This uses exactly the verified hooks — `remove_Xinstruction` ([base_editor.py:775](#)), `add_instruction` ([base_editor.py:732](#)), `set_parameter` ([base_editor.py:492](#)). The **only** ngspice-specific syntax (`.lib` / `.options temp`) is confined to this one method; a future Spectre backend overrides it.

### Cross-product execution model

The evaluation grid is **{enabled testbenches} × {enabled corners}**. Today it is implicitly `{testbenches} × {the one hardcoded corner}`.

- **Enumeration.** `simulate_circuit()` iterates `self.spicelib_wrappers.items()` (one wrapper per enabled testbench, built by [`orchestrator.create_spicelib_wrappers()` :85-124](src/spicexplorer/optimization/orchestrator.py#L85)) at [base.py:485](src/spicexplorer/optimization/base.py#L485). The cross-product nests a corner loop outside it: for each enabled `Corner`, `apply_corner(corner)` on each wrapper, run, collect.
- **Run labeling.** `NGSpice_Wrapper._move_to_run_folder(label=…)` ([spicelib.py:263](src/spicexplorer/spice_engine/spicelib.py#L263)) already names run folders. Pass a composite label `f"{tb_name}__{corner.name}"` so artifacts are physically keyed by (testbench, corner).
- **Result keying.** Results are `Dict[str, RawRead]` keyed by `tb_name` ([base.py:481](src/spicexplorer/optimization/base.py#L481)). Re-key to `Dict[Tuple[str, str], RawRead]` = `(tb_name, corner_name)`. Metric extraction at [base.py:850-853](src/spicexplorer/optimization/base.py#L850-L853) (`self.spicelib_wrappers[target.testbench].extract_scalar_variable_from_raw(...)`) becomes `performance_array[(corner_name, target.name)] = …`, so each spec yields one value **per corner**.
- **Checkpoint shape.** `fit_summary` ([base.py:899-902](src/spicexplorer/optimization/base.py#L899-L902)) is keyed by spec name today; under multi-corner it becomes `f"{corner_name}::{spec_name}"`. Single-corner keys stay effectively unchanged. **Caveat:** dotted/`::` keys interact with the existing CSV-column-mangling issue handled by `.iterrows()` in [checkpoint_reader.py](ui/backend/services/checkpoint_reader.py) — keep that.

**Important:** the cross-product only changes *how many* `(metric, value)` pairs feed the scorer. How those collapse into one scalar objective is the Phase 2 open question.

### Phasing

#### Phase 1 (this round) — single chosen corner

**Goal:** corners become first-class and actually drive the sim, but the optimization loop, scorer, and per-trial flow are **untouched**. Only the *one-time* netlist preparation gains a corner-apply step. Load `pvt.active_corner`, resolve to a `Corner`, apply its `.lib`/temp/supply to each wrapper **once**, before the loop. Every subsequent trial inherits the corner because `SpiceEditor` netlist state persists across runs.

**Exact change list:**

1. **[`core/domains.py`](src/spicexplorer/core/domains.py)**
   - Add `ModelInclude`, `SupplyOverride`, `Corner`, `PVTConfig` dataclasses (Part A → abstraction boundary).
   - Add field `pvt: Optional[PVTConfig] = None` to `Project_Setup` ([domains.py:595-616](src/spicexplorer/core/domains.py#L595-L616)).
   - In `from_yaml()` ([domains.py:638-682](src/spicexplorer/core/domains.py#L638-L682)), **before** `safe_from_dict`, pre-process the raw `proj` dict to expand each corner's `process:` bundle reference into a resolved `model_includes` list (bundles are sugar, not a retained field). Insert this near the existing `ws_root` resolution at [domains.py:655-661](src/spicexplorer/core/domains.py#L655-L661).
   - Leave `TechSpec` ([domains.py:149](src/spicexplorer/core/domains.py#L149)), `PVT` ([domains.py:161](src/spicexplorer/core/domains.py#L161)), and `pvt_corners` ([domains.py:607](src/spicexplorer/core/domains.py#L607)) **as-is** so existing YAMLs don't break (they remain silently ignored — no regression). Optionally have `summary()` also log `pvt.active_corner` ([domains.py:745](src/spicexplorer/core/domains.py#L745)).

2. **[`spice_engine/spicelib.py`](src/spicexplorer/spice_engine/spicelib.py)**
   - Add `NGSpice_Wrapper.apply_corner(self, corner: Corner)` (body above), plus an optional `self.model_lib_root` set in `__init__`. **No change** to `update_params` ([spicelib.py:291](src/spicexplorer/spice_engine/spicelib.py#L291)), `run_and_wait` ([spicelib.py:388](src/spicexplorer/spice_engine/spicelib.py#L388)), or run-folder logic.

3. **[`optimization/base.py`](src/spicexplorer/optimization/base.py)**
   - In `Spice_Base_Optimizer.__post_init__()` ([base.py:463-475](src/spicexplorer/optimization/base.py#L463-L475)) — the existing one-time tb-param setup — add one block after the param loop:
     ```python
     if getattr(self.setup_obj, "pvt", None) is not None:
         corner = self.setup_obj.pvt.get_active()
         for tb in self.setup_obj.testbenches:
             if tb.enable:
                 self.spicelib_wrappers[tb.name].apply_corner(corner)
     ```

4. **[`optimization/orchestrator.py`](src/spicexplorer/optimization/orchestrator.py)** — **no change required** (it builds wrappers; the corner is applied by the optimizer's `__post_init__`). Optionally thread `model_lib_root` into the `NGSpice_Wrapper(...)` constructor ([orchestrator.py:115](src/spicexplorer/optimization/orchestrator.py#L115)).

5. **UI (optional, ephemeral override mirroring the existing `_apply_overrides` pattern):**
   - [`routes/project.py:78-93`](ui/backend/routes/project.py#L78-L93) — surface `pvt.active_corner` + the corner list (currently returns only `pvt_corners` temp/corner/supply).
   - `services/optimizer_runner.py` `_apply_overrides` — accept an ephemeral `active_corner` override (same in-memory, never-rewrite-YAML pattern documented in CLAUDE.md). This gives operating-mode (ii) "switch to a defined corner" for free.

**Stays untouched (optimizer core):**

- The nevergrad / Ax optimizer classes ([stochastic/nevergrad.py](src/spicexplorer/optimization/stochastic/nevergrad.py), [stochastic/bayesian_ax.py](src/spicexplorer/optimization/stochastic/bayesian_ax.py)).
- `optimization_step` ([nevergrad.py:176-184](src/spicexplorer/optimization/stochastic/nevergrad.py#L176-L184)), the `optimize()` loop, `simulate_circuit()` ([base.py:479-517](src/spicexplorer/optimization/base.py#L479-L517)), `evaluate()` ([base.py:837](src/spicexplorer/optimization/base.py#L837)).
- `compute_fitness()` / `compute_fitness_for_spec()` scoring math ([base.py:877-913](src/spicexplorer/optimization/base.py#L877-L913)), `core/utils.py` error/reward functions.
- Checkpoint schema, all charts. Single-corner runs are a strict superset of today (today ≈ "active_corner = whatever the netlist hardcodes").

#### Phase 2 (deferred, research) — multi-corner aggregation

Phase 1 makes one corner drive the sim. Phase 2 runs the **{tb × corner} cross-product** and must collapse **N corner-scores into one scalar** that the optimizer consumes (`self.optimizer.tell(candidate, -1 * curr_score)`, [nevergrad.py:183](src/spicexplorer/optimization/stochastic/nevergrad.py#L183)).

**The open question:** `compute_fitness()` ([base.py:877-913](src/spicexplorer/optimization/base.py#L877-L913)) produces one `total_score` from one `performance_array`. With multiple corners there are multiple `performance_array`s. How they combine sets the optimizer's incentive structure. Candidate strategies (**not committing to one**):

| Strategy | Behavior | Trade-off |
|---|---|---|
| **Worst-case / min** | `score = min_c score_c` | Robust, classic for PVT sign-off; pessimistic, can stall on one pathological corner. |
| **Weighted mean** | `score = Σ w_c · score_c` | Smooth gradient; can mask a single failing corner. |
| **Sum of penalties** | accumulate per-corner penalties before reward | Aligns with the existing penalty/reward split ([base.py:904-908](src/spicexplorer/optimization/base.py#L904-L908)); double-counts shared violations. |
| **Must-pass-all (constraint)** | any corner missing spec → large penalty (`MAX_PENALTY`, [base.py:43](src/spicexplorer/optimization/base.py#L43)); reward only when all pass | Matches real tape-out sign-off semantics. |
| **Pareto / multi-objective** | keep corners as separate objectives | Natural fit for the Ax backend; changes the optimizer contract from scalar `tell` to multi-objective. |

Secondary Phase 2 items: where the corner loop lives (inside `simulate_circuit` vs. a new outer driver); parallelism — the existing `parallel_sim` path ([base.py:506-516](src/spicexplorer/optimization/base.py#L506-L516)) would now fan out tb × corner (N× more concurrent ngspice processes); and the `fit_summary` / checkpoint key namespacing (`{corner}::{spec}`) plus the dotted-column caveat in [checkpoint_reader.py](ui/backend/services/checkpoint_reader.py). **All out of scope for this round.**

---

## Part B — Manual simulation feature (shares sim infra)

**Goal:** "evaluate one design point on demand" — run all enabled testbenches once for a user- or checkpoint-supplied param vector and return metrics + score. **No new sim path is needed**; the optimizer already contains the exact primitive.

### The shared primitive

`Spice_Constraint_Satisfaction.evaluate(parameterization, append_to_log=False)` ([base.py:837](src/spicexplorer/optimization/base.py#L837)) does precisely the four steps a manual sim needs:

1. `simulate_circuit(parameterization)` — injects params into every enabled testbench and runs ngspice once each ([base.py:844](src/spicexplorer/optimization/base.py#L844), loop at [base.py:485](src/spicexplorer/optimization/base.py#L485) — no repetition, run-count is structurally 1).
2. extracts each `TargetSpec`'s scalar metric from the RAW ([base.py:850-853](src/spicexplorer/optimization/base.py#L850-L853)).
3. `compute_fitness(...)` → `(total_score, fit_summary)` ([base.py:857](src/spicexplorer/optimization/base.py#L857)).
4. `append_to_log=False` skips polluting `optimization_log`; `clean_up(delete_raw_only=True)` runs ([base.py:873](src/spicexplorer/optimization/base.py#L873)).

`Spice_Base_Optimizer.plot_solution` already calls `self.evaluate(parameterization, append_to_log=False)` — so one-shot evaluation of a given vector is an **existing, exercised** path. **Key subtlety:** `evaluate` expects **denormalized engineering-real values** (e.g. `x_dut_W_1 = 72e-6`); `denormalize_params` ([nevergrad.py:180](src/spicexplorer/optimization/stochastic/nevergrad.py#L180)) only exists to undo the optimizer's normalized space. Manual sim **bypasses `denormalize_params`** and feeds real values straight to `evaluate`. (`update_params` adds `k`/`p` suffixes for `C*`/`R*`-prefixed names — [spicelib.py:291](src/spicexplorer/spice_engine/spicelib.py#L291) — so manual values must follow the same convention.)

### Data flow

```
[ Studio center view (OptimizeTab) "Manual Sim" panel ]
        │  param vector (engineering-real) + yaml_path + source mode
        ▼
POST /api/simulate/once   (new route, ui/backend/routes/simulate.py — sibling of sanity.py)
        │  loop.run_in_executor (sync, like sanity.py:241)
        ▼
_run_single_sim(yaml_path, params):
   Project_Setup.from_yaml(yaml_path)                          # domains.py:639
   wrappers = _build_spicelib_wrappers(project)                # reuse optimizer_runner.py
   opt = Nevergrad_Spice_Single_Objective(project, wrappers)   # nevergrad.py:205  (NO parameterize/optimize)
   score, fit_summary = opt.evaluate(params, append_to_log=False)   # base.py:837  ← all enabled TBs once
        │  { ok, score, metrics{spec:{curr_val,score,passes}}, params_used, log_tails, elapsed_ms, pdk_ok }
        ▼
[ Per-spec table (value vs target, pass/fail), total score, log tails ]
```

### Integration points

**Reuse (no change):**

| What | Location |
|---|---|
| `evaluate(parameterization, append_to_log=False)` — the primitive | [base.py:837](src/spicexplorer/optimization/base.py#L837) |
| `Nevergrad_Spice_Single_Objective(setup_obj, spicelib_wrappers)` — instantiate only | [nevergrad.py:205](src/spicexplorer/optimization/stochastic/nevergrad.py#L205) |
| wrapper construction (`_build_spicelib_wrappers`) — promote to shared helper | [optimizer_runner.py](ui/backend/services/optimizer_runner.py) (mirrored at [sanity.py:110-125](ui/backend/routes/sanity.py#L110-L125)) |
| PDK probe + log-tail helpers | `probe_pdk` ([sanity.py:84,91](ui/backend/routes/sanity.py#L84)) + `_tail_log` ([sanity.py:57](ui/backend/routes/sanity.py#L57)) |

Do **not** call `parameterize()` / `_create_optimizer_obj()` / `optimize()` — they build the nevergrad optimizer and pick a **random** candidate via `ask()` ([nevergrad.py:178](src/spicexplorer/optimization/stochastic/nevergrad.py#L178)), which is wrong for "evaluate THIS point."

**New backend (minimal):**

1. `ui/backend/routes/simulate.py` → `POST /api/simulate/once`: request `{ yaml_path, params }` (or `{ yaml_path, checkpoint_id, point }`); body = `_run_single_sim` above; mirror sanity.py's `run_in_executor` + PDK fold-in + try/except shape ([sanity.py:208-235](ui/backend/routes/sanity.py#L208-L235)).
2. Register the router in `ui/backend/main.py`.
3. [`ui/src/lib/api.ts`](ui/src/lib/api.ts): add `simulateOnce(...)` modeled on `sanityCheck` ([api.ts:130](ui/src/lib/api.ts#L130)).
4. `ui/src/types/api.ts`: add a `SimulateOnceResponse` type.

### Two input modes

**Mode A — load a prior optimization result.** A checkpoint row carries the full param vector. `read_json_checkpoint` reads `entry.get_params()` into `params_out[name] = [values...]` ([checkpoint_reader.py:47-48](ui/backend/services/checkpoint_reader.py#L47-L48)); `read_csv_checkpoint` parses `point.params.<name>` columns ([checkpoint_reader.py:64-80](ui/backend/services/checkpoint_reader.py#L64)). These values are written by `evaluate` as `params=parameterization` ([base.py:863](src/spicexplorer/optimization/base.py#L863)) — already **engineering-real**, so they feed `evaluate` with **no transform**. Pick the best iteration by argmax of `scores`. Cleanest: have the route accept `{checkpoint_id, point}` and resolve server-side via `_resolve_checkpoint_path` ([checkpoint.py:24](ui/backend/routes/checkpoint.py#L24)) + `read_checkpoint` ([checkpoint_reader.py:107](ui/backend/services/checkpoint_reader.py#L107)). Re-simulating the best point should reproduce its stored `curr_val`s — doubling as a result-validation tool.

**Mode B — manual user-supplied values.** Pre-fill a form from `project.dut_params` using each `Param.init` ([domains.py:173](src/spicexplorer/core/domains.py#L173)) as default, with `min_val`/`max_val` ([domains.py:170-171](src/spicexplorer/core/domains.py#L170-L171)) as bound hints. A **partial** dict is valid: `update_params` only sets provided keys; unset params keep their netlist `.param` defaults. Respect `is_integer` and the `C*`/`R*` suffix convention. Both modes converge on the identical `evaluate(params, append_to_log=False)` call.

**Scoring identity:** a manual sim through `evaluate` produces the **same `fit_summary` and `score`** as a real trial (identical `compute_fitness`), so results are directly comparable to checkpoint rows and to Score Shaping (`score_service.compute_score` uses the same `core/utils` error fns).

### Required UI changes

- **Placement:** a collapsible "Manual Sim / Evaluate Point" panel in **OptimizeTab** ([components/tabs/OptimizeTab.tsx](ui/src/components/tabs/OptimizeTab.tsx)) — design-centric, PDK-gated, next to live runs and the runs rail (which lists checkpoints for Mode A). (HealthTab is where `sanityCheck` lives, but it's diagnostics, not design iteration.)
- **Controls:** a `Segmented` source toggle ("From checkpoint" | "Manual entry"); Mode A → a `Select` of checkpoints (from `api.listCheckpoints`, [api.ts:105](ui/src/lib/api.ts#L105)) + "Best point" default; Mode B → a `Table` of dut_params with numeric inputs pre-filled from `init` (+ min/max/integer hints, "Reset to init"). A primary "Simulate" `Button` disabled when `pdk_ok === false` (reuse the OptimizeTab live-Start gating). Spinner/elapsed during the synchronous call.
- **Result display:** per-spec table (spec, `curr_val` via `formatEng`, target, goal, pass/fail badge, per-spec score) + total score `Stat`; collapsible per-testbench log tails (reuse the HealthTab pattern). Optionally a "Send to Score Shaping" action handing `metrics{spec: curr_val}` to `computeScore` ([api.ts:67](ui/src/lib/api.ts#L67)).

### How close is the existing sanity path?

`ui/backend/routes/sanity.py:_run_sanity` ([sanity.py:80](ui/backend/routes/sanity.py#L80)) is **~90% of the feature**: it loads the project, builds wrappers per enabled testbench, instantiates `Nevergrad_Spice_Single_Objective` ([sanity.py:168](ui/backend/routes/sanity.py#L168)), then calls `parameterize()` + `_create_optimizer_obj()` + `optimization_step()` ([sanity.py:169-180](ui/backend/routes/sanity.py#L169-L180)), returning `score` + per-spec `metrics` ([sanity.py:185-207](ui/backend/routes/sanity.py#L185-L207)).

What blocks it from *being* the feature:

1. **The param vector is random, not chosen.** `optimization_step` calls `self.optimizer.ask()` ([nevergrad.py:178](src/spicexplorer/optimization/stochastic/nevergrad.py#L178)) — it sims a random in-bounds point. **This is the core gap.**
2. It pays for `_create_optimizer_obj()` purely as a precondition for `ask()`/`tell()` — unneeded for a chosen point.
3. It runs `run_sanity_check` (a no-metric smoke sim) on **every** testbench first ([sanity.py:122](ui/backend/routes/sanity.py#L122)) before the trial — extra work a manual sim doesn't want.
4. **It does not return a usable param vector.** `optimization_step` returns `candidate.value` — the **normalized** value ([nevergrad.py:184](src/spicexplorer/optimization/stochastic/nevergrad.py#L184)) — and sanity.py discards it (`_params` at [sanity.py:180](ui/backend/routes/sanity.py#L180) is unused). So you can't reproduce the simulated point.

→ The manual-sim route is a **sibling** of `sanity.py` that (a) accepts a param vector, (b) calls `evaluate(vector, append_to_log=False)` instead of `optimization_step()`, (c) skips the per-tb `run_sanity_check` and `_create_optimizer_obj`, (d) echoes `params_used` back. Wrapper construction, PDK fold-in, log-tail handling, and `run_in_executor` are directly copyable.

### Interface gaps

1. **No "evaluate a chosen point" façade.** `evaluate(append_to_log=False)` exists but is only reachable on a fully-constructed optimizer. A `Base_Optimizer.simulate_point(params) -> (score, fit_summary)` convenience method would be the clean library seam; the route can construct the optimizer directly in the meantime.
2. **No input-domain validation.** `evaluate` trusts real values; nothing checks they lie within `min_val`/`max_val` ([domains.py:170-171](src/spicexplorer/core/domains.py#L170-L171)) or respects `is_integer`. The route should range-check.
3. **`freeze` / `init` are dead in the optimizer path.** `parameterize()` iterates **all** `dut_params` and ignores `freeze` ([nevergrad.py:127-130](src/spicexplorer/optimization/stochastic/nevergrad.py#L127-L130)); `init` never seeds initial points (the `suggest()` block is commented out, [nevergrad.py:168](src/spicexplorer/optimization/stochastic/nevergrad.py#L168)). So "a full design vector" = all dut_params; manual sim should treat all as inputs and use `init` only as a form default. Worth flagging that `freeze` is effectively dead.
4. **No best-point accessor on a loaded checkpoint.** JSON checkpoints don't persist a best index, so "best point" must be recomputed by argmax of `scores`. An `OptimizationLog.get_best_point()` (or persisting the index) would make it exact.
5. **No corner/PVT selection seam (today).** Corners are hardcoded in netlists; a manual sim inherits whatever the netlist bakes in — **until Part A Phase 1 lands**, after which a manual sim can reuse `apply_corner(active_corner)` for free. Per-corner manual sim is the natural follow-on.
6. **Wrapper construction is destructive.** `_validate` rmtree's `output_folder` on construction ([spicelib.py:228-236](src/spicexplorer/spice_engine/spicelib.py#L228)), and `output_folder = ws_root/outdir` is shared with live runs. Running manual sim while a live run is active would clobber outputs. *Mitigation:* give manual-sim wrappers a distinct subfolder (e.g. `outdir/manual_sim`) or serialize against live runs.

---

## Load-bearing findings (summary)

- **`pvt_map` is dead config** — silently dropped by non-strict dacite ([DECITE_CONFIG domains.py:757](src/spicexplorer/core/domains.py#L757)); `TechSpec` has no such field; zero `src/` consumers.
- **`pvt_corners` is parsed but never drives a sim** — only logged ([domains.py:745](src/spicexplorer/core/domains.py#L745)) and displayed ([project.py:79](ui/backend/routes/project.py#L79)).
- **No `.lib`/temp/supply injection exists** — the engine only does `set_parameter` ([spicelib.py:291](src/spicexplorer/spice_engine/spicelib.py#L291)); corner/temp/VDD are hardcoded in the netlists.
- **A corner = ordered set of `(lib_file, section)`** across MOS/RES/CAP — verified real in the `cornerMOS/RES/CAP.lib` files.
- **Clean Phase-1 seam:** add `apply_corner()` to `NGSpice_Wrapper` (using verified `add_instruction`/`remove_Xinstruction`/`set_parameter`) and call it once in `Spice_Base_Optimizer.__post_init__` ([base.py:463](src/spicexplorer/optimization/base.py#L463)); the optimize loop, scorer, and `simulate_circuit` stay untouched.
- **Manual sim reuses `evaluate(params, append_to_log=False)`** ([base.py:837](src/spicexplorer/optimization/base.py#L837)) verbatim — the only new code is a thin route + UI; `sanity.py` is the 90%-there template, blocked only by its random `ask()` point and its discarded/normalized vector.
- **Multi-corner aggregation is the genuine research item** — deferred to Phase 2.

> **Data bug flagged (orthogonal):** [`project_setup.yaml`](examples/OTA/folded_cascode/ihp-sg13g2/sizing/project_setup.yaml) defines `x_dut_Vb1` **twice** (last-wins in the `dut_params` list, since `parameterize` and the manual form both iterate `dut_params`), and `x_dut_Vb2` is used in the netlist but never declared as a DUT param. Worth fixing before relying on the manual form's name→value mapping.
