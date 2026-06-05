# SpiceXplorer Webapp — Functional Bug Audit (static re-audit, current HEAD)

> **Static analysis only.** No app/docker/ngspice/uvicorn/npm/pytest was run this round. Every finding below was re-derived by reading the file **at HEAD** (`e6b68ed`, branch `feat/pvt`) and confirming the cited code is still present. Items the prior audit raised that are now fixed are not re-reported as active — they are listed separately under *Previously reported — now fixed*. Findings whose final confirmation would require running the app/sim are marked **needs runtime verification (deferred — no docker/live-sim on server this round)**.
>
> **Server grounding (do not re-test):** ngspice / xschem / openvaf are installed natively; the PDK is at `/home/noorizad/local/pdks` with `PDK=ihp-sg13g2`, `PDK_ROOT=/home/noorizad/local/pdks`, and the 40 `sg13g2_pr` `.sym` files exist under `libs.tech/xschem/sg13g2_pr/`. So on **this** server the PDK is present and live runs / symbol resolution work here — unlike the prior Docker/PDK-less audit. Several findings below are therefore **latent/config-conditional**: real code-level defects that only bite a different (but supported) environment or an as-yet-unshipped config.

**Totals (active, this round):** 17 active findings — **4 major, 13 minor** — plus the two pre-flagged regressions, covered first (one of them, Score Shaping, is **already fixed** at HEAD and is documented as such, not counted in the 17).

---

## 1. Webapp map

SpiceXplorer's core workflow is **YAML → `Project_Setup.from_yaml()` → `Nevergrad_Spice_Single_Objective` → `NGSpice_Wrapper` → ngspice → scorer → checkpoints + Plotly reports**; the web UI is a thin shell over that library.

### Topology

```
Browser :4000                  FastAPI :8000                 spicexplorer library            ngspice
─────────────────  REST+SSE  ─────────────────  imports  ──────────────────────  subprocess  ───────
Next.js 15            ───►     ui/backend/          ───►    src/spicexplorer/         ───►     (PDK:
(App Router, ui/src,          main.py mounts 12            core.domains.Project_Setup          ihp-sg13g2 at
 Zustand, EventSource)        routers under /api          → optimization.stochastic           $PDK_ROOT;
                              + GET /health                 .nevergrad                          live_runs_enabled
                                                          → spice_engine.spicelib              = true here)
```

Cross-cutting: a live run executes in a **background daemon thread** (`optimizer_runner._run_live`) that pushes events onto an `asyncio.Queue` via `run_coroutine_threadsafe`; `GET /api/optimize/stream/{run_id}` (SSE) drains it (60 s heartbeat, `None` sentinel = done). Replay drip-feeds checkpoint rows at 50 ms with no PDK needed. `next.config.mjs` rewrites `/api/*` to the backend (afterFiles), so all `/api` traffic hits FastAPI.

### Frontend (`ui/src`)

- **Studio shell** — [(studio)/layout.tsx](ui/src/app/(studio)/layout.tsx) renders `StudioShell` (activity bar + left rail + tab strip + right rail + bottom panel + status bar + overlays). Each `(studio)/<view>/page.tsx` renders one center view; the layout never remounts, so rails + the live SSE stream survive navigation. [app/page.tsx](ui/src/app/page.tsx) redirects to `/setup`. [components/shell/nav.ts](ui/src/components/shell/nav.ts) is the single source of truth for the 7 views (setup / scoring / optimize / compare / schematic / pipeline / health).
- **Views** ([components/tabs/](ui/src/components/tabs/)): `SetupTab`, `ScoreShapingTab`, `OptimizeTab`, `ExplorerTab` (compare), `SchematicTab`, `PipelineView`, `HealthTab`.
- **Stores** (Zustand, [stores/](ui/src/stores/)): `projectStore` (loaded/applied project: `summary`, `yamlPath`, `isApplied`), `runStore` (active run **and** the SSE `EventSource` hoisted module-level, `history` persisted to localStorage), `explorerStore` (compare A/B), `uiStore` (`selectedSpec`/`selectedRunId`, panel toggles, overlay flags, the shared ephemeral `RunConfig`), `wizardStore` (new-project form across 7 steps).
- **PVT** ([components/pvt/](ui/src/components/pvt/)): `CornerSelect` (Run popover / Optimize toolbar / Health), `ManualSimPanel` (single-point sim → `/api/simulate/once`), [lib/pvt.ts](ui/src/lib/pvt.ts).
- **Plumbing**: [lib/api.ts](ui/src/lib/api.ts) (single typed fetch client), [types/api.ts](ui/src/types/api.ts) (mirrors every FastAPI shape), [lib/launchRun.ts](ui/src/lib/launchRun.ts), [lib/xschem/](ui/src/lib/xschem/) (`parser`/`render`/`types`).
- **Charts** ([components/charts/](ui/src/components/charts/)): `PlotlyChart` base (react-plotly.js, SSR-disabled) + `MetricConvergence`/`MetricHistogram`/`MetricScatter`/`ScoreConvergence`/`PenaltyCurve`/`Sensitivity`.
- **Wizard** ([components/wizard/](ui/src/components/wizard/)): `WizardShell` + `steps/*` (basic/pdk/dut/pvt/testbenches/specs/optimizer) + `optimizer-registry.ts`.

### Backend (`ui/backend`) — 12 routers under `/api` + `GET /health`

| Method | Path | Purpose | File |
|---|---|---|---|
| GET | `/api/config` | demo presets + default yaml | [config.py](ui/backend/routes/config.py) |
| GET | `/api/env` | ngspice + PDK probe; `live_runs_enabled` | [env.py](ui/backend/routes/env.py) |
| POST/GET | `/api/project/load` · `/yaml-text` · `/validate` · `/generate` · `/parse-to-form` | load + wizard ↔ YAML | [project.py](ui/backend/routes/project.py) |
| POST | `/api/score` | per-spec + aggregate penalties | [score.py](ui/backend/routes/score.py) |
| POST/GET | `/api/optimize/start` · `/stop/{id}` · `/stream/{id}` | launch / stop / SSE | [optimize.py](ui/backend/routes/optimize.py) |
| GET/DELETE | `/api/checkpoint[...]` | list / load / envelope / scatter / delete | [checkpoint.py](ui/backend/routes/checkpoint.py) |
| GET | `/api/schematic` · `/xschem/{file,resolve,list,project}` | schematic + symbol resolution | [schematic.py](ui/backend/routes/schematic.py) · [xschem.py](ui/backend/routes/xschem.py) |
| POST | `/api/sanity-check` | per-tb sanity + one optimizer trial | [sanity.py](ui/backend/routes/sanity.py) |
| POST | `/api/netlist/parse` | extract `.param` rows | [netlist.py](ui/backend/routes/netlist.py) |
| GET | `/api/spec/{name}/sensitivity` | finite-diff metric vs DUT param | [sensitivity.py](ui/backend/routes/sensitivity.py) |
| POST | `/api/simulate/once` | evaluate ONE chosen design point | [simulate.py](ui/backend/routes/simulate.py) |

### Backend services ([ui/backend/services/](ui/backend/services/))

- **`optimizer_runner.py`** — `_run_live` (daemon thread; subclass `_StreamingOpt`; pushes SSE events to an `asyncio.Queue`), `_run_replay` (drip-feed), `_apply_overrides` (algorithm/budget/seed/active_corner), `_build_spicelib_wrappers`, module-level `_runs` registry.
- **`env_probe.py`** — `probe_ngspice` / `probe_pdk` / `probe_env`; `live_runs_enabled = ngspice_ok and pdk_ok`.
- **`checkpoint_reader.py`** — CSV via `.iterrows()` (dotted columns like `point.score`); JSON via `Optimization_Log_Visualizer`; `compute_envelope` / `compute_scatter`.
- **`score_service.py`** — per-spec linear/sigmoid penalties + weighted aggregate + penalty curve.
- **`yaml_generator.py`** (wizard form ↔ YAML), **`netlist_parser.py`** (`.param` regex), **`num.py`** (`safe_float`), **`app_config.py`** (`REPO_ROOT`, preset paths).

### Library (`src/spicexplorer`)

- **`core/domains.py`** — `Project_Setup.from_yaml` (resolves `ws_root`, `_normalize_pvt_block`, rejects duplicate `dut_param` names), `Param` (`freeze=False` default, `resolve_min_max` / `ressolve_val`), `TargetSpec`/`ListTargetSpec`, `Corner`/`PVTConfig` (`.get_active`).
- **`core/utils.py`** — engineering-string parsing + `compute_relative_absolute_error` / `compute_relative_sigmoid_error`.
- **`spice_engine/spicelib.py`** — `NGSpice_Wrapper` (param injection, subprocess, `_validate` rmtree, `apply_corner` — the only ngspice PVT seam).
- **`optimization/base.py`** (`Spice_Base_Optimizer.__post_init__` applies `pvt.get_active()` once; `evaluate(params, append_to_log)` primitive), **`optimization/stochastic/nevergrad.py`** (`parameterize` skips frozen params; `optimization_step`).

### Key end-to-end flows

1. **Load/Apply** — SetupTab → `POST /api/project/load` (path or content) → `_summarise` → `projectStore.apply(summary, yamlPath)` unlocks `requiresProject` views.
2. **Live run** — OptimizeTab/RunControl → `launchRun.startLive` → `POST /api/optimize/start` → `_run_live` thread (`_StreamingOpt`) → SSE `/api/optimize/stream/{id}` → `runStore.pushEvent`.
3. **Replay** — preset checkpoint → `_run_replay` drip → same SSE.
4. **Manual sim** — `ManualSimPanel` → `POST /api/simulate/once` → `opt.evaluate(params, append_to_log=False)`.
5. **Score shaping** — ScoreShapingTab → `POST /api/score` → `score_service` (reloads YAML each call).
6. **Schematic** — SchematicTab → `/api/xschem/project` + `/file` + `/resolve` → `parser` → `SchematicViewer`; DeviceInspector → `/api/spec/{name}/sensitivity`.
7. **Health / env** — StudioShell hydrates `/api/env` once; HealthTab → `/api/sanity-check`.

---

## 2. Confirmed flagged issues

The two issues called out in the audit brief, each traced end-to-end against HEAD.

### FLAG-1 · Device symbols render as "missing symbol" placeholders when the backend has `PDK_ROOT` but not the `PDK` env var (env-contract divergence vs. the sim/PDK probe)

- **Area:** Schematic / xschem rendering (backend route + env probe)
- **Location:** [xschem.py:47-54](ui/backend/routes/xschem.py#L47-L54) (`_pdk_xschem_dir`, `@lru_cache` at :47) and [xschem.py:57-66](ui/backend/routes/xschem.py#L57-L66) (`_search_roots`); cf. [env_probe.py:25](ui/backend/services/env_probe.py#L25), [env_probe.py:42-58](ui/backend/services/env_probe.py#L42-L58), [env_probe.py:79-111](ui/backend/services/env_probe.py#L79-L111) (`probe_pdk` / `_candidate_pdk_roots` / `live_runs_enabled`). Frontend manifestation: [SchematicTab.tsx](ui/src/components/tabs/SchematicTab.tsx) (resolve loop + missing-symbol banner) and [SchematicViewer.tsx](ui/src/components/schematic/SchematicViewer.tsx) (`SymbolPlaceholder`).
- **Severity:** 🟡 **minor** (downgraded from the flag's "major": the defect is latent/config-conditional, see below) · **Confidence:** medium
- **Symptom:** In the Schematic tab the PDK device bodies (`sg13g2_pr/sg13_lv_nmos.sym`, `sg13g2_pr/sg13_lv_pmos.sym`, `sg13g2_pr/annotate_fet_params.sym`, …) draw as red "?" placeholder boxes and the amber "*N symbol(s) unavailable — the xschem symbol library (PDK) is not reachable*" banner appears, **even though live sims/sensitivity work** (`pdk_ok: true`). `devices/*.sym` and bare library symbols still resolve, so only the PDK-prefixed device symbols vanish.
- **Root cause:** The xschem symbol resolver and the PDK-presence probe use **divergent env contracts**:
  - `_pdk_xschem_dir()` ([xschem.py:48-54](ui/backend/routes/xschem.py#L48-L54)) reads `pdk_root = os.environ.get("PDK_ROOT")` **and** `pdk = os.environ.get("PDK")`, then `if not pdk_root or not pdk: return None`. It strictly requires **both** env vars and builds `Path(pdk_root)/pdk/libs.tech/xschem`. `_search_roots()` appends the PDK dir only when this returns non-`None`.
  - By contrast, `_PDK_ENV_VARS = ("PDK_ROOT", "PDK", "IHP_PDK_ROOT")` ([env_probe.py:25](ui/backend/services/env_probe.py#L25)) and `_candidate_pdk_roots()` also adds `app_config.pdk_root`; `probe_pdk()` returns `pdk_ok: true` as soon as `cornerMOSlv.lib` is found under **any single** candidate root, and `probe_env()` sets `live_runs_enabled = ngspice_ok and pdk_ok`.

  So a backend started with **only** `PDK_ROOT` (no `PDK`), or with `IHP_PDK_ROOT`, or via `app_config.pdk_root`, reports `pdk_ok: true` / `live_runs_enabled: true` while `_pdk_xschem_dir()` returns `None`, dropping the PDK root from the resolver path. A symref like `sg13g2_pr/sg13_lv_nmos.sym` then only tries `base_dir` (the project `xschem/` dir) and the xschem share libs — none of which contain `sg13g2_pr/` — so `/api/xschem/resolve` 404s and the frontend renders a `SymbolPlaceholder` and increments the "N symbols unavailable" banner. The `@lru_cache(maxsize=1)` on `_pdk_xschem_dir` / `_xschem_share_dirs` also freezes a `None` result for the process lifetime, so a late-exported `PDK` won't take effect without a restart.
- **Code path:** `SchematicTab` resolve loop → `api.xschemResolve(ref='sg13g2_pr/sg13_lv_nmos.sym', base=<abs .sch>)` → `GET /api/xschem/resolve` → `resolve_ref` → `_search_roots(base_dir)` → `_pdk_xschem_dir()` returns `None` (PDK unset) → candidate never matches → `HTTPException(404)` → frontend catch returns `null` → ref absent from `symMap` → `SchematicViewer` renders `SymbolPlaceholder` + banner. Parallel: `GET /api/env` → `probe_pdk` finds `cornerMOSlv.lib` via `PDK_ROOT` → `pdk_ok: true` (misleading "PDK reachable").
- **Why this is only minor here:** On **this** server both `PDK_ROOT` and `PDK` are exported and the 40 `sg13g2_pr` `.sym` files exist, so symbols render correctly. The bug only manifests in deployments that supply the PDK via `PDK_ROOT`-only / `IHP_PDK_ROOT` / `app_config.pdk_root` without exporting the singular `PDK`. The env-contract divergence is fully provable statically; reproducing the placeholder rendering would require a backend launched with `PDK` unset — **needs runtime verification (deferred — no live-sim this round).**
- **Fix direction (not implemented):** Make the symbol resolver use the **same** PDK-root discovery as `env_probe` instead of requiring the singular `PDK`: derive `libs.tech/xschem` from `probe_pdk()`'s resolved `pdk_root` (or accept `PDK_ROOT` alone, `IHP_PDK_ROOT`, and `app_config.pdk_root`, joining the tech name when needed), so the symbol search path can never be a strict subset of the sim PDK path. Also stop `lru_cache`-ing a `None`/empty result (or clear the caches on env change) so a late-exported PDK takes effect without a process restart.

### FLAG-2 · "Score Shaping computes only ONE score at a time" — **NOT present at HEAD; already fixed by `62fd49d` (tagged BUG-02)**

- **Area:** Score Shaping (frontend + backend)
- **Verdict:** This previously-reported defect is **fixed**. No active remediation required; no runtime verification needed (fully decidable by reading source + git history).
- **What the bug was:** The frontend `ScoreShapingTab` used to POST a **one-key** `metric_values` map (only the selected spec's value). The backend [`compute_score`](ui/backend/services/score_service.py) already iterated **all** enabled specs (`enabled_targets()` then a loop, byte-identical since the first commit `2e82124`), but any spec absent from the payload hit `metric_values.get(spec.name) → None`, fell into the null-placeholder + `continue` branch ([score_service.py:46-52](ui/backend/services/score_service.py#L46-L52)), and contributed `0` to `total_linear`/`total_sigmoid`. So the displayed `F(x)` reflected only the single selected spec — the reported symptom. The defect was **purely frontend** (the API client and `ScoreRequest` always accepted a full dict).
- **The fix at HEAD (confirmed present):** [ScoreShapingTab.tsx:31](ui/src/components/tabs/ScoreShapingTab.tsx#L31) now holds `values: Record<string, number>` (the full per-spec vector); an effect seeds `values[s.name] = s.target` for every enabled spec; `computeScore` calls `api.computeScore(yamlPath, vals, specName)` with the **entire** `vals` map ([ScoreShapingTab.tsx:77](ui/src/components/tabs/ScoreShapingTab.tsx#L77)), `specName` driving only the highlighted curve. `git show 62fd49d -- …/ScoreShapingTab.tsx` shows the exact `{ [specName]: value }` → `vals` change, and the commit body lists it as "BUG-02, flagged."
- **Residual note (separate, see SCR-1 below):** The footer still labels the aggregate `F(x) = Σ wᵢ · P̂ᵢ` while the backend returns the **negated** sum — a sign/label inconsistency tracked as **SCR-1**, distinct from the "only one score" bug.

---

## 3. Additional bugs (grouped by area)

### 3a. Setup / Wizard / project summary

#### WIZ-1 · `freeze: true` is silently dropped on YAML generation — a frozen DUT param is emitted as an optimized (swept) dimension · 🟠 **major**

- **Location:** [yaml_generator.py:46-47](ui/backend/services/yaml_generator.py#L46-L47) (`_build_dut_param`) · **Confidence:** high
- **Symptom:** Checking the "frz" (freeze) checkbox for a DUT param in the wizard has no effect on the generated/saved YAML: the emitted `dut_param` has no `freeze` key. On load, `Param.freeze` defaults to `False` ([domains.py:308](src/spicexplorer/core/domains.py#L308)), so the param is added to the optimizer search space and swept between `min_val`/`max_val` instead of being fixed. The live YAML preview also never shows `freeze: true`.
- **Root cause:** The emit condition is **inverted** relative to the dataclass default. The only value that *must* be serialized is `freeze: true`, but `_build_dut_param` writes the key only when `row.get("freeze") is False` — i.e. it emits the redundant `freeze: false` and **never** emits `freeze: true`. The wizard checkbox ([DutParamsStep.tsx](ui/src/components/wizard/steps/DutParamsStep.tsx)) sets `freeze: e.target.checked` (a real bool); when checked, `True is False → False`, so the key is dropped.
- **Code path:** DutParamsStep checkbox → `wizardStore.setDutParams` → `WizardShell` → `api.generateProject(form)` → `POST /api/project/generate` → `generate_yaml` → `_build_dut_param` drops `freeze: true` → `Param(freeze=False)` → `NevergradMixin.parameterize` ([nevergrad.py:136](src/spicexplorer/optimization/stochastic/nevergrad.py#L136) `if getattr(param, "freeze", False):`) treats it as a search dimension. The parse-to-form side reads `freeze` correctly, so only the emit side is broken.
- **Fix direction (not implemented):** Emit `freeze` whenever it is truthy, e.g. `if row.get("freeze"): out["freeze"] = True`. Optionally drop the redundant `freeze: false` branch entirely (False is the dataclass default).

#### WIZ-2 · `pvt.model_lib_root` is dropped on the YAML→form→YAML round-trip, breaking `.lib` path resolution · 🟠 **major**

- **Location:** [yaml_generator.py:174-206](ui/backend/services/yaml_generator.py#L174-L206) (`_build_pvt_block`, no `model_lib_root` emitted) and [yaml_generator.py:253-293](ui/backend/services/yaml_generator.py#L253-L293) (`_pvt_block_to_form`, returns only `{active_corner, corners}` at :290-293, never reads `block.get("model_lib_root")`). Form-type gap: [api.ts](ui/src/types/api.ts) `WizardPVTConfig`, [wizardStore.ts](ui/src/stores/wizardStore.ts) default, [PVTStep.tsx](ui/src/components/wizard/steps/PVTStep.tsx). Caller: [SetupTab.tsx](ui/src/components/tabs/SetupTab.tsx) `editInWizard`. · **Confidence:** high
- **Symptom:** Loading a project whose `pvt:` block sets `model_lib_root: <dir>`, clicking "Edit in wizard", then Save produces a regenerated YAML with **no** `model_lib_root`. At sim time the corner's includes are emitted as bare `lib_file` strings instead of `<model_lib_root>/<lib_file>`, so the `.lib` paths no longer resolve.
- **Root cause:** `_pvt_block_to_form` carries only `active_corner` and `corners`; it never reads `model_lib_root` (which survives `_normalize_pvt_block` untouched, so the value *is* present to read but is ignored). `_build_pvt_block` has no `model_lib_root` field at all. The wizard form type also has no slot for it. Unlike the legacy display-only flat corners, `model_lib_root` is **load-bearing**: [base.py:499](src/spicexplorer/optimization/base.py#L499) passes it to `apply_corner`, and `spicelib.py` prepends it to each include's `lib_file` (`path = inc.lib_file if not model_lib_root else str(Path(model_lib_root) / inc.lib_file)`).
- **Code path:** `editInWizard` → `POST /api/project/parse-to-form` → `project_dict_to_form` → `_pvt_block_to_form` (drops `model_lib_root`) → `wizardStore.setForm` → user Saves → `POST /api/project/generate` → `_build_pvt_block` (no `model_lib_root` emitted) → regenerated YAML → `apply_corner(... model_lib_root=None)` → bare `.lib` path.
- **Why this doesn't bite shipped examples:** the only shipped `pvt:` block (cascode `project_setup.yaml`) sets `model_lib_root: null`. The loss only bites projects that set a non-null value, none of which ship. Confirming the downstream sim breakage end-to-end would need a live run — **needs runtime verification (deferred)** — but the code-level data-loss is fully verifiable statically.
- **Fix direction (not implemented):** Carry `model_lib_root` through both directions: have `_pvt_block_to_form` read `block.get("model_lib_root")` into the form, add a field to the wizard PVT form/type and PVTStep, and have `_build_pvt_block` emit `model_lib_root` when set.

#### WIZ-3 · `target_spec` weight of `0` is misreported as `1.0` in the parsed project summary · 🟡 **minor**

- **Location:** [project.py:47](ui/backend/routes/project.py#L47) (`_summarise`). Frontend consumer: [SetupTab.tsx](ui/src/components/tabs/SetupTab.tsx) weight Σ + per-spec column. · **Confidence:** high
- **Symptom:** A spec authored with `weight: 0` (a legitimate way to keep a spec defined but contribute nothing to the aggregate) is shown in the Setup summary's per-spec weight column and the "weight Σ" total as `1.0` instead of `0`. Display-only; the optimizer uses the real weight (`0`).
- **Root cause:** The summary uses a truthiness guard: `"weight": float(s.weight) if s.weight else 1.0`. Because `0`/`0.0`/`np.float64(0.0)` is falsy, a genuine zero weight falls through to the `1.0` fallback. `TargetSpec.weight` is never re-validated/clamped in `__post_init__` (unlike `range`/`tolerance`, which are forced positive), so `0` reaches this line intact and is masked. The optimizer reads `target_spec.weight` directly ([base.py:996](src/spicexplorer/optimization/base.py#L996), :1063), so only the panel misreports.
- **Code path:** `SetupTab` apply/load → `POST /api/project/load` → `_summarise` (weight `0` → `1.0`) → `ProjectSummary.target_specs` → SetupTab weight Σ + per-spec column (the `?? 0` there only guards null/undefined and does not undo the `1.0` coercion).
- **Fix direction (not implemented):** Guard on `None`, not truthiness: `float(s.weight) if s.weight is not None else 1.0` — matching the `isinstance(p.val, (int, float))` / `p.min_val is not None` patterns already used in the same function.

#### WIZ-4 · Multi-rail PVT corners lose supply rails 2..N when round-tripped through the wizard · 🟡 **minor**

- **Location:** [yaml_generator.py:276-283](ui/backend/services/yaml_generator.py#L276-L283) (`_pvt_block_to_form`, keeps `supplies[0]` only) and [yaml_generator.py:197-199](ui/backend/services/yaml_generator.py#L197-L199) (`_build_pvt_block`, emits singular `supply`). Single-rail model: [api.ts](ui/src/types/api.ts) `WizardPVTCorner`, [PVTStep.tsx](ui/src/components/wizard/steps/PVTStep.tsx). · **Confidence:** medium
- **Symptom:** A `pvt:` corner with more than one supply (after `supplies: [...]` widening) is reduced to a single rail by "Edit in wizard". Regenerating/Saving from the wizard emits only that first rail, permanently dropping the additional supply overrides.
- **Root cause:** `_pvt_block_to_form` deliberately keeps only `supplies[0]` (`first = supplies[0] if supplies else {}`), and the wizard PVT model (`WizardPVTCorner`: `supply_node`/`supply_value`) holds one rail. `_build_pvt_block` correspondingly emits a singular `supply: {node, value}`, so rails 2..N have no representation. Multi-rail **is** representable in the engine (`_normalize_pvt_block` widens singular `supply` and accepts a native `supplies` list, `SupplyOverride`/`supplies: List` is the canonical core shape) — the loss is purely in the wizard layer.
- **Mitigating context:** This is **documented as intentional** (docstring at [yaml_generator.py:258-259](ui/backend/services/yaml_generator.py#L258-L259) steers multi-rail users to the raw editor), and it is currently only reachable by hand-authoring a multi-rail `pvt:` YAML (no committed example uses `supplies:`, no wizard UI to create N>1 rails). Still a real silent data-loss path: there is **no** user-facing warning that Save will drop rails.
- **Fix direction (not implemented):** If multi-rail wizard editing is desired, extend `WizardPVTCorner` to a list of rails and emit `supplies: [...]`. Otherwise, at minimum surface a non-destructive warning when a loaded corner has >1 supply.

#### SCH-2 · `dut_param` string `val` is never resolved (`ressolve_val` only runs on testbench params) — a numeric-but-eng-string operating point serializes as `null` and the inspector slider falls back to range-center · 🟡 **minor**

- **Location:** [domains.py:832-846](src/spicexplorer/core/domains.py#L832-L846) (`resolve_all_parameter_ranges`; dut_params loop :835-839 never calls `ressolve_val`) + [domains.py:313-325](src/spicexplorer/core/domains.py#L313-L325) (`resolve_min_max` resolves `init` but not `val`; `ressolve_val` defined at :323 but only called at :844 for tb params); [project.py:63](ui/backend/routes/project.py#L63) (`val` serialization isinstance check); [sensitivity.py:84-99](ui/backend/routes/sensitivity.py#L84-L99) (`_nominal` via `num.safe_float`, can't parse eng-strings); [DeviceInspector.tsx](ui/src/components/schematic/DeviceInspector.tsx) (midpoint fallback). · **Confidence:** medium
- **Symptom:** For a DUT param whose YAML `val:` is an engineering string (e.g. `"0.18u"`) or a constraint reference, the Device Inspector W/L sliders seed from the parameter's range midpoint rather than the intended operating point, and the project summary reports `val: null`. The on-screen nominal can disagree with the design's declared `val`.
- **Root cause:** `resolve_all_parameter_ranges` resolves dut_param `min_val`/`max_val`/`init` (via `resolve_min_max`) but never calls `Param.ressolve_val()` for dut_params — `ressolve_val` is invoked only in the testbench-param loop ([domains.py:844](src/spicexplorer/core/domains.py#L844), its sole call site). So a `dut_param.val` that is a `str` stays a string. `project.py:63` then serializes `val` as `float(p.val)` only if `isinstance(p.val, (int, float))`; a leftover string fails the check → `val: null`. The frontend `midpoint()` and backend `_nominal()` (via `safe_float`, which can't parse eng-strings) both fall back to range center. The asymmetry is genuine: the sibling `init` field **is** resolved in the same method while `val` is not, and a dedicated `ressolve_val` exists but is never wired into the dut_params loop.
- **Latency:** the shipped cascode example sets **no** `dut_param` `val:` (only `min`/`max` via constraint refs), so no current example triggers it.
- **Fix direction (not implemented):** Call `ressolve_val` (or `parse_value` the `dut_param.val`) inside `resolve_all_parameter_ranges`' dut_params loop so `val` becomes a resolved `np.float64`, parallel to how `init` is resolved; the existing `project.py:63` isinstance check then serializes it (`np.float64` subclasses `float`).

### 3b. Explore / Compare / Score

#### EXP-1 · `exact`-goal specs get the wrong "best" value (max instead of closest-to-target) in the envelope and spec-summary tables · 🟠 **major**

- **Location:** [ExplorerTab.tsx:26-31](ui/src/components/tabs/ExplorerTab.tsx#L26-L31) (`bestOf`), used at :191-192 + :198 (envelope/winner) and :495-498 (spec-summary pass/fail). The server-side raw envelope has the identical bug (see EXP-2). · **Confidence:** high
- **Symptom:** For an `exact`-goal spec (the cascode `pm`, target 60° ±10°), the "performance envelope" and "spec summary" tables surface the run's **maximum** phase-margin sample as "best". A run that hit ~60° but also sampled, say, 120° is shown with best=120°, declared FAIL by `passesSpec(s, 120)` (|120−60|>10), and the A/B "winner" is decided by `aBest > bBest` — meaningless for an exact target. So the head-to-head winner, the highlighted best value, and the pass/fail badge are all wrong for any `exact` spec.
- **Root cause:** `bestOf(values, goal)` is `values.reduce((m, v) => (goal === "minimize" ? (v < m ? v : m) : v > m ? v : m), values[0])` — only `minimize` is special-cased (pick min); every other goal, including `exact`, takes the **max**. There is no `exact` branch returning the sample minimizing `|v − target|`. Note `passesSpec` → `statusForGoal` is itself correct for exact; the bug is that `bestOf` hands it the wrong representative sample.
- **Code path:** `loadBoth` → `api.loadCheckpoint` → `GET /api/checkpoint/{id}` → store `runA`/`runB` → envelope rows / spec-summary call `bestOf(per_metric[s.name], s.goal)` with the unhandled `exact` goal.
- **Fix direction (not implemented):** Give `bestOf` the spec target and add an `exact` branch returning the argmin of `|v − target|`; make the `winner` comparison compare `|aBest−target|` vs `|bBest−target|` for exact goals.

#### EXP-2 · `compute_envelope` returns max (not closest-to-target) and a wrong `passes` for `exact`-goal specs · 🟠 **major**

- **Location:** [checkpoint_reader.py:131-144](ui/backend/services/checkpoint_reader.py#L131-L144) (`best_ever` + `passes`); reached via [checkpoint.py:143-151](ui/backend/routes/checkpoint.py#L143-L151) (`GET /checkpoint/{id}/envelope`). Frontend consumer: [ExplorerTab.tsx](ui/src/components/tabs/ExplorerTab.tsx) "envelope (run A · raw)" panel, fed by [api.ts](ui/src/lib/api.ts) `api.envelope`. · **Confidence:** high
- **Symptom:** The server-computed "envelope (run A · raw)" panel reports, for the `exact` `pm` spec, `best_ever = max(pm samples)` and `passes` evaluated against that outlier max. A run whose best in-band point passes ±10° of 60° can be shown as `best_ever=120°` with `passes=false`, contradicting the actual feasibility.
- **Root cause:** The goal switch is `if goal == "minimize": best_ever = min(clean) else: best_ever = max(clean)` — the `exact` case falls into `else` → `max(clean)`. The `passes` block for `exact` (`abs(best_ever - target) <= tol`) is itself correct but evaluates the already-wrong `best_ever`. (`compute_scatter` handles `exact` correctly per-point, confirming this is an omission, not intended design.) All three shipped examples contain exact-goal specs, so this is routinely reachable.
- **Code path:** ExplorerTab effect → `api.envelope(runAId, yamlPath)` → `GET /api/checkpoint/{id}/envelope` → `_target_specs_from_yaml` (real goal/target/tolerance) + `compute_envelope` → `else: best_ever = max(clean)` for the `exact` `pm` spec.
- **Fix direction (not implemented):** Add an `exact` branch: `best_ever = min(clean, key=lambda v: abs(v - target))` (`target` already available from `spec_map`). The `passes` line then evaluates the correct point.

#### EXP-3 · `MetricConvergenceChart` best-so-far curve treats `exact` goals as maximize · 🟡 **minor**

- **Location:** [MetricConvergenceChart.tsx:19-31](ui/src/components/charts/MetricConvergenceChart.tsx#L19-L31) (`bestSoFar`). Call sites: [ExplorerTab.tsx](ui/src/components/tabs/ExplorerTab.tsx) and [OptimizeTab.tsx](ui/src/components/tabs/OptimizeTab.tsx) (both pass `goal`+`target`). · **Confidence:** high
- **Symptom:** The metric "best-so-far" convergence panel for an `exact` spec (e.g. `pm`) draws a monotonically non-decreasing running-max curve instead of a curve converging toward the target band. The trace climbs past the target and never reflects that values overshooting 60° are getting worse.
- **Root cause:** `bestSoFar(values, goal)` only special-cases `goal === "minimize"` (running min); every other goal, including `exact`, falls into the `else` (`Math.max`). The function never receives `target`, so it has no notion of distance-to-target — even though the chart already uses `target` for the dashed reference line.
- **Code path:** `metricRuns` → `<MetricConvergenceChart goal={selectedSpecObj.goal} target=…>` → `bestSoFar(run.values, goal)` with `goal === "exact"` falling through to running max.
- **Fix direction (not implemented):** Thread `target` into `bestSoFar` and, for `exact` goals, track the running best as the sample minimizing `|v − target|`. Apply at both call sites (ExplorerTab + OptimizeTab).

#### SCR-1 · "F(x) aggregate" footer shows a negated value under a header that defines `F(x)` as a sum of non-negative penalties · 🟡 **minor**

- **Location:** [score_service.py:103](ui/backend/services/score_service.py#L103) (`aggregate` negated) vs [ScoreShapingTab.tsx](ui/src/components/tabs/ScoreShapingTab.tsx) header `F(x) = Σ wᵢ · P̂ᵢ` and footer rendering `aggregate.sigmoid`/`aggregate.linear` raw. · **Confidence:** medium
- **Symptom:** The per-spec columns show non-negative penalties `P̂` (0..1) and the header reads `F(x) = Σ wᵢ · P̂ᵢ`, but the aggregate footer prints **negative** numbers (e.g. sigmoid −0.234) because the backend returns `{"linear": -total_linear, "sigmoid": -total_sigmoid}`. The displayed aggregate therefore does not equal the visible weighted sum of the `P̂` column and reads as a sign error.
- **Root cause:** `compute_score` accumulates `total_linear`/`total_sigmoid` as non-negative weighted penalty sums, then returns them **negated** (the optimizer's maximize-score convention), while the UI labels and surrounds the value as the non-negative `Σ wᵢ · P̂ᵢ`. The "dominant spec" callout uses positive `sigmoid * weight`, so it's consistent with the header but contradicts the negated footer, reinforcing the mismatch. Cosmetic only — the live optimizer uses the core scorer (`core/utils.py`), not this UI service.
- **Code path:** `ScoreShapingTab.computeScore` → `api.computeScore` → `POST /api/score` → `compute_score` returns negated `aggregate` → footer renders `aggregate.sigmoid.toFixed(3)` under the `Σ wᵢ · P̂ᵢ` header.
- **Fix direction (not implemented):** Make the contract consistent: either return the non-negated `Σ wᵢ · P̂ᵢ` for the footer (keeping a separate explicit `score = -penalty` field if the negated value is wanted), or relabel the footer to `F(x) = −Σ wᵢ · P̂ᵢ` / "score". Do not change optimizer semantics.

### 3c. Live run / Right rail / Pipeline

#### OPT-1 · Right-rail "Spec status" / Pipeline pass-fail uses latest-seen metrics, not the best-scoring trial's — inconsistent with the displayed best params · 🟡 **minor**

- **Location:** [runStore.ts:193-209](ui/src/stores/runStore.ts#L193-L209) (`pushEvent` merge); consumed at [RightRail.tsx:39](ui/src/components/shell/RightRail.tsx#L39) and [PipelineView.tsx](ui/src/components/tabs/PipelineView.tsx) (`bestMetrics` reads in `specPass` and the DAG tint); emitted at [optimizer_runner.py:152-162](ui/backend/services/optimizer_runner.py#L152-L162). · **Confidence:** medium
- **Symptom:** During a live run, the right-rail "Spec status" chips (and the Pipeline DAG pass/fail tint) can show metric values from a later, lower-scoring iteration while the adjacent "Best params" table and "best score" stat show the best-scoring trial. A user can see all-green spec chips that do not correspond to the best design shown next to them.
- **Root cause:** The server emits `best_score`/`best_params` from the instance-tracked running best (updated only on improvement) but `metrics` from the **current** trial's `fit_summary` curr_val. `pushEvent` merges each event's `metrics` into `bestMetrics` keeping the last non-null per spec (latest trial), while `bestParams` reflects the best trial. There is no `best_metrics` channel, so `bestMetrics` and `bestParams` describe different iterations; RightRail/Pipeline treat `bestMetrics` as the best design's metrics.
- **Scope:** Live-run display only; the replay path pairs `metrics[i]`/`params[i]` from the same row, so it is internally consistent. No checkpoint/optimization-data corruption.
- **Fix direction (not implemented):** Either (a) emit the best trial's `fit_summary` as a `best_metrics` field alongside `best_params` (snapshot when `_best_score` updates) and key RightRail/Pipeline off it, or (b) relabel the right-rail status as "current" to match latest-trial semantics. Do not accumulate latest metrics under a name implying "best".

#### RAIL-1 · RightRail spec pass/fail ignores tolerance, contradicting HealthTab / `statusForGoal` · 🟡 **minor**

- **Location:** [RightRail.tsx:40-46](ui/src/components/shell/RightRail.tsx#L40-L46) · **Confidence:** high
- **Symptom:** During a live run, the always-on RightRail "Spec status" panel colors a spec green/red using a **strict** comparison to target (`exceed: val >= target`, `minimize: val <= target`). HealthTab and every other surface use `statusForGoal`, which admits the tolerance band (`exceed: val >= target − tol`, defaulting tol to 5% of target). A metric within tolerance just below an `exceed` target shows FAIL in RightRail but PASS in HealthTab for the identical value — the two surfaces disagree.
- **Root cause:** `RightRail.specStatuses` hand-rolls `spec.goal === "exceed" ? val >= spec.target : spec.goal === "minimize" ? val <= spec.target : Math.abs(val - spec.target) <= (spec.tolerance ?? Infinity)`. Only the `exact` branch consults tolerance; the `exceed`/`minimize` branches omit `spec.tolerance`, unlike [utils.ts](ui/src/lib/utils.ts) `statusForGoal` which subtracts/adds the tolerance. The `exact` branch also diverges (defaults to `Infinity`, too lenient, vs `statusForGoal`'s `target*0.05`). Purely frontend (no backend).
- **Fix direction (not implemented):** Reuse `statusForGoal(spec.goal, val, spec.target, spec.tolerance ?? undefined)` in RightRail instead of the inline comparison — `summary.target_specs` is the same `TargetSpec[]` HealthTab already feeds to `statusForGoal`, so it is a drop-in.

#### RAIL-2 · RightRail "Spec status" shows last-iteration metric values, not the best point's metrics · 🟡 **minor**

- **Location:** [optimizer_runner.py:152-162](ui/backend/services/optimizer_runner.py#L152-L162) (`_emit`), [runStore.ts:193-208](ui/src/stores/runStore.ts#L193-L208) (`pushEvent`), [RightRail.tsx:38-52](ui/src/components/shell/RightRail.tsx#L38-L52). · **Confidence:** medium
- **Symptom:** The RightRail pairs `best_score`/`best_params` (the run's best point) with per-spec metric values taken from `bestMetrics`, but `bestMetrics` is the most-recent non-null value of each metric across all events, not the metrics of the best design. After the optimizer moves past its best point, the displayed metric values and pass/fail flags can describe a different design than the `best_params` shown right below.
- **Root cause:** Same emit/merge mismatch as OPT-1: the streaming optimizer emits `metrics` as the current step's `fit_summary` curr_val while `best_params`/`best_score` are the running best; `pushEvent` keeps the last non-null per key under the misleadingly-named `bestMetrics`. (This entry is the RightRail-specific framing; OPT-1 covers the same root cause as it also affects the Pipeline DAG.) Live-run only; replay is consistent. Display-only, no data corruption.
- **Fix direction (not implemented):** Same as OPT-1 — add a `best_metrics` payload keyed off the best point, or rename the field to "latest metrics" in the UI.

### 3d. Backend / library plumbing

#### OPT-2 · Starting a live run while a manual sim is in flight rmtree's the manual sim's working directory (one-directional isolation only) · 🟡 **minor**

- **Location:** [optimizer_runner.py:53-77](ui/backend/services/optimizer_runner.py#L53-L77) (`_build_spicelib_wrappers`; live run calls it with no `output_subdir` at :257) + [simulate.py:177](ui/backend/routes/simulate.py#L177) (manual sim uses `output_subdir="manual_sim"`) + [spicelib.py:234-242](src/spicexplorer/spice_engine/spicelib.py#L234-L242) (`NGSpice_Wrapper._validate` / `rmtree` at :238). · **Confidence:** medium
- **Symptom:** If a live optimization run is started while a Manual Sim is still executing, the manual sim can fail or return a confusing error because its run directory is deleted out from under ngspice mid-evaluation.
- **Root cause:** The manual sim isolates itself under `ws_root/outdir/manual_sim`, and the comment at `_build_spicelib_wrappers` only reasons about manual-not-clobbering-live. But a live run's wrappers use `output_folder = ws_root/outdir` (no subdir), and `NGSpice_Wrapper._validate` calls `shutil.rmtree(self.output_folder)` at construction — recursively removing `ws_root/outdir` **including** the nested `manual_sim/` subtree. No lock/guard serializes the executor-thread manual sim against the live-run thread.
- **Reproducing the actual race requires concurrent execution** — **needs runtime verification (deferred)**; the rmtree-of-parent code path is confirmed statically.
- **Fix direction (not implemented):** Give the live run its own isolated subfolder (e.g. `ws_root/outdir/live` or a per-run id), or scope `_validate`'s rmtree to its own run subtree rather than the shared parent, or add a mutex preventing concurrent live-run + manual-sim wrapper construction.

#### OPT-3 · Autosave checkpoints are written CWD-relative (`./auto_save`) but discovered under `REPO_ROOT/auto_save` — Resume and live checkpoint listing silently break when the backend CWD is not the repo root · 🟡 **minor**

- **Location:** [base.py:63](src/spicexplorer/optimization/base.py#L63) (write, CWD-relative `./auto_save`) and `base.py:449-450` (`get_auto_save_name` builds under `autosave_checkpoint_dir`) vs [checkpoint.py:53](ui/backend/routes/checkpoint.py#L53) + [checkpoint.py:63](ui/backend/routes/checkpoint.py#L63) (read, `REPO_ROOT / "auto_save"`); resume 404 at [optimize.py:70-76](ui/backend/routes/optimize.py#L70-L76); launch at [run_newcas_ui.sh:71-73](scripts/run_newcas_ui.sh#L71-L73). · **Confidence:** low
- **Symptom:** Live autosaved checkpoints (and the FINAL checkpoint) may not appear in the checkpoint list, and Resume-from-checkpoint can 404 ("Resume checkpoint '<id>' not found"), whenever the uvicorn backend was launched from a directory other than the repo root.
- **Root cause:** `Base_Optimizer` creates the autosave dir as a CWD-relative `./auto_save/...` path, while the FastAPI checkpoint resolver/listing only ever `rglob` under `REPO_ROOT/auto_save` (`UI_DIR.parent`, an absolute `__file__`-derived path). These coincide only when the backend CWD equals `REPO_ROOT`. `run_newcas_ui.sh` starts uvicorn before its only `cd` (which is for the frontend), so alignment depends on the operator running the script from the repo root; the Docker image happens to align (`WORKDIR /app == REPO_ROOT`). The SSE checkpoint event's id (latest `.json` stem) is therefore unresolvable when CWDs diverge.
- **Latency:** Under the documented happy path (operator runs the script from the repo root) CWD == REPO_ROOT and it works by accident. Manifesting it requires launching the backend from a non-repo-root directory — **needs runtime verification (deferred)**; the path mismatch itself is confirmed statically.
- **Fix direction (not implemented):** Anchor the optimizer autosave dir to an absolute, project-derived root (e.g. `ws_root` or `REPO_ROOT`) instead of `./auto_save`; or make the checkpoint resolver/listing also search the process CWD's `./auto_save`; or have `run_newcas_ui.sh` `cd` to `ROOT_DIR` before launching uvicorn.

#### ENV-1 · PDK fast-path subpaths miss the IHP `libs.tech/ngspice/models/` layout, forcing a full-tree rglob on every probe · 🟡 **minor**

- **Location:** [env_probe.py:28-33](ui/backend/services/env_probe.py#L28-L33) (`_PDK_LIB_SUBPATHS`) and [env_probe.py:61-76](ui/backend/services/env_probe.py#L61-L76) (`_find_model_lib`). · **Confidence:** high
- **Symptom:** On this server `cornerMOSlv.lib` lives at `$PDK_ROOT/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib`, but none of the four explicit fast-path subpaths match that combination — they cover `ihp-sg13g2/libs.tech/ngspice/<lib>`, `libs.tech/ngspice/<lib>`, and `libs.tech/ngspice/models/<lib>`, never the tech-prefixed `ihp-sg13g2/libs.tech/ngspice/models/<lib>`. The probe still returns `pdk_ok: true`, but only via the bounded `rglob` fallback, which walks the ~15k-file / ~423-dir PDK tree on **every** `GET /api/env` and every `probe_pdk()` call inside `/api/sanity-check`, `/api/optimize/start`, and `/api/simulate/once` (there is no caching in `env_probe`).
- **Root cause:** `_PDK_LIB_SUBPATHS` enumerates explicit candidates but omits the `{_PDK_TECH}/libs.tech/ngspice/models/{_PDK_MODEL_LIB}` layout that the installed IHP sg13g2 PDK actually uses, so `_find_model_lib` falls through to `root.rglob(_PDK_MODEL_LIB)` for the real install. Functionally correct (returns the first match) but does a recursive directory walk instead of a single `stat()`.
- **Fix direction (not implemented):** Add `f"{_PDK_TECH}/libs.tech/ngspice/models/{_PDK_MODEL_LIB}"` to `_PDK_LIB_SUBPATHS` so the common IHP layout hits the O(1) fast path and the rglob becomes a rare fallback.

#### CKPT-1 · JSON checkpoint reader ignores the `limit` parameter that CSV and the route honor · 🟡 **minor**

- **Location:** [checkpoint_reader.py:15](ui/backend/services/checkpoint_reader.py#L15) (`read_json_checkpoint`, no `limit` param) and [checkpoint_reader.py:104-105](ui/backend/services/checkpoint_reader.py#L104-L105) (`read_checkpoint` JSON branch drops `limit`, CSV branch forwards it); route at [checkpoint.py](ui/backend/routes/checkpoint.py) `load_checkpoint`. · **Confidence:** high
- **Symptom:** `GET /api/checkpoint/{id}?limit=N` truncates the returned series for CSV checkpoints (`df.head(limit)`) but returns the **full** series for JSON checkpoints regardless of `limit`. A caller requesting a capped slice of a large JSON autosave (the format live autosaves write) gets every row, so the cap is silently ineffective.
- **Root cause:** `read_json_checkpoint(path)` has no `limit` parameter and iterates the entire `OptimizationLog`, while `read_checkpoint` forwards `limit` only on the CSV branch: `if path.suffix == ".json": return read_json_checkpoint(path)` drops `limit`. The route passes `limit` to `read_checkpoint`, but it is discarded for JSON. **Latent:** the only caller (`api.loadCheckpoint`) defaults `limit=0` and both real call sites pass no limit, so no current caller triggers it — but any future `?limit=N` against a JSON checkpoint would be a no-op.
- **Fix direction (not implemented):** Add a `limit` parameter to `read_json_checkpoint` (slice the log to the first `limit` entries) and forward it from `read_checkpoint`'s JSON branch, matching the CSV behavior.

---

## 4. Previously reported — now fixed (verified)

The prior audit raised a large set of findings (its own BUG-01..BUG-39 numbering). Several have since been resolved by `a0fbc6f` & `62fd49d` (audit bug fixes in lib/backend/frontend), `c57b005` (merge), the PVT Phase-1 + manual-sim work (`ebc8e9d` / `2375f45` / `37952bf` / `c409031` / `a15b420`), `dc8b6f5` (cascode PVT block), `42dd636` (scroll-clip fix), `53a587f` (skip disabled testbenches), and `e6b68ed` (xschem `.sym` files vendored into the Docker PDK). Verified-fixed items relevant to this round:

- **Score Shaping single-spec aggregate (prior BUG-02 / flagged):** **Fixed** by `62fd49d`. The frontend now sends the full per-spec vector (`api.computeScore(yamlPath, vals, specName)`); the backend always summed all enabled specs. See **FLAG-2** above for the full trace.
- **`bestOf` stack overflow on long runs (prior BUG-20 class):** `62fd49d` replaced `Math.min(...spread)` with a `reduce` in `ExplorerTab.bestOf` ([ExplorerTab.tsx:28-30](ui/src/components/tabs/ExplorerTab.tsx#L28-L30)), removing the RangeError on >~65k-point arrays. (The separate `exact`-goal logic error in the same function is **still open** — see EXP-1.)
- **Schematic "all symbols are red ?" under Docker (prior BUG-01):** the Docker-only resolution gap was addressed by `e6b68ed` vendoring the `sg13g2_pr` `.sym` files into the Docker PDK. On this native PDK-equipped server symbols resolve; the *remaining* env-contract divergence (`PDK_ROOT`-only deployments) is reclassified and tracked as **FLAG-1** (minor) above.
- **PVT corners not driving the sim / manual-sim gaps (prior planning items):** PVT Phase 1 (`apply_corner` seam, `_normalize_pvt_block`, `CornerSelect`, `ManualSimPanel`, `/api/simulate/once`) landed via the PVT commits; the manual-sim "sims a random point" `sanity.py` gap is closed (it now routes through `evaluate(params, append_to_log=False)`).
- **Disabled-testbench `KeyError` in `__post_init__` (prior planning item):** **Fixed** by `53a587f` (skips disabled testbenches).
- **Scroll container clipped tall panels (couldn't reach Manual Sim logs):** **Fixed** by `42dd636`.

> The prior audit's "rejected candidates" list is empty for this round (no candidates were supplied for re-judgement here); the bullets above are the prior findings that the fix commits demonstrably resolve.

---

## 5. Investigated, not confirmed

- **Score Shaping fidelity vs. the real optimizer score (out of scope for the "only one score" bug):** `score_service`'s `_raw_directional_error` + relative-absolute/sigmoid penalty math is a simplified re-implementation rather than a call into the library's `compute_fitness` / `compute_error`; the real path adds a reward term and `log_scale` / tolerance-adjusted-target handling that the preview only approximates. This is a **fidelity** difference (a known design simplification), **not** a correctness bug, and is independent of the now-fixed multi-spec iteration. Not reported as an active bug.
- **xschem parser/renderer as the cause of placeholder symbols:** Ruled out. None of the referenced symbols nest sub-symrefs, and `SchematicViewer` renders only an instance's own `sym.primitives` / `sym.texts` (never recursing into `sym.instances`); the renderer handles all primitive kinds (`L/B/P/A/T`) present in these symbols. The parser/renderer is **not** the regression — the placeholder cause is purely backend symbol resolution (FLAG-1).
- **`_pdk_xschem_dir` `lru_cache` freezing a stale value (within FLAG-1):** confirmed as a contributing factor (a late-exported `PDK` won't take effect without a process restart), folded into FLAG-1's fix direction rather than reported separately.
