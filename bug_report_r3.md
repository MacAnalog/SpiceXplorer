# Bug report — functional audit, THIRD round (2026-06, encapsulation + NEWCAS + PVT)

> **Method.** Multi-agent static-analysis pass over `feat/pvt` at HEAD. 15 subsystem finders read the
> code (no app/sim run); **every** candidate was then handed to an independent adversarial verifier
> that re-read the cited file and tried to *refute* it. Only verifier-confirmed findings appear below.
> **73 raised → 73 confirmed REAL after dedup into 52 distinct bugs; 17 refuted** (listed at the end).
> Verifiers corrected several finder severities (mostly *down*, where reachability is gated behind a
> non-default config) — the severities here are the verifier's corrected ones.
>
> **Scope priority (as requested):** the just-landed **encapsulation / run-lifecycle** code (un-audited
> before this round), the **core library functions exercised by `examples/OTA/cascode`** (NEWCAS), and
> **PVT** (`apply_corner` + `__post_init__` + `PVTConfig`), plus the UI + backend. Already-closed
> BUG-01..39 / BUG-A1..A16 were excluded by construction. IDs are `BUG-B*`.
>
> **Reachability legend:** 🟢 reachable with the shipped cascode example / default UI flow · 🟡 reachable
> with valid user-authored config or a non-default toggle · ⚪ latent (correct code path exists; needs a
> not-yet-wired caller or contrived input).

---

## Tier 0 — Security & data-integrity (fix first)

> **✅ STATUS: all four FIXED 2026-06-06** (uncommitted in the worktree; see TODO §19 Tier 0). Verified by
> `pytest` (121 pass), `tsc`/`eslint`, +21 regression tests, and a 5-agent adversarial bypass review.
> The review surfaced (and these fixes now cover): a B3 *cross-run* residual (catalog dedups same-stemmed
> checkpoints from different runs → fixed with a precise `?path=` delete) and a B2 *sibling* LFI on
> `project.py /yaml-text` (fixed via the shared validator). One residual is **deferred**: a B4
> *start-after-stop* TOCTOU (a run starting in the join→move gap can still resurrect the dir — needs a
> per-project delete tombstone; low risk on single-user localhost).

### BUG-B1 — `eval()` on checkpoint JSON → server-side RCE  🟡 major *(verifier: "could argue critical")*
`viz/plotting.py:54-61` `load_checkpoint` runs `eval(entry["log_file"])` (the author's own comment: *"Only
use this on trusted checkpoint files"*). `checkpoint_reader.read_json_checkpoint:16-18` calls exactly that,
and it backs **`GET /api/checkpoint/{id}`, `/envelope`, `/scatter`, `/report`**. `_resolve_checkpoint_path`
globs `runs_root()/projects_root().rglob` over `WORK_ROOT` (the host bind-mount `/work`, writable outside
the app), so a checkpoint with `"log_file": "__import__('os').system('…')"` executes on any analysis
request. The safe reader the codebase already has (`optimizer_runner._load_checkpoint_log`, which sets
`log_file=None`) and the catalog's `_count_iters` deliberately avoid this path — the analysis routes did not.
**Fix:** use `ast.literal_eval` (or set `log_file=None`) in `load_checkpoint`, or route these endpoints
through the non-eval reader. *(The replay reach is gated to preset checkpoints only, so it is the
load/envelope/scatter/report autosave paths that are exposed.)*

### BUG-B2 — Arbitrary file read (LFI) via `yaml_path` in checkpoint report  🟡 major
`checkpoint.py:339-342` (`GET /api/checkpoint/{id}/report`) does `yp = Path(yaml_path); if yp.exists() and
yp.is_file(): z.writestr("project_setup.yaml", yp.read_text())` with **no** suffix/root check, so
`?yaml_path=/etc/passwd` (or `.env` with API keys, SSH keys) is returned verbatim inside the zip. Contrast
`/yaml-text` (`project.py:219`, gated to `.yaml/.yml`) and `_assert_under_work_root`. **Fix:** resolve and
require the path under an allowed root (REPO_ROOT/examples or WORK_ROOT) and/or restrict to `.yaml/.yml`.

### BUG-B3 — Unscoped `DELETE /checkpoint/{id}` wipes same-named checkpoints across **all** projects  🟢 major *(irreversible)*
`checkpoint.py:223-264`: with `project_id` omitted, `all_projects=True` → roots span every project's
`checkpoints/`. Stems are `{name}_{algo}_{budget}_trial{N}[_FINAL]` (no project/run id), and
`copy_example`/`fork` reproduce identical stems, so the route `unlink()`s the match under **every** root.
`RunsRail.tsx:157` calls `deleteCheckpoint(id, undefined)` whenever no project is active (global view), so a
user deleting one checkpoint there silently and irreversibly destroys a fork/copy's identically-named file.
The route's own docstring acknowledges the hazard yet still defaults to the global search. **Fix:** require a
`project_id` (or delete only an unambiguous single match; 409 on multi-root collision).

### BUG-B4 — Soft-delete/delete-run races a slow trial → worker resurrects the moved-away dir outside `.trash`  🟡 major *(medium confidence)*
`optimizer_runner.stop_runs_for` (`:601-606`) sets `stop_event` then `thread.join(timeout=10.0)` and returns
the target **count**, not liveness. `stop_event` is only read at a trial boundary (`optimization_step` top),
so a single multi-testbench ngspice trial >10 s outlives the join. `delete_project`/`delete_run`
(`projects.py:130,176`) then `shutil.move` the dir to `.trash`; the still-alive worker's final
`save_checkpoint` does `path.parent.mkdir(parents=True, exist_ok=True)` (`base.py:428`) at the cached old
path, **re-creating `projects/<P>/runs/<id>/` outside the trash** (the exact resurrection the docstring
claims to prevent) — incomplete soft-delete + later `dst.exists()` 409 on restore. **Fix:** check
`thread.is_alive()` after join and 409/retry, or poll `stop_event` inside the sim wait and guard
`save_checkpoint` against writing once the run dir is gone.

---

## Tier 1 — Major: NEWCAS core library (on the `examples/OTA/cascode` path)

### BUG-B5 — Base `optimize()` autosave empties the log but not `global_best_index` → IndexError mid-run  🟡 major
`base.py:178-198`: on autosave `self.optimization_log = OptimizationLog()` (empty), then the same iteration
reads `self.optimization_log[self.global_best_index]` (`:189`) → **IndexError**, escaping the loop's
`except KeyboardInterrupt`. Fires at the first autosave boundary. Default `autosave_checkpoint_freqeucny` is
2500 so `budget ≥ 2500` triggers it; the reference script `nevergrad_single_obj_opt.py:40` literally shows
the commented `optimizer.autosave_checkpoint_freqeucny = 5`, which crashes at trial 4. The UI's
`_StreamingOpt.optimize()` override exists precisely to dodge this; the bare library/CLI path does not.
**Fix:** reset `global_best_index` on log-reset and track best on the instance (or index by
`len(log)-1`).

### BUG-B6 — `denormalize_params` maps `val/range` instead of `(val−min)/range` → log/non-zero-min params escape their bounds  🟡 major
`base.py:130-151`: `parameterize` (`nevergrad.py:148-155`) samples `ng.p.Log/Scalar` in `[min,max]`, but the
normalized coordinate is `val/range` (`range=max−min`), which only lands in `[0,1]` when `min==0`. With the
example's `log_variable_bounds {min:1,max:100}`, a `log_scale:true` dut_param at `val=100` → `x=1.0101` → a
**physical value above the declared `max_val`** (and `val=1` never reaches `min_val`). Linear has the same
off-by-offset for any non-zero `lin_variable_bounds.min`; it's masked today only because the example uses
`min:0`. `log_scale` is exposed per-param in the wizard. **Fix:** normalize as `(val − bounds.min)/range`.

### BUG-B7 — `target` in `XeY` YAML notation stays a `str` → load crash / type-contract break  🟢 major
`domains.py:380-468`: `TargetSpec.__post_init__` runs `parse_value` on `range` and `tolerance` but **never on
`target`**, and the `list_target_spec_hook` path bypasses dacite casting. PyYAML 1.1 parses `200e6`/`25e-6`/
`15e-6` as **strings** (unsigned-exponent, dot-less mantissa). The shipped cascode YAML loads with
`ugf.target=='200e6'` etc. (reproduced). If any such target *omits* tolerance, `from_yaml` crashes at `:458`
(`abs(0.05 * self.target)` → `TypeError`); `meets_spec`/`get_simple_penalty` raise `UFuncTypeError` on a
string target. The example survives only because each `XeY` target carries explicit tolerance. **Fix:**
`self.target = parse_value(self.target)` in `__post_init__` (and `self.weight`).

### BUG-B8 — All-numeric `dut_param` bounds skip the `min_val ≥ max_val` validation → silent reversed/zero-width range  🟡 major
`domains.py:319-327,836-848`: the only `min ≥ max` guard is inside `resolve_min_max`, which
`resolve_all_parameter_ranges` calls **only** when `needs_resolution()` (a *string* bound). A param with
plain-numeric `min_val:5,max_val:1` (or `min==max`) is never validated → `linear_denormalize(pmin=5,pmax=1)`
silently inverts the search range (the integer path late-crashes in nevergrad). **Fix:** always validate
`min ≥ max` after coercing numeric bounds (in `resolve_all_parameter_ranges` or `Param.__post_init__`).

### BUG-B9 — `weight: null` / blank in a target spec → `np.float64(None)=NaN` poisons every trial's score  🟡 major
`domains.py:380-468` never normalizes a `None` weight (the `1.0` default applies only to an *omitted* key,
and the hook bypasses dacite type enforcement); `base.py:1007/1075` then computes `spec_penalty *
np.float64(None)` → NaN, so aggregate fitness is NaN and Nevergrad cannot rank it. The preview
`score_service` guards (`weight if not None else 1.0`) but the **library scorer does not**, so the UI
preview and the live run disagree. **Fix:** coerce `if self.weight is None: self.weight = 1.0` in
`__post_init__`.

### BUG-B10 — Frozen `dut_param` with eng-string `val`/`init` and no min/max crashes `from_yaml`  🟡 major
`domains.py:316-321,836-847`: a string `val`/`init` makes `needs_resolution()` True, so
`resolve_all_parameter_ranges` calls `resolve_min_max`, which raises *"missing min or max value"* — even
though a frozen constant legitimately omits bounds (it's excluded from the search). `freeze` is never
consulted in that loop. A plain-numeric `val: 0.18e-6` loads fine; the eng-string form `val: "0.18u"` (the
encouraged format) fails the whole load with a misleading message. **Fix:** skip range resolution for frozen
params (resolve only their `val`/`init`).

---

## Tier 2 — Major: PVT

### BUG-B11 — `apply_corner` collapses multiple `model_includes` sharing one `lib_file` to only the last section  🟡 major/pvt
`spicelib.py:369-376`: the strip key is the lib-file **basename only** (`\.lib\s+\S*<basename>\s+\S+`). A
corner with `[{models.lib, nmos_tt}, {models.lib, pmos_tt}]` ends up with only `.lib models.lib pmos_tt` —
iteration 2's strip deletes the just-added `nmos_tt` line. Silent wrong device models, no error. The IHP
example dodges it only because each family uses a distinct file. **Fix:** key the strip on
`(basename, section)`, or snapshot the original `.lib` lines to strip once before the add-loop.

### BUG-B12 — PVT supply override silently no-ops on a rail not declared as a `.param`  🟡 major/pvt
`spicelib.py:383-389`: a supply is applied via `ed.set_parameter(s.node, s.value)`, which (per spicelib)
**inserts a new `.PARAM`** when no matching `.param` exists rather than erroring. So a corner naming
`{node: VSS, value: 0}` or the source instance `Vdd` (instead of the param `VDD`) adds a dangling,
unreferenced `.PARAM` and the sim runs at the **netlist-default supply** with no diagnostic. The temp branch
above it strips-then-asserts authoritatively; supplies have no such guard or post-check, and node names are
not validated against actual sources. PVT supply variation is a core purpose of the feature. **Fix:** after
`set_parameter`, verify the param existed (`get_parameter` in try/except) and warn/raise; document that
`node` must match a `.param` name.

---

## Tier 3 — Major: backend & UI (encapsulation / lifecycle)

### BUG-B13 — Concurrent runs cross-contaminate `run.log` + SSE log via the shared `spicexplorer` logger  🟡 major
`optimizer_runner.py:372-412`: each `_run_live` attaches its `_QueueLogHandler` **and** a per-run
`FileHandler` to the process-global `logging.getLogger("spicexplorer")` with **no run-scoping filter**
(`_QueueLogHandler.emit` forwards every record). Nothing serializes runs (`start_run` spawns a daemon thread
per call; no 409 gate), and report.md Phase 2 explicitly designs for "two concurrent runs produce disjoint
dirs" — but the **log** artifact, unlike sim/checkpoints/events, is not isolated, so each run's library lines
land in the other's `runs/<id>/run.log` and bottom-panel SSE tab, defeating the "zip-and-hand-over" contract.
**Fix:** scope each handler with a `logging.Filter` keyed on the worker thread id (or a per-run child
logger), or reject a second concurrent live run with 409.

### BUG-B14 — xschem `_allowed_roots()` omits `WORK_ROOT` → every encapsulated project's schematic 403s under Docker  🟢 major
`xschem.py:124-142`: `_allowed_roots()` whitelists only `REPO_ROOT` (+ PDK/share). Encapsulated projects
live under `work_root()` = `/work` in Docker (outside `/app` = REPO_ROOT). After `from-example`,
`get_project` returns `yaml_path=/work/projects/<id>/...`; `SchematicTab` → `/api/xschem/project` →
`_validate_under_allowed` raises **HTTP 403** for `/work/...`. The whole schematic viewer is dead for
encapsulated projects under the documented portable artifact; native works only by accident (WORK_ROOT
defaults under REPO_ROOT). **Fix:** add `work_root()` (resolved) to `_allowed_roots()` and `_search_roots`.

### BUG-B15 — Wizard YAML round-trip drops `dut_param.val` → un-pins frozen parameters  🟡 major
`yaml_generator.py:36-53,326-339` (+ `types/api.ts` `WizardDutParam`, which has **no `val` field**):
`_build_dut_param` emits `init` but never `val`; `project_dict_to_form` carries `init` but never `val`. A
frozen pinned param (`freeze:true, val:0.18u`) loaded into the wizard and re-saved loses `val`, so
`parameterize` (`nevergrad.py:138`, `fixed = param.val if not None else param.init`) silently falls back to
`init`/netlist default — **changing the frozen device's operating point and the optimized design**. The
`copy_example`/wizard flow makes parse→edit→save a normal path. **Fix:** add `val` to `WizardDutParam`, carry
it in `project_dict_to_form`, emit it in `_build_dut_param`.

### BUG-B16 — Project switch mid-run never tears down the `runStore` EventSource → old run streams under the new project  🟢 major
`projectStore.ts:73-97` (`switchProject`) rebinds `summary`/`yamlPath`/`projectId` but never touches
`runStore`; the module-level EventSource keeps streaming. Start a run on A, ⌘P → click B: RightRail/StatusBar
keep showing "running" and the **still-running A optimizer's** `best_metrics`/`iter` events display as if
they belong to B; A is never told to stop (no `api.stopRun`) and keeps consuming SPICE. `RunsRail` refetches
B's runs while A streams. **Fix:** in `switchProject` (and fork-then-switch), if `useRunStore.isRunning`
either block the switch or `useRunStore.getState().stopRun()` before rebinding.

---

## Tier 4 — Minor (grouped by subsystem)

### Core / scoring (NEWCAS path)
- **BUG-B17** 🟡 `domains.py:457-462` — tolerance fallback yields `0` when `target==0`, defeating its own
  `>0` guard (zero-width band; `minimize`/`exact` then treat any nonzero value as a violation). Floor to a
  small positive epsilon.
- **BUG-B18** 🟡/pvt `domains.py:180-186` (via `parse_value` `:113-123`) — a present-but-blank `temp:` /
  supply `value:` / `params` entry calls `parse_value(None)` → bare `AttributeError`, surfacing as the opaque
  "Unexpected error while loading". Guard `None` and raise a descriptive `ValueError`.
- **BUG-B19** 🟡 `base.py:967-970,1039-1042` — for `log_scale:true` specs the scorer log10's the **tolerance
  width** (`convert_linear_to_log(tolerance)`), giving an absurd / negative band (`tolerance<1` → negative →
  pass/fail inverts). Opt-in; no shipped config enables `log_scale`. Derive the band from transformed bounds.
- **BUG-B20** 🟡 `base.py:1048-1052` — EXCEED reward `elif spec_curr_val > target_val + tolerance` is dead
  code (the first `if` already fired), and EXCEED vs MINIMIZE rewards are structured asymmetrically. Remove
  the dead branch or implement the intended within-tolerance region.
- **BUG-B21** 🟡 `utils.py:211-218` — `compute_relative_log_reward`/`compute_log_reward` produce `±inf` when
  `curr==target` (or `curr,target ≤ 0`) on the MINIMIZE reward path (clipped to MAX_REWARD, so one point gets
  a spuriously huge bounded reward). Opt-in `reward_type`. Add an epsilon floor / positivity guard.
- **BUG-B22** 🟡 `utils.py:181-182,202-203` — `compute_exponential_error` (and the relative variant)
  overflow to `inf` on raw SI magnitudes → saturated `-MAX_PENALTY` flattens the optimizer gradient between
  distinguishably-bad points. Opt-in `error_type`. Clamp the exponent argument.
- **BUG-B23** 🟡/design `base.py:943` — `total_score = reward if penalty > -EPSILON else penalty` discards
  **all** reward while any spec is violated, so improving a satisfied spec never changes the score until all
  constraints pass (large flat regions can stall Nevergrad). Likely intentional constraint-first shaping —
  document it, or blend `penalty + α·reward`.
- **BUG-B24** ⚪ `domains.py:345-346` vs `utils.py:294-307` — `compute_log_normalization` (RL backend path)
  uses base-`e`; the Nevergrad `log_denormalize` uses base-10, so the two backends search different spaces
  for the same `log_scale` param. (RL backend is dormant.) Pick one base.
- **BUG-B25** ⚪ `base.py:164,178-179,189` — `global_best_index = trial` (absolute) is wrong under
  `keep_history=True` (resume / 2nd `optimize()` on a loaded instance): it indexes a stale old entry, so
  `get_best_params()` returns the wrong best. No shipped caller chains this today. Track best by
  `len(log)-1`.

### Optimizer / lifecycle (backend)
- **BUG-B26** ⚪ `base.py:699-707` + `nevergrad.py:209-215` — `Spice_Bode_Optimizer.__init__` **and**
  `Nevergrad_Spice_Constraint_Satisfaction.__init__` drop the `output_root` kwarg their bases accept, so
  per-run checkpoint isolation is unavailable and constructing them with `output_root=` raises `TypeError`.
  Latent (UI uses the single-objective class). Forward `output_root`, mirroring the single-objective sibling.
- **BUG-B27** 🟡 `optimizer_runner.py:311,435` — resume loops `range(budget)` (full budget) while
  `optimization_log` is pre-seeded with N prior trials, so a resumed budget-B run does B *more* (total N+B)
  and `iter` climbs past budget. Iterate `max(0, budget − len(log))` or document incremental resume.
- **BUG-B28** 🟡 `base.py:529-531` — in the **serial** sim path a `curr_raw is None` (ngspice
  non-convergence on an extreme candidate) raises `RuntimeError` and aborts the whole run (losing in-memory
  trials since the last checkpoint); the parallel default path scores it as NaN→MAX_PENALTY gracefully. Catch
  the no-RAW case per testbench and assign NaN instead of raising.
- **BUG-B29** 🟡 `base.py:884-888,933-937` + `domains.py` `ListTargetSpec` — duplicate target-spec **names**
  across testbenches collide in `performance_array`/`fit_summary` (name-keyed) and silently overwrite — no
  uniqueness guard, unlike `dut_params` and PVT corners. Reject duplicate names or key by `(testbench, name)`.
- **BUG-B36** 🟡 `optimizer_runner.py:570-575` + `optimize.py:64` — `POST /api/optimize/start {"replay":true}`
  with no `checkpoint_id` schedules **no** task, never sets `state.done` or queues the sentinel → the SSE
  stream heartbeats "running" forever and the run is never pruned. The UI always supplies an id (direct-API
  only). Reject replay-without-id at the route, or enqueue an error + sentinel.
- **BUG-B41** 🟡 `simulate.py:185`, `sanity.py:185`, `sensitivity.py:148` (via `base.py:69-73`) — these
  one-off routes construct the optimizer with **no `output_root`**, so each request `mkdir`s an empty
  CWD-relative `./auto_save/<name>_<algo>_<ts>` (`/app` in Docker), unbounded, and these dirs are then
  uselessly rglob'd on every checkpoint listing. Pass `output_root` under `scratch/` or set
  `disable_autosave=True`.
- **BUG-B42** ⚪ `optimizer_runner.py:407,474-489` + `project_service.py:136-154` — on SIGKILL/OOM the
  `finally` that finalizes `run.json` never runs, and `reconcile_stale_runs` only repairs `running`→`error`
  at the **next startup** — so a hard-killed run read before any restart (or via a fresh container with a
  different `WORK_ROOT`) shows perpetually "running" (no heartbeat/PID liveness in `list_runs`).
- **BUG-B43** 🟡 `project_service.py:136-154` — `reconcile_stale_runs` scans only `projects_root()`/
  `runs_root()`, **not `.trash`**, so a crashed `running` run that was soft-deleted before restart stays
  "running" forever and reappears as such on restore. Also walk `trash_root()`, or re-reconcile on restore.

### Routes / scoping (backend)
- **BUG-B35** 🟡 `checkpoint.py:59-63` + `checkpoint_reader.py:144,194` — `_target_specs_from_yaml` emits
  `"tolerance": float(s.tolerance) if s.tolerance else None`, so a spec with `target:0` (tolerance defaults
  to `0.0`, falsy) serializes `tolerance:None`; `spec.get("tolerance", default)` returns the present `None`
  and `target − None` → **TypeError 500** on `/envelope`, `/scatter`, `/report` (the wizard's blank-tolerance
  default also produces `tolerance:None` specs). Coalesce `spec.get("tolerance") or abs(0.05*target)`.
- **BUG-B37** 🟡 `optimize.py:48-50` + `project.py:167-172` — a malformed `project_id` (`/`, `\`, `..`) makes
  `project_dir` raise `ValueError`, which `start_run`/`load_project` don't catch (only `FileNotFoundError`) →
  opaque **500** instead of 404/400 (no traversal occurs — only the status is wrong). `get_project`/
  `project_runs` funnel through `project_exists` and 404 correctly; mirror that.
- **BUG-B38** 🟡 `checkpoint.py:66-80` — `_resolve_checkpoint_path` does `rglob(f"{id}*.json")[0]` (prefix
  glob, order-dependent), so `..._trial2` matches `..._trial2`, `..._trial20`, `..._trial200`; combined with
  cross-project stem collisions (forks share stems) and no `project_id` threading, load/envelope/scatter/
  report/resume can silently operate on the **wrong** run. Match exact `p.stem` (as `delete_checkpoint` does)
  and thread `project_id`.
- **BUG-B39** 🟡 `checkpoint.py:59-63` — `_target_specs_from_yaml` iterates `.targets` (all specs) not
  `enabled_targets()`, so a `enable:false` spec gates envelope `passes`/scatter `feasible` even though the
  optimizer never scored it → analysis views disagree with the objective. Filter to `enabled_targets()`.

### PVT
- **BUG-B30** 🟡/pvt `spicelib.py:378-381` — the temp-strip regex `^\s*\.options?\s+temp\s*=` anchors `temp`
  as the *first* token, so a combined `.options reltol=1e-3 temp=27` is **not** stripped (stale temp lingers,
  idempotency broken) — or, where it *does* match a `.options temp=27 gmin=1e-12` line, `remove_Xinstruction`
  deletes the **whole line**, dropping `gmin`/`reltol` (violating the "single-corner = strict superset of
  legacy" contract). No shipped netlist puts `temp` first. Inject `temp=` as its own line / rewrite only the
  `temp=` token.
- **BUG-B31** 🟡/pvt `optimizer_runner.py:356-363` + `sanity.py:112-116` + `simulate.py:169-177` — an unknown
  `active_corner` override is handled **three different ways**: manual-sim appends a user-visible warning; the
  live run logs to `getLogger(__name__)` (NOT the `spicexplorer` logger the SSE handler listens on, so it's
  **not surfaced**); sanity silently falls back to the YAML default **and reports the fallback in
  `active_corner` as if honored** (no `warnings` field). Reachable when a stale/typo'd corner lingers
  client-side after a YAML/project change. Surface it uniformly (422, or a `warnings` field everywhere).
- **BUG-B32** 🟡/pvt `domains.py:161-172` — a corner declaring both `process: <bundle>` and inline
  `model_includes:` silently `pop`s and discards `process` (the dangling-bundle `ValueError` only fires when
  `model_includes` is absent) → wrong process models with no diagnostic if the inline list is incomplete.
  Raise/warn on the ambiguous both-keys case.
- **BUG-B33** 🟡/pvt `sanity.py:126-141` + `base.py:501-509` — the per-testbench sanity sims call
  `run_sanity_check(use_editor=False)`, which runs the **raw on-disk netlist**, so `apply_corner` is **never
  applied** to the per-tb Health pass/fail rows the user reads (only the single trial step uses the corner) —
  yet the response advertises one `active_corner`. Apply the corner before the per-tb sim, or label those
  rows corner-independent. *(The finder's secondary rmtree/log-loss claim was a misdiagnosis — `curr_log` is
  simply never set on the RUN_NOW sanity path.)*
- **BUG-B34** 🟡/pvt `domains.py:286-294` + `base.py:501-509` — `get_active()` resolves by name and ignores a
  corner's `enabled:false`, and `__post_init__` applies it unconditionally, so an `active_corner` pointing at
  a disabled corner still drives the sim (inconsistent with `enabled_corners()`). *(Dissent: one verifier
  judged the log-and-apply behavior intentional; treat as a consistency wart, latent until Phase 2.)*

### Frontend
- **BUG-B40** 🟡/pvt `yaml_generator.py:163-213,259-308` — the wizard round-trip drops per-corner PVT
  `params:` overrides (neither read in `_pvt_block_to_form` nor emitted in `_build_pvt_block`), so a corner's
  `params:{vbias:0.7}` reverts to netlist defaults after a wizard Save (same class as the fixed
  model_lib_root/multi-rail drops). Carry + re-emit `params`.
- **BUG-B44** 🟡 `projectStore.ts:109-123` — deleting the **active** project calls only `projectStore.reset()`
  and never touches `runStore`, so the EventSource + `isRunning` survive → RightRail/StatusBar show a phantom
  running run for a deleted project until the backend sentinel arrives. Call `useRunStore.getState().reset()`
  when `projectId === id`.
- **BUG-B45** 🟡 `pareto.ts:17-20` + `MetricScatterChart.tsx:43-46` — `paretoFront`/`feasibleRect` branch only
  on `goal==="minimize"`, so an **exact** goal axis is treated as larger-is-better → wrong Pareto frontier and
  an unbounded half-plane feasible rect instead of a band around target (per-point feasibility from the
  backend is correct, so the overlay visibly contradicts the point colors). Same closest-to-target rule that
  was fixed for best-of/convergence, not applied to the overlays.
- **BUG-B46** 🟢 `lib/utils.ts:10-19` — `formatEng(0)` returns `"0.000 p"` (zero falls through to the pico
  fallback branch), so any zero metric/param/best-score shows a spurious pico prefix in RunsRail/RightRail/
  DeviceInspector. Special-case `value===0`.
- **BUG-B47** 🟡 `runStore.ts:142-213` — superseding an in-flight run (new live run, or clicking a different
  replay) fires `stopRun`+`closeStream` but never `finishRun()`, the only writer of a localStorage history
  `RunRecord`, so the superseded run vanishes from history despite producing trials. Snapshot+record the prior
  run before resetting.
- **BUG-B48** 🟡 `ScoreShapingTab.tsx:53-60,143-150,292-300` — typing an engineering string (the editor
  invites "250u") into the **target** field makes `Number("250u")=NaN`, which `??` does not recover, so
  `range`/`sliderMin`/`sliderMax`/`step`/marker all go NaN and the slider breaks (the penalty *curve* is
  server-parsed and stays correct). Parse with the same eng-string parser as the backend; guard non-finite.
- **BUG-B49** ⚪ `ExplorerTab.tsx:194-197,509-510` — the scatter point-click resolves the run by
  `runLabel === runB?.label ? "B" : "A"`, so when A and B carry the **same label** every click routes to B
  (wrong run in the inspector / re-simulate). Disambiguate by trace index/`customdata`, not label equality.
- **BUG-B50** 🟡 `simulate.py:184` + `spicelib.py:236-242` — `/simulate/once` always uses the fixed
  `ws_root/outdir/manual_sim`, dispatched on the shared executor pool, so two overlapping manual sims (or a
  manual sim + the Explorer "Re-simulate", same route) race the per-wrapper `rmtree` of that folder →
  corrupted result or `FileExistsError`. Give each manual sim a unique subfolder.
- **BUG-B51** 🟡 `RunsRail.tsx:204-212` — the inline run-rename input commits `onBlur`, and Escape only clears
  `editingRunId` (no cancel flag), so the unmount-triggered blur still calls `commitRunRename` → **Escape
  saves the draft instead of cancelling** (ProjectsOverlay's rename has no `onBlur` and cancels correctly).
  Set a cancelled flag in the Escape branch and suppress the blur-commit.
- **BUG-B52** 🟡 `ProjectsOverlay.tsx:71-101` — `doLoadExample`/`doCreate` only `closeProjects()` on
  `switchProject` success and set **no error** on failure (e.g. a 422 when the copied/wizard YAML fails
  `from_yaml`), so the project is created on disk but the overlay gives no feedback and the user may retry,
  creating duplicates (`doFork` handles this correctly). Mirror `doFork`'s `setError` + `refreshProjects`.

---

## Cross-cutting root causes (fix once, resolve several)

1. **`tolerance:None` serialization** (`checkpoint.py:61`) is the single root cause of B35 (and the `target:0`
   500s). Emit a resolved tolerance / coalesce on read.
2. **`output_root` not threaded** through `Spice_Bode_Optimizer`, `Nevergrad_Spice_Constraint_Satisfaction`,
   and the one-off `/simulate`-`/sanity`-`/sensitivity` routes (B26, B41) — the encapsulation work covered
   only the single-objective live path.
3. **Cross-project checkpoint stem collisions** (B3, B38) — stems carry no project/run id, so global-scope
   resolution/delete can hit the wrong (or every) project. Add an id to the stem or always scope by
   `project_id`.
4. **`stop_runs_for` bounded join ignores liveness** (B4) — the lifecycle DELETE/move/fork paths assume the
   worker stopped; it may not have.
5. **Shared `spicexplorer` logger has no run scoping** (B13) + the live-run corner warning goes to a logger
   the SSE handler doesn't listen on (B31) — per-run log isolation and PVT warning surfacing both depend on it.
6. **Exact-goal "closest-to-target" rule** (fixed for best-of/convergence/envelope in §17) was **not** applied
   to the FE Pareto/feasible-region overlays (B45).
7. **No cross-store teardown** between `projectStore` and `runStore` on switch/delete (B16, B44).

---

## Considered and refuted (17) — recorded so they aren't re-raised

- *log_scale base-e vs base-10* — real code divergence but RL backend is dormant (folded into **B24** as a
  minor note, not a live bug).
- *Streaming resume leaves `global_best_index` at 0* — the resume seeds `_best_score`/`_best_params`/
  `_best_metrics` from history, which is what's used; `global_best_index` is not read on that path.
- *`_StreamingOpt.optimize()` resets log but not seeded `_abs_iter`/best* — no reachable trigger (non-resume
  branch builds a fresh optimizer with an empty log).
- *`compute_fitness` numpy-truthiness short-circuit* — refuted: the only writer of `performance_array`
  produces scalars, never 0-d arrays.
- *Base `optimize()` unbound `trial` / wrong-best on resume when `budget==0`* — guarded by
  `not optimization_log.is_empty()`; the reachable base path is `keep_history=False` (empty-start), so the
  global-index-vs-trial coincidence holds (the *latent* `keep_history=True` variant is kept as **B25**).
- *`_config_snapshot` writes `active_corner` under only the first root* — false; the loop iterates both roots.
- *`soft_delete_project` trash-id lacks a uniqueness suffix (same-second re-delete clobbers)* — flagged by
  two finders; verifiers judged it not currently reachable / mitigated. **Watch item**, not confirmed.
- *`restore_project` joins an unvalidated `meta["name"]`* — `_assert_under_work_root` blocks escape *above*
  WORK_ROOT (not lateral); verifiers found a mitigating factor. **Hardening candidate**, not confirmed.
- *History `bestScore` assumes higher-is-better* — false; the replay emitter sends a per-row `best_score`.
- *`rerun`/resume/manual-sim resolve checkpoints across all projects (foreign load)* — the cross-project
  scoping is real but is captured by **B3/B38**; the standalone "foreign load" framing was over-stated.

> **Verification footing.** Findings above were each confirmed by an adversarial verifier re-reading the
> cited code; line numbers were spot-corrected by the verifiers. Nothing here was run against a live sim,
> so the few items tagged "needs a live trial to confirm the visible symptom" (B4, B13, B31) are confirmed
> at the source/data-flow level. Recommended next step: land **Tier 0** (security + data-loss) first, then
> the NEWCAS-core majors (**B5–B10**) and PVT majors (**B11–B12**), each with a regression test, before the
> minors.
