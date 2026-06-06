# Strategy Evaluation — Project Encapsulation & Run Isolation under `/work`

> **Scope of this document.** Strategy and end-goal clarity only — **no code**. It evaluates the
> idea of giving the web app a well-defined, persistent filesystem under the Docker `/work` mount so
> that (a) each project is an encapsulated, self-contained directory separate from the git repo, and
> (b) each optimization *run* is an isolated, named, debuggable unit. It was produced by five
> independent expert analyses (filesystem, backend/API, lifecycle-UX, Docker/ops, migration/risk),
> reconciled here against the actual code at `HEAD`. Every factual claim below was verified by
> reading the cited file.

---

## 1. Verdict & End-Goal Vision

**Yes — do this. It is the right idea, and it is also fixing a real bug you have today, not just a
UX nicety.** The caveat is scope: build the *durable, isolated filesystem* in full, but ship the
*"project manager"* (fork/rename/tags/retention) only as far as one user and a handful of projects
actually need. The encapsulation model is desirable precisely because ~70% of it is *relocating
behavior you already have* (`outdir`, `auto_save`, the `/work` mount, the absolute-Save hook) onto a
clean root, and only ~30% is genuinely new (a per-run workspace + a small project registry).

**The end state.** You open the Studio, click the project name in the title bar (or ⌘P), and pick
from a list of projects — each one a folder under `/work/projects/` with its own config, netlists,
schematics, and run history. You create a new project from the wizard and it scaffolds a clean
template directory; you "load example" and it **copies** the example into a fresh, mutable project
(the repo's `examples/` stays pristine and read-only). You start an optimization and it writes into
`runs/<run_id>/` *inside that project* — checkpoints, the full log, the exact config it ran with,
and the sim outputs, all in one place. When a run misbehaves, you open its folder (or its row in the
Runs panel) and everything needed to debug it is right there, self-contained. Nothing lands in the
git tree; nothing is lost when the container is recreated.

---

## 2. The Core Problem Today

Three concrete defects make this work load-bearing rather than cosmetic:

1. **Run checkpoints are silently destroyed in Docker.** [base.py:63](../../src/spicexplorer/optimization/base.py#L63)
   sets `autosave_checkpoint_dir = Path("./auto_save/…")` — **relative to the process CWD**. The
   backend's CWD in the container is `/app` ([Dockerfile.backend:138](../../docker/Dockerfile.backend#L138)),
   and the entrypoint even pre-creates `/app/auto_save` ([entrypoint-backend.sh:30-31](../../docker/entrypoint-backend.sh#L30-L31)).
   So every live-run checkpoint is written **inside the image layer, not the `/work` bind-mount**,
   and is **gone on `docker compose down` / `docker rm`**. This is data loss, today.

2. **The CWD-coupling forces a fragile dual-root search.** Because the write path is non-deterministic,
   [checkpoint.py:60-68](../../ui/backend/routes/checkpoint.py#L60-L68) has to scan **both**
   `REPO_ROOT/auto_save` and `Path.cwd()/auto_save`, and the native launcher
   [run_newcas_ui.sh:74](../../scripts/run_newcas_ui.sh#L74) does a forced `cd "${ROOT_DIR}"` *just* to make
   the two coincide (its own comment at [:72](../../scripts/run_newcas_ui.sh#L72) admits this). That is the
   live evidence the path is broken.

3. **There is no "project" — only a loose YAML path, and runs all pool together.** `POST /api/project/load`
   ([project.py:129-162](../../ui/backend/routes/project.py#L129-L162)) takes an arbitrary `yaml_path` or
   raw `yaml_content` (written to a `/tmp` temp file). There is no per-project home, so every
   project's autosaves pile into one flat `auto_save/` keyed only by `name_timestamp`. Multiple
   projects bleed into each other; a run is a 25-item `localStorage` blob, not an inspectable unit.
   And `auto_save/` is **not in `.gitignore`** (only `work/` is, [.gitignore:31](../../.gitignore#L31)), so
   the repo-root pile is git-visible and committable.

**Isolation is the real win.** Fixing (1)–(3) makes runs durable, deterministic, per-project, and
self-contained — which is exactly your stated goal of "organize and isolate runs for UX *and*
debugging." Encapsulating projects is the natural container for that isolation.

---

## 3. Target Filesystem Layout

The unit of encapsulation is **one directory per project**; the unit of isolation is **one directory
per run** *inside* that project. Examples and the PDK are **referenced, never copied**.

```
/work/                                       # WORK_ROOT — the ONLY mutable tree (bind mount, host-visible)
├── projects/
│   └── cascode-ota-a1b2c3d4/                # project dir = <slug>-<id8>  (slug human-readable, id8 = uuid4 hex[:8])
│       ├── project.yaml                      # the Project_Setup DSL.  ws_root: .   (this dir)
│       ├── manifest.json                     # {id, slug, name, created, source:{kind,ref}, last_run_id, schema_version}
│       │                                     #   UI/backend metadata only — core/domains.py never reads it
│       ├── spice/                            # COPIED netlists.   netlist: spice/ota-improved.spice  (ws_root-relative)
│       ├── xschem/                           # COPIED schematics + .svg.  schematic: xschem/ota-improved.sch
│       ├── scratch/                          # outdir for EPHEMERAL one-off sims (out of the source tree)
│       │   ├── manual_sim/   sanity/   sensitivity/
│       └── runs/                             # append-only, per-RUN isolation — REPLACES the global ./auto_save
│           └── 2026-06-05_14-53_LogBFGSCMAPlus_7f3a9c21/   # <ts>_<algo>_<runid8>
│               ├── run.json                  # {label, algo, seed, budget, active_corner, status, started/ended,
│               │                             #  best_score, parent_run_id?, override_diff}  ← reproducibility
│               ├── config_snapshot.yaml      # exact applied Project_Setup at launch (ephemeral overrides BAKED IN)
│               ├── checkpoints/*.json        # autosave_checkpoint_dir → here (per-run, not global)
│               ├── run.log                   # the full DEBUG library log (was SSE-ephemeral)
│               ├── events.ndjson             # the SSE event stream, replayable offline
│               └── sim/                       # live sim outputs (the wrapper rmtree's only THIS run's dir)
├── .trash/                                   # soft-deleted projects/runs (mv here, GC later) — only if/when delete ships
└── (logs/ optionally)

examples/   (IN-REPO, READ-ONLY — NOT under /work, unchanged)
   examples/OTA/cascode/ihp-sg13g2/{sizing/project_setup.yaml, spice/*, xschem/*}
   → keeps  ws_root: ..  and runs from a fresh clone with zero edits.   (the portability contract — §8)

/opt/pdk  (ihp-sg13g2, IN-IMAGE)  → referenced via PDK_ROOT; corner .lib files resolve at sim time.  NEVER copied.
```

**Copied vs referenced.** On "load example" / "new project": **copy** `project.yaml` (rewriting
`ws_root → .`, `outdir → scratch`), `spice/`, and `xschem/` into the new project dir. **Never copy**
the PDK (it's large and external — `pvt.model_lib_root` / `PDK_ROOT=/opt/pdk` stays a reference) and
**never copy** the in-repo `examples/` themselves or the preset demo checkpoints (those stay served
read-only from the repo). Isolation is worth the duplicated netlists; dedup is a non-goal.

---

## 4. The `WORK_ROOT` Resolution Rule (resolves the cross-lens conflict)

The lenses split on *where* runs live (per-project vs a global `/work/runs/`). **Decision:
per-project `runs/`** — a global pool would isolate runs from each other but *not per project*, which
fails your explicit goal. But the ops lens's `WORK_ROOT` rule is correct and adopted wholesale,
because it's the one thing that has to work identically in Docker and native dev:

```
WORK_ROOT =
   1. os.environ["WORK_ROOT"]      if set        # Docker compose sets it to /work
   2. else  <REPO_ROOT>/work                      # native default — already gitignored (.gitignore:31)
   → .expanduser().resolve();  mkdir(parents=True, exist_ok=True) on first use
```

- Add `work_root()` to [app_config.py](../../ui/backend/app_config.py) as the single source of truth.
  `resolve()` stays `REPO_ROOT`-relative **for read-only assets** (examples, presets, schematic SVG);
  `work_root()` is the base for **all mutable state**.
- [compose.yaml](../../compose.yaml) backend `environment:` adds `WORK_ROOT: /work`; the bind-mount
  `${WORKDIR:-./work}:/work` stays. (Optionally rename the host var `WORKDIR → WORK_ROOT` for
  symmetry, keeping `WORKDIR` as a deprecated alias.)
- Native `run_newcas_ui.sh` needs **no** `WORK_ROOT` → falls back to `<repo>/work`. The forced
  `cd "${ROOT_DIR}"` at [:74](../../scripts/run_newcas_ui.sh#L74) is then **no longer needed to place
  autosave** — but audit for other CWD dependencies before removing it.
- **Document a real out-of-repo path** (`WORKDIR=~/spicex-work`) in `.env.example` to literally honor
  "separate from the git repo"; keep `./work` only as the zero-config fallback. Keep the **bind mount,
  not a named volume** — a named volume hides data in Docker's managed area (`/var/lib/docker/…`),
  defeating host-visible, backup-able, git-separate project data.

---

## 5. Backend Strategy

The backend stays a **thin adapter**; the optimizer library and its scorer stay canonical. The only
new business logic is filesystem bookkeeping.

**(a) The load-bearing fix — de-CWD the autosave root (additive, default unchanged).** Give
`Base_Optimizer.__init__` ([base.py:60-66](../../src/spicexplorer/optimization/base.py#L60-L66)) an
**optional** kwarg (`output_root: Path | None = None`); when present,
`autosave_checkpoint_dir = output_root / "checkpoints"`, else keep today's `Path("./auto_save/…")`.
Then [optimizer_runner.py](../../ui/backend/services/optimizer_runner.py) passes the per-run dir at
construction.

> **Why a constructor kwarg, not a post-construction attribute set.** Setting `opt.autosave_checkpoint_dir`
> *after* `__init__` is tempting (it's public mutable state) but `__init__` already `mkdir`s the
> default at [base.py:64](../../src/spicexplorer/optimization/base.py#L64), leaving a stray `/app/auto_save/…`
> dir every run. The kwarg avoids that and **preserves the CLI/example scripts** (they omit it → byte-identical
> behavior), which is essential — `examples/.../sizing/*.py` run the optimizer directly.

**(b) A single project service.** New `ui/backend/services/project_service.py` owns *all* `/work`
bookkeeping: `list_projects()` (scan `WORK_ROOT/projects/*/project.yaml`), `resolve_yaml(id)`,
`run_dir(id, run_id)`, `create_project()` (scaffold + `generate_yaml`), `copy_example()`
(`shutil.copytree` from `examples/` + rewrite `ws_root`/`outdir`), and — when/if needed —
`fork`/`delete` (guarded so `rmtree`/`mv` can only touch paths strictly under `WORK_ROOT.resolve()`,
the same defense-in-depth as `delete_checkpoint`). No FS logic leaks into route handlers.

**(c) `project_id` everywhere, with `yaml_path` back-compat.** Add an optional `project_id` to the
path-keyed request bodies (`LoadRequest`, `StartRequest`, score/sanity/sensitivity) and a shared
`resolve_project(project_id, yaml_path)` helper: prefer `project_id → resolve_yaml`, else fall back
to `yaml_path`, else `default_yaml_path()`. This lets the frontend pass **one stable id** without a
breaking rewrite, while the uploaded-YAML temp-file flow and existing tests that pass raw paths keep
working. **Keep ephemeral overrides ephemeral** — `_apply_overrides` and `spec_overrides` must stay
in-memory; only `create`/`fork`/`from-example` ever write YAML.

**(d) New router `routes/projects.py`** (registered with `/api` prefix — **must** be a backend route,
not a Next `app/api/**` handler, or it 404s under Docker):

```
GET    /api/projects                 → [{id, name, updated, run_count, best_score, source}]
POST   /api/projects                 {name, template?}        → scaffold dir + template YAML
POST   /api/projects/from-example    {example_key, name?}     → COPY example into /work/projects/<id>
GET    /api/projects/{id}            → {summary (_summarise), manifest}
GET    /api/projects/{id}/runs       → run.json list  (replaces localStorage history)
DELETE /api/projects/{id}            → soft-delete (deferred — see §10)
POST   /api/projects/{id}/fork       → cp -r           (deferred — see §10)
```

**(e) Checkpoint discovery follows the runs.** Extend [checkpoint.py](../../ui/backend/routes/checkpoint.py)
`_autosave_roots()` to include per-project `runs/*/checkpoints/`, **keeping** the legacy
`REPO_ROOT/auto_save` + `cwd/auto_save` roots (so old runs aren't orphaned) and the repo-relative
preset catalog ([app_config.py](../../ui/backend/app_config.py) `preset_checkpoint_paths()`) as a separate
read-only source. Scope by `project_id` where the caller knows it, to avoid an O(projects×runs)
`rglob` on every list.

---

## 6. Run Isolation & Debuggability

**What makes one run self-contained** — the answer to "ease of debugging." A run dir is debuggable
*from itself alone* when it carries:

| Artifact | Purpose | Today |
|---|---|---|
| `config_snapshot.yaml` | the **exact** `Project_Setup` it ran, with ephemeral algo/budget/seed/corner overrides baked in | overrides are in-memory only ([_apply_overrides](../../ui/backend/services/optimizer_runner.py)) — **no durable record of what actually ran** |
| `run.json` | label, status, best score, timing, `parent_run_id`, the override diff | a `localStorage` blob, not per-run |
| `checkpoints/*.json` | the optimizer's autosaves | pooled globally in `./auto_save`, CWD-dependent |
| `run.log` | full DEBUG library log | SSE-ephemeral, lost on disconnect |
| `events.ndjson` | replayable per-trial event stream | streamed, never persisted |
| `sim/` | the simulator outputs for this run | shared `outdir/live`, can collide |

The **per-run directory contract**: `runs/<ts>_<algo>_<runid8>/` is created at Start; the optimizer's
`output_root` points at it; the wrapper's `output_subdir` writes `sim/` *inside* it (so the wrapper's
`rmtree` can never touch another run); the SSE stream is tee'd to `events.ndjson` and the log to
`run.log`; on finish `run.json` is committed with the final score. **A run becomes a folder you can
zip and hand to someone.**

**What the user can do with a run:** open it (load its checkpoint into Explore), compare two runs A/B
(Explorer already does A/B), resume from its latest checkpoint (`parent_run_id` records lineage),
read its log inline (BottomPanel "Run log" tab), and delete it (trash its folder). A startup
**reconciler** flips any `status: running` left by a crashed run to `error`, so the list is honest.

---

## 7. UX & Lifecycle

**Two new surfaces, both reusing existing patterns — and crucially *not* a new ActivityBar view**
(that would collide with the `nav.ts` 1..8 shortcut + `CommandPalette` `/^[1-8]$/` invariant):

1. **Title-bar Project switcher + ⌘P Projects overlay.** The active project name shows in the title
   bar; click or ⌘P opens a `ProjectsOverlay` (same `dynamic(... ssr:false)` pattern as
   [CommandPalette](../../ui/src/components/overlays/CommandPalette.tsx)): search, rows
   (name · last run · best score · origin badge), and footer actions **+ New** (existing
   `WizardOverlay`) and **Load example…** (copy-on-load). This **retires** `SetupTab`'s raw-path /
   "Load example" `<select>` affordances — Setup becomes pure edit/apply.

2. **Per-project Runs panel** — rescope the existing
   [RunsRail](../../ui/src/components/shell/rails/RunsRail.tsx) to `GET /api/projects/{id}/runs`: status
   pills (running/done/stopped/error), best score, algo·iter, sparkline; row actions Open-in-Explore,
   Resume, Delete, **Open log → BottomPanel**. Checkpoints become children of their owning run, not a
   global flat list.

**Store changes.** `projectStore` gains `{id, name, projects[], switchProject(id)}`; `yamlPath`
becomes derived, never user-typed. **`runStore` keeps its live-SSE machinery client-side** (that's
correct), but its `localStorage` history (`HISTORY_KEY`) is **replaced by a server fetch keyed by
`project_id`**, refetched on run-finish and project-switch. Switching projects mid-run keeps the run
streaming (it's tagged to its origin project) with a confirm dialog.

**Lifecycle flows:** Create → wizard → `POST /api/projects` (scaffold+copy) → auto-switch.
Load → ⌘P → `switchProject`. Load+copy example → ⌘P → "Load example" → `POST from-example`.
Fork/Rename/Delete → deferred (§10), but cheap when wanted (fork = `copytree`; rename =
manifest-only edit so paths never break; delete = `mv` to `.trash`).

---

## 8. Migration & Backward-Compat

**The single biggest hazard is the `ws_root` portability contract — do not touch it.**
[from_yaml](../../src/spicexplorer/core/domains.py#L802-L808) has three battle-tested branches:
absolute→as-is, relative→YAML-dir, omitted→YAML-dir, with `~` expansion. The committed examples ship
`ws_root: ..` and bundle netlists in-repo so a fresh clone runs unedited. **Encapsulation is purely
additive:** a *new* project's copied `project.yaml` simply uses `ws_root: .` (resolves to the project
dir via the *existing* relative branch — zero core change); examples keep `ws_root: ..` and resolve
into the repo as before. "Legacy example" vs "encapsulated project" becomes a property of the YAML
the user wrote, not a global mode switch.

> **Why `ws_root: .` and not an absolute `/work/<id>`.** A relative `.` keeps the project *directory*
> itself portable (copy/move it, it still resolves); an absolute path bakes in `/work` and breaks the
> moment the mount differs (native vs Docker). Same encapsulation, no portability loss.

**The proven failure mode to guard against:** `project.py` once returned `yaml_path=""` and live runs
silently optimized the *default cascode example* instead of the applied YAML (the fix is documented at
[project.py:134-151](../../ui/backend/routes/project.py#L134-L151)). A registry that resolves "current
project" by anything other than the exact path threaded through every endpoint can resurrect this —
now *harder to notice* because it "looks encapsulated." Hence `project_id → resolve_yaml` must be the
single resolver, and a regression test must pin it.

**Migrating existing artifacts:** on startup, if `REPO_ROOT/auto_save` is non-empty and differs from
`work_root()/…`, log a one-time WARNING with a `mv` suggestion (**don't auto-move** — could clobber an
in-flight run). Add `auto_save/` to `.gitignore`. Offer a one-time importer that backfills the old
`runStore` `localStorage` history into `run.json` stubs and adopts orphan checkpoints into an
"Unsorted" project, so the upgrade loses nothing.

---

## 9. Phased Roadmap

Each phase ships and is testable on its own; ordered load-bearing-first, gold-plating-last.

**Phase 0 — Pin the contract (no product change).** Add `tests/test_ws_root_contract.py` (fast, no
SPICE): assert all three `from_yaml` branches + `~` expansion, **plus** a guard that for an
absolute-`ws_root` project the resolved `output_folder` and autosave root are **under that root and
`REPO_ROOT not in resolved.parents`**. *This is the test that would have caught the `yaml_path=""`
bug.* Wire it beside the existing [test_ui_restructure_2026_06.py](../../tests/test_ui_restructure_2026_06.py).
**Proof: green against today's code.**

**Phase 1 — `WORK_ROOT` + deterministic autosave (backend-only, default unchanged).** Add
`work_root()`/`auto_save_root()` to `app_config.py`; add the optional `output_root` kwarg to
`Base_Optimizer.__init__`; have `optimizer_runner` pass it; set `WORK_ROOT=/work` in compose. Keep
checkpoint.py's multi-root search (widen, don't narrow). **Fixes the Docker data-loss bug.**
*Proof: a live run in the container writes checkpoints under `/work` and they survive `docker compose
down && up`; CLI optimizer with no kwarg is byte-identical.*

**Phase 2 — Per-run isolation leaf.** Route each run's checkpoints/log/sim/events into
`runs/<run_id>/`; tee SSE → `events.ndjson`, log → `run.log`; write `run.json` + `config_snapshot.yaml`;
add the startup reconciler. *Proof: two concurrent runs produce disjoint dirs; each run dir alone
reproduces its config; neither under `REPO_ROOT`.*

**Phase 3 — Project registry + scaffold + copy-example + UI switcher.** `project_service.py`, the
`routes/projects.py` GET/POST/from-example endpoints, the `project_id` resolver, the title-bar
switcher + ⌘P overlay, and the rescoped Runs panel. Move run history server-side; one-time
`localStorage` import. *Proof: create two projects, copy an example into one, run each, see disjoint
per-project run lists; examples still run from a fresh clone.*

**Phase 4 (deferred) — Lifecycle niceties.** Fork (`copytree`), rename (manifest-only),
soft-delete + `.trash` + TTL sweeper, retention/GC, `/work` disk-usage surfacing in `/api/env`. Build
when a real second project/user makes them load-bearing.

---

## 10. Scope Discipline — Must-Build vs Defer

**Must-build (the two stated goals need exactly these):**

- `WORK_ROOT` resolution + de-CWD'd autosave (Phase 1) — *fixes a real data-loss bug.*
- Per-run isolation dir with config snapshot + log + checkpoints + sim (Phase 2) — *the debugging win.*
- A **flat** project registry + scaffold + copy-example + a project switcher (Phase 3) — *the
  encapsulation win.* A directory convention **is** the registry — no DB.

**Defer (premature for a single-user, localhost-CORS, few-projects tool):**

- Fork, rename, tags, search, soft-delete/trash, retention/GC, quotas. Each adds a schema + migration
  + test surface guarding behavior no one exercises yet. They're *cheap to add later* (fork = `cp -r`,
  rename = manifest edit, delete = `mv`), so there's no penalty for waiting.
- **Multi-user / ACLs — actively wrong to build now.** CORS is localhost-only and the container runs
  as a single gosu-dropped host user; multi-user contradicts both.

The skeptic's rule for any path-touching change: **if it alters `from_yaml`'s resolution branches or
makes a writer CWD-relative, stop — it's not additive.**

---

## 11. Open Decisions For You

These are genuine forks where your call changes the design. My recommendation is first.

1. **Per-project `runs/` vs a global `/work/runs/`.** → **Per-project** (recommended): it's what
   "isolate runs *per project*" means; a global pool isolates runs from each other but not by project.

2. **`ws_root: .` (relative, portable) vs absolute `/work/<id>`.** → **`ws_root: .`** (recommended):
   keeps the project dir movable and works identically native/Docker. Absolute only if you want the
   YAML to be self-locating outside the app.

3. **Project dir id scheme: `<slug>-<id8>` immutable vs bare slug with rename.** → **`<slug>-<id8>`,
   dir frozen, name lives in `manifest.json`** (recommended): rename never breaks paths or run lineage.

4. **Where does `outdir` point in a copied project: `scratch/` (out of source tree) vs keep
   `spice/temp_spice_out`?** → **`scratch/`** (recommended): keeps ephemeral manual/sanity sims out of
   the netlist source tree; the wrapper's `rmtree` can't touch sources. (Examples keep their current
   `outdir` untouched.)

5. **Native default `WORK_ROOT`: `<repo>/work` (zero-config, gitignored, but physically in the repo
   tree) vs require an explicit out-of-repo path.** → **`<repo>/work` fallback + documented out-of-repo
   override** (recommended): zero-config for dev, strict separation when you want it.

---

### Appendix — Provenance

Synthesized from five grounded expert analyses (filesystem, backend/API, lifecycle-UX, Docker/ops,
migration/risk) plus direct verification at `HEAD`. The three headline facts —
`/app/auto_save` ephemerality in Docker, `auto_save/` absent from `.gitignore`, and the forced native
`cd` — were each confirmed by reading [Dockerfile.backend:138](../../docker/Dockerfile.backend#L138),
[entrypoint-backend.sh:30-31](../../docker/entrypoint-backend.sh#L30-L31), [.gitignore:31](../../.gitignore#L31),
and [run_newcas_ui.sh:72-74](../../scripts/run_newcas_ui.sh#L72-L74). The adversarial critique phase
(cross-lens conflict reconciliation) is folded into §4, §5(a), §8, and §10.
