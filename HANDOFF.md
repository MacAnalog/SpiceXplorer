# Handoff — SpiceXplorer Dockerization + Setup-editor fix

**For: the agent running on the Mac (M1/arm64) where Docker builds happen.**
**From: the agent on the research server (no Docker daemon access there).**
Date: 2026-06-04 · Repo: `github.com/MacAnalog/SpiceXplorer` · Branch: `dev/ui`

---

## TL;DR — do this first

Everything below is **already committed and pushed** to `origin/dev/ui` (HEAD `f65a544`).
The Mac just needs to sync and rebuild:

```bash
docker info                       # ensure Docker Desktop is running
git checkout dev/ui && git pull   # gets f65a544 — includes the editor fix
docker compose up --build         # see "Build modes" before a long build
# open http://localhost:4000  →  Setup → "Load example…"  → YAML now renders
```

The earlier "empty Setup editor" was a running image built **before** the fix
commit. A pull + rebuild resolves it.

---

## Why this handoff exists

The research server has **no Docker daemon access** for this user (account not in
the `docker` group, no rootless, no passwordless sudo). All container builds/tests
happen on the **Mac**, which is a separate clone. Code is synced **via git**
(`origin/dev/ui`), so the rule is: *server commits + pushes → Mac pulls + builds.*

## What this project is

SpiceXplorer = a uv-managed Python circuit-optimization library + **FastAPI backend**
+ **Next.js "Studio" UI**, driving **ngspice** with the **IHP `ihp-sg13g2` PDK**.
The goal of this work: **containerize** it so other researchers run it without
installing ngspice/PDK, and **provision for an LLM-agent layer + API keys** later.

---

## What was built (commits `8103083` → `f65a544`)

A self-contained two-service stack — `docker compose up --build` gives a working,
live-SPICE UI on x86-64 **and** arm64, no host install.

| Area | File(s) | What |
|---|---|---|
| Orchestration | `compose.yaml`, `.env.example` | 2 services (backend `:8000`, frontend `:4000`); all knobs via `.env`. |
| Backend image | `docker/Dockerfile.backend` | Compiles **ngspice 45 from source** (OSDI + XSPICE, headless); installs the uv venv. |
| OSDI models | `docker/Dockerfile.backend` (`OSDI_MODE` arg) | `compile` (default) builds **openvaf** (Rust+LLVM-18) and compiles OSDI from vendored Verilog-A **for the host arch** → native multi-arch. `vendor` reuses prebuilt **x86-64** OSDI (fast, x86-64/emulation only). |
| PDK (vendored) | `docker/pdk/` (~3 MB, Apache-2.0) | ngspice `models/*.lib` + `.spiceinit` + prebuilt `osdi/*.osdi` (x86-64) + `verilog-a/` source. |
| torch = CPU-only | `pyproject.toml` (`[tool.uv.sources]` + `pytorch-cpu` index), `uv.lock` | No CUDA. Lock carries both `x86_64` + `aarch64` linux wheels. |
| Frontend image | `ui/Dockerfile`, `ui/next.config.mjs` | 3-stage Next.js **standalone** build, non-root. (`output: 'standalone'` added.) |
| Host-UID mapping | `docker/entrypoint-backend.sh` | Aligns runtime user to host `UID`/`GID` via `gosu` so `/work` bind-mount files aren't root-owned (Linux). |
| LLM-agent provisioning | `pyproject.toml` (`agents` extra), `compose.yaml`, `.env.example` | `INSTALL_AGENTS=true` installs the extra; `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`GOOGLE_API_KEY` pass through at **runtime only** (never baked into the image). |
| Netlist portability | `examples/OTA/{cascode,5t-ota}/.../*_tb-ac.spice` | 2 hardcoded `/home/noorizad/...` `.include` paths made relative. |
| Docs | `README.md`, `CLAUDE.md`, `docker/pdk/README.md` | "Run in Docker" section + OSDI/openvaf notes. |
| **Bug fix** | `ui/backend/routes/project.py` (`f65a544`) | Added missing **`GET /api/yaml-text`** endpoint (see below). |

---

## Verified state (from the Mac's emulated-amd64 build)

✅ Build succeeds; **live SPICE works**: all three testbench sanity checks passed
(`tb_ac`, `tb_noise`, `tb_tran`), OSDI/PSP103 models loaded, CPU torch initialized
(`Using device: cpu`), project parsed from `/app/examples/...`.

## The bug that was fixed (`f65a544`)

"Load example…" populated the left rail (project/testbenches/devices/specs) but the
**Monaco editor stayed empty**. Cause: the Setup view fetches `GET /api/yaml-text?path=…`
to fill the editor, but **that backend route never existed** (pre-existing
frontend/backend mismatch — unrelated to Docker). Added it to
`ui/backend/routes/project.py` (returns raw YAML as `text/plain`, gated to existing
`.yaml`/`.yml`). The 404 seen in devtools was the old running image; **rebuild to apply.**

---

## Build modes (pick before a long build)

`docker compose up --build` defaults to **`OSDI_MODE=compile`** → native to the host:

- **Native arm64 (default on the Mac):** compiles openvaf (Rust + LLVM-18) then ngspice
  then the venv. First build ~20–30 min, cached after; **sims run at native speed.**
  ⚠️ **The `osdi-compile` stage (openvaf) has NOT been tested yet.** If `cargo build`
  or an `openvaf --target_cpu generic … -o x.osdi …` step errors, that's the thing to
  debug (recipe + pins in `docker/pdk/README.md`).
- **Faster fallback — emulated amd64, prebuilt OSDI** (this is what was verified working):
  ```bash
  OSDI_MODE=vendor DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up --build
  ```
  Slower at runtime (QEMU), but no openvaf build and known-good.

## Verify after build

```bash
# 1) environment probe — expect pdk_ok:true, live_runs_enabled:true
docker compose exec backend python3 -c "import urllib.request as u; print(u.urlopen('http://127.0.0.1:8000/api/env').read().decode())"
# 2) the fixed endpoint — expect 200 + YAML text
docker compose exec backend python3 -c "import urllib.request as u; print(u.urlopen('http://127.0.0.1:8000/api/yaml-text?path=/app/examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml').read()[:120])"
# 3) UI: http://localhost:4000 → Setup → Load example → editor shows YAML
```

`.env` knobs (copy `.env.example` → `.env`): `WORKDIR`, `UID`/`GID` (Linux only),
`FRONTEND_PORT`/`BACKEND_PORT`, `LOG_LEVEL`, `OSDI_MODE`, `INSTALL_AGENTS`, API keys.

---

## Open items / next steps

1. **Confirm the native arm64 `OSDI_MODE=compile` build** actually works end-to-end
   (the openvaf stage is the only unverified piece). If it fails, use the `vendor`
   fallback above and report the exact `cargo`/`openvaf` error.
2. **Umbrella repo:** the parent `TCAD` repo pins this submodule — bump its pointer to
   `f65a544` (a commit in the parent) when promoting.
3. **Optional slimming:** `torch` is imported eagerly on the SPICE path; making it lazy
   would shrink the image further (out of scope so far).
4. This `HANDOFF.md` is disposable — delete once the Mac is in sync.

## Reference

- Architecture & constraints: `CLAUDE.md` (has a "Docker" subsection).
- Docker details: file headers in `docker/`, and `docker/pdk/README.md` (openvaf recipe).
- "Run in Docker": `README.md`.
