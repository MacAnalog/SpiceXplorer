#!/usr/bin/env bash
#
# Launch the SpiceXplorer UI: FastAPI backend (:8000) + Next.js frontend (:4000).
#
# RUN IT, DON'T SOURCE IT:   ./scripts/run_newcas_ui.sh
# Sourcing would run the cleanup trap (and `set -e`) in your interactive shell,
# which tears the servers down the moment the shell hits any error or you Ctrl-C
# — surfacing as "Internal Server Error" (a dead :4000 behind the VS Code port
# forward). The guard below refuses to be sourced, before touching shell options.

# --- refuse to be sourced (must come before `set -e` so the caller's shell is
#     left untouched if someone does `source` this) ---
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Don't source this script — run it instead:  ./scripts/run_newcas_ui.sh" >&2
  return 1 2>/dev/null || exit 1
fi

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="${ROOT_DIR}/ui"

# Console log level for the spicexplorer library (DEBUG/INFO/WARNING/ERROR/CRITICAL)
LOG_LEVEL="${LOG_LEVEL:-INFO}"
# Ports are overridable but default to the documented pair (4000 avoids VS Code
# Remote SSH, which occupies :3000 on the server).
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-4000}"

# Kill any listener on a port. `ss` is used to find the PID because lsof can't
# see Next's IPv6 wildcard bind (`*:4000`), and next-server sets a process
# *title* so it can't be matched by name (`pkill -f next-server` misses it) —
# it must be killed by PID. Only ever targets the given port; never :3000.
kill_port() {
  local port="$1" pids=""
  command -v ss >/dev/null 2>&1 || return 0
  # `|| true` keeps an empty match (grep exit 1 under pipefail) from tripping set -e.
  pids="$(ss -ltnH "sport = :${port}" 2>/dev/null \
    | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u | tr '\n' ' ' || true)"
  if [[ -n "${pids// }" ]]; then
    echo "Freeing :${port} (pids: ${pids})"
    kill -9 ${pids} 2>/dev/null || true
  fi
}

# Stop any leftover SpiceXplorer servers and free both ports. Kills the
# supervisors first — by their exact command line, scoped to our ports, so other
# projects' servers are untouched — so Next can't respawn next-server, then frees
# whatever still holds the ports by PID. Used both before start (clear a crashed
# previous run) and on exit (clean shutdown).
stop_servers() {
  # Kill supervisors before their children so nothing respawns: uvicorn's `uv run`
  # wrapper + reloader, and npm (which re-spawns `next dev`/next-server) before
  # `next dev` itself. All scoped to our ports so other projects are untouched.
  pkill -f "uvicorn ui.backend.main:app .*--port ${BACKEND_PORT}" 2>/dev/null || true
  pkill -f "npm run dev.*-p ${FRONTEND_PORT}" 2>/dev/null || true
  pkill -f "next dev .*-p ${FRONTEND_PORT}" 2>/dev/null || true
  sleep 1
  kill_port "${BACKEND_PORT}"
  kill_port "${FRONTEND_PORT}"
}

stop_servers  # clear anything a crashed previous run left behind

# Start FastAPI backend.
# --reload is scoped to the source dirs: watching the whole repo (the default)
# reloads on every backend log write (logs/), checkpoint autosave (auto_save/),
# and Next build artifact (ui/.next) — an endless reload loop that thrashes the
# backend ("watchfiles: change detected" spam, a new log file per restart).
echo "Starting FastAPI backend on :${BACKEND_PORT}… (LOG_LEVEL=${LOG_LEVEL})"
# Run the backend from the repo root so its CWD == REPO_ROOT: the optimizer autosaves to a
# CWD-relative ./auto_save while the checkpoint API reads REPO_ROOT/auto_save — keep them
# aligned regardless of where this script was invoked from (BUG-A9 / OPT-3).
cd "${ROOT_DIR}"
LOG_LEVEL="${LOG_LEVEL}" uv run --extra ui uvicorn ui.backend.main:app \
  --reload --reload-dir "${ROOT_DIR}/ui/backend" --reload-dir "${ROOT_DIR}/src" \
  --port "${BACKEND_PORT}" &
BACKEND_PID=$!

# Start Next.js frontend on :4000 (avoids VS Code Remote SSH occupying :3000)
echo "Starting Next.js frontend on :${FRONTEND_PORT}…"
cd "${UI_DIR}"
if [ ! -d "node_modules" ]; then
  npm install
fi
npm run dev -- -p "${FRONTEND_PORT}" &
FRONTEND_PID=$!

# Clean up on Ctrl-C / termination / normal exit. Safe because we refused to be
# sourced above, so this trap only ever fires for THIS process — not your shell.
cleanup() {
  trap - EXIT INT TERM
  echo ""
  echo "Shutting down…"
  kill "${BACKEND_PID}" "${FRONTEND_PID}" 2>/dev/null || true
  stop_servers  # also reap the reload worker / next-server and free the ports
}
trap cleanup EXIT INT TERM

echo ""
echo "SpiceXplorer UI is starting:"
echo "  Backend  → http://localhost:${BACKEND_PORT}"
echo "  Frontend → http://localhost:${FRONTEND_PORT}"
echo ""
echo "Press Ctrl+C to stop both processes."

wait
