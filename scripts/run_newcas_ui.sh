#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="${ROOT_DIR}/ui"

# Console log level for the spicexplorer library (DEBUG/INFO/WARNING/ERROR/CRITICAL)
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Start FastAPI backend
echo "Starting FastAPI backend on :8000… (LOG_LEVEL=${LOG_LEVEL})"
LOG_LEVEL="${LOG_LEVEL}" uv run --extra ui uvicorn ui.backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start Next.js frontend on :4000 (avoids VS Code Remote SSH occupying :3000)
echo "Starting Next.js frontend on :4000…"
cd "${UI_DIR}"
if [ ! -d "node_modules" ]; then
  npm install
fi
npm run dev -- -p 4000 &
FRONTEND_PID=$!

trap "kill ${BACKEND_PID} ${FRONTEND_PID} 2>/dev/null || true" EXIT INT TERM

echo ""
echo "SpiceXplorer UI is starting:"
echo "  Backend  → http://localhost:8000"
echo "  Frontend → http://localhost:4000"
echo ""
echo "Press Ctrl+C to stop both processes."

wait
