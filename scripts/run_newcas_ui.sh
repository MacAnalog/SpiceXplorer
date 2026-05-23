#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="${ROOT_DIR}/ui"

# Start FastAPI backend
echo "Starting FastAPI backend on :8000…"
uv run --extra ui uvicorn ui.backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start Next.js frontend
echo "Starting Next.js frontend on :3000…"
cd "${UI_DIR}"
if [ ! -d "node_modules" ]; then
  npm install
fi
npm run dev &
FRONTEND_PID=$!

trap "kill ${BACKEND_PID} ${FRONTEND_PID} 2>/dev/null || true" EXIT INT TERM

echo ""
echo "SpiceXplorer UI is starting:"
echo "  Backend  → http://localhost:8000"
echo "  Frontend → http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both processes."

wait
