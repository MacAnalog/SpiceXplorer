#!/usr/bin/env bash
# SpiceXplorer backend entrypoint.
#
# Aligns the in-container runtime user to the *host* UID/GID (passed as
# HOST_UID/HOST_GID by docker compose from .env) so files written to the /work
# bind-mount — and the bundled example outputs — are owned by the host user
# rather than root. On Linux, container UIDs map 1:1 to the host, so without
# this a root container leaves root-owned files the host user can't edit.
# (macOS/Windows remap ownership automatically; this is a no-op harm there.)
set -euo pipefail

HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"

# Create matching group/user only if those numeric IDs don't already exist
# (Ubuntu 24.04 ships a UID/GID 1000 'ubuntu' account). gosu accepts a numeric
# uid:gid regardless, so creation is best-effort.
if ! getent group "$HOST_GID" >/dev/null 2>&1; then
  groupadd -g "$HOST_GID" appgrp || true
fi
if ! getent passwd "$HOST_UID" >/dev/null 2>&1; then
  useradd -u "$HOST_UID" -g "$HOST_GID" -M -d /app -s /bin/bash appusr || true
fi

# Writable locations. Everything else in the image is read-only to the app.
#   /work            — the user's bind-mounted project/output drive
#   /app/logs        — library DEBUG logs (always written)
#   /app/auto_save   — optimizer checkpoints
#   /app/examples    — bundled demos write run outputs under ws_root here
mkdir -p /work /app/logs /app/auto_save
chown "$HOST_UID:$HOST_GID" /work /app/logs /app/auto_save 2>/dev/null || true
chown -R "$HOST_UID:$HOST_GID" /app/examples 2>/dev/null || true

exec gosu "$HOST_UID:$HOST_GID" "$@"
