"""Optimization run management: start, stop, and SSE stream."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ui.backend.app_config import preset_checkpoint_paths
from ui.backend.services import optimizer_runner as runner
from ui.backend.services import project_service
from ui.backend.services.env_probe import probe_env

router = APIRouter()


class StartRequest(BaseModel):
    yaml_path: str | None = None
    # The owning project (report.md P3). When set, the run is resolved + isolated
    # under WORK_ROOT/projects/<id>/runs/; yaml_path stays as a back-compat fallback.
    project_id: str | None = None
    label: str | None = None
    replay: bool = False
    checkpoint_id: str | None = None
    budget: int = 200
    # Ephemeral live-run overrides (applied in-memory to the loaded project; the
    # YAML on disk is never rewritten). Ignored for replay runs.
    algorithm: str | None = None
    seed: int | None = None
    # PVT corner to optimize against (must match a corner in the project's `pvt:`
    # block). None keeps the YAML's active_corner.
    active_corner: str | None = None
    # Checkpointing (live runs only). autosave_every writes a cumulative
    # checkpoint every N trials; resume_checkpoint_id continues a prior run from
    # a saved checkpoint (load_checkpoint + optimize(keep_history=True)).
    autosave_every: int | None = None
    resume_checkpoint_id: str | None = None


@router.post("/optimize/start")
async def start_run(body: StartRequest, request: Request):
    loop = asyncio.get_event_loop()

    # A replay needs a checkpoint to replay — reject the no-id case up front instead of scheduling
    # nothing and leaving the SSE stream heartbeating "running" forever (BUG-B36).
    if body.replay and not body.checkpoint_id:
        raise HTTPException(400, "replay requires a checkpoint_id")

    # Refuse to start a run for a project whose delete is in flight — otherwise this run could
    # re-create the project tree the delete is about to move to trash (start-after-stop TOCTOU,
    # the B4 residual).
    if runner.is_project_deleting(body.project_id):
        raise HTTPException(409, "project is being deleted; cannot start a run")

    # Single resolver: project_id → its project.yaml; else yaml_path; else default.
    try:
        yaml_path = str(project_service.resolve_yaml(body.project_id, body.yaml_path))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        # malformed project_id (path separators / '..') → 400, not an opaque 500 (BUG-B37)
        raise HTTPException(400, str(e))

    # Live and resume runs need real SPICE: refuse cleanly (409) when the
    # environment can't run it, instead of failing deep in the engine. Replay
    # needs no PDK, so it is exempt. (The client also gates this; this is the
    # server-side enforcement so direct/programmatic callers get a clear error.)
    if not body.replay:
        env = probe_env()
        if not env.get("live_runs_enabled", False):
            raise HTTPException(409, env.get("pdk_detail") or "Live runs disabled: ngspice/PDK unavailable.")

    checkpoint_path: Path | None = None
    replay_len: int | None = None
    if body.replay and body.checkpoint_id:
        presets = preset_checkpoint_paths()
        checkpoint_path = presets.get(body.checkpoint_id)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise HTTPException(404, f"Checkpoint '{body.checkpoint_id}' not found")
        # Report the row count so the UI's progress denominator is the checkpoint
        # length, not the unrelated live-run budget (default 200).
        try:
            from ui.backend.services.checkpoint_reader import read_checkpoint
            replay_len = read_checkpoint(checkpoint_path).get("n_iters")
        except Exception:
            replay_len = None

    # Resume: resolve the checkpoint to continue a live run from (presets or autosaves).
    resume_path: str | None = None
    if body.resume_checkpoint_id and not body.replay:
        from ui.backend.routes.checkpoint import _resolve_checkpoint_path

        resolved = _resolve_checkpoint_path(body.resume_checkpoint_id)
        if resolved is None:
            raise HTTPException(404, f"Resume checkpoint '{body.resume_checkpoint_id}' not found")
        resume_path = str(resolved)

    run_id = runner.start_run(
        project_path=yaml_path if not body.replay else None,
        project_id=body.project_id,
        label=body.label,
        replay=body.replay,
        checkpoint_id=body.checkpoint_id,
        checkpoint_path=checkpoint_path,
        budget=body.budget,
        algorithm=body.algorithm,
        seed=body.seed,
        active_corner=body.active_corner,
        autosave_every=body.autosave_every,
        resume_path=resume_path,
        loop=loop,
    )
    return {"run_id": run_id, "replay": body.replay, "resumed": resume_path is not None, "n_iters": replay_len}


@router.post("/optimize/stop/{run_id}")
def stop_run(run_id: str):
    runner.stop_run(run_id)
    return {"ok": True}


@router.get("/optimize/stream/{run_id}")
async def stream_run(run_id: str):
    state = runner.get_run(run_id)
    if state is None:
        raise HTTPException(404, f"Run '{run_id}' not found")

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(state.queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                yield "data: {\"heartbeat\": true}\n\n"
                continue

            if event is None:
                yield 'data: {"done": true}\n\n'
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
