"""Optimization run management: start, stop, and SSE stream."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ui.backend.app_config import default_yaml_path, preset_checkpoint_paths
from ui.backend.services import optimizer_runner as runner

router = APIRouter()


class StartRequest(BaseModel):
    yaml_path: str | None = None
    replay: bool = False
    checkpoint_id: str | None = None
    budget: int = 200
    # Ephemeral live-run overrides (applied in-memory to the loaded project; the
    # YAML on disk is never rewritten). Ignored for replay runs.
    algorithm: str | None = None
    seed: int | None = None
    # Checkpointing (live runs only). autosave_every writes a cumulative
    # checkpoint every N trials; resume_checkpoint_id continues a prior run from
    # a saved checkpoint (load_checkpoint + optimize(keep_history=True)).
    autosave_every: int | None = None
    resume_checkpoint_id: str | None = None


@router.post("/optimize/start")
async def start_run(body: StartRequest, request: Request):
    loop = asyncio.get_event_loop()

    yaml_path = body.yaml_path or str(default_yaml_path())

    checkpoint_path: Path | None = None
    if body.replay and body.checkpoint_id:
        presets = preset_checkpoint_paths()
        checkpoint_path = presets.get(body.checkpoint_id)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise HTTPException(404, f"Checkpoint '{body.checkpoint_id}' not found")

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
        replay=body.replay,
        checkpoint_id=body.checkpoint_id,
        checkpoint_path=checkpoint_path,
        budget=body.budget,
        algorithm=body.algorithm,
        seed=body.seed,
        autosave_every=body.autosave_every,
        resume_path=resume_path,
        loop=loop,
    )
    return {"run_id": run_id, "replay": body.replay, "resumed": resume_path is not None}


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
