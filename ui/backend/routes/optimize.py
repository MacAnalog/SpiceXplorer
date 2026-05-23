"""Optimization run management: start, stop, and SSE stream."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ui.backend.app_config import default_yaml_path, demo_checkpoint_paths
from ui.backend.services import optimizer_runner as runner

router = APIRouter()


class StartRequest(BaseModel):
    yaml_path: str | None = None
    replay: bool = False
    checkpoint_id: str | None = None
    budget: int = 200


@router.post("/optimize/start")
async def start_run(body: StartRequest, request: Request):
    loop = asyncio.get_event_loop()

    yaml_path = body.yaml_path or str(default_yaml_path())

    checkpoint_path: Path | None = None
    if body.replay and body.checkpoint_id:
        demos = demo_checkpoint_paths()
        checkpoint_path = demos.get(body.checkpoint_id)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise HTTPException(404, f"Demo checkpoint '{body.checkpoint_id}' not found")

    run_id = runner.start_run(
        project_path=yaml_path if not body.replay else None,
        replay=body.replay,
        checkpoint_id=body.checkpoint_id,
        checkpoint_path=checkpoint_path,
        budget=body.budget,
        loop=loop,
    )
    return {"run_id": run_id, "replay": body.replay}


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
