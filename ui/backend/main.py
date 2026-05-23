"""SpiceXplorer UI — FastAPI backend."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Configure root logger so uvicorn/fastapi messages appear consistently.
# basicconfig is idempotent so it's safe to call at import time in both
# the reloader parent and the worker child.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_LOG_LEVEL_NAME = os.environ.get("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)
_REPO_ROOT = Path(__file__).parent.parent.parent

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ui.backend.routes import config, project, score, optimize, checkpoint, schematic, sanity


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs only in the uvicorn worker process, not the WatchFiles reloader parent,
    # so exactly one log file is created per server start.
    from spicexplorer.logging import setup_loggers
    setup_loggers(
        out_logname="SpiceXplorer",
        parent_folder=_REPO_ROOT,
        console_level=_LOG_LEVEL,
    )
    yield


app = FastAPI(title="SpiceXplorer API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(config.router, prefix="/api")
app.include_router(project.router, prefix="/api")
app.include_router(score.router, prefix="/api")
app.include_router(optimize.router, prefix="/api")
app.include_router(checkpoint.router, prefix="/api")
app.include_router(schematic.router, prefix="/api")
app.include_router(sanity.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
