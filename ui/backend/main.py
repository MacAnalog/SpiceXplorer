"""SpiceXplorer UI — FastAPI backend."""
import logging
import os
from pathlib import Path

# Configure root logger so uvicorn/fastapi messages appear consistently
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Wire up the spicexplorer library logger.
# Console verbosity is controlled by LOG_LEVEL env var (default: INFO).
# The log file (in logs/) always captures everything at DEBUG.
_LOG_LEVEL_NAME = os.environ.get("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)

from spicexplorer.logging import setup_loggers
_REPO_ROOT = Path(__file__).parent.parent.parent
setup_loggers(
    out_logname="SpiceXplorer",
    parent_folder=_REPO_ROOT,
    console_level=_LOG_LEVEL,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ui.backend.routes import config, project, score, optimize, checkpoint, schematic

app = FastAPI(title="SpiceXplorer API", version="1.0.0")

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


@app.get("/health")
def health():
    return {"status": "ok"}
