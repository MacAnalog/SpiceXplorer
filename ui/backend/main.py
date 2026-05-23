"""SpiceXplorer NEWCAS Demo — FastAPI backend."""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ui.backend.routes import config, project, score, optimize, checkpoint, schematic

app = FastAPI(title="SpiceXplorer Demo API", version="1.0.0")

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
