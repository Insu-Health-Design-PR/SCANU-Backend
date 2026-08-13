"""
Layer 8 sensor dashboard — legacy entrypoint (prefer ``app:app`` at repo root).

Run from ``Backend/`` (not the parent ``New_Backend/`` folder):

  cd Backend
  make api
  # or:
  python -m uvicorn app:app --host 0.0.0.0 --port 8088

This module remains for compatibility with ``layer8_ui.app:app`` when cwd is ``Backend/``.
Routes are implemented in ``api.routes``; this file re-exports the same app surface.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from layer8_ui.dashboard_routes import build_router, index_handler

LAYER8_DIR = Path(__file__).resolve().parent
STATIC = LAYER8_DIR / "static"
ARTIFACTS = LAYER8_DIR / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SCANU Layer 8 — sensor runners", version="0.1.0")

if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    assets_dir = STATIC / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

app.include_router(build_router(LAYER8_DIR))


@app.get("/")
def index():
    return index_handler(STATIC)
