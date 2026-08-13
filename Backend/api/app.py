"""FastAPI application composition for the migrated Layer 8 backend."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import build_router, index_handler

BACKEND_DIR = Path(__file__).resolve().parent.parent
LEGACY_LAYER8_DIR = BACKEND_DIR / "layer8_ui"
STATIC = LEGACY_LAYER8_DIR / "static"
ARTIFACTS = LEGACY_LAYER8_DIR / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


class _SuppressCancelledErrorFilter(logging.Filter):
    """Avoid noisy tracebacks when Ctrl+C cancels MJPEG/SSE streaming tasks."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.exc_info:
            # websockets legacy keepalive can race with high-rate binary JPEG pushes.
            msg = record.getMessage()
            if "keepalive ping failed" in msg or "AssertionError" in msg:
                return False
            return True
        exc = record.exc_info[1]
        if isinstance(exc, asyncio.CancelledError):
            return False
        if isinstance(exc, AssertionError) and "keepalive ping failed" in record.getMessage():
            return False
        return True


def _configure_uvicorn_logging() -> None:
    flt = _SuppressCancelledErrorFilter()
    for name in ("uvicorn.error", "uvicorn"):
        logging.getLogger(name).addFilter(flt)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _configure_uvicorn_logging()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="SCANU Backend — Layer 8 API",
        version="0.1.0",
        lifespan=_lifespan,
    )

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
        assets_dir = STATIC / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    app.router.routes.extend(build_router(LEGACY_LAYER8_DIR).routes)

    @app.get("/health")
    @app.get("/health/")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def index():
        return index_handler(STATIC)

    return app


app = create_app()
