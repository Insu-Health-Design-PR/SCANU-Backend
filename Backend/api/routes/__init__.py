"""FastAPI route registry for the migrated backend."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.routes.alignment import build_alignment_router
from api.routes.jetson_back import build_jetson_back_router
from api.routes.camera_roles import build_camera_roles_router
from api.routes.config import build_config_router
from api.routes.context import create_router_context
from api.routes.devices import build_devices_router
from api.routes.global_id import build_global_id_router
from api.routes.info import build_info_router
from api.routes.metrics import build_metrics_router
from api.routes.mmwave_fusion import build_mmwave_fusion_router
from api.routes.playground import build_playground_router
from api.routes.preview import build_preview_router
from api.routes.profiles import build_profiles_router
from api.routes.recording import build_recording_router
from api.routes.screenshot import build_screenshot_router
from api.routes.sensors import build_sensors_router


def build_router(layer8_dir: Path) -> APIRouter:
    """Compose modular API routers with endpoint parity to the legacy dashboard."""
    ctx = create_router_context(layer8_dir)
    router = APIRouter()
    child_routers = (
        build_metrics_router(ctx),
        build_devices_router(ctx),
        build_config_router(ctx),
        build_sensors_router(ctx),
        build_camera_roles_router(ctx),
        build_info_router(ctx),
        build_profiles_router(ctx),
        build_playground_router(ctx),
        build_preview_router(ctx),
        build_recording_router(ctx),
        build_screenshot_router(ctx),
        build_global_id_router(ctx),
        build_alignment_router(ctx),
        build_jetson_back_router(ctx),
        build_mmwave_fusion_router(ctx),
    )
    for child in child_routers:
        router.routes.extend(child.routes)
    return router


def index_handler(static_dir: Path) -> FileResponse:
    index_path = Path(static_dir) / "index.html"
    if not index_path.is_file():
        raise HTTPException(404, "static/index.html missing")
    return FileResponse(index_path)
