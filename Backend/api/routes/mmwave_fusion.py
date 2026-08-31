"""mmWave MMWAVE_ROOT adapter + live metrics + fusion overlay API."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.routes.context import RouterContext
from services.mmwave_metrics_service import live_metrics_snapshot
from services.mmwave_root import mmwave_root_status, preflight


class MmwaveRootConfigBody(BaseModel):
    path: str | None = None
    config_path: str | None = None
    sensor_distance_m: float | None = None
    radar_a_usb_location: str | None = None
    radar_b_usb_location: str | None = None
    calibration_session_a: str | None = None
    calibration_session_b: str | None = None


class MmwaveFusionConfigBody(BaseModel):
    enable: int | None = Field(None, ge=0, le=1)
    metrics_path: str | None = None
    depth_gate_m: float | None = None
    lateral_gate_m: float | None = None
    corridor_half_width_m: float | None = None
    mount_lateral_m: float | None = None
    mount_height_m: float | None = None
    webcam_side: str | None = None
    multi_camera_side: str | None = None


def build_mmwave_fusion_router(ctx: RouterContext) -> APIRouter:
    router = APIRouter(tags=["mmwave_fusion"])

    @router.get("/api/mmwave/root/status")
    def mmwave_root_status_get() -> dict[str, Any]:
        settings = ctx.settings.get()
        return mmwave_root_status(settings=settings, backend_root=ctx.layer8_dir.parent)

    @router.post("/api/mmwave/preflight")
    def mmwave_preflight() -> dict[str, Any]:
        settings = ctx.settings.get()
        return preflight(settings=settings, backend_root=ctx.layer8_dir.parent)

    @router.post("/api/mmwave/probe")
    def mmwave_probe() -> dict[str, Any]:
        """Full stack probe: radars, replay, live metrics, fusion, cameras, runners."""
        from layer8_ui.artifact_paths import software_root_from_settings
        from services.mmwave_probe import probe_mmwave_stack

        settings = ctx.settings.get()
        return probe_mmwave_stack(
            settings=settings,
            layer8_dir=ctx.layer8_dir,
            software_root=software_root_from_settings(settings),
            backend_root=ctx.layer8_dir.parent,
        )

    @router.get("/api/mmwave/live_metrics")
    def mmwave_live_metrics() -> dict[str, Any]:
        from layer8_ui.artifact_paths import software_root_from_settings

        settings = ctx.settings.get()
        return live_metrics_snapshot(
            settings,
            layer8_dir=ctx.layer8_dir,
            software_root=software_root_from_settings(settings),
        )

    @router.get("/api/mmwave/live/status")
    def mmwave_live_status() -> dict[str, Any]:
        path = ctx.layer8_dir / "artifacts" / "live_status.json"
        runtime = ctx.sensors.status("mmwave")
        payload: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text())
                payload = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                payload = {}
        return {
            "ok": bool(runtime.get("running")) and payload.get("state") not in ("FAULT",),
            "runtime": runtime,
            "live": payload or {"state": "STOPPED"},
            "status_path": str(path),
        }

    @router.post("/api/mmwave/live/start")
    def mmwave_live_start() -> Any:
        return ctx.sensors.start("mmwave")

    @router.post("/api/mmwave/live/stop")
    def mmwave_live_stop() -> dict[str, Any]:
        return ctx.sensors.stop("mmwave")

    @router.post("/api/mmwave/live/recalibrate")
    def mmwave_live_recalibrate() -> Any:
        # Recalibration deliberately restarts the single sensor-owner process.
        return ctx.sensors.restart("mmwave")

    @router.get("/api/mmwave/fusion/config")
    def mmwave_fusion_config_get() -> dict[str, Any]:
        settings = ctx.settings.get()
        fusion = settings.get("mmwave_fusion") if isinstance(settings.get("mmwave_fusion"), dict) else {}
        root = settings.get("mmwave_root") if isinstance(settings.get("mmwave_root"), dict) else {}
        return {"mmwave_fusion": fusion, "mmwave_root": root}

    @router.put("/api/mmwave/fusion/config")
    def mmwave_fusion_config_put(body: MmwaveFusionConfigBody) -> dict[str, Any]:
        current = deepcopy(ctx.settings.get())
        block = current.get("mmwave_fusion") if isinstance(current.get("mmwave_fusion"), dict) else {}
        block = dict(block)
        block.update(body.model_dump(exclude_none=True))
        current["mmwave_fusion"] = block
        saved = ctx.settings.replace(current)
        fusion = saved.get("mmwave_fusion") if isinstance(saved.get("mmwave_fusion"), dict) else {}
        return {"ok": True, "mmwave_fusion": fusion}

    @router.put("/api/mmwave/root/config")
    def mmwave_root_config_put(body: MmwaveRootConfigBody) -> dict[str, Any]:
        current = deepcopy(ctx.settings.get())
        block = current.get("mmwave_root") if isinstance(current.get("mmwave_root"), dict) else {}
        block = dict(block)
        block.update(body.model_dump(exclude_none=True))
        current["mmwave_root"] = block
        saved = ctx.settings.replace(current)
        root = saved.get("mmwave_root") if isinstance(saved.get("mmwave_root"), dict) else {}
        status = mmwave_root_status(settings=saved, backend_root=ctx.layer8_dir.parent)
        if not status.get("ok"):
            raise HTTPException(400, detail=f"MMWAVE_ROOT invalid: {status.get('missing')}")
        return {"ok": True, "mmwave_root": root, "status": status}

    return router
