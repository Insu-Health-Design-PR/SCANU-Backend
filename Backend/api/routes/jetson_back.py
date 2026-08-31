"""Back Camera Jetson SSH / MediaMTX / cam-rtsp remote control."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.routes.context import RouterContext
from services.jetson_ssh import (
    all_services_status,
    config_public_dict,
    daemon_reload,
    jetson_config_from_settings,
    list_v4l2_devices,
    service_action,
    service_journal,
    set_cam_rtsp_video_device,
    tail_cam_log,
    test_connection,
)


class JetsonBackConfigBody(BaseModel):
    host: str | None = None
    user: str | None = None
    port: int | None = Field(None, ge=1, le=65535)
    identity_file: str | None = None
    connect_timeout_s: float | None = Field(None, ge=3.0, le=60.0)
    cam_rtsp_log: str | None = None


class JetsonVideoDeviceBody(BaseModel):
    device: str = Field(..., description="e.g. /dev/video1")


def build_jetson_back_router(ctx: RouterContext) -> APIRouter:
    router = APIRouter(tags=["jetson_back"])

    def _cfg() -> Any:
        return jetson_config_from_settings(ctx.settings.get(), ctx.layer8_dir)

    @router.get("/api/back_camera/jetson/config")
    def jetson_config_get() -> dict[str, Any]:
        cfg = _cfg()
        return {"jetson_back": config_public_dict(cfg)}

    @router.put("/api/back_camera/jetson/config")
    def jetson_config_put(body: JetsonBackConfigBody) -> dict[str, Any]:
        current = deepcopy(ctx.settings.get())
        block = current.get("jetson_back") if isinstance(current.get("jetson_back"), dict) else {}
        block = dict(block)
        data = body.model_dump(exclude_none=True)
        block.update(data)
        current["jetson_back"] = block
        # Keep multi_camera jetson_ip in sync when host changes.
        if "host" in data and isinstance(current.get("multi_camera"), dict):
            mc = dict(current["multi_camera"])
            mc["jetson_ip"] = str(data["host"])
            if not str(mc.get("jetson_stream_url") or "").strip() and data["host"]:
                port = str(mc.get("jetson_stream_port") or "8554")
                path = str(mc.get("jetson_stream_path") or "/cam")
                if not path.startswith("/"):
                    path = "/" + path
                mc["jetson_stream_url"] = f"rtsp://{data['host']}:{port}{path}"
            current["multi_camera"] = mc
        saved = ctx.settings.replace(current)
        cfg = jetson_config_from_settings(saved, ctx.layer8_dir)
        return {"ok": True, "jetson_back": config_public_dict(cfg)}

    @router.get("/api/back_camera/jetson/connect")
    def jetson_connect_test() -> dict[str, Any]:
        return test_connection(_cfg())

    @router.get("/api/back_camera/jetson/services")
    def jetson_services_status() -> dict[str, Any]:
        return all_services_status(_cfg())

    @router.post("/api/back_camera/jetson/services/{service_key}/{action}")
    def jetson_service_control(service_key: str, action: str) -> dict[str, Any]:
        res = service_action(_cfg(), service_key, action)
        if not res.get("ok") and res.get("error"):
            raise HTTPException(503, detail=str(res.get("error")))
        return res

    @router.post("/api/back_camera/jetson/daemon-reload")
    def jetson_daemon_reload() -> dict[str, Any]:
        res = daemon_reload(_cfg())
        if not res.get("ok"):
            raise HTTPException(503, detail=res.get("stderr") or res.get("error") or "daemon-reload failed")
        return res

    @router.get("/api/back_camera/jetson/logs/{service_key}")
    def jetson_service_logs(service_key: str, lines: int = 80) -> dict[str, Any]:
        res = service_journal(_cfg(), service_key, lines=lines)
        if not res.get("ok"):
            raise HTTPException(503, detail=res.get("stderr") or "journal fetch failed")
        return res

    @router.get("/api/back_camera/jetson/cam-log")
    def jetson_cam_publish_log(lines: int = 80) -> dict[str, Any]:
        res = tail_cam_log(_cfg(), lines=lines)
        if not res.get("ok"):
            raise HTTPException(503, detail=res.get("stderr") or "log tail failed")
        return res

    @router.get("/api/back_camera/jetson/v4l2-devices")
    def jetson_v4l2_devices() -> dict[str, Any]:
        res = list_v4l2_devices(_cfg())
        if not res.get("ok"):
            raise HTTPException(503, detail=res.get("stderr") or "v4l2-ctl failed")
        return res

    @router.post("/api/back_camera/jetson/cam-rtsp/device")
    def jetson_set_cam_device(body: JetsonVideoDeviceBody) -> dict[str, Any]:
        res = set_cam_rtsp_video_device(_cfg(), body.device)
        if not res.get("ok"):
            raise HTTPException(
                503,
                detail=res.get("stderr") or res.get("error") or "failed to set VIDEO_DEVICE",
            )
        return res

    return router
