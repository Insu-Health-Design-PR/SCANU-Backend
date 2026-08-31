"""Front Camera / Back Camera API aliases for operators and integrations.

Front Camera maps to the ``webcam`` sensor block (local USB + infer_objects).
Back Camera maps to the ``multi_camera`` sensor block (USB or Jetson stream).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.routes.context import RouterContext
from api.schemas.profiles import ApplyModelProfileBody
from api.schemas.webrtc import WebRTCOfferBody
from api.streaming.live_mjpeg import ai_camera_live_mjpeg, multi_camera_live_mjpeg
from api.streaming.webrtc import handle_multi_camera_offer, handle_webcam_offer

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from layer8_ui import v4l2_camera_controls  # noqa: E402


class CameraControlsSetBody(BaseModel):
    index: int | None = Field(None, ge=0, le=64, description="Optional /dev/videoN; defaults to saved camera device")
    controls: dict[str, Any] = Field(..., description="Control id → value, e.g. brightness: 128")


def _front_device_index(settings: dict[str, Any]) -> int:
    sec = settings.get("webcam") or {}
    return max(0, int(sec.get("webcam_device", 0) or 0))


def _back_device_index(settings: dict[str, Any]) -> int:
    sec = settings.get("multi_camera") or {}
    return max(0, int(sec.get("webcam_device", 0) or 0))


def build_camera_roles_router(ctx: RouterContext) -> APIRouter:
    router = APIRouter(tags=["cameras"])

    # --- Front Camera (webcam) ---

    @router.get("/api/front_camera/config")
    def front_camera_config() -> dict[str, Any]:
        s = ctx.settings.get()
        return {"front_camera": dict(s.get("webcam") or {}), "webcam": dict(s.get("webcam") or {})}

    @router.put("/api/front_camera/config")
    def put_front_camera_config(body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        patch = body.get("front_camera")
        if patch is None:
            patch = body.get("webcam")
        if patch is None:
            raise HTTPException(400, "body must include front_camera or webcam object")
        if not isinstance(patch, dict):
            raise HTTPException(400, "front_camera must be an object")
        current = ctx.settings.get()
        current["webcam"] = {**(current.get("webcam") or {}), **patch}
        return ctx.settings.replace(current)

    @router.get("/api/front_camera/status")
    def front_camera_status() -> dict[str, Any]:
        return ctx.sensors.status("webcam")

    @router.post("/api/front_camera/run")
    def front_camera_run() -> Any:
        return ctx.sensors.run_sensor("webcam")

    @router.post("/api/front_camera/stop")
    def front_camera_stop() -> dict[str, Any]:
        return ctx.sensors.stop("webcam")

    @router.post("/api/front_camera/restart")
    def front_camera_restart() -> Any:
        return ctx.sensors.restart_sensor("webcam")

    @router.get("/api/front_camera/preview/live")
    async def front_camera_preview_live() -> StreamingResponse:
        return await ai_camera_live_mjpeg(ctx)

    @router.post("/api/front_camera/webrtc/offer")
    async def front_camera_webrtc_offer(body: WebRTCOfferBody) -> dict[str, str]:
        return await handle_webcam_offer(ctx, body)

    @router.get("/api/front_camera/command")
    def front_camera_command() -> dict[str, Any]:
        return ctx.sensors.build_command("webcam", ctx.settings.get())

    @router.get("/api/front_camera/controls")
    def front_camera_controls() -> dict[str, Any]:
        index = _front_device_index(ctx.settings.get())
        out = v4l2_camera_controls.list_camera_controls(index)
        if not out.get("ok"):
            raise HTTPException(status_code=503, detail=out.get("error") or "list-ctrls failed")
        out["camera"] = "front"
        out["settings_key"] = "webcam"
        return out

    @router.post("/api/front_camera/controls/set")
    def front_camera_controls_set(body: CameraControlsSetBody) -> dict[str, Any]:
        if not body.controls:
            raise HTTPException(400, "controls object is required")
        index = body.index if body.index is not None else _front_device_index(ctx.settings.get())
        out = v4l2_camera_controls.set_camera_controls(int(index), body.controls)
        if not out.get("ok"):
            raise HTTPException(status_code=400, detail=out.get("error") or "set-ctrl failed")
        out["camera"] = "front"
        return out

    @router.post("/api/front_camera/controls/reset")
    def front_camera_controls_reset(index: int | None = None) -> dict[str, Any]:
        idx = int(index) if index is not None else _front_device_index(ctx.settings.get())
        out = v4l2_camera_controls.reset_camera_controls(idx)
        if not out.get("ok"):
            raise HTTPException(status_code=400, detail=out.get("error") or "reset failed")
        out["camera"] = "front"
        return out

    # --- Back Camera (multi_camera) ---

    @router.get("/api/back_camera/config")
    def back_camera_config() -> dict[str, Any]:
        s = ctx.settings.get()
        return {
            "back_camera": dict(s.get("multi_camera") or {}),
            "multi_camera": dict(s.get("multi_camera") or {}),
        }

    @router.put("/api/back_camera/config")
    def put_back_camera_config(body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        patch = body.get("back_camera")
        if patch is None:
            patch = body.get("multi_camera")
        if patch is None:
            raise HTTPException(400, "body must include back_camera or multi_camera object")
        if not isinstance(patch, dict):
            raise HTTPException(400, "back_camera must be an object")
        current = ctx.settings.get()
        current["multi_camera"] = {**(current.get("multi_camera") or {}), **patch}
        return ctx.settings.replace(current)

    @router.get("/api/back_camera/status")
    def back_camera_status() -> dict[str, Any]:
        return ctx.sensors.status("multi_camera")

    @router.post("/api/back_camera/run")
    def back_camera_run() -> Any:
        return ctx.sensors.run_sensor("multi_camera")

    @router.post("/api/back_camera/stop")
    def back_camera_stop() -> dict[str, Any]:
        return ctx.sensors.stop("multi_camera")

    @router.post("/api/back_camera/restart")
    def back_camera_restart() -> Any:
        return ctx.sensors.restart_sensor("multi_camera")

    @router.get("/api/back_camera/preview/live")
    async def back_camera_preview_live() -> StreamingResponse:
        return await multi_camera_live_mjpeg(ctx)

    @router.post("/api/back_camera/webrtc/offer")
    async def back_camera_webrtc_offer(body: WebRTCOfferBody) -> dict[str, str]:
        return await handle_multi_camera_offer(ctx, body)

    @router.get("/api/back_camera/command")
    def back_camera_command() -> dict[str, Any]:
        return ctx.sensors.build_command("multi_camera", ctx.settings.get())

    @router.get("/api/back_camera/controls")
    def back_camera_controls() -> dict[str, Any]:
        index = _back_device_index(ctx.settings.get())
        out = v4l2_camera_controls.list_camera_controls(index)
        if not out.get("ok"):
            raise HTTPException(status_code=503, detail=out.get("error") or "list-ctrls failed")
        out["camera"] = "back"
        out["settings_key"] = "multi_camera"
        return out

    @router.post("/api/back_camera/controls/set")
    def back_camera_controls_set(body: CameraControlsSetBody) -> dict[str, Any]:
        if not body.controls:
            raise HTTPException(400, "controls object is required")
        index = body.index if body.index is not None else _back_device_index(ctx.settings.get())
        out = v4l2_camera_controls.set_camera_controls(int(index), body.controls)
        if not out.get("ok"):
            raise HTTPException(status_code=400, detail=out.get("error") or "set-ctrl failed")
        out["camera"] = "back"
        return out

    @router.post("/api/back_camera/controls/reset")
    def back_camera_controls_reset(index: int | None = None) -> dict[str, Any]:
        idx = int(index) if index is not None else _back_device_index(ctx.settings.get())
        out = v4l2_camera_controls.reset_camera_controls(idx)
        if not out.get("ok"):
            raise HTTPException(status_code=400, detail=out.get("error") or "reset failed")
        out["camera"] = "back"
        return out

    @router.get("/api/front_camera/profiles")
    def front_camera_profiles() -> dict[str, Any]:
        from services import model_profiles as profiles

        return {"profiles": profiles.get_model_profiles_normalized(ctx.layer8_dir)}

    @router.post("/api/front_camera/profiles/apply")
    def front_camera_apply_profile(body: ApplyModelProfileBody) -> dict[str, Any]:
        from services import model_profiles as profiles

        pid = body.id.strip()
        if not pid:
            raise HTTPException(400, "id is required")
        norm = profiles.get_model_profiles_normalized(ctx.layer8_dir)
        prof = norm.get(pid)
        if prof is None:
            raise HTTPException(404, "profile not found")
        values = prof.get("values") or {}
        if not isinstance(values, dict):
            raise HTTPException(400, "profile.values must be an object")
        current = ctx.settings.get()
        w = profiles.apply_values_to_webcam({**(current.get("webcam") or {})}, values)
        w["active_model_profile_id"] = pid
        current["webcam"] = w
        return ctx.settings.replace(current)

    @router.get("/api/back_camera/profiles")
    def back_camera_profiles() -> dict[str, Any]:
        from services import model_profiles as profiles

        return {"profiles": profiles.get_model_profiles_normalized(ctx.layer8_dir)}

    @router.post("/api/back_camera/profiles/apply")
    def back_camera_apply_profile(body: ApplyModelProfileBody) -> dict[str, Any]:
        from services import model_profiles as profiles

        pid = body.id.strip()
        if not pid:
            raise HTTPException(400, "id is required")
        norm = profiles.get_model_profiles_normalized(ctx.layer8_dir)
        prof = norm.get(pid)
        if prof is None:
            raise HTTPException(404, "profile not found")
        values = prof.get("values") or {}
        if not isinstance(values, dict):
            raise HTTPException(400, "profile.values must be an object")
        current = ctx.settings.get()
        mc = profiles.apply_values_to_multi_camera({**(current.get("multi_camera") or {})}, values)
        mc["active_model_profile_id"] = pid
        current["multi_camera"] = mc
        return ctx.settings.replace(current)

    @router.get("/api/front_camera/threat/metrics")
    def front_camera_threat_metrics() -> dict[str, Any]:
        return ctx.metrics.threat_metrics_for("webcam")

    @router.get("/api/back_camera/threat/metrics")
    def back_camera_threat_metrics() -> dict[str, Any]:
        return ctx.metrics.threat_metrics_for("multi_camera")

    @router.get("/api/cameras")
    def list_camera_apis() -> dict[str, Any]:
        """Discovery helper for Front / Back camera namespaces."""
        return {
            "front_camera": {
                "label": "Front Cam",
                "settings_key": "webcam",
                "legacy_aliases": ["/api/ai_camera", "/api/run/webcam"],
                "endpoints": {
                    "config": "/api/front_camera/config",
                    "status": "/api/front_camera/status",
                    "run": "/api/front_camera/run",
                    "stop": "/api/front_camera/stop",
                    "restart": "/api/front_camera/restart",
                    "preview_live": "/api/front_camera/preview/live",
                    "webrtc_offer": "/api/front_camera/webrtc/offer",
                    "controls": "/api/front_camera/controls",
                    "profiles": "/api/front_camera/profiles",
                },
            },
            "back_camera": {
                "label": "Back Cam",
                "settings_key": "multi_camera",
                "legacy_aliases": ["/api/multi_camera", "/api/run/multi_camera"],
                "endpoints": {
                    "config": "/api/back_camera/config",
                    "status": "/api/back_camera/status",
                    "run": "/api/back_camera/run",
                    "stop": "/api/back_camera/stop",
                    "restart": "/api/back_camera/restart",
                    "preview_live": "/api/back_camera/preview/live",
                    "webrtc_offer": "/api/back_camera/webrtc/offer",
                    "controls": "/api/back_camera/controls",
                    "profiles": "/api/back_camera/profiles",
                },
            },
        }

    return router
