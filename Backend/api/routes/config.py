"""Settings and per-sensor configuration routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.routes.context import RouterContext
from api.schemas.sensors import SensorName

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from layer8_ui import v4l2_tools  # noqa: E402
from layer8_ui.thermal_device import detect_working_thermal_device  # noqa: E402


def build_config_router(ctx: RouterContext) -> APIRouter:
    router = APIRouter(tags=["config"])

    @router.get("/api/config")
    def get_config() -> dict[str, Any]:
        return ctx.settings.get()

    @router.put("/api/config")
    def put_config(body: dict[str, Any]) -> dict[str, Any]:
        """Accept either ``{"settings": {...}}`` or a bare settings object.

        Older / cached UI builds sometimes PUT the config dict at the top level;
        that used to 422 on SettingsBody and look like a save failure.
        """
        if not isinstance(body, dict) or not body:
            raise HTTPException(400, "body must be a non-empty object")
        if isinstance(body.get("settings"), dict) and (
            "thermal" in body["settings"]
            or "webcam" in body["settings"]
            or "multi_camera" in body["settings"]
            or "mmwave" in body["settings"]
            or "software_root" in body["settings"]
        ):
            incoming = body["settings"]
        elif any(k in body for k in ("thermal", "webcam", "multi_camera", "mmwave", "software_root")):
            # Bare settings document (no wrapper).
            incoming = body
        elif isinstance(body.get("settings"), dict):
            incoming = body["settings"]
        else:
            raise HTTPException(
                422,
                "Expected {\"settings\": {...}} or a bare settings object with sensor keys",
            )
        return ctx.settings.replace(incoming)

    @router.post("/api/config/reset")
    def reset_config() -> dict[str, Any]:
        return ctx.settings.reset_all()

    @router.post("/api/config/reset/{sensor}")
    def reset_sensor_config(sensor: SensorName) -> dict[str, Any]:
        return ctx.settings.reset_sensor(sensor)

    @router.post("/api/config/reset/model")
    def reset_model_weapon_defaults() -> dict[str, Any]:
        """Reset weapon / verbose keys only (stored under ``webcam`` in JSON)."""
        return ctx.settings.reset_webcam_model()

    @router.post("/api/config/reset/thermal-model")
    def reset_thermal_model_weapon_defaults() -> dict[str, Any]:
        """Reset weapon / verbose keys only (stored under ``thermal`` in JSON)."""
        return ctx.settings.reset_thermal_model()

    @router.post("/api/config/reset/multi-camera-model")
    def reset_multi_camera_model_weapon_defaults() -> dict[str, Any]:
        """Reset weapon / verbose keys only (stored under ``multi_camera`` in JSON)."""
        return ctx.settings.reset_multi_camera_model()

    @router.get("/api/thermal/config")
    def get_thermal_config() -> dict[str, Any]:
        s = ctx.settings.get()
        return {"thermal": dict(s.get("thermal") or {})}

    @router.put("/api/thermal/config")
    def put_thermal_config(body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        patch = body.get("thermal")
        if patch is None or not isinstance(patch, dict):
            raise HTTPException(400, "body must include a thermal object")
        current = ctx.settings.get()
        current["thermal"] = {**(current.get("thermal") or {}), **patch}
        return ctx.settings.replace(current)

    @router.post("/api/thermal/auto_configure")
    def thermal_auto_configure() -> Any:
        current = ctx.settings.get()
        t = dict(current.get("thermal") or {})
        width = int(t.get("thermal_width", 160))
        height = int(t.get("thermal_height", 120))
        fps = int(t.get("thermal_fps", 9))
        preferred = int(t.get("thermal_device", 0))
        max_idx = int(t.get("thermal_detect_max_index", 12))
        detected = detect_working_thermal_device(
            preferred=preferred,
            width=width,
            height=height,
            fps=fps,
            search_max_index=max_idx,
        )
        if detected is None:
            return JSONResponse(
                status_code=404,
                content={
                    "ok": False,
                    "error": "No working thermal V4L2 device found",
                    "thermal": t,
                },
            )
        t["thermal_device"] = int(detected)
        t["thermal_auto_detect"] = 1
        current["thermal"] = t
        ctx.settings.replace(current)
        out = ctx.settings.get()
        return {
            "ok": True,
            "thermal": dict(out.get("thermal") or {}),
            "detected_device": int(detected),
        }

    @router.get("/api/mmwave/config")
    def get_mmwave_config() -> dict[str, Any]:
        s = ctx.settings.get()
        return {"mmwave": dict(s.get("mmwave") or {})}

    @router.put("/api/mmwave/config")
    def put_mmwave_config(body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        patch = body.get("mmwave")
        if patch is None or not isinstance(patch, dict):
            raise HTTPException(400, "body must include a mmwave object")
        current = ctx.settings.get()
        current["mmwave"] = {**(current.get("mmwave") or {}), **patch}
        return ctx.settings.replace(current)

    @router.post("/api/mmwave/auto_configure")
    def mmwave_auto_configure() -> Any:
        """Set ``cli_port`` / ``data_port`` from ``/api/devices/serial`` heuristics (same as dashboard Auto-detect)."""
        cand = v4l2_tools.list_serial_port_candidates()
        if not cand.get("ok"):
            return JSONResponse(
                status_code=404,
                content={
                    "ok": False,
                    "error": str(cand.get("error") or "serial discovery failed"),
                    "mmwave": dict(ctx.settings.get().get("mmwave") or {}),
                },
            )
        current = ctx.settings.get()
        m = dict(current.get("mmwave") or {})
        cli = cand.get("suggested_cli")
        data = cand.get("suggested_data")
        if cli:
            m["cli_port"] = str(cli)
        if data:
            m["data_port"] = str(data)
        current["mmwave"] = m
        ctx.settings.replace(current)
        out = ctx.settings.get()
        return {
            "ok": True,
            "mmwave": dict(out.get("mmwave") or {}),
            "suggested_cli": cli,
            "suggested_data": data,
            "ports": cand.get("ports") or [],
        }

    @router.get("/api/ai_camera/config")
    def get_ai_camera_config() -> dict[str, Any]:
        return ctx.settings.get()

    @router.put("/api/ai_camera/config")
    def put_ai_camera_config(body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        current = ctx.settings.get()
        patch = body.get("webcam")
        if patch is not None:
            if not isinstance(patch, dict):
                raise HTTPException(400, "webcam must be an object when provided")
            current["webcam"] = {**(current.get("webcam") or {}), **patch}
        return ctx.settings.replace(current)

    return router
