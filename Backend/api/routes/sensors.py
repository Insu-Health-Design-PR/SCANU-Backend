"""Sensor control and status routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.routes.context import RouterContext
from api.schemas.sensors import SensorName
from api.streaming.status_stream import status_events


def build_sensors_router(ctx: RouterContext) -> APIRouter:
    router = APIRouter(tags=["sensors"])

    @router.get("/api/thermal/status")
    def thermal_status() -> dict[str, Any]:
        return ctx.sensors.status("thermal")

    @router.post("/api/thermal/run")
    def thermal_run() -> Any:
        return ctx.sensors.start("thermal")

    @router.post("/api/thermal/stop")
    def thermal_stop() -> dict[str, Any]:
        return ctx.sensors.stop("thermal")

    @router.post("/api/thermal/restart")
    def thermal_restart() -> Any:
        return ctx.sensors.restart("thermal")

    @router.get("/api/mmwave/status")
    def mmwave_status() -> dict[str, Any]:
        return ctx.sensors.status("mmwave")

    @router.post("/api/mmwave/run")
    def mmwave_run() -> Any:
        return ctx.sensors.start("mmwave")

    @router.post("/api/mmwave/stop")
    def mmwave_stop() -> dict[str, Any]:
        return ctx.sensors.stop("mmwave")

    @router.post("/api/mmwave/restart")
    def mmwave_restart() -> Any:
        return ctx.sensors.restart("mmwave")

    @router.get("/api/status")
    def all_status() -> dict[str, Any]:
        return ctx.sensors.all_status()

    @router.get("/api/status/stream")
    async def stream_status() -> StreamingResponse:
        return StreamingResponse(status_events(ctx.layer8_dir), media_type="text/event-stream")

    @router.get("/api/status/{sensor}")
    def one_status(sensor: SensorName) -> dict[str, Any]:
        return ctx.sensors.status(sensor)

    @router.get("/api/ai_camera/status")
    def ai_camera_status() -> dict[str, Any]:
        return ctx.sensors.status("webcam")

    @router.post("/api/run/{sensor}")
    def run_sensor(sensor: SensorName) -> Any:
        return ctx.sensors.run_sensor(sensor)

    @router.post("/api/ai_camera/run")
    def ai_camera_run() -> Any:
        return ctx.sensors.run_sensor("webcam")

    @router.post("/api/stop/{sensor}")
    def stop_sensor(sensor: SensorName) -> dict[str, Any]:
        return ctx.sensors.stop(sensor)

    @router.post("/api/ai_camera/stop")
    def ai_camera_stop() -> dict[str, Any]:
        return ctx.sensors.stop("webcam")

    @router.post("/api/restart/{sensor}")
    def restart_sensor(sensor: SensorName) -> Any:
        return ctx.sensors.restart_sensor(sensor)

    @router.post("/api/ai_camera/restart")
    def ai_camera_restart() -> Any:
        return ctx.sensors.restart_sensor("webcam")

    @router.get("/api/multi_camera/status")
    def multi_camera_status() -> dict[str, Any]:
        return ctx.sensors.status("multi_camera")

    @router.post("/api/multi_camera/run")
    def multi_camera_run() -> Any:
        return ctx.sensors.run_sensor("multi_camera")

    @router.post("/api/multi_camera/stop")
    def multi_camera_stop() -> dict[str, Any]:
        return ctx.sensors.stop("multi_camera")

    @router.post("/api/multi_camera/restart")
    def multi_camera_restart() -> Any:
        return ctx.sensors.restart_sensor("multi_camera")

    @router.post("/api/run_all")
    def run_all_sensors() -> Any:
        return ctx.sensors.run_all()

    @router.post("/api/stop_all")
    def stop_all_sensors() -> dict[str, Any]:
        return ctx.sensors.stop_all()

    @router.post("/api/restart_all")
    def restart_all_sensors() -> Any:
        return ctx.sensors.restart_all()

    return router
