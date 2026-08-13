"""Live stream record start/stop/download routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.routes.context import RouterContext

RecordSensor = Literal["thermal", "webcam"]


class RecordStartBody(BaseModel):
    playback_speed: float = Field(1.5, ge=1.0, le=4.0, description="MP4 FPS = source_fps × speed")
    source_fps: float | None = Field(
        None,
        ge=1.0,
        le=120.0,
        description="Captured stream rate (default from sensor settings, usually ~15–30)",
    )
    output_path: str | None = Field(
        None,
        description="Optional path relative to software root, e.g. layer8_ui/artifacts/recordings/foo.mp4",
    )


def build_recording_router(ctx: RouterContext) -> APIRouter:
    router = APIRouter(tags=["recording"])

    @router.get("/api/record/status")
    def all_record_status() -> dict[str, Any]:
        return ctx.recording.status()

    @router.get("/api/record/{sensor}/status")
    def record_status(sensor: RecordSensor) -> dict[str, Any]:
        return ctx.recording.status(sensor)

    @router.post("/api/record/{sensor}/start")
    def record_start(sensor: RecordSensor, body: RecordStartBody | None = None) -> dict[str, Any]:
        opts = body or RecordStartBody()
        settings = ctx.settings.get()
        return ctx.recording.start(
            sensor,
            settings,
            playback_speed=opts.playback_speed,
            source_fps=opts.source_fps,
            output_path=opts.output_path,
        )

    @router.post("/api/record/{sensor}/stop")
    def record_stop(sensor: RecordSensor) -> dict[str, Any]:
        return ctx.recording.stop(sensor)

    @router.get("/api/record/{sensor}/download")
    def record_download(sensor: RecordSensor) -> FileResponse:
        path = ctx.recording.latest_recording_path(sensor)
        if path is None or not path.is_file():
            raise HTTPException(
                404,
                "No recording file found. Start a recording, stop it, then download.",
            )
        return FileResponse(path, media_type="video/mp4", filename=Path(path).name)

    return router
