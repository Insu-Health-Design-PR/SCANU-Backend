"""Save a single live preview frame to Desktop/Screenshots."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from api.routes.context import RouterContext

ScreenshotSensor = Literal["thermal", "webcam", "multi_camera"]


def build_screenshot_router(ctx: RouterContext) -> APIRouter:
    router = APIRouter(tags=["screenshot"])

    @router.post("/api/screenshot/{sensor}")
    def capture_screenshot(sensor: ScreenshotSensor) -> dict[str, Any]:
        result = ctx.screenshots.capture(sensor)
        if not result.get("ok"):
            err = str(result.get("error") or "capture_failed")
            if err == "no_frame":
                raise HTTPException(503, result.get("message") or "No live frame available")
            raise HTTPException(500, result.get("message") or err)
        return result

    @router.get("/api/screenshot/{sensor}/frame")
    def screenshot_frame(sensor: ScreenshotSensor) -> Response:
        """Single JPEG still from IPC — avoids opening a second MJPEG stream in the browser."""
        jpg = ctx.screenshots.capture_jpeg_bytes(sensor)
        if not jpg:
            raise HTTPException(
                503,
                f"No live {sensor} frame available. Start the sensor runner and wait for preview.",
            )
        return Response(content=jpg, media_type="image/jpeg")

    return router
