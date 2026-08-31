"""MJPEG, WebRTC, WebSocket preview, and embed page routes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, WebSocket
from fastapi.responses import FileResponse, StreamingResponse

from api.routes.context import RouterContext
from api.schemas.sensors import SensorName
from api.schemas.webrtc import WebRTCOfferBody
from api.streaming.live_mjpeg import (
    ai_camera_live_mjpeg,
    mmwave_live_mjpeg,
    multi_camera_live_mjpeg,
    thermal_live_mjpeg,
)
from api.streaming.mmwave_websocket import websocket_mmwave
from api.streaming.webrtc import handle_multi_camera_offer, handle_webcam_offer
from api.streaming.websocket import websocket_multi_camera, websocket_thermal, websocket_webcam

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from layer8_ui.artifact_paths import resolved_artifact_path  # noqa: E402


def build_preview_router(ctx: RouterContext) -> APIRouter:
    layer8_dir = ctx.layer8_dir
    router = APIRouter(tags=["preview"])

    @router.get("/api/preview/video/{sensor}")
    def preview_video(sensor: Literal["thermal", "webcam", "multi_camera", "mmwave"]) -> FileResponse:
        s = ctx.settings.get()
        key = "video"
        sub = (s.get(sensor) or {}).get(key) or ""
        path = resolved_artifact_path(s, relative_to_software=str(sub), layer8_dir=layer8_dir)
        if path is None or not path.is_file():
            raise HTTPException(
                404,
                "Video file not found. Run capture first, or fix the video path in settings "
                "(must be under software/ or layer8_ui/).",
            )
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @router.get("/api/preview/live/{sensor}")
    async def preview_live(
        sensor: SensorName,
        side: str = Query("fused", description="mmWave view: fused|front|back"),
    ) -> StreamingResponse:
        if sensor == "webcam":
            return await ai_camera_live_mjpeg(ctx)
        if sensor == "multi_camera":
            return await multi_camera_live_mjpeg(ctx)
        if sensor == "thermal":
            return await thermal_live_mjpeg(ctx)
        return await mmwave_live_mjpeg(ctx, side=side)

    @router.get("/api/ai_camera/preview/live")
    async def ai_camera_preview_live() -> StreamingResponse:
        return await ai_camera_live_mjpeg(ctx)

    @router.get("/api/multi_camera/preview/live")
    async def multi_camera_preview_live() -> StreamingResponse:
        return await multi_camera_live_mjpeg(ctx)

    @router.get("/api/thermal/preview/live")
    @router.get("/api/preview/live_direct/thermal")
    async def thermal_preview_live_aliases() -> StreamingResponse:
        return await thermal_live_mjpeg(ctx)

    @router.get("/api/mmwave/preview/live")
    @router.get("/api/preview/live_direct/mmwave")
    async def mmwave_preview_live_aliases(
        side: str = Query("fused", description="mmWave view: fused|front|back"),
    ) -> StreamingResponse:
        return await mmwave_live_mjpeg(ctx, side=side)

    @router.websocket("/ws/thermal")
    async def ws_thermal(websocket: WebSocket) -> None:
        await websocket_thermal(websocket, ctx)

    @router.websocket("/ws/webcam")
    async def ws_webcam(websocket: WebSocket) -> None:
        await websocket_webcam(websocket, ctx)

    @router.websocket("/ws/multi_camera")
    async def ws_multi_camera(websocket: WebSocket) -> None:
        await websocket_multi_camera(websocket, ctx)

    @router.websocket("/ws/mmwave")
    async def ws_mmwave(websocket: WebSocket) -> None:
        await websocket_mmwave(websocket, ctx)

    @router.post("/api/webrtc/multi_camera/offer")
    async def webrtc_multi_camera_offer(body: WebRTCOfferBody) -> dict[str, str]:
        return await handle_multi_camera_offer(ctx, body)

    @router.post("/api/webrtc/webcam/offer")
    @router.post("/api/ai_camera/webrtc/offer")
    @router.post("/api/webrtc/ai_camera/offer")
    async def webrtc_webcam_offer(body: WebRTCOfferBody) -> dict[str, str]:
        return await handle_webcam_offer(ctx, body)

    @router.get("/embed/thermal")
    def embed_thermal_page() -> FileResponse:
        """Minimal full-page thermal viewer for a second screen or another operator."""
        p = layer8_dir / "static" / "embed_thermal.html"
        if not p.is_file():
            raise HTTPException(404, "static/embed_thermal.html missing")
        return FileResponse(p, media_type="text/html")

    @router.get("/embed/webcam")
    def embed_webcam_page() -> FileResponse:
        """Minimal full-page webcam viewer; use alongside ``/embed/thermal`` on another device."""
        p = layer8_dir / "static" / "embed_webcam.html"
        if not p.is_file():
            raise HTTPException(404, "static/embed_webcam.html missing")
        return FileResponse(p, media_type="text/html")

    @router.get("/embed/webcam_mjpeg")
    def embed_webcam_mjpeg_page() -> FileResponse:
        """Webcam preview via ``/api/ai_camera/preview/live`` (MJPEG); handy for second screens / Tailscale."""
        p = layer8_dir / "static" / "embed_webcam_mjpeg.html"
        if not p.is_file():
            raise HTTPException(404, "static/embed_webcam_mjpeg.html missing")
        return FileResponse(p, media_type="text/html")

    def _mmwave_output_file_response() -> FileResponse:
        s = ctx.settings.get()
        sub = (s.get("mmwave") or {}).get("output") or ""
        path = resolved_artifact_path(s, relative_to_software=str(sub), layer8_dir=layer8_dir)
        if path is None or not path.is_file():
            raise HTTPException(
                404,
                "Output JSON not found. Run mmWave capture or set output path in settings.",
            )
        return FileResponse(path, media_type="application/json", filename=path.name)

    @router.get("/api/preview/output/mmwave")
    def preview_mmwave_output() -> FileResponse:
        return _mmwave_output_file_response()

    @router.get("/api/mmwave/preview/output")
    def mmwave_preview_output() -> FileResponse:
        return _mmwave_output_file_response()

    return router
