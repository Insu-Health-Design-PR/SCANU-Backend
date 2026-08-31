"""Multipart MJPEG stream builders for live sensor previews."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi.responses import StreamingResponse

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from runtime import sensor_runner  # noqa: E402
from layer8_ui.artifact_paths import resolved_artifact_path  # noqa: E402
from layer8_ui.preview_media import (  # noqa: E402
    THERMAL_JPEG_WAITING,
    WEBCAM_JPEG_WAITING,
    mjpeg_headers,
    mjpeg_placeholder_jpeg,
)

from api.streaming.mjpeg import mjpeg_chunk, mjpeg_media_type
from api.streaming.webrtc_frames import preview_frame_interval_s

if TYPE_CHECKING:
    from api.routes.context import RouterContext


async def ai_camera_live_mjpeg(ctx: RouterContext) -> StreamingResponse:
    """AI webcam MJPEG: same frame selection as ``vid-main-webrtc`` / ``WebcamJpegTrack``."""
    layer8_dir = ctx.layer8_dir
    st = sensor_runner.status("webcam", layer8_dir)
    s = ctx.settings.get()
    infer_runner = bool(st.get("running")) and not bool(st.get("preview_only"))

    if infer_runner:
        rel = (s.get("webcam") or {}).get("live_frame") or ""
        rpath = resolved_artifact_path(s, relative_to_software=str(rel), layer8_dir=layer8_dir)
        runner_missing = mjpeg_placeholder_jpeg(
            "Webcam runner is ON (subprocess holds the camera).",
            "Waiting for BGR/JPEG IPC or webcam.live_frame (same order as WebRTC).",
            (rpath.name if rpath else "configure webcam.live_frame"),
        )

        async def mjpeg_runner_file():
            boundary = "frame"
            last_good: bytes | None = None
            try:
                while True:
                    jpg = ctx.frame_sources.runner_frame_jpeg_webcam(
                        rpath,
                        allow_disk_fallback=False,
                    )
                    if jpg:
                        last_good = jpg
                    elif last_good:
                        jpg = last_good
                    else:
                        jpg = runner_missing
                    yield mjpeg_chunk(jpg, boundary)
                    await asyncio.sleep(
                        preview_frame_interval_s(ctx.settings.get().get("webcam") or {})
                    )
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            mjpeg_runner_file(),
            media_type=mjpeg_media_type(),
            headers=mjpeg_headers(),
        )

    async def mjpeg_stream_shared():
        boundary = "frame"
        last_good: bytes | None = None
        ctx.webcam_stream.add_client(s)
        try:
            while True:
                st_now = sensor_runner.status("webcam", layer8_dir)
                if bool(st_now.get("running")) and not bool(st_now.get("preview_only")):
                    break
                payload = ctx.webcam_stream.latest_jpg()
                if payload:
                    last_good = payload
                else:
                    payload = last_good or WEBCAM_JPEG_WAITING
                yield mjpeg_chunk(payload, boundary)
                await asyncio.sleep(
                    preview_frame_interval_s(ctx.settings.get().get("webcam") or {})
                )
        except asyncio.CancelledError:
            return
        finally:
            ctx.webcam_stream.remove_client()

    return StreamingResponse(
        mjpeg_stream_shared(),
        media_type=mjpeg_media_type(),
        headers=mjpeg_headers(),
    )


async def multi_camera_live_mjpeg(ctx: RouterContext) -> StreamingResponse:
    """Multi-camera MJPEG: same frame selection as webcam but uses multi_camera IPC/settings."""
    layer8_dir = ctx.layer8_dir
    st = sensor_runner.status("multi_camera", layer8_dir)
    s = ctx.settings.get()

    if bool(st.get("running")):
        rel = (s.get("multi_camera") or {}).get("live_frame") or ""
        rpath = resolved_artifact_path(s, relative_to_software=str(rel), layer8_dir=layer8_dir)
        runner_missing = mjpeg_placeholder_jpeg(
            "Multi-camera runner is ON (subprocess holds the camera).",
            "Waiting for BGR/JPEG IPC (same order as WebRTC).",
            (rpath.name if rpath else "configure multi_camera.live_frame"),
        )

        async def mjpeg_runner_file():
            boundary = "frame"
            last_good: bytes | None = None
            try:
                while True:
                    jpg = ctx.frame_sources.runner_frame_jpeg_multi_camera(
                        rpath,
                        allow_disk_fallback=False,
                    )
                    if jpg:
                        last_good = jpg
                    elif last_good:
                        jpg = last_good
                    else:
                        jpg = runner_missing
                    yield mjpeg_chunk(jpg, boundary)
                    await asyncio.sleep(
                        preview_frame_interval_s(ctx.settings.get().get("multi_camera") or {})
                    )
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            mjpeg_runner_file(),
            media_type=mjpeg_media_type(),
            headers=mjpeg_headers(),
        )

    async def mjpeg_stream_shared():
        boundary = "frame"
        last_good: bytes | None = None
        ctx.multi_camera_stream.add_client(s)
        try:
            while True:
                st_now = sensor_runner.status("multi_camera", layer8_dir)
                if bool(st_now.get("running")) and not bool(st_now.get("preview_only")):
                    break
                payload = ctx.multi_camera_stream.latest_jpg()
                if payload:
                    last_good = payload
                else:
                    payload = last_good or WEBCAM_JPEG_WAITING
                yield mjpeg_chunk(payload, boundary)
                await asyncio.sleep(
                    preview_frame_interval_s(ctx.settings.get().get("multi_camera") or {})
                )
        except asyncio.CancelledError:
            return
        finally:
            ctx.multi_camera_stream.remove_client()

    return StreamingResponse(
        mjpeg_stream_shared(),
        media_type=mjpeg_media_type(),
        headers=mjpeg_headers(),
    )


async def thermal_live_mjpeg(ctx: RouterContext) -> StreamingResponse:
    """Thermal MJPEG: IPC overlay when runner on, else shared V4L2 colormap stream."""
    layer8_dir = ctx.layer8_dir
    fs = ctx.frame_sources

    async def mjpeg_stream():
        boundary = "frame"
        held = False
        last_good: bytes | None = None
        try:
            while True:
                s = ctx.settings.get()
                runner_on = bool(sensor_runner.status("thermal", layer8_dir).get("running"))
                rel = (s.get("thermal") or {}).get("live_frame") or ""
                rpath = resolved_artifact_path(s, relative_to_software=str(rel), layer8_dir=layer8_dir)
                payload: bytes | None = None
                if runner_on:
                    if held:
                        ctx.thermal_stream.remove_client()
                        held = False
                    payload = await asyncio.to_thread(
                        lambda: fs.runner_frame_jpeg_thermal(
                            rpath, allow_disk_fallback=False
                        ),
                    )
                    if payload:
                        last_good = payload
                    elif last_good:
                        payload = last_good
                    else:
                        payload = fs.thermal_runner_ipc_placeholder(layer8_dir)
                elif fs.thermal_uses_v4l2_preview(s):
                    if not held:
                        ctx.thermal_stream.add_client(s)
                        held = True
                    else:
                        ctx.thermal_stream.sync_settings(s)
                    payload = ctx.thermal_stream.latest_jpg() or THERMAL_JPEG_WAITING
                    if payload and payload is not THERMAL_JPEG_WAITING:
                        last_good = payload
                else:
                    payload = fs.thermal_disk_jpeg(s, layer8_dir) or mjpeg_placeholder_jpeg(
                        "Thermal idle — start Run for live AI overlay.",
                        "Preview does not open the camera (direct/shared pipeline).",
                        "layer8_ui/artifacts/live_thermal.jpg",
                    )
                yield mjpeg_chunk(payload, boundary)
                if runner_on:
                    await asyncio.sleep(
                        preview_frame_interval_s(ctx.settings.get().get("thermal") or {})
                    )
                else:
                    await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            return
        finally:
            if held:
                ctx.thermal_stream.remove_client()

    return StreamingResponse(
        mjpeg_stream(),
        media_type=mjpeg_media_type(),
        headers=mjpeg_headers(),
    )


async def mmwave_live_mjpeg(ctx: RouterContext, *, side: str = "fused") -> StreamingResponse:
    """mmWave MJPEG; calibrated fused dashboard is the default view."""
    s = ctx.settings.get()
    m = s.get("mmwave") or {}
    side_l = str(side or "front").strip().lower()
    if side_l in ("fused", "fusion", "dashboard", ""):
        rel = m.get("live_frame_fused") or m.get("live_frame") or ""
        label = "mmWave Fused"
    elif side_l in ("back", "b", "rear"):
        rel = m.get("live_frame_back") or m.get("live_frame") or ""
        label = "mmWave Back"
    else:
        rel = m.get("live_frame") or ""
        label = "mmWave Front"
    missing = mjpeg_placeholder_jpeg(
        f"{label}: no live JPEG yet",
        f"Expected file: {Path(str(rel)).name if rel else 'configure mmwave.live_frame(_back)'}",
        "Start the mmWave runner or fix live_frame in settings.",
    )

    async def mjpeg_stream():
        boundary = "frame"
        last_good: bytes | None = None
        try:
            while True:
                payload: bytes | None = None
                # Resolve on every frame. A dashboard may be created after the
                # browser opened the stream; keeping an earlier ``None`` would
                # otherwise make the placeholder permanent for that client.
                rpath = resolved_artifact_path(
                    ctx.settings.get(),
                    relative_to_software=str(rel),
                    layer8_dir=ctx.layer8_dir,
                )
                if rpath is not None and rpath.is_file():
                    try:
                        raw = rpath.read_bytes()
                        if raw:
                            payload = raw
                            last_good = raw
                    except OSError:
                        payload = None
                if not payload:
                    payload = last_good or missing
                yield mjpeg_chunk(payload, boundary)
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        mjpeg_stream(),
        media_type=mjpeg_media_type(),
        headers=mjpeg_headers(),
    )
