"""WebSocket binary JPEG stream handlers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from runtime import sensor_runner  # noqa: E402
from layer8_ui.artifact_paths import resolved_artifact_path  # noqa: E402
from layer8_ui.preview_media import (  # noqa: E402
    THERMAL_JPEG_WAITING,
    WEBCAM_JPEG_WAITING,
    mjpeg_placeholder_jpeg,
)

from api.streaming.webrtc_frames import preview_frame_interval_s

if TYPE_CHECKING:
    from api.routes.context import RouterContext

# Drop a frame rather than block the event loop / congest the write buffer.
_WS_SEND_TIMEOUT_S = 0.35
_DEFAULT_INTERVAL_S = 1.0 / 30.0


async def _send_jpeg(websocket: WebSocket, payload: bytes) -> bool:
    """Send one JPEG; return False if the socket is dead."""
    try:
        await asyncio.wait_for(websocket.send_bytes(payload), timeout=_WS_SEND_TIMEOUT_S)
        return True
    except asyncio.TimeoutError:
        # Client/backpressure: skip this frame, keep the connection.
        return True
    except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError, AssertionError):
        return False
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "close message" in msg or "not connected" in msg:
            return False
        raise


async def _jpeg_push_loop(
    websocket: WebSocket,
    *,
    next_frame: Callable[[], Awaitable[bytes | None]],
    frame_interval_s: Callable[[], float],
    initial_placeholder: bytes | None = None,
) -> None:
    """Push binary JPEGs, holding the last good frame through brief IPC gaps."""
    await websocket.accept()
    last_sent: bytes | None = None
    last_good: bytes | None = None
    try:
        while True:
            interval = max(1.0 / 60.0, float(frame_interval_s() or _DEFAULT_INTERVAL_S))
            frame = await next_frame()
            if frame:
                last_good = frame
                payload = frame
            elif last_good is not None:
                payload = last_good
            elif initial_placeholder is not None:
                payload = initial_placeholder
            else:
                await asyncio.sleep(interval)
                continue
            if payload is not last_sent and payload != last_sent:
                if not await _send_jpeg(websocket, payload):
                    break
                last_sent = payload
            await asyncio.sleep(interval)
    except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError, AssertionError):
        pass


async def websocket_thermal(websocket: WebSocket, ctx: RouterContext) -> None:
    """Binary JPEG stream; IPC overlay when infer runner is on."""
    layer8_dir = ctx.layer8_dir
    fs = ctx.frame_sources
    held = False

    async def next_frame() -> bytes | None:
        nonlocal held
        s = ctx.settings.get()
        runner_on = bool(sensor_runner.status("thermal", layer8_dir).get("running"))
        rel = (s.get("thermal") or {}).get("live_frame") or ""
        rpath = resolved_artifact_path(s, relative_to_software=str(rel), layer8_dir=layer8_dir)
        if runner_on:
            if held:
                ctx.thermal_stream.remove_client()
                held = False
            return await asyncio.to_thread(
                lambda: fs.runner_frame_jpeg_thermal(rpath, allow_disk_fallback=False),
            )
        if fs.thermal_uses_v4l2_preview(s):
            if not held:
                ctx.thermal_stream.add_client(s)
                held = True
            else:
                ctx.thermal_stream.sync_settings(s)
            return ctx.thermal_stream.latest_jpg() or THERMAL_JPEG_WAITING
        return fs.thermal_disk_jpeg(s, layer8_dir) or THERMAL_JPEG_WAITING

    try:
        await _jpeg_push_loop(
            websocket,
            next_frame=next_frame,
            frame_interval_s=lambda: preview_frame_interval_s(ctx.settings.get().get("thermal") or {}),
            initial_placeholder=fs.thermal_runner_ipc_placeholder(layer8_dir),
        )
    finally:
        if held:
            ctx.thermal_stream.remove_client()


async def websocket_webcam(websocket: WebSocket, ctx: RouterContext) -> None:
    """Binary JPEG stream; fan-out from one shared V4L2 reader (parallel with ``/ws/thermal``)."""
    layer8_dir = ctx.layer8_dir
    held = False
    waiting = mjpeg_placeholder_jpeg(
        "Webcam runner is ON (subprocess holds the camera).",
        "Waiting for BGR/JPEG IPC (same path as WebRTC fan-out).",
        "configure webcam.live_frame",
    )

    async def next_frame() -> bytes | None:
        nonlocal held
        s = ctx.settings.get()
        runner_on = bool(sensor_runner.status("webcam", layer8_dir).get("running"))
        if runner_on:
            if held:
                ctx.webcam_stream.remove_client()
                held = False
            rel = (s.get("webcam") or {}).get("live_frame") or ""
            rpath = resolved_artifact_path(s, relative_to_software=str(rel), layer8_dir=layer8_dir)
            return await asyncio.to_thread(
                lambda: ctx.frame_sources.runner_frame_jpeg_webcam(
                    rpath, allow_disk_fallback=False
                ),
            )
        if not held:
            ctx.webcam_stream.add_client(s)
            held = True
        else:
            ctx.webcam_stream.sync_settings(s)
        return ctx.webcam_stream.latest_jpg() or WEBCAM_JPEG_WAITING

    try:
        await _jpeg_push_loop(
            websocket,
            next_frame=next_frame,
            frame_interval_s=lambda: preview_frame_interval_s(ctx.settings.get().get("webcam") or {}),
            initial_placeholder=waiting,
        )
    finally:
        if held:
            ctx.webcam_stream.remove_client()


async def websocket_multi_camera(websocket: WebSocket, ctx: RouterContext) -> None:
    """Binary JPEG stream for the Multi_Camera panel."""
    layer8_dir = ctx.layer8_dir
    held = False
    waiting = mjpeg_placeholder_jpeg(
        "Multi-camera runner is ON (subprocess holds the camera).",
        "Waiting for BGR/JPEG IPC (same path as WebRTC fan-out).",
        "configure multi_camera.live_frame",
    )

    async def next_frame() -> bytes | None:
        nonlocal held
        s = ctx.settings.get()
        runner_on = bool(sensor_runner.status("multi_camera", layer8_dir).get("running"))
        if runner_on:
            if held:
                ctx.multi_camera_stream.remove_client()
                held = False
            rel = (s.get("multi_camera") or {}).get("live_frame") or ""
            rpath = resolved_artifact_path(s, relative_to_software=str(rel), layer8_dir=layer8_dir)
            return await asyncio.to_thread(
                lambda: ctx.frame_sources.runner_frame_jpeg_multi_camera(
                    rpath, allow_disk_fallback=False
                ),
            )
        if not held:
            ctx.multi_camera_stream.add_client(s)
            held = True
        else:
            ctx.multi_camera_stream.sync_settings(s)
        return ctx.multi_camera_stream.latest_jpg() or WEBCAM_JPEG_WAITING

    try:
        await _jpeg_push_loop(
            websocket,
            next_frame=next_frame,
            frame_interval_s=lambda: preview_frame_interval_s(
                ctx.settings.get().get("multi_camera") or {}
            ),
            initial_placeholder=waiting,
        )
    finally:
        if held:
            ctx.multi_camera_stream.remove_client()
