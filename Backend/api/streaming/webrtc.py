"""WebRTC offer/answer and track setup."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import HTTPException

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from runtime import sensor_runner  # noqa: E402
from layer8_ui.webcam_ipc_fanout import get_webcam_ipc_fanout  # noqa: E402
from layer8_ui.multi_camera_ipc_fanout import get_multi_camera_ipc_fanout  # noqa: E402
from media.encode.jpeg import is_valid_jpeg

from api.schemas.webrtc import WebRTCOfferBody
from api.streaming.webrtc_frames import (
    frame_pts_step,
    frame_time_base,
    next_frame_delay,
    prepare_bgr_for_webrtc,
    webrtc_fps,
    webrtc_ipc_poll_fps,
    webrtc_max_width,
    webrtc_smooth_display,
)

if TYPE_CHECKING:
    from api.routes.context import RouterContext


def patch_aiortc_h264_encoder_for_jetson() -> str:
    """
    Prefer Jetson hardware H264 encoder when aiortc encodes outbound video.

    aiortc 1.9 uses h264_omx -> libx264 by default. On Jetson, ffmpeg's
    h264_v4l2m2m can reduce CPU usage significantly.
    """
    # NOTE:
    # We temporarily disable custom encoder monkey-patching here because probing
    # h264_v4l2m2m in this runtime has shown instability (Bus error/core dump)
    # while still failing to acquire a valid device. Keep aiortc defaults for
    # stability; revisit Jetson hardware encode via a dedicated GStreamer path.
    return "disabled_for_stability"


def _jpeg_to_bgr(jpg: bytes | None) -> np.ndarray | None:
    import cv2

    if not is_valid_jpeg(jpg):
        return None
    arr = np.frombuffer(jpg, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _prefer_h264_for_sender(pc: Any) -> None:
    """Prefer H.264 on outbound video when aiortc/browser both support it."""
    try:
        from aiortc import RTCRtpSender
    except Exception:
        return
    caps = RTCRtpSender.getCapabilities("video")
    h264 = [c for c in caps.codecs if (c.mimeType or "").lower() == "video/h264"]
    if not h264:
        return
    for transceiver in pc.getTransceivers():
        if getattr(transceiver, "kind", None) != "video":
            continue
        try:
            transceiver.setCodecPreferences(h264)
        except Exception:
            pass


async def _wait_for_ice_gathering(pc: Any, timeout_s: float = 1.0) -> None:
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    async def _on_ice_gathering_state_change() -> None:
        if pc.iceGatheringState == "complete":
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return


async def handle_webcam_offer(ctx: RouterContext, body: WebRTCOfferBody) -> dict[str, str]:
    if body.type != "offer" or not body.sdp.strip():
        raise HTTPException(status_code=400, detail="WebRTC request must contain an SDP offer")
    try:
        from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
        from av import VideoFrame
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"WebRTC backend unavailable; install aiortc+av in runtime env ({exc})",
        ) from exc
    patch_aiortc_h264_encoder_for_jetson()

    layer8_dir = ctx.layer8_dir
    _wcam = ctx.settings.get().get("webcam") or {}
    _webrtc_capture_fps = webrtc_fps(_wcam)
    _webrtc_max_width = webrtc_max_width(_wcam)
    _webrtc_smooth = webrtc_smooth_display(_wcam)
    _webrtc_ipc_poll_fps = webrtc_ipc_poll_fps(_wcam)
    _wcam_st = sensor_runner.status("webcam", layer8_dir)
    runner_on = bool(_wcam_st.get("running"))
    infer_runner = runner_on and not bool(_wcam_st.get("preview_only"))
    fanout = get_webcam_ipc_fanout()

    class WebcamDirectTrack(VideoStreamTrack):
        """Fallback when infer subprocess is off: shared V4L2 reader."""

        def __init__(self, fps: float) -> None:
            super().__init__()
            self._fps = float(fps)
            self._max_width = int(_webrtc_max_width)
            self._smooth = bool(_webrtc_smooth)
            self._next_t = 0.0
            self._pts = 0
            self._pts_step = frame_pts_step(self._fps)
            self._display_bgr: np.ndarray | None = None
            self._last_jpg_sig: tuple[int, bytes] | None = None

        @staticmethod
        def _jpg_signature(jpg: bytes | None) -> tuple[int, bytes] | None:
            if not jpg:
                return None
            return (len(jpg), jpg[:32])

        async def recv(self) -> Any:
            if self._next_t <= 0:
                import time

                self._next_t = time.monotonic()
            delay, self._next_t = next_frame_delay(self._next_t, self._fps)
            if delay > 0:
                await asyncio.sleep(delay)
            jpg = ctx.webcam_stream.latest_jpg()
            sig = self._jpg_signature(jpg)
            if jpg and (not self._smooth or sig != self._last_jpg_sig):
                bgr = _jpeg_to_bgr(jpg)
                if bgr is not None:
                    self._display_bgr = prepare_bgr_for_webrtc(bgr, max_width=self._max_width)
                    self._last_jpg_sig = sig
            if self._smooth and self._display_bgr is not None:
                bgr = self._display_bgr
            else:
                bgr = prepare_bgr_for_webrtc(
                    _jpeg_to_bgr(jpg) if jpg else None,
                    max_width=self._max_width,
                )
            frame = VideoFrame.from_ndarray(bgr, format="bgr24")
            frame.pts = self._pts
            frame.time_base = frame_time_base()
            self._pts += self._pts_step
            return frame

    pc = RTCPeerConnection()
    ctx.webrtc_peers.add(pc)
    hold_shared_webcam_preview: list[bool] = [False]
    used_fanout = False

    def _release_shared_webcam_preview() -> None:
        if not hold_shared_webcam_preview[0]:
            return
        ctx.webcam_stream.remove_client()
        hold_shared_webcam_preview[0] = False

    @pc.on("connectionstatechange")
    async def _on_state_change() -> None:
        if pc.connectionState in ("failed", "closed", "disconnected"):
            _release_shared_webcam_preview()
            if used_fanout:
                await fanout.unregister_peer()
            ctx.webrtc_peers.discard(pc)
            await pc.close()

    try:
        if infer_runner:
            track = await fanout.register_peer(
                _webrtc_capture_fps,
                max_width=_webrtc_max_width,
                smooth_display=_webrtc_smooth,
                ipc_poll_fps=_webrtc_ipc_poll_fps,
            )
            used_fanout = True
        else:
            preview_only = bool(_wcam_st.get("preview_only"))
            if not preview_only:
                ctx.webcam_stream.add_client(ctx.settings.get())
                hold_shared_webcam_preview[0] = True
            track = WebcamDirectTrack(_webrtc_capture_fps)
        pc.addTrack(track)
        _prefer_h264_for_sender(pc)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_for_ice_gathering(pc)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    except Exception:
        _release_shared_webcam_preview()
        if used_fanout:
            await fanout.unregister_peer()
        ctx.webrtc_peers.discard(pc)
        await pc.close()
        raise


async def handle_multi_camera_offer(ctx: RouterContext, body: WebRTCOfferBody) -> dict[str, str]:
    if body.type != "offer" or not body.sdp.strip():
        raise HTTPException(status_code=400, detail="WebRTC request must contain an SDP offer")
    try:
        from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
        from av import VideoFrame
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"WebRTC backend unavailable; install aiortc+av in runtime env ({exc})",
        ) from exc
    patch_aiortc_h264_encoder_for_jetson()

    layer8_dir = ctx.layer8_dir
    _wcam = ctx.settings.get().get("multi_camera") or {}
    _webrtc_capture_fps = webrtc_fps(_wcam)
    _webrtc_max_width = webrtc_max_width(_wcam)
    _webrtc_smooth = webrtc_smooth_display(_wcam)
    _webrtc_ipc_poll_fps = webrtc_ipc_poll_fps(_wcam)
    runner_on = bool(sensor_runner.status("multi_camera", layer8_dir).get("running"))
    fanout = get_multi_camera_ipc_fanout()

    class MultiCameraDirectTrack(VideoStreamTrack):
        """Fallback when infer subprocess is off: shared V4L2 reader."""

        def __init__(self, fps: float) -> None:
            super().__init__()
            self._fps = float(fps)
            self._max_width = int(_webrtc_max_width)
            self._smooth = bool(_webrtc_smooth)
            self._next_t = 0.0
            self._pts = 0
            self._pts_step = frame_pts_step(self._fps)
            self._display_bgr: np.ndarray | None = None
            self._last_jpg_sig: tuple[int, bytes] | None = None

        @staticmethod
        def _jpg_signature(jpg: bytes | None) -> tuple[int, bytes] | None:
            if not jpg:
                return None
            return (len(jpg), jpg[:32])

        async def recv(self) -> Any:
            if self._next_t <= 0:
                import time

                self._next_t = time.monotonic()
            delay, self._next_t = next_frame_delay(self._next_t, self._fps)
            if delay > 0:
                await asyncio.sleep(delay)
            jpg = ctx.multi_camera_stream.latest_jpg()
            sig = self._jpg_signature(jpg)
            if jpg and (not self._smooth or sig != self._last_jpg_sig):
                bgr = _jpeg_to_bgr(jpg)
                if bgr is not None:
                    self._display_bgr = prepare_bgr_for_webrtc(bgr, max_width=self._max_width)
                    self._last_jpg_sig = sig
            if self._smooth and self._display_bgr is not None:
                bgr = self._display_bgr
            else:
                bgr = prepare_bgr_for_webrtc(
                    _jpeg_to_bgr(jpg) if jpg else None,
                    max_width=self._max_width,
                )
            frame = VideoFrame.from_ndarray(bgr, format="bgr24")
            frame.pts = self._pts
            frame.time_base = frame_time_base()
            self._pts += self._pts_step
            return frame

    pc = RTCPeerConnection()
    ctx.webrtc_peers.add(pc)
    hold_shared_multi_camera_preview: list[bool] = [False]
    used_fanout = False

    def _release_shared_multi_camera_preview() -> None:
        if not hold_shared_multi_camera_preview[0]:
            return
        ctx.multi_camera_stream.remove_client()
        hold_shared_multi_camera_preview[0] = False

    @pc.on("connectionstatechange")
    async def _on_state_change() -> None:
        if pc.connectionState in ("failed", "closed", "disconnected"):
            _release_shared_multi_camera_preview()
            if used_fanout:
                await fanout.unregister_peer()
            ctx.webrtc_peers.discard(pc)
            await pc.close()

    try:
        if runner_on:
            track = await fanout.register_peer(
                _webrtc_capture_fps,
                max_width=_webrtc_max_width,
                smooth_display=_webrtc_smooth,
                ipc_poll_fps=_webrtc_ipc_poll_fps,
            )
            used_fanout = True
        else:
            ctx.multi_camera_stream.add_client(ctx.settings.get())
            hold_shared_multi_camera_preview[0] = True
            track = MultiCameraDirectTrack(_webrtc_capture_fps)
        pc.addTrack(track)
        _prefer_h264_for_sender(pc)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_for_ice_gathering(pc)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    except Exception:
        _release_shared_multi_camera_preview()
        if used_fanout:
            await fanout.unregister_peer()
        ctx.webrtc_peers.discard(pc)
        await pc.close()
        raise
