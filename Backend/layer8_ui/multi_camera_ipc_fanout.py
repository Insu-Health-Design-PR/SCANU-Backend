"""
Shared webcam IPC → WebRTC fan-out.

One asyncio pump reads BGR/JPEG IPC from the infer subprocess; many WebRTC peers
subscribe without each opening their own reader or racing on mmap.

Set ``webrtc_smooth_display: 0`` in multi_camera settings to revert to the legacy
synced pump/display behavior.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from api.streaming.webrtc_frames import (
    frame_pts_step,
    frame_time_base,
    next_frame_delay,
    prepare_bgr_for_webrtc,
)
from media.encode.jpeg import is_valid_jpeg
from media.ipc import LiveBgrFrameReader, LiveFrameReader


class MultiCameraIpcFanout:
    """Single producer (IPC pump) → many WebRTC ``VideoStreamTrack`` consumers."""

    def __init__(
        self,
        *,
        bgr_path: Path,
        jpg_path: Path,
    ) -> None:
        self._bgr_reader = LiveBgrFrameReader(bgr_path)
        self._jpg_reader = LiveFrameReader(jpg_path)
        self._lock = threading.Lock()
        self._latest_bgr: np.ndarray | None = None
        self._display_bgr: np.ndarray | None = None
        self._display_max_width = 0
        self._last_ipc_seq: int | None = None
        self._smooth_mode = False
        self._peers = 0
        self._pump_task: asyncio.Task[None] | None = None
        self._pump_fps = 30.0
        self._stop = asyncio.Event()

    def peer_count(self) -> int:
        with self._lock:
            return int(self._peers)

    def _bgr_from_payload(self, raw: bytes, height: int, width: int, channels: int) -> np.ndarray | None:
        expected = int(height) * int(width) * int(channels)
        if channels != 3 or len(raw) != expected:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        return arr.reshape((int(height), int(width), 3)).copy()

    def _read_ipc_bgr(self) -> np.ndarray | None:
        bgr, _seq = self._read_ipc_bgr_with_seq()
        return bgr

    def _read_ipc_bgr_with_seq(self) -> tuple[np.ndarray | None, int | None]:
        payload = self._bgr_reader.read_latest_with_seq()
        if payload is not None:
            (raw, height, width, channels), seq = payload
            bgr = self._bgr_from_payload(raw, height, width, channels)
            if bgr is not None:
                return bgr, seq
        jpg_payload = self._jpg_reader.read_latest_with_seq()
        if jpg_payload is not None:
            jpg, seq = jpg_payload
            if jpg and is_valid_jpeg(jpg):
                arr = np.frombuffer(jpg, dtype=np.uint8)
                if arr.size:
                    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if decoded is not None:
                        return decoded, seq
        return None, None

    def _update_display_cache_locked(self, bgr: np.ndarray) -> None:
        self._latest_bgr = bgr
        if self._display_max_width > 0:
            self._display_bgr = prepare_bgr_for_webrtc(bgr, max_width=self._display_max_width)

    async def _pump_loop(self) -> None:
        interval = 1.0 / max(5.0, min(60.0, float(self._pump_fps)))
        while not self._stop.is_set():
            if self._smooth_mode:
                bgr, seq = await asyncio.to_thread(self._read_ipc_bgr_with_seq)
                if bgr is not None and seq is not None and seq != self._last_ipc_seq:
                    self._last_ipc_seq = seq
                    with self._lock:
                        self._update_display_cache_locked(bgr)
            else:
                bgr, seq = await asyncio.to_thread(self._read_ipc_bgr_with_seq)
                if bgr is not None and seq is not None and seq != self._last_ipc_seq:
                    self._last_ipc_seq = seq
                    with self._lock:
                        self._latest_bgr = bgr
            await asyncio.sleep(interval)

    async def register_peer(
        self,
        fps: float,
        *,
        max_width: int,
        smooth_display: bool = False,
        ipc_poll_fps: float | None = None,
    ) -> Any:
        """Return an aiortc ``VideoStreamTrack`` for one viewer."""
        from aiortc import VideoStreamTrack
        from av import VideoFrame

        if smooth_display:
            self._smooth_mode = True
            poll_fps = float(ipc_poll_fps or fps)
            self._pump_fps = max(self._pump_fps, poll_fps, float(fps))
        else:
            self._pump_fps = max(self._pump_fps, float(fps))
        with self._lock:
            self._peers += 1
            self._display_max_width = max(self._display_max_width, int(max_width))
            if self._smooth_mode and self._latest_bgr is not None and self._display_bgr is None:
                self._display_bgr = prepare_bgr_for_webrtc(
                    self._latest_bgr,
                    max_width=self._display_max_width,
                )
            if self._pump_task is None or self._pump_task.done():
                self._stop = asyncio.Event()
                self._pump_task = asyncio.create_task(self._pump_loop())

        parent = self

        class _FanoutTrack(VideoStreamTrack):
            def __init__(self) -> None:
                super().__init__()
                self._fps = float(fps)
                self._max_width = int(max_width)
                self._smooth = bool(smooth_display)
                self._next_t = time.monotonic()
                self._pts = 0
                self._pts_step = frame_pts_step(self._fps)

            async def recv(self) -> Any:
                delay, self._next_t = next_frame_delay(self._next_t, self._fps)
                if delay > 0:
                    await asyncio.sleep(delay)
                with parent._lock:
                    src: np.ndarray | None = None
                    if self._smooth and parent._display_bgr is not None:
                        if self._max_width == parent._display_max_width:
                            src = parent._display_bgr
                        elif parent._latest_bgr is not None:
                            src = prepare_bgr_for_webrtc(
                                parent._latest_bgr, max_width=self._max_width
                            )
                    elif parent._latest_bgr is not None:
                        src = prepare_bgr_for_webrtc(
                            parent._latest_bgr, max_width=self._max_width
                        )
                    elif parent._display_bgr is not None:
                        src = parent._display_bgr
                    bgr = None if src is None else np.ascontiguousarray(src)
                if bgr is None:
                    bgr = prepare_bgr_for_webrtc(None, max_width=self._max_width)
                frame = VideoFrame.from_ndarray(bgr, format="bgr24")
                frame.pts = self._pts
                frame.time_base = frame_time_base()
                self._pts += self._pts_step
                return frame

        return _FanoutTrack()

    async def unregister_peer(self) -> None:
        with self._lock:
            self._peers = max(0, self._peers - 1)
            if self._peers > 0:
                return
        self._stop.set()
        if self._pump_task is not None:
            try:
                await asyncio.wait_for(self._pump_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._pump_task.cancel()
            self._pump_task = None
        with self._lock:
            self._latest_bgr = None
            self._display_bgr = None
            self._display_max_width = 0
            self._last_ipc_seq = None
            self._smooth_mode = False

    def latest_bgr(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_bgr is None:
                return None
            return self._latest_bgr.copy()

    def refresh_from_ipc_sync(self) -> np.ndarray | None:
        bgr = self._read_ipc_bgr()
        if bgr is not None:
            with self._lock:
                self._latest_bgr = bgr
                if self._smooth_mode and self._display_max_width > 0:
                    self._display_bgr = prepare_bgr_for_webrtc(
                        bgr,
                        max_width=self._display_max_width,
                    )
        return bgr


_MultiCameraFanoutSingleton: MultiCameraIpcFanout | None = None
_FanoutLock = threading.Lock()


def get_multi_camera_ipc_fanout(
    *,
    bgr_path: Path = Path("/dev/shm/scanu_multi_camera_live_bgr_frame.bin"),
    jpg_path: Path = Path("/dev/shm/scanu_multi_camera_live_frame.bin"),
) -> MultiCameraIpcFanout:
    global _MultiCameraFanoutSingleton
    with _FanoutLock:
        if _MultiCameraFanoutSingleton is None:
            _MultiCameraFanoutSingleton = MultiCameraIpcFanout(bgr_path=bgr_path, jpg_path=jpg_path)
        return _MultiCameraFanoutSingleton
