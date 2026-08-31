"""Frame pacing and sizing helpers for WebRTC video tracks."""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any

import cv2
import numpy as np

CLOCK_HZ = 90000
DEFAULT_WEBRTC_FPS = 24.0
MIN_WEBRTC_FPS = 5.0
# Allow 60fps profiles (smooth display / WebRTC). Infer can still run slower via stride.
MAX_WEBRTC_FPS = 60.0
DEFAULT_WEBRTC_MAX_WIDTH = 1280
MIN_WEBRTC_MAX_WIDTH = 320
MAX_WEBRTC_MAX_WIDTH = 3840


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def webrtc_fps(webcam_settings: dict[str, Any]) -> float:
    """Prefer explicit WebRTC FPS, then camera FPS, while keeping encoder load bounded."""
    raw = (
        webcam_settings.get("webrtc_fps")
        or webcam_settings.get("webrtc_capture_fps")
        or webcam_settings.get("fps")
    )
    return max(MIN_WEBRTC_FPS, min(MAX_WEBRTC_FPS, _number(raw, DEFAULT_WEBRTC_FPS)))


def webrtc_max_width(webcam_settings: dict[str, Any]) -> int:
    raw = webcam_settings.get("webrtc_max_width") or webcam_settings.get("webcam_webrtc_max_width")
    value = int(_number(raw, DEFAULT_WEBRTC_MAX_WIDTH))
    return max(MIN_WEBRTC_MAX_WIDTH, min(MAX_WEBRTC_MAX_WIDTH, value))


def webrtc_smooth_display(webcam_settings: dict[str, Any]) -> bool:
    """When enabled, WebRTC display FPS is decoupled from infer FPS (hold last good frame)."""
    raw = webcam_settings.get("webrtc_smooth_display")
    if raw is None:
        raw = webcam_settings.get("webcam_webrtc_smooth_display")
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(int(_number(raw, 0)))


def webrtc_ipc_poll_fps(webcam_settings: dict[str, Any]) -> float:
    """IPC poll rate while smooth display is enabled (independent of infer FPS)."""
    raw = (
        webcam_settings.get("webrtc_ipc_poll_fps")
        or webcam_settings.get("webcam_webrtc_ipc_poll_fps")
        or webcam_settings.get("fps")
    )
    return max(MIN_WEBRTC_FPS, min(MAX_WEBRTC_FPS, _number(raw, MAX_WEBRTC_FPS)))


def preview_frame_interval_s(sensor_settings: dict[str, Any] | None, *, default_fps: float = 30.0) -> float:
    """Sleep between MJPEG/WebSocket preview frames from webrtc_fps / fps settings."""
    fps = webrtc_fps(sensor_settings or {}) if sensor_settings else default_fps
    return 1.0 / max(MIN_WEBRTC_FPS, min(MAX_WEBRTC_FPS, float(fps)))


def next_frame_delay(next_t: float, fps: float) -> tuple[float, float]:
    now = time.monotonic()
    delay = max(0.0, next_t - now)
    return delay, max(next_t + (1.0 / fps), now)


def make_black_bgr(max_width: int) -> np.ndarray:
    width = min(DEFAULT_WEBRTC_MAX_WIDTH, int(max_width))
    height = max(1, int(width * 9 / 16))
    return np.zeros((height, width, 3), dtype=np.uint8)


def prepare_bgr_for_webrtc(bgr: np.ndarray | None, *, max_width: int) -> np.ndarray:
    """Return a contiguous BGR frame sized for realtime WebRTC encoding."""
    if bgr is None:
        return make_black_bgr(max_width)
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return make_black_bgr(max_width)
    height, width = bgr.shape[:2]
    if width > max_width:
        scale = float(max_width) / float(width)
        new_size = (int(max_width), max(1, int(height * scale)))
        bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)
    if bgr.dtype != np.uint8:
        bgr = bgr.astype(np.uint8, copy=False)
    return np.ascontiguousarray(bgr)


def frame_time_base() -> Fraction:
    return Fraction(1, CLOCK_HZ)


def frame_pts_step(fps: float) -> int:
    return max(1, int(CLOCK_HZ / fps))
