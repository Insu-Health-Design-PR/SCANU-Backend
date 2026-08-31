"""JPEG encode and validation helpers."""

from __future__ import annotations

import cv2
import numpy as np

# Preview MJPEG: cap the *long* side. Portrait 1080×1920 ignored a max-width=1280
# cap and the browser then sat around 10–15 fps decoding two full-height JPEGs.
DEFAULT_PREVIEW_JPEG_QUALITY = 62
DEFAULT_PREVIEW_JPEG_MAX_WIDTH = 960
DEFAULT_PREVIEW_JPEG_MAX_SIDE = 960


def is_valid_jpeg(payload: bytes | None) -> bool:
    """Cheap sanity check to avoid decoding torn mmap/disk writes."""
    if not payload or len(payload) < 4:
        return False
    if payload[0] != 0xFF or payload[1] != 0xD8:
        return False
    if payload[-2] != 0xFF or payload[-1] != 0xD9:
        return False
    return True


def _even(n: int) -> int:
    n = max(2, int(n))
    return n if n % 2 == 0 else n - 1


def _downscale_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    if frame is None or frame.size == 0 or max_side <= 0:
        return frame
    h, w = int(frame.shape[0]), int(frame.shape[1])
    long_side = max(h, w)
    if long_side <= max_side:
        return frame
    scale = float(max_side) / float(long_side)
    nw = _even(max(2, int(round(w * scale))))
    nh = _even(max(2, int(round(h * scale))))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame, quality: int = 85) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    payload = buf.tobytes()
    return payload if is_valid_jpeg(payload) else None


def encode_preview_jpeg(
    frame,
    *,
    quality: int = DEFAULT_PREVIEW_JPEG_QUALITY,
    max_width: int = DEFAULT_PREVIEW_JPEG_MAX_WIDTH,
    max_side: int | None = None,
) -> bytes | None:
    """Fast UI JPEG: shrink the long side then encode. Do not use for archival stills."""
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.size == 0:
        return None
    cap = int(max_side) if max_side is not None else int(max_width)
    small = _downscale_max_side(arr, cap)
    return encode_jpeg(small, quality=int(quality))
