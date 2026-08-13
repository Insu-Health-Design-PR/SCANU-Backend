"""JPEG placeholders and MJPEG response headers for live previews."""

from __future__ import annotations

import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

_MJPEG_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0, no-transform",
    "Pragma": "no-cache",
    "Expires": "0",
    # Allow chunked multipart streams through reverse proxies (nginx, etc.).
    "X-Accel-Buffering": "no",
}


def mjpeg_headers() -> dict[str, str]:
    return dict(_MJPEG_HEADERS)


def mjpeg_placeholder_jpeg(*lines: str) -> bytes:
    """Valid JPEG bytes so MJPEG clients never hang on an empty multipart stream."""
    h, w = 360, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (28, 32, 40)
    y = 80
    for raw in lines:
        for part in (raw[i : i + 72] for i in range(0, len(raw), 72)):
            cv2.putText(
                img,
                part,
                (24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (190, 198, 210),
                1,
                cv2.LINE_AA,
            )
            y += 28
            if y > h - 40:
                break
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return buf.tobytes() if ok else b""


THERMAL_JPEG_WAITING = mjpeg_placeholder_jpeg(
    "Thermal direct: waiting for camera…",
    "If this persists: wrong /dev/video index, device busy,",
    "or another app is using the thermal node.",
)
WEBCAM_JPEG_WAITING = mjpeg_placeholder_jpeg(
    "Webcam direct: waiting for camera…",
    "If this persists: wrong webcam_device, device busy,",
    "or the thermal runner is using the same /dev/video node.",
)
