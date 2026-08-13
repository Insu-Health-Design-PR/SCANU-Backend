"""JPEG encode and validation helpers."""

from __future__ import annotations

import cv2


def is_valid_jpeg(payload: bytes | None) -> bool:
    """Cheap sanity check to avoid decoding torn mmap/disk writes."""
    if not payload or len(payload) < 4:
        return False
    if payload[0] != 0xFF or payload[1] != 0xD8:
        return False
    if payload[-2] != 0xFF or payload[-1] != 0xD9:
        return False
    return True


def encode_jpeg(frame, quality: int = 85) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    payload = buf.tobytes()
    return payload if is_valid_jpeg(payload) else None
