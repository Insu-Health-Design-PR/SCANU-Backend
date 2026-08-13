"""Multipart MJPEG stream helpers."""

from __future__ import annotations


def mjpeg_chunk(jpg: bytes, boundary: str = "frame") -> bytes:
    return (
        f"--{boundary}\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(jpg)}\r\n\r\n"
    ).encode("utf-8") + jpg + b"\r\n"


def mjpeg_media_type(boundary: str = "frame") -> str:
    return f"multipart/x-mixed-replace; boundary={boundary}"
