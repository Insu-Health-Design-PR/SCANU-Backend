"""Camera device helpers for stable Layer 8 capture."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2


def device_from_env_or_settings(env_name: str, settings_value: Any, default: int | str = 0) -> str:
    """Return a camera source from env first, then settings, preserving path values."""
    env_value = os.environ.get(env_name)
    raw = env_value if env_value is not None and env_value.strip() != "" else settings_value
    if raw is None or str(raw).strip() == "":
        raw = default
    return str(raw).strip()


def is_numeric_device(source: Any) -> bool:
    val = str(source).strip()
    return val.isdigit() or (val.startswith("-") and val[1:].isdigit())


def open_v4l2_capture(source: Any) -> cv2.VideoCapture:
    """Open either a numeric V4L2 index or a stable /dev/v4l/by-* path."""
    src = str(source).strip()
    if is_numeric_device(src):
        return cv2.VideoCapture(int(src), cv2.CAP_V4L2)

    expanded = str(Path(src).expanduser())
    cap = cv2.VideoCapture(expanded, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(expanded)
    return cap


def display_device(source: Any) -> str:
    src = str(source).strip()
    return f"/dev/video{src}" if is_numeric_device(src) else src
