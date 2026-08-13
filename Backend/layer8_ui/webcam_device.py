"""Webcam device auto-detection helpers."""

from __future__ import annotations

import os
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2


def video_nodes_present() -> list[str]:
    """Return existing ``/dev/videoN`` paths (sorted)."""
    out: list[str] = []
    for i in range(0, 64):
        p = f"/dev/video{i}"
        if os.path.exists(p):
            out.append(p)
    return out


def _probe_device(index: int, width: int, height: int, fps: float = 30.0) -> bool:
    cap = cv2.VideoCapture(int(index), cv2.CAP_V4L2)
    if not cap.isOpened():
        return False
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        return bool(ok and frame is not None)
    finally:
        cap.release()


def detect_working_webcam_device(
    *,
    preferred: int,
    width: int,
    height: int,
    search_max_index: int = 8,
    fps: float = 30.0,
) -> int | None:
    """
    Return first working V4L2 index for webcam capture.

    Strategy:
    1) Try preferred index first.
    2) Fallback scan from 0..search_max_index excluding preferred.
    """
    if preferred >= 0 and _probe_device(preferred, width, height, fps=float(fps)):
        return preferred

    for idx in range(0, int(search_max_index) + 1):
        if idx == preferred:
            continue
        if _probe_device(idx, width, height, fps=float(fps)):
            return idx
    return None


def wait_for_webcam_device(
    *,
    preferred: int,
    width: int,
    height: int,
    search_max_index: int = 8,
    fps: float = 30.0,
    timeout_s: float = 20.0,
    poll_s: float = 0.5,
) -> int | None:
    """
    Poll until a webcam node appears and can deliver a frame, or timeout.

    Helps with USB cams (e.g. NexiGo N950P) that enumerate a few seconds after plug-in.
    """
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        nodes = video_nodes_present()
        if nodes:
            found = detect_working_webcam_device(
                preferred=preferred,
                width=width,
                height=height,
                search_max_index=search_max_index,
                fps=fps,
            )
            if found is not None:
                return found
            # Node exists but cannot read yet (driver still settling) — keep waiting.
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.1, float(poll_s)))


def camera_missing_hint() -> str:
    """Short operator hint when no V4L2 webcam is available."""
    try:
        import subprocess

        p = subprocess.run(
            ["lsusb"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        text = (p.stdout or "") + (p.stderr or "")
        if "3443:950a" in text or "NexiGo" in text:
            return (
                "NexiGo N950P is on USB but has no /dev/video node yet (or is reconnecting). "
                "Use a rear USB3 port / short cable; wait until `ls /dev/video*` is stable."
            )
    except Exception:
        pass
    return (
        "No /dev/video* on this machine. Plug in the webcam, confirm with "
        "`ls /dev/video*` and `v4l2-ctl --list-devices`, then retry."
    )
