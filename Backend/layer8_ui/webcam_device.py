"""Webcam / HDMI-UVC device helpers.

HDMI USB capture dongles expose a *capture* node and a *metadata* node per
physical camera. Opening the metadata node fails; OpenCV often cannot probe
H.264 capture either. Dual local runs must therefore:

1. Prefer V4L2 Video Capture nodes only.
2. Never steal the peer camera's index.
3. Accept an existing capture node without an OpenCV frame probe.
"""

from __future__ import annotations

import fcntl
import os
import struct
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2

_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_DEVICE_CAPS = 0x80000000
_VIDIOC_QUERYCAP = 0x80685600


def video_nodes_present() -> list[str]:
    """Return existing ``/dev/videoN`` paths (sorted)."""
    out: list[str] = []
    for i in range(0, 64):
        p = f"/dev/video{i}"
        if os.path.exists(p):
            out.append(p)
    return out


def v4l2_device_caps(path: str) -> int | None:
    """Return V4L2 device_caps (or capabilities) for ``path``, or None."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    buf = bytearray(104)
    try:
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
    except OSError:
        return None
    finally:
        os.close(fd)
    caps, device_caps = struct.unpack_from("<II", buf, 84)
    if caps & _V4L2_CAP_DEVICE_CAPS:
        return int(device_caps)
    return int(caps)


def is_video_capture_node(index: int) -> bool:
    """True if ``/dev/videoN`` is a capture node (not UVC metadata)."""
    path = f"/dev/video{int(index)}"
    if not os.path.exists(path):
        return False
    caps = v4l2_device_caps(path)
    return bool(caps is not None and (caps & _V4L2_CAP_VIDEO_CAPTURE))


def list_capture_indices(max_index: int = 63) -> list[int]:
    """Sorted indices of Video Capture nodes only."""
    out: list[int] = []
    for i in range(0, int(max_index) + 1):
        if is_video_capture_node(i):
            out.append(i)
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
        try:
            ok, frame = cap.read()
        except cv2.error:
            return False
        return bool(ok and frame is not None and getattr(frame, "size", 0) > 0)
    finally:
        cap.release()


def detect_working_webcam_device(
    *,
    preferred: int,
    width: int,
    height: int,
    search_max_index: int = 8,
    fps: float = 30.0,
    exclude: set[int] | frozenset[int] | None = None,
) -> int | None:
    """
    Return first working V4L2 capture index.

    Prefers capture-node existence over OpenCV probe (HDMI H.264 often fails probe).
    """
    skip = {int(i) for i in (exclude or ())}
    if preferred >= 0 and preferred not in skip and is_video_capture_node(preferred):
        return preferred

    for idx in list_capture_indices(search_max_index):
        if idx == preferred or idx in skip:
            continue
        # Capture node present is enough for HDMI; probe is best-effort for MJPEG cams.
        if _probe_device(idx, width, height, fps=float(fps)) or is_video_capture_node(idx):
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
    exclude: set[int] | frozenset[int] | None = None,
) -> int | None:
    """Poll until a capture node is ready, or timeout."""
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        found = detect_working_webcam_device(
            preferred=preferred,
            width=width,
            height=height,
            search_max_index=search_max_index,
            fps=fps,
            exclude=exclude,
        )
        if found is not None:
            return found
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.1, float(poll_s)))


def resolve_local_webcam_device(
    preferred: int,
    *,
    auto_detect: bool = False,
    exclude: set[int] | frozenset[int] | None = None,
    search_max_index: int = 8,
    label: str = "Camera",
) -> int:
    """
    Resolve a local USB/HDMI camera index for Front or Back.

    Rules:
    - Never return a metadata-only node.
    - Never return an index in ``exclude`` (peer camera).
    - If preferred is a free capture node, use it immediately (no OpenCV probe).
    - If ``auto_detect``, fall back to the next free capture node.
    """
    skip = {int(i) for i in (exclude or ())}
    pref = int(preferred)
    capture = list_capture_indices(search_max_index)

    if pref not in skip and is_video_capture_node(pref):
        return pref

    if auto_detect:
        # Prefer other capture nodes; keep preferred order when possible.
        ordered = [pref] + [i for i in capture if i != pref] if pref >= 0 else list(capture)
        for idx in ordered:
            if idx in skip:
                continue
            if is_video_capture_node(idx):
                return int(idx)

    hint = camera_missing_hint()
    if pref in skip:
        raise ValueError(
            f"{label}: /dev/video{pref} is reserved for the other camera. "
            f"Pick a different capture node. {hint}"
        )
    if os.path.exists(f"/dev/video{pref}") and not is_video_capture_node(pref):
        raise ValueError(
            f"{label}: /dev/video{pref} is a metadata node, not Video Capture. "
            f"Use one of {capture or 'none'}. {hint}"
        )
    raise ValueError(
        f"{label}: webcam not ready (wanted /dev/video{pref}). {hint}"
    )


def peer_webcam_exclude(settings: dict, *, self_sensor: str) -> set[int]:
    """Indices owned by the other local camera pipeline (exclude when resolving)."""
    other = "multi_camera" if self_sensor == "webcam" else "webcam"
    block = settings.get(other) if isinstance(settings, dict) else None
    if not isinstance(block, dict):
        return set()
    mode = str(block.get("source_mode") or "local").strip().lower()
    if mode in ("jetson", "network", "rtsp", "url"):
        return set()
    try:
        return {int(block.get("webcam_device", -1))}
    except (TypeError, ValueError):
        return set()


def camera_missing_hint() -> str:
    """Short operator hint when no suitable V4L2 webcam is available."""
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
    nodes = video_nodes_present()
    if not nodes:
        return (
            "No /dev/video* on this machine. Plug in the webcam, confirm with "
            "`ls /dev/video*` and `v4l2-ctl --list-devices`, then retry."
        )
    capture = [f"/dev/video{i}" for i in list_capture_indices()]
    meta = [p for p in nodes if p not in capture]
    return (
        f"V4L2 nodes {nodes}; capture={capture or 'none'}; metadata={meta or 'none'}. "
        "HDMI dongles expose a metadata node next to capture — use the capture node "
        "(Device Caps Video Capture), not the metadata node."
    )
