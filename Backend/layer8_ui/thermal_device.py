"""Thermal camera device auto-detection helpers."""

from __future__ import annotations

import glob
import os
import re
import subprocess

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2

_THERMAL_NAME_PAT = re.compile(r"(?i)purethermal|pure.?thermal|thermal|flir|seek|infrared")
_WEBCAM_NAME_PAT = re.compile(
    r"(?i)usb\s+video|uvc|webcam|nexigo|logitech|hdmi|face\s*cam|realsense|ai\s*camera|camera\s*\("
)
# PureThermal native sizes — used for probing so HD webcams are not mistaken for thermal.
_THERMAL_PROBE_SIZES: tuple[tuple[int, int], ...] = ((160, 120), (120, 160), (80, 60), (60, 80))


def v4l2_device_path(device: int | str) -> str:
    """Linux V4L2 nodes must be opened by path; numeric indices often fail on PureThermal."""
    s = str(device).strip()
    if s.startswith("/dev/video"):
        return s
    return f"/dev/video{int(s)}"


def _device_exists(device: int | str) -> bool:
    return os.path.exists(v4l2_device_path(device))


def _existing_video_indices(max_index: int = 12) -> list[int]:
    out: list[int] = []
    for path in sorted(glob.glob("/dev/video*")):
        m = re.search(r"/dev/video(\d+)$", path)
        if not m:
            continue
        idx = int(m.group(1))
        if idx <= int(max_index):
            out.append(idx)
    return sorted(set(out))


def _v4l2_device_groups() -> list[tuple[str, list[int]]]:
    """Return ``(device_name, [video indices])`` from ``v4l2-ctl --list-devices``."""
    try:
        p = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        text = p.stdout or ""
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    if not text.strip():
        return []
    groups: list[tuple[str, list[int]]] = []
    current_name = ""
    current_indices: list[int] = []
    for line in text.splitlines():
        if line.strip() and not line.startswith(("\t", " ")):
            if current_name and current_indices:
                groups.append((current_name, sorted(set(current_indices))))
            current_name = line.strip()
            current_indices = []
            continue
        m = re.match(r"^\s*/dev/video(\d+)", line)
        if m and current_name:
            current_indices.append(int(m.group(1)))
    if current_name and current_indices:
        groups.append((current_name, sorted(set(current_indices))))
    return groups


def _indices_from_v4l2(name_pat: re.Pattern[str]) -> list[int]:
    indices: list[int] = []
    for name, inds in _v4l2_device_groups():
        if name_pat.search(name):
            indices.extend(inds)
    return sorted(set(indices))


def _thermal_indices_from_v4l2() -> list[int]:
    """PureThermal / thermal UVC nodes from ``v4l2-ctl --list-devices``."""
    return _indices_from_v4l2(_THERMAL_NAME_PAT)


def _webcam_indices_from_v4l2() -> list[int]:
    """UVC / USB webcam nodes — must not be used for thermal capture."""
    thermal = set(_thermal_indices_from_v4l2())
    out: list[int] = []
    for name, inds in _v4l2_device_groups():
        if _WEBCAM_NAME_PAT.search(name) and not _THERMAL_NAME_PAT.search(name):
            for idx in inds:
                if idx not in thermal:
                    out.append(idx)
    return sorted(set(out))


def _blocked_for_thermal(index: int) -> bool:
    """True when this index is a known webcam node and not a thermal camera."""
    thermal = set(_thermal_indices_from_v4l2())
    if int(index) in thermal:
        return False
    return int(index) in set(_webcam_indices_from_v4l2())


def _effective_thermal_probe_sizes(width: int, height: int) -> list[tuple[int, int]]:
    """Probe at native thermal sizes; ignore HD settings that belong on AI/webcam cameras."""
    w, h = int(width), int(height)
    sizes: list[tuple[int, int]] = []
    if w <= 320 and h <= 240 and w > 0 and h > 0:
        sizes.append((w, h))
    for pair in _THERMAL_PROBE_SIZES:
        if pair not in sizes:
            sizes.append(pair)
    return sizes


def _probe_device_v4l2_ctl(index: int, width: int, height: int) -> bool:
    path = v4l2_device_path(index)
    if not os.path.exists(path):
        return False
    for pixfmt in ("GREY", "Y16 "):
        try:
            p = subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    path,
                    f"--set-fmt-video=width={int(width)},height={int(height)},pixelformat={pixfmt}",
                    "--stream-mmap",
                    "--stream-count=1",
                    "--stream-to=/dev/null",
                ],
                capture_output=True,
                timeout=2.5,
            )
            if p.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            continue
    return False


def _probe_device_opencv(index: int, width: int, height: int, fps: int) -> bool:
    path = v4l2_device_path(index)
    if not os.path.exists(path):
        return False
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        return False
    try:
        try:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "1", "6", " "))
        except Exception:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"GREY"))
            except Exception:
                pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, int(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok or frame is None or getattr(frame, "size", 0) <= 0:
            return False
        fh, fw = int(frame.shape[0]), int(frame.shape[1])
        # Reject obvious HD RGB frames from webcams probed at wrong size.
        if fw > 400 or fh > 400:
            return False
        return True
    finally:
        cap.release()


def _probe_device(index: int, width: int, height: int, fps: int) -> bool:
    if _probe_device_v4l2_ctl(index, width, height):
        return True
    return _probe_device_opencv(index, width, height, fps)


def _probe_thermal_device(index: int, width: int, height: int, fps: int) -> bool:
    if _blocked_for_thermal(index):
        return False
    for pw, ph in _effective_thermal_probe_sizes(width, height):
        if _probe_device(index, pw, ph, fps):
            return True
    return False


def detect_working_thermal_device(
    *,
    preferred: int,
    width: int,
    height: int,
    fps: int,
    search_max_index: int = 12,
) -> int | None:
    """
    Return a V4L2 index for PureThermal capture.

    Never returns known UVC/webcam nodes (e.g. ``/dev/video0`` NexiGo). Probes at
    native thermal resolutions (160×120) even when settings mistakenly use HD sizes.
    """
    named = _thermal_indices_from_v4l2()
    candidates: list[int] = []
    if named:
        candidates.extend(named)
    pref = int(preferred)
    if pref >= 0 and pref not in candidates and not _blocked_for_thermal(pref):
        candidates.insert(0, pref)
    for idx in _existing_video_indices(int(search_max_index)):
        if idx not in candidates and not _blocked_for_thermal(idx):
            candidates.append(idx)

    for idx in candidates:
        if not _device_exists(idx):
            continue
        if _probe_thermal_device(idx, width, height, fps):
            return idx

    # Last resort: return a named PureThermal node even if probe failed (USB replug).
    for idx in named:
        if _device_exists(idx):
            return idx
    return None


def resolve_thermal_device_or_raise(t: dict) -> int:
    """Resolve thermal index from settings; raise if no ``/dev/video*`` node exists."""
    preferred = int(t.get("thermal_device", 2))
    width = int(t.get("thermal_width", 160))
    height = int(t.get("thermal_height", 120))
    fps = int(t.get("thermal_fps", 9))
    max_idx = int(t.get("thermal_detect_max_index", 12))

    if _blocked_for_thermal(preferred):
        preferred = -1

    if int(t.get("thermal_auto_detect", 1)):
        detected = detect_working_thermal_device(
            preferred=preferred,
            width=width,
            height=height,
            fps=fps,
            search_max_index=max_idx,
        )
        if detected is not None:
            return int(detected)

    if preferred >= 0 and not _blocked_for_thermal(preferred) and _device_exists(preferred):
        return int(preferred)

    named = _thermal_indices_from_v4l2()
    webcam = _webcam_indices_from_v4l2()
    hint = ""
    if named:
        hint = f" PureThermal reported at {named}."
    elif webcam:
        hint = f" Webcam nodes (not thermal): {webcam}."
    raise RuntimeError(
        f"Thermal device {v4l2_device_path(preferred)} is not a thermal camera.{hint} "
        "Set thermal_device to your PureThermal index (often /dev/video2), not the AI webcam. "
        "Use Thermal → Auto-configure or POST /api/thermal/auto_configure."
    )
