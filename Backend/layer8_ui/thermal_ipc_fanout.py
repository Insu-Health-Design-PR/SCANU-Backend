"""Shared thermal IPC → WebRTC fan-out (same pattern as webcam_ipc_fanout)."""

from __future__ import annotations

import threading
from pathlib import Path

from layer8_ui.webcam_ipc_fanout import WebcamIpcFanout

_ThermalFanoutSingleton: WebcamIpcFanout | None = None
_ThermalFanoutLock = threading.Lock()


def get_thermal_ipc_fanout(
    *,
    bgr_path: Path = Path("/dev/shm/scanu_thermal_live_bgr_frame.bin"),
    jpg_path: Path = Path("/dev/shm/scanu_thermal_live_frame.bin"),
) -> WebcamIpcFanout:
    global _ThermalFanoutSingleton
    with _ThermalFanoutLock:
        if _ThermalFanoutSingleton is None:
            _ThermalFanoutSingleton = WebcamIpcFanout(bgr_path=bgr_path, jpg_path=jpg_path)
        return _ThermalFanoutSingleton
