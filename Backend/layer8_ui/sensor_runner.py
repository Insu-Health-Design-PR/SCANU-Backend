"""Compatibility shim — implementation lives in ``runtime.sensor_runner``."""

from __future__ import annotations

from runtime.sensor_runner import *  # noqa: F403
from runtime.sensor_runner import (  # noqa: F401
    SensorId,
    build_command,
    build_mmwave_command,
    command_cwd,
    mmwave_capture_script,
    resolved_software_root,
    restart,
    start,
    status,
    stop,
)
