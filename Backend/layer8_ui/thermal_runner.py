"""Compatibility shim — implementation lives in ``runtime.thermal_runner``."""

from __future__ import annotations

from runtime.thermal_runner import *  # noqa: F403
from runtime.thermal_runner import (  # noqa: F401
    ThermalSharedStream,
    build_thermal_command,
    get_thermal_shared_stream,
    layer1_examples_dir,
    thermal_capture_script,
    thermal_command_cwd,
    thermal_preview_only,
    thermal_uses_inprocess_v4l2_preview,
)
