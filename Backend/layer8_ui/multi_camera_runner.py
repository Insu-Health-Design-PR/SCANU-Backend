"""Compatibility shim — implementation lives in ``runtime.multi_camera_runner``."""

from __future__ import annotations

from runtime.multi_camera_runner import *  # noqa: F403
from runtime.multi_camera_runner import (  # noqa: F401
    MultiCameraSharedStream,
    build_multi_camera_command,
    get_multi_camera_shared_stream,
    multi_camera_command_cwd,
)
