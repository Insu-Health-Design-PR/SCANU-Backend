"""Compatibility shim — implementation lives in ``runtime.webcam_runner``."""

from __future__ import annotations

from runtime.webcam_runner import *  # noqa: F403
from runtime.webcam_runner import (  # noqa: F401
    WebcamSharedStream,
    begin_webcam_preview_only,
    build_webcam_command,
    end_webcam_preview_only,
    get_webcam_shared_stream,
    webcam_command_cwd,
    webcam_preview_only,
)
