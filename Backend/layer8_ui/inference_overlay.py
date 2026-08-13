"""
Optional BGR overlays from weapon inference (``infer_thermal_objects`` pipeline).

The live subprocess uses ``runtime.webcam_layer8_runner``; this module is for
server-side preview composition when the UI requests an overlay without a full infer pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_weapon_overlay(
    bgr: np.ndarray,
    settings: dict[str, Any],
    *,
    source: str = "webcam",
) -> np.ndarray:
    """
    Return ``bgr`` unchanged for now. Hook: decode sidecar JSON or call infer helpers
    from migrated ``weapon_ai.infer_thermal_objects`` when wired.
    """
    del settings, source
    return bgr
