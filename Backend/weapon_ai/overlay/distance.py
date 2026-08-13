"""Monocular person distance from bbox height (beta overlay labels)."""

from __future__ import annotations

import math


def depth_from_bbox_height(
    bbox_h_px: float,
    frame_h_px: float,
    *,
    person_height_m: float = 1.70,
    vertical_fov_deg: float = 60.0,
) -> float | None:
    """Monocular depth (meters) from person bbox height and assumed height + VFOV."""
    if bbox_h_px <= 1 or frame_h_px <= 1:
        return None
    vfov = math.radians(max(1.0, float(vertical_fov_deg)))
    denom = float(bbox_h_px) * 2.0 * math.tan(vfov / 2.0)
    if denom <= 1e-9:
        return None
    return float(person_height_m) * float(frame_h_px) / denom


def smooth_depth_m(
    prev_m: float | None,
    measured_m: float | None,
    *,
    alpha: float = 0.35,
) -> float | None:
    """EMA-smooth monocular depth to reduce label flicker (beta)."""
    if measured_m is None or measured_m <= 0:
        return prev_m
    if prev_m is None or prev_m <= 0:
        return float(measured_m)
    a = min(1.0, max(0.05, float(alpha)))
    return a * float(measured_m) + (1.0 - a) * float(prev_m)
