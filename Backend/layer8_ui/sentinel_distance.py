"""Sentinel corridor distance estimate between facing Front/Back cameras.

When Front and Back face each other with known baseline B, a person between them
has approximate depths:

  d = (H_person * frame_h) / (bbox_h * 2 * tan(VFOV/2))

If both cameras see the person, d_front + d_back ≈ B (consistency check).
"""

from __future__ import annotations

import math
from typing import Any


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


def estimate_sentinel(
    *,
    baseline_m: float,
    front_bbox_h: float | None = None,
    back_bbox_h: float | None = None,
    front_frame_h: float = 1080,
    back_frame_h: float = 1080,
    person_height_m: float = 1.70,
    vertical_fov_deg: float = 60.0,
) -> dict[str, Any]:
    """Return per-side depths and optional corridor consistency vs baseline."""
    d_f = (
        depth_from_bbox_height(
            float(front_bbox_h),
            float(front_frame_h),
            person_height_m=person_height_m,
            vertical_fov_deg=vertical_fov_deg,
        )
        if front_bbox_h is not None
        else None
    )
    d_b = (
        depth_from_bbox_height(
            float(back_bbox_h),
            float(back_frame_h),
            person_height_m=person_height_m,
            vertical_fov_deg=vertical_fov_deg,
        )
        if back_bbox_h is not None
        else None
    )
    out: dict[str, Any] = {
        "baseline_m": float(baseline_m),
        "d_front_m": None if d_f is None else round(d_f, 3),
        "d_back_m": None if d_b is None else round(d_b, 3),
        "sum_m": None,
        "residual_m": None,
        "ok_consistency": None,
    }
    if d_f is not None and d_b is not None and baseline_m > 0:
        s = d_f + d_b
        residual = s - float(baseline_m)
        out["sum_m"] = round(s, 3)
        out["residual_m"] = round(residual, 3)
        # Within 25% of baseline → rough OK for uncalibrated estimate
        out["ok_consistency"] = abs(residual) <= 0.25 * float(baseline_m)
    return out


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


def format_depth_label_m(distance_m: float | None) -> str:
    """Short UI suffix without leading space, e.g. ``5m``."""
    if distance_m is None or distance_m <= 0.05 or distance_m > 200.0:
        return ""
    meters = int(round(float(distance_m)))
    if meters < 1:
        meters = 1
    return f"{meters}m"
