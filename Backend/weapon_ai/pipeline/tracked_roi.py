"""Decide when to re-run person YOLO vs reuse tracked ROI geometry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackedRoiConfig:
    """Conservative defaults: person YOLO every 3 frames, max 100 ms age."""

    enabled: bool = True
    person_interval_frames: int = 3
    max_person_age_ms: float = 100.0
    min_track_score: float = 0.25
    max_box_shift_frac: float = 0.35
    high_risk_every_cycle: bool = True
    mmwave_high_risk: float = 0.45
    fallback_fixed_stride: bool = False


def box_shift_frac(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)
    cx_a = 0.5 * (ax1 + ax2)
    cy_a = 0.5 * (ay1 + ay2)
    cx_b = 0.5 * (bx1 + bx2)
    cy_b = 0.5 * (by1 + by2)
    diag = (aw * aw + ah * ah) ** 0.5
    dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
    area_a = float(aw * ah)
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    area_ratio = abs(area_a - area_b) / max(area_a, area_b)
    return max(dist / max(1.0, diag), area_ratio)


def should_refresh_person(
    *,
    cfg: TrackedRoiConfig,
    frames_since_person: int,
    person_age_ms: float | None,
    unmatched_or_new: bool,
    track_lost: bool,
    low_track_score: bool,
    box_shifted: bool,
    mmwave_high_risk: bool,
    force: bool = False,
) -> bool:
    """Return True when person YOLO must run. Never caches gun decisions."""
    if force:
        return True
    if not cfg.enabled or cfg.fallback_fixed_stride:
        return True
    if unmatched_or_new or track_lost or low_track_score or box_shifted:
        return True
    if cfg.high_risk_every_cycle and mmwave_high_risk:
        return True
    if person_age_ms is not None and person_age_ms >= float(cfg.max_person_age_ms):
        return True
    if frames_since_person >= max(1, int(cfg.person_interval_frames)):
        return True
    return False
