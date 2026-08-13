"""Threat score formatting and bucketing."""

from __future__ import annotations

SAFE_MAX = 0.01


def overlay_score(score: float) -> str:
    """Clip to [0,1] and format for on-frame text."""
    value = min(1.0, max(0.0, float(score)))
    return f"{value:.2f}"


def threat_bucket(score: float, unsafe_min: float, safe_max: float = SAFE_MAX) -> str:
    del safe_max
    if score >= float(unsafe_min):
        return "unsafe"
    return "safe"


def bucket_threat(score: float) -> str:
    """Coarse generic bucket retained for older callers."""
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"
