"""Typed contracts shared by the dual-radar live runtime and its consumers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "scanu_mmwave_dual_live_v2"


@dataclass(frozen=True)
class LiveQuality:
    radar_a_ok: bool
    radar_b_ok: bool
    frames_a: int
    frames_b: int
    alignment_error_ms: float | None
    calibration_valid: bool
    dropped_frames_a: int = 0
    dropped_frames_b: int = 0
    message: str = ""


@dataclass(frozen=True)
class ReflectiveCandidate:
    center_m: list[float]
    score: float
    source_views: list[str]
    global_track_id: str | None = None
    persistent: bool = False
    classification: str = "reflective_anomaly"
    material_confirmed: bool = False


@dataclass(frozen=True)
class GlobalTrackPayload:
    global_track_id: str
    centroid_m: list[float]
    velocity_mps: list[float]
    extent_m: list[float]
    point_count: int
    source_views: list[str]
    age_windows: int
    missed_windows: int


@dataclass
class FusedLiveFrame:
    timestamp_ns: int
    points_a: list[list[float]]
    points_b: list[list[float]]
    fused_points: list[list[float]]
    tracks: list[GlobalTrackPayload]
    reflective_candidates: list[ReflectiveCandidate]
    quality: LiveQuality
    range_profile_a: list[float] = field(default_factory=list)
    range_profile_b: list[float] = field(default_factory=list)
    state: str = "LIVE"
    calibration_id: str = ""
    experimental: bool = True
    material_confirmed: bool = False
    weapon_classification: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        payload["fused"] = {
            "global_person_count": len(
                [track for track in self.tracks if track.missed_windows == 0]
            ),
            "active_tracks": [asdict(track) for track in self.tracks],
            "reflective_candidates": [
                asdict(candidate) for candidate in self.reflective_candidates
            ],
            "point_count": len(self.fused_points),
        }
        return payload


def status_payload(
    state: str,
    *,
    detail: str = "",
    calibration_progress: float = 0.0,
    calibration_id: str = "",
    started_utc: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experimental": True,
        "state": state,
        "detail": detail,
        "calibration_progress": max(0.0, min(1.0, float(calibration_progress))),
        "calibration_id": calibration_id,
        "started_utc": started_utc,
        "material_confirmed": False,
        "weapon_classification": False,
    }
