"""Global multi-person tracks for two facing, independent mmWave radars.

This laboratory module consumes the *processed* point/TLV sessions produced by
the two existing AWR1843 paths.  It deliberately does not combine ADC samples:
two independent radars do not share an RF clock, trigger, or carrier phase, so
their raw samples cannot be treated as one coherent virtual array.

Instead, each time-matched window is transformed into one measured world frame,
clustered independently, associated across views, and then associated over time
to stable global IDs.  Reflective events remain unverified radar evidence; they
are never labelled metal, weapon, or firearm by this module.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lab.dual_mmwave77_stereo.point_cloud_fusion import (
    SensorSession,
    _clutter_mask_for,
    _load_clutter,
    _load_clutter_explicit,
    anomaly_centers,
    cluster_points,
    load_session,
    matched_window_pairs,
    transform_b_to_a,
    window_points,
)


@dataclass
class GlobalTrack:
    """One world-frame track.  Coordinates are in sensor A's frame."""

    track_id: str
    centroid_m: np.ndarray
    velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    first_time_s: float = 0.0
    last_time_s: float = 0.0
    age_windows: int = 0
    missed_windows: int = 0
    hits: int = 0
    last_source_views: tuple[str, ...] = ()

    def predicted(self, time_s: float) -> np.ndarray:
        return self.centroid_m + self.velocity_mps * max(0.0, time_s - self.last_time_s)


def _inside_portal(points: np.ndarray, distance_m: float) -> np.ndarray:
    """Reject behind-radar and implausibly high/low returns before clustering."""
    if not len(points):
        return points
    keep = (
        (points[:, 1] >= -0.15)
        & (points[:, 1] <= distance_m + 0.15)
        & (points[:, 2] >= -1.7)
        & (points[:, 2] <= 1.9)
    )
    return points[keep]


def _view_clusters(points: np.ndarray, distance_m: float, min_points: int) -> list[dict[str, Any]]:
    clusters = cluster_points(_inside_portal(points, distance_m), min_points=min_points)
    return [
        {
            "centroid_m": np.asarray(cluster.centroid_m, dtype=np.float32),
            "extent_m": [float(value) for value in cluster.extent_m],
            "point_count": int(cluster.point_count),
        }
        for cluster in clusters
    ]


def _associate_views(
    clusters_a: list[dict[str, Any]],
    clusters_b: list[dict[str, Any]],
    gate_m: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Greedily form one-to-one cross-view matches using world-frame distance."""
    candidates: list[tuple[float, int, int]] = []
    for ia, a in enumerate(clusters_a):
        for ib, b in enumerate(clusters_b):
            distance = float(np.linalg.norm(a["centroid_m"] - b["centroid_m"]))
            if distance <= gate_m:
                candidates.append((distance, ia, ib))
    candidates.sort(key=lambda item: (item[0], -clusters_a[item[1]]["point_count"], -clusters_b[item[2]]["point_count"]))
    matches: list[tuple[int, int]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for _, ia, ib in candidates:
        if ia not in used_a and ib not in used_b:
            matches.append((ia, ib))
            used_a.add(ia)
            used_b.add(ib)
    return matches, used_a, used_b


def _make_observations(
    clusters_a: list[dict[str, Any]], clusters_b: list[dict[str, Any]], gate_m: float
) -> list[dict[str, Any]]:
    matches, used_a, used_b = _associate_views(clusters_a, clusters_b, gate_m)
    observations: list[dict[str, Any]] = []
    for ia, ib in matches:
        a, b = clusters_a[ia], clusters_b[ib]
        weight_a, weight_b = max(1, a["point_count"]), max(1, b["point_count"])
        centroid = (weight_a * a["centroid_m"] + weight_b * b["centroid_m"]) / (weight_a + weight_b)
        observations.append(
            {
                "centroid_m": centroid,
                "extent_m": np.maximum(a["extent_m"], b["extent_m"]).tolist(),
                "point_count": int(a["point_count"] + b["point_count"]),
                "source_views": ("A", "B"),
                "cross_view_match": True,
            }
        )
    for ia, a in enumerate(clusters_a):
        if ia not in used_a:
            observations.append({"centroid_m": a["centroid_m"], "extent_m": a["extent_m"], "point_count": a["point_count"], "source_views": ("A",), "cross_view_match": False})
    for ib, b in enumerate(clusters_b):
        if ib not in used_b:
            observations.append({"centroid_m": b["centroid_m"], "extent_m": b["extent_m"], "point_count": b["point_count"], "source_views": ("B",), "cross_view_match": False})
    return observations


def _update_tracks(
    tracks: list[GlobalTrack], observations: list[dict[str, Any]], time_s: float, gate_m: float, max_missed: int, next_id: int
) -> tuple[list[GlobalTrack], list[dict[str, Any]], int]:
    """One-to-one nearest-neighbour update; unmatched observations start a track."""
    candidates: list[tuple[float, int, int]] = []
    for it, track in enumerate(tracks):
        predicted = track.predicted(time_s)
        for io, observation in enumerate(observations):
            distance = float(np.linalg.norm(predicted - observation["centroid_m"]))
            if distance <= gate_m:
                candidates.append((distance, it, io))
    candidates.sort(key=lambda item: item[0])
    matched_track: set[int] = set()
    matched_observation: set[int] = set()
    for _, it, io in candidates:
        if it in matched_track or io in matched_observation:
            continue
        track, observation = tracks[it], observations[io]
        elapsed = max(1e-3, time_s - track.last_time_s)
        measured_velocity = (observation["centroid_m"] - track.centroid_m) / elapsed
        track.velocity_mps = (0.65 * track.velocity_mps + 0.35 * measured_velocity).astype(np.float32)
        track.centroid_m = np.asarray(observation["centroid_m"], dtype=np.float32)
        track.last_time_s = time_s
        track.age_windows += 1
        track.hits += 1
        track.missed_windows = 0
        track.last_source_views = tuple(observation["source_views"])
        observation["global_track_id"] = track.track_id
        matched_track.add(it)
        matched_observation.add(io)
    for it, track in enumerate(tracks):
        if it not in matched_track:
            track.age_windows += 1
            track.missed_windows += 1
    for io, observation in enumerate(observations):
        if io in matched_observation:
            continue
        track = GlobalTrack(
            track_id=f"G{next_id:03d}",
            centroid_m=np.asarray(observation["centroid_m"], dtype=np.float32),
            first_time_s=time_s,
            last_time_s=time_s,
            age_windows=1,
            hits=1,
            last_source_views=tuple(observation["source_views"]),
        )
        next_id += 1
        tracks.append(track)
        observation["global_track_id"] = track.track_id
    tracks = [track for track in tracks if track.missed_windows <= max_missed]
    return tracks, observations, next_id


def _track_for_point(tracks: list[GlobalTrack], point: np.ndarray, gate_m: float) -> str | None:
    if not tracks:
        return None
    distances = [float(np.linalg.norm(track.centroid_m - point)) for track in tracks]
    index = int(np.argmin(distances))
    return tracks[index].track_id if distances[index] <= gate_m else None


def _associate_reflective_evidence(
    centers_a: np.ndarray, centers_b: np.ndarray, tracks: list[GlobalTrack], track_gate_m: float, cross_view_gate_m: float
) -> list[dict[str, Any]]:
    """Associate unverified reflective events to global person tracks.

    The strongest statement emitted is a same-window, two-view reflective
    anomaly.  It is still not a material identification.
    """
    evidence: list[dict[str, Any]] = []
    matches, used_a, used_b = _associate_views(
        [{"centroid_m": point, "point_count": 1} for point in centers_a],
        [{"centroid_m": point, "point_count": 1} for point in centers_b],
        cross_view_gate_m,
    )
    for ia, ib in matches:
        center = (centers_a[ia] + centers_b[ib]) / 2.0
        evidence.append({"center_m": center.tolist(), "source_views": ["A", "B"], "classification": "multiview_reflective_anomaly", "material_confirmed": False, "global_track_id": _track_for_point(tracks, center, track_gate_m)})
    for ia, center in enumerate(centers_a):
        if ia not in used_a:
            evidence.append({"center_m": center.tolist(), "source_views": ["A"], "classification": "single_view_reflective_anomaly", "material_confirmed": False, "global_track_id": _track_for_point(tracks, center, track_gate_m)})
    for ib, center in enumerate(centers_b):
        if ib not in used_b:
            evidence.append({"center_m": center.tolist(), "source_views": ["B"], "classification": "single_view_reflective_anomaly", "material_confirmed": False, "global_track_id": _track_for_point(tracks, center, track_gate_m)})
    return evidence


def _track_row(track: GlobalTrack) -> dict[str, Any]:
    return {
        "global_track_id": track.track_id,
        "centroid_m": [float(value) for value in track.centroid_m],
        "velocity_mps": [float(value) for value in track.velocity_mps],
        "age_windows": track.age_windows,
        "missed_windows": track.missed_windows,
        "hits": track.hits,
        "last_source_views": list(track.last_source_views),
    }


def run_global_tracking(
    session_a: SensorSession,
    session_b: SensorSession,
    *,
    distance_m: float,
    clock_offset_b_minus_a_s: float = 0.0,
    window_tolerance_s: float = 0.5,
    clutter_a: dict[str, np.ndarray] | None = None,
    clutter_b: dict[str, np.ndarray] | None = None,
    min_cluster_points: int = 15,
    cross_view_gate_m: float = 0.8,
    track_gate_m: float = 1.0,
    max_missed_windows: int = 3,
) -> dict[str, Any]:
    """Create globally deduplicated tracks from two time-matched sessions."""
    rows: list[dict[str, Any]] = []
    tracks: list[GlobalTrack] = []
    next_id = 1
    for ia, ib in matched_window_pairs(session_a, session_b, clock_offset_b_minus_a_s, window_tolerance_s):
        raw_a = window_points(session_a, session_a.window_frame_start[ia])
        raw_b = window_points(session_b, session_b.window_frame_start[ib])
        points_a = _inside_portal(raw_a[~_clutter_mask_for(raw_a, clutter_a)], distance_m)
        points_b_local = raw_b[~_clutter_mask_for(raw_b, clutter_b)]
        points_b = _inside_portal(transform_b_to_a(points_b_local, distance_m), distance_m)
        clusters_a = _view_clusters(points_a, distance_m, min_cluster_points)
        clusters_b = _view_clusters(points_b, distance_m, min_cluster_points)
        observations = _make_observations(clusters_a, clusters_b, cross_view_gate_m)
        time_s = float(session_a.window_time_s[ia])
        tracks, observations, next_id = _update_tracks(tracks, observations, time_s, track_gate_m, max_missed_windows, next_id)
        centers_a = anomaly_centers(session_a.window_anomalies[ia] if ia < len(session_a.window_anomalies) else [], transform=False)
        centers_b = anomaly_centers(session_b.window_anomalies[ib] if ib < len(session_b.window_anomalies) else [], transform=True, distance_m=distance_m)
        evidence = _associate_reflective_evidence(centers_a, centers_b, tracks, track_gate_m, cross_view_gate_m)
        rows.append({
            "window_index": int(session_a.window_index[ia]),
            "time_s": time_s,
            "global_person_count": len(tracks),
            "observations": [{**observation, "centroid_m": [float(value) for value in observation["centroid_m"]]} for observation in observations],
            "active_tracks": [_track_row(track) for track in tracks],
            "reflective_evidence": evidence,
        })
    return {
        "schema_version": "scanu_lab_dual_global_tracks_v1",
        "experimental": True,
        "world_frame": "sensor_A; B is transformed with x'=-x, y'=D-y, z'=z",
        "sensor_distance_m": distance_m,
        "matched_windows": len(rows),
        "maximum_simultaneous_global_person_count": max((row["global_person_count"] for row in rows), default=0),
        "unique_global_track_ids": sorted({track["global_track_id"] for row in rows for track in row["active_tracks"]}),
        "parameters": {"min_cluster_points": min_cluster_points, "cross_view_gate_m": cross_view_gate_m, "track_gate_m": track_gate_m, "max_missed_windows": max_missed_windows, "clock_offset_b_minus_a_s": clock_offset_b_minus_a_s, "window_tolerance_s": window_tolerance_s},
        "limitations": [
            "post-CFAR/TLV track-level fusion only; raw ADC is not coherently combined",
            "deduplication quality depends on measured extrinsics, host-clock offset, clutter calibration and association gates",
            "a global track is an experimental spatial hypothesis, not a biometric identity or validated person-count metric",
            "reflective evidence is not a metal, weapon, or firearm classification; controlled material calibration is required",
        ],
        "windows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental global multi-person tracks from two facing AWR1843 sessions")
    parser.add_argument("--session-a", required=True, type=Path)
    parser.add_argument("--session-b", required=True, type=Path)
    parser.add_argument("--distance-m", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-a", type=Path)
    parser.add_argument("--calibration-b", type=Path)
    parser.add_argument("--clock-offset-b-minus-a-s", type=float, default=0.0)
    parser.add_argument("--window-tolerance-s", type=float, default=0.5)
    parser.add_argument("--min-cluster-points", type=int, default=15)
    parser.add_argument("--cross-view-gate-m", type=float, default=0.8)
    parser.add_argument("--track-gate-m", type=float, default=1.0)
    parser.add_argument("--max-missed-windows", type=int, default=3)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = run_global_tracking(
        load_session(args.session_a), load_session(args.session_b),
        distance_m=args.distance_m,
        clock_offset_b_minus_a_s=args.clock_offset_b_minus_a_s,
        window_tolerance_s=args.window_tolerance_s,
        clutter_a=_load_clutter_explicit(args.calibration_a) if args.calibration_a else _load_clutter(args.session_a),
        clutter_b=_load_clutter_explicit(args.calibration_b) if args.calibration_b else _load_clutter(args.session_b),
        min_cluster_points=args.min_cluster_points,
        cross_view_gate_m=args.cross_view_gate_m,
        track_gate_m=args.track_gate_m,
        max_missed_windows=args.max_missed_windows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("matched_windows", "maximum_simultaneous_global_person_count", "unique_global_track_ids")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
