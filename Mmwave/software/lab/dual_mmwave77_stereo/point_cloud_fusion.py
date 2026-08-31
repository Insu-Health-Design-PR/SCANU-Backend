"""Point-cloud fusion for two facing AWR1843BOOST sensors.

Transforms sensor B's detections into sensor A's frame using a measured
rigid geometry (distance D and equal height), merges the point clouds per
matched window, and clusters the fused cloud to describe the person with
both sensors' evidence. Output is descriptive only: no ground-truth label
exists for these lab captures, so fused counts/extents are not accuracy.

Canonical geometry for this test:
  D  = distance between the two sensors (12 ft = 3.6576 m)
  h  = antenna height, equal for both (40 in = 1.016 m)
Facing sensors: B's +y (range) points at A, so in A's frame a B point
(x_B, y_B, z_B) maps to (-x_B, D - y_B, z_B).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lab.mmwave77_usb.perception import (
    PerceptionSpec,
    _person_clusters,
    _update_track,
)


def _utc_ns(text: str) -> int:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return int(dt.timestamp() * 1_000_000_000)


@dataclass
class SensorSession:
    """Frame-level point cloud + perception windows for one sensor."""
    frames_utc_ns: list[int]
    frame_numbers: list[int]          # one per frame, matching points list order
    frames_monotonic_ns: list[int]    # host clock per frame, for camera alignment
    points: list[np.ndarray]           # one (N,6) float32 array per frame (x,y,z,doppler,snr,noise)
    window_index: list[int]
    window_frame_start: list[int]
    window_time_s: list[float]
    window_present: list[bool]         # track observed_this_window
    window_cluster_points: list[int]
    window_input_points: list[int]
    window_state: list[str] = field(default_factory=list)          # screening_state per window
    window_anomalies: list[list[dict[str, Any]]] = field(default_factory=list)  # anomalies per window


POINT_FIELDS = ("x", "y", "z", "doppler", "snr", "noise")


def _load_frames(frames_path: Path) -> tuple[list[int], list[int], list[int], list[np.ndarray]]:
    utc_ns: list[int] = []
    frame_numbers: list[int] = []
    monotonic_ns: list[int] = []
    points: list[np.ndarray] = []
    with frames_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            utc_ns.append(_utc_ns(row["host_utc"]))
            frame_numbers.append(int(row["frame_number"]))
            monotonic_ns.append(int(row.get("host_monotonic_ns", 0)))
            pts = row.get("points", [])
            values: list[list[float]] = []
            for p in pts:
                try:
                    item = [float(p.get(field, 0.0)) for field in POINT_FIELDS]
                except (TypeError, ValueError):
                    continue
                if all(math.isfinite(value) for value in item):
                    values.append(item)
            points.append(
                np.asarray(values, dtype=np.float32)
                if values
                else np.empty((0, len(POINT_FIELDS)), dtype=np.float32)
            )
    return utc_ns, frame_numbers, monotonic_ns, points


def _load_perception(
    perception_path: Path,
) -> tuple[list[int], list[int], list[float], list[bool], list[int], list[int], list[str], list[list[dict[str, Any]]]]:
    idx: list[int] = []
    fstart: list[int] = []
    tsec: list[float] = []
    present: list[bool] = []
    cluster_pts: list[int] = []
    input_pts: list[int] = []
    states: list[str] = []
    anomalies: list[list[dict[str, Any]]] = []
    with perception_path.open() as stream:
        for line in stream:
            if not line.strip() or '"window_index"' not in line:
                continue
            row = json.loads(line)
            idx.append(int(row["window_index"]))
            fstart.append(int(row["frame_start"]))
            tsec.append(float(row.get("time_s", 0.0)))
            track = row.get("track")
            present.append(bool(track.get("observed_this_window", False)) if track else False)
            cluster_pts.append(int(track.get("cluster_point_count", 0)) if track else 0)
            input_pts.append(int(row.get("input_point_count", 0)))
            states.append(str(row.get("screening_state", "")))
            anomalies.append(list(row.get("anomalies", [])))
    return idx, fstart, tsec, present, cluster_pts, input_pts, states, anomalies


def load_session(session_dir: Path) -> SensorSession:
    frames_utc_ns, frame_numbers, monotonic_ns, points = _load_frames(
        session_dir / "frames.jsonl"
    )
    idx, fstart, tsec, present, cluster_pts, input_pts, states, anomalies = (
        _load_perception(session_dir / "perception.jsonl")
    )
    return SensorSession(
        frames_utc_ns=frames_utc_ns,
        frame_numbers=frame_numbers,
        frames_monotonic_ns=monotonic_ns,
        points=points,
        window_index=idx,
        window_frame_start=fstart,
        window_time_s=tsec,
        window_present=present,
        window_cluster_points=cluster_pts,
        window_input_points=input_pts,
        window_state=states,
        window_anomalies=anomalies,
    )


def _frame_position(session: SensorSession, frame_number: int) -> int:
    """Index into session points for a 1-based frame number.

    Performs a binary search over frame_numbers so frames that start at
    non-1 numbers or have gaps still map to the correct position.
    """
    import bisect

    index = bisect.bisect_left(session.frame_numbers, frame_number)
    if index < len(session.frame_numbers) and session.frame_numbers[index] == frame_number:
        return index
    return -1


def window_start_ns(session: SensorSession, frame_start: int) -> int:
    position = _frame_position(session, frame_start)
    if position >= 0:
        return session.frames_utc_ns[position]
    return session.frames_utc_ns[0]


def window_center_monotonic_ns(
    session: SensorSession,
    frame_start: int,
    frame_span: int = 10,
) -> int:
    """Host-monotonic timestamp at the temporal center of a radar window.

    A rendered perception row summarizes ``frame_span`` radar frames.  Camera
    lookup must therefore use the median timestamp of those observations, not
    the first frame timestamp; using the start makes the visualization lead
    the camera by roughly half a window (about 0.45--0.50 s for the current
    AWR1843 profile).
    """
    i0 = _frame_position(session, frame_start)
    if i0 < 0:
        return session.frames_monotonic_ns[0]
    i1 = min(i0 + frame_span, len(session.frames_monotonic_ns))
    values = sorted(
        int(value)
        for value in session.frames_monotonic_ns[i0:i1]
        if int(value) > 0
    )
    if not values:
        return session.frames_monotonic_ns[i0]
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) // 2


def window_points(session: SensorSession, frame_start: int, frame_span: int = 10) -> np.ndarray:
    i0 = _frame_position(session, frame_start)
    if i0 < 0:
        return np.empty((0, len(POINT_FIELDS)), dtype=np.float32)
    i1 = min(i0 + frame_span, len(session.points))
    return np.concatenate(session.points[i0:i1], axis=0) if i1 > i0 else np.empty((0, len(POINT_FIELDS)), dtype=np.float32)


def _window_state_at(session: SensorSession, index: int) -> str:
    if index < 0 or index >= len(session.window_state):
        return ""
    return session.window_state[index]


def _window_anomalies_at(session: SensorSession, index: int) -> list[dict[str, Any]]:
    if index < 0 or index >= len(session.window_anomalies):
        return []
    return session.window_anomalies[index]


def matched_window_pairs(
    session_a: SensorSession,
    session_b: SensorSession,
    clock_offset_b_minus_a_s: float,
    window_tolerance_s: float,
) -> list[tuple[int, int]]:
    """Return aligned (i_a, i_b) window-pair indices.

    Window numbers are local counters, not a synchronization signal: the two
    capture processes can start on different scheduler ticks.  Pair every A
    window with the nearest *unused* B window in corrected wall-clock time.
    This keeps a camera frame associated with the radar evidence measured at
    the same physical time, rather than with an arbitrary equal counter.

    ``clock_offset_b_minus_a_s`` is the measured host-clock difference.  It
    must not be used to hide a capture-start delay; an unsynchronised start
    simply leaves unmatched leading/trailing windows.
    """
    offset_ns = int(clock_offset_b_minus_a_s * 1e9)
    tol_ns = int(window_tolerance_s * 1e9)

    b_starts_on_a = [
        window_start_ns(session_b, frame_start) - offset_ns
        for frame_start in session_b.window_frame_start
    ]
    used_b: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i_a, frame_start in enumerate(session_a.window_frame_start):
        a_start = window_start_ns(session_a, frame_start)
        candidates = [
            (abs(b_start - a_start), i_b)
            for i_b, b_start in enumerate(b_starts_on_a)
            if i_b not in used_b
        ]
        if not candidates:
            continue
        delta_ns, i_b = min(candidates)
        if delta_ns <= tol_ns:
            pairs.append((i_a, i_b))
            used_b.add(i_b)
    return pairs


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    edges = np.empty(centers.shape[0] + 1, dtype=np.float64)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return edges


def _clutter_mask_for(points: np.ndarray, clutter: dict[str, np.ndarray] | None) -> np.ndarray:
    """Return True per point falling in calibrated static clutter bins."""
    if clutter is None or not len(points):
        return np.zeros(len(points), dtype=bool)
    mask = np.asarray(clutter["clutter_mask"], dtype=bool)
    if mask.ndim != 3 or not mask.size:
        return np.zeros(len(points), dtype=bool)
    x, y, z = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64), points[:, 2].astype(np.float64)
    range_m = np.sqrt(x * x + y * y + z * z)
    azimuth = np.degrees(np.arctan2(x, y))
    elevation = np.degrees(np.arctan2(z, np.hypot(x, y)))
    edges = (
        _centers_to_edges(np.asarray(clutter["range_centers_m"])),
        _centers_to_edges(np.asarray(clutter["azimuth_centers_deg"])),
        _centers_to_edges(np.asarray(clutter["elevation_centers_deg"])),
    )
    indexes = [
        np.searchsorted(axis_edges, values, side="right") - 1
        for axis_edges, values in zip(edges, (range_m, azimuth, elevation))
    ]
    valid = np.ones(len(points), dtype=bool)
    for index, axis_edges in zip(indexes, edges):
        valid &= (index >= 0) & (index < len(axis_edges) - 1)
    result = np.zeros(len(points), dtype=bool)
    result[valid] = mask[tuple(index[valid] for index in indexes)]
    return result


def _load_clutter(session_dir: Path) -> dict[str, np.ndarray] | None:
    summary_path = session_dir / "perception_summary.json"
    calibration = None
    if summary_path.is_file():
        with summary_path.open() as f:
            calibration = json.load(f).get("calibration_session")
    if not calibration:
        return None
    path = Path(calibration) / "clutter_mask.npz"
    if not path.is_file():
        return None
    with np.load(path) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _load_clutter_explicit(calibration: Path | None) -> dict[str, np.ndarray] | None:
    if calibration is None:
        return None
    path = calibration / "clutter_mask.npz"
    if not path.is_file():
        raise ValueError(f"calibration has no clutter_mask.npz: {calibration}")
    with np.load(path) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def transform_b_to_a(points: np.ndarray, distance_m: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    out = points.copy()
    out[:, 0] = -points[:, 0]
    out[:, 1] = distance_m - points[:, 1]
    return out


def anomaly_centers(
    anomalies: list[dict[str, Any]], *, transform: bool, distance_m: float = 0.0
) -> np.ndarray:
    """Return anomaly center_m as an (N,3) array in sensor A's frame."""
    if not anomalies:
        return np.empty((0, 3), dtype=np.float32)
    out = np.asarray(
        [list(a["center_m"]) for a in anomalies], dtype=np.float32
    ).reshape(-1, 3)
    if transform:
        out = transform_b_to_a(out, distance_m)
    return out


@dataclass
class FusedCluster:
    centroid_m: list[float]
    extent_m: list[float]
    point_count: int
    energy: float = 0.0


def cluster_points(points: np.ndarray, voxel_m: float = 0.18, min_points: int = 5) -> list[FusedCluster]:
    """Connected components on a voxel grid (numpy only)."""
    if points.shape[0] == 0:
        return []
    xyz = points[:, :3]
    vox = np.floor(xyz / voxel_m).astype(np.int64)
    keys = vox[:, 0] * 1_000_000 + vox[:, 1] * 1_000 + vox[:, 2]
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    xyz_sorted = xyz[order]
    # Runs of the same voxel => super-points.
    change = np.empty(keys_sorted.shape[0], dtype=bool)
    change[0] = True
    change[1:] = keys_sorted[1:] != keys_sorted[:-1]
    run_id = np.cumsum(change) - 1
    vox_sorted = vox[order]
    run_vox = vox_sorted[change]
    # ``np.add.at`` accumulates into the existing array. Starting from
    # ``empty`` made voxel centroids depend on uninitialized memory and could
    # produce NaN/Inf values in an otherwise finite capture.
    run_cent = np.zeros((run_id[-1] + 1, 3), dtype=np.float32)
    run_count = np.bincount(run_id, minlength=run_id[-1] + 1)
    np.add.at(run_cent, run_id, xyz_sorted)
    run_cent /= run_count[:, None]

    # Union-find over neighboring voxels (26-connectivity).
    key_to_idx = {tuple(k.tolist()): i for i, k in enumerate(run_vox)}
    parent = list(range(len(run_vox)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, (kx, ky, kz) in enumerate(run_vox.tolist()):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    j = key_to_idx.get((kx + dx, ky + dy, kz + dz))
                    if j is not None:
                        union(i, j)

    # Accumulate members per root.
    roots = [find(i) for i in range(len(run_vox))]
    groups: dict[int, list[int]] = {}
    for i, r in enumerate(roots):
        groups.setdefault(r, []).append(i)

    clusters: list[FusedCluster] = []
    for members in groups.values():
        total = sum(int(run_count[m]) for m in members)
        if total < min_points:
            continue
        pts = np.concatenate(
            [run_cent[m:m + 1] for m in members], axis=0
        ) if len(members) > 1 else run_cent[members[0]:members[0] + 1]
        # Include original points that belong to these voxels for extent.
        mask = np.isin(run_id, members)
        orig = xyz_sorted[mask]
        centroid = orig.mean(axis=0).tolist()
        lo = orig.min(axis=0)
        hi = orig.max(axis=0)
        extent = (hi - lo).tolist()
        clusters.append(FusedCluster(centroid_m=centroid, extent_m=extent, point_count=total))
    clusters.sort(key=lambda c: c.point_count, reverse=True)
    return clusters


def _fuse_window(
    session_a: SensorSession,
    session_b: SensorSession,
    i_a: int,
    i_b: int,
    distance_m: float,
    clutter_a: dict[str, np.ndarray] | None,
    clutter_b: dict[str, np.ndarray] | None,
    person_min_points: int,
    spec: PerceptionSpec,
    fused_track: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fuse one matched window pair and return its report row plus point clouds.

    Returns (row, pts_a, pts_b, fused, updated_track) so video rendering reuses
    the exact per-window computation that produced the fusion report while the
    temporal track state is preserved across windows.
    """
    w = session_a.window_index[i_a]
    pts_a_raw = window_points(session_a, session_a.window_frame_start[i_a])
    pts_b_raw = window_points(session_b, session_b.window_frame_start[i_b])
    pts_a = pts_a_raw[~_clutter_mask_for(pts_a_raw, clutter_a)]
    pts_b_local = pts_b_raw[~_clutter_mask_for(pts_b_raw, clutter_b)]
    pts_b = transform_b_to_a(pts_b_local, distance_m)
    fused = np.concatenate([pts_a, pts_b], axis=0) if pts_b.shape[0] else pts_a

    window_time = session_a.window_time_s[i_a]
    fused_clusters = _person_clusters(fused, spec)
    fused_track, fused_selected = _update_track(
        fused_track, fused_clusters, time_s=window_time, spec=spec
    )
    fused_track_present = fused_track is not None and fused_selected is not None

    # Person region: between the sensors along y, near the antenna height.
    mid = distance_m / 2.0
    region_a = pts_a[
        (np.abs(pts_a[:, 1] - mid) < mid * 0.9)
        & (np.abs(pts_a[:, 2]) < 1.4)
    ]
    region_b = pts_b[
        (np.abs(pts_b[:, 1] - mid) < mid * 0.9)
        & (np.abs(pts_b[:, 2]) < 1.4)
    ]
    region_fused = fused[
        (np.abs(fused[:, 1] - mid) < mid * 0.9)
        & (np.abs(fused[:, 2]) < 1.4)
    ]
    clusters_a = cluster_points(region_a)
    clusters_b = cluster_points(region_b)
    clusters_fused = cluster_points(region_fused)
    person_a = clusters_a[0] if clusters_a else None
    person_b = clusters_b[0] if clusters_b else None
    person_fused = clusters_fused[0] if clusters_fused else None

    a_present = person_a is not None and person_a.point_count >= person_min_points
    b_present = person_b is not None and person_b.point_count >= person_min_points
    fused_present = person_fused is not None and person_fused.point_count >= person_min_points

    row = {
        "window_index": w,
        "time_s": round(session_a.window_time_s[i_a], 3),
        "fused_point_count": int(fused.shape[0]),
        "fused_region_point_count": int(region_fused.shape[0]),
        "fused_person_present": fused_present,
        "fused_person_points": person_fused.point_count if person_fused else 0,
        "fused_centroid_m": person_fused.centroid_m if person_fused else None,
        "fused_extent_m": person_fused.extent_m if person_fused else None,
        "fused_track_present": fused_track_present,
        "fused_track_confidence": (
            float(fused_track.confidence) if fused_track is not None else 0.0
        ),
        "fused_track_cluster_points": (
            int(fused_selected["count"]) if fused_selected is not None else 0
        ),
        "a_person_present": a_present,
        "b_person_present": b_present,
        "either_person_present": bool(a_present or b_present),
        "both_person_present": bool(a_present and b_present),
        "a_perception_track_present": bool(session_a.window_present[i_a]),
        "b_perception_track_present": bool(session_b.window_present[i_b]),
        "a_person_points": person_a.point_count if person_a else 0,
        "b_person_points": person_b.point_count if person_b else 0,
        "a_region_points": int(region_a.shape[0]),
        "b_region_points": int(region_b.shape[0]),
        "a_screening_state": _window_state_at(session_a, i_a),
        "b_screening_state": _window_state_at(session_b, i_b),
        "a_anomaly_centers": anomaly_centers(
            _window_anomalies_at(session_a, i_a), transform=False
        ).tolist(),
        "b_anomaly_centers": anomaly_centers(
            _window_anomalies_at(session_b, i_b), transform=True, distance_m=distance_m
        ).tolist(),
        "a_anomaly_count": len(_window_anomalies_at(session_a, i_a)),
        "b_anomaly_count": len(_window_anomalies_at(session_b, i_b)),
    }
    return row, pts_a, pts_b, fused, fused_track


def run_fusion(
    session_a: SensorSession,
    session_b: SensorSession,
    distance_m: float,
    clock_offset_b_minus_a_s: float,
    window_tolerance_s: float,
    clutter_a: dict[str, np.ndarray] | None = None,
    clutter_b: dict[str, np.ndarray] | None = None,
    person_min_points: int = 15,
) -> dict[str, Any]:
    spec = PerceptionSpec()
    fused_track: Any = None
    rows: list[dict[str, Any]] = []
    for i_a, i_b in matched_window_pairs(
        session_a,
        session_b,
        clock_offset_b_minus_a_s,
        window_tolerance_s,
    ):
        row, _, _, _, fused_track = _fuse_window(
            session_a,
            session_b,
            i_a,
            i_b,
            distance_m,
            clutter_a,
            clutter_b,
            person_min_points,
            spec,
            fused_track,
        )
        rows.append(row)

    n = len(rows)
    if n == 0:
        return {"matched_window_pairs": 0}

    fused_present = [r["fused_person_present"] for r in rows]
    a_present = [r["a_person_present"] for r in rows]
    b_present = [r["b_person_present"] for r in rows]

    fused_count = [r["fused_person_points"] for r in rows]
    fused_extent = [r["fused_extent_m"] for r in rows if r["fused_extent_m"]]
    fused_centroid = [r["fused_centroid_m"] for r in rows if r["fused_centroid_m"]]

    jumps = []
    for i in range(1, len(fused_centroid)):
        dx = np.asarray(fused_centroid[i]) - np.asarray(fused_centroid[i - 1])
        jumps.append(float(np.linalg.norm(dx)))
    track_continuity_s = float(np.median(jumps)) if jumps else 0.0

    return {
        "schema_version": "scanu_lab_dual_point_cloud_fusion_v1",
        "experimental": True,
        "not_a_ground_truth_accuracy_measurement": True,
        "geometry": {
            "sensor_distance_m": distance_m,
            "sensor_height_m": 1.016,
            "transform_b_to_a": "x'=-x, y'=D-y, z'=z",
        },
        "clock_offset_b_minus_a_s_applied": clock_offset_b_minus_a_s,
        "window_match_tolerance_s": window_tolerance_s,
        "matched_window_pairs": n,
        "fused_person_present_fraction": float(np.mean(fused_present)),
        "a_person_present_fraction": float(np.mean(a_present)),
        "b_person_present_fraction": float(np.mean(b_present)),
        "either_person_present_fraction": float(np.mean([a or b for a, b in zip(a_present, b_present)])),
        "fused_when_either_failed": int(
            sum(1 for r in rows if r["fused_person_present"] and not r["either_person_present"])
        ),
        "either_when_fused_failed": int(
            sum(1 for r in rows if r["either_person_present"] and not r["fused_person_present"])
        ),
        "median_fused_person_points": float(np.median(fused_count)) if fused_count else 0.0,
        "median_best_single_person_points": float(
            np.median([max(r["a_person_points"], r["b_person_points"]) for r in rows])
        ) if rows else 0.0,
        "fusion_point_gain_median_ratio": float(
            np.median(
                [
                    (r["fused_person_points"] / max(r["a_person_points"], r["b_person_points"], 1))
                    for r in rows
                    if r["fused_person_points"] > 0 and max(r["a_person_points"], r["b_person_points"]) > 0
                ]
            )
        ) if rows else 0.0,
        "fused_denser_than_best_single_fraction": float(
            np.mean(
                [
                    r["fused_person_points"] > max(r["a_person_points"], r["b_person_points"])
                    for r in rows
                ]
            )
        ) if rows else 0.0,
        "median_fused_extent_m": [
            float(np.median([e[0] for e in fused_extent])),
            float(np.median([e[1] for e in fused_extent])),
            float(np.median([e[2] for e in fused_extent])),
        ] if fused_extent else None,
        "median_centroid_m": [
            float(np.median([c[0] for c in fused_centroid])),
            float(np.median([c[1] for c in fused_centroid])),
            float(np.median([c[2] for c in fused_centroid])),
        ] if fused_centroid else None,
        "track_continuity_median_jump_m": track_continuity_s,
        "fused_track_present_fraction": float(
            np.mean([r["fused_track_present"] for r in rows])
        ),
        "a_perception_track_present_fraction": float(
            np.mean([r["a_perception_track_present"] for r in rows])
        ),
        "b_perception_track_present_fraction": float(
            np.mean([r["b_perception_track_present"] for r in rows])
        ),
        "fused_track_when_both_perception_failed": int(
            sum(
                1
                for r in rows
                if r["fused_track_present"]
                and not r["a_perception_track_present"]
                and not r["b_perception_track_present"]
            )
        ),
        "both_perception_when_fused_track_failed": int(
            sum(
                1
                for r in rows
                if r["a_perception_track_present"]
                and r["b_perception_track_present"]
                and not r["fused_track_present"]
            )
        ),
        "median_fused_track_cluster_points": float(
            np.median([r["fused_track_cluster_points"] for r in rows])
        ) if rows else 0.0,
        "median_fused_track_confidence": float(
            np.median([r["fused_track_confidence"] for r in rows])
        ) if rows else 0.0,
        "anomaly_windows_a": int(sum(1 for r in rows if r["a_screening_state"] == "suspicious_metal")),
        "anomaly_windows_b": int(sum(1 for r in rows if r["b_screening_state"] == "suspicious_metal")),
        "anomaly_windows_either": int(
            sum(
                1
                for r in rows
                if r["a_screening_state"] == "suspicious_metal"
                or r["b_screening_state"] == "suspicious_metal"
            )
        ),
        "anomaly_windows_both": int(
            sum(
                1
                for r in rows
                if r["a_screening_state"] == "suspicious_metal"
                and r["b_screening_state"] == "suspicious_metal"
            )
        ),
        "total_anomaly_detections_a": int(sum(r["a_anomaly_count"] for r in rows)),
        "total_anomaly_detections_b": int(sum(r["b_anomaly_count"] for r in rows)),
        "limitations": [
            "descriptive point-cloud fusion of two post-CFAR single-radar pipelines; not raw-return coherent fusion",
            "rigid transform uses measured sensor distance and equal height; residual yaw/height misalignment is not modeled",
            "views are complementary (front/back), not redundant; fused counts are not an accuracy metric",
            "fused_track_* reuses the single-sensor perception tracker (PerceptionSpec, DBSCAN + association) on the fused cloud, so it is directly comparable to each sensor's perception track",
            "the fused tracker holds a single track and associates by gate; it can lock onto the strongest (e.g. far-wall) cluster and miss the person in windows where a single sensor's own tracker succeeds",
            "per-window voxel presence (a/b/fused_person_present) is not the temporal track and is expected to be lower than the perception track presence",
            "no controlled ground-truth label in this capture; fused person presence is a description, not detection accuracy",
            "cross-host time alignment depends on the externally supplied clock_offset_b_minus_a_s",
        ],
        "windows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Point-cloud fusion of two facing AWR1843 sensors")
    parser.add_argument("--session-a", required=True, type=Path)
    parser.add_argument("--session-b", required=True, type=Path)
    parser.add_argument("--calibration-a", type=Path, default=None)
    parser.add_argument("--calibration-b", type=Path, default=None)
    parser.add_argument("--distance-m", required=True, type=float)
    parser.add_argument("--clock-offset-b-minus-a-s", type=float, default=0.0)
    parser.add_argument("--window-tolerance-s", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    session_a = load_session(args.session_a)
    session_b = load_session(args.session_b)
    report = run_fusion(
        session_a,
        session_b,
        distance_m=args.distance_m,
        clock_offset_b_minus_a_s=args.clock_offset_b_minus_a_s,
        window_tolerance_s=args.window_tolerance_s,
        clutter_a=(
            _load_clutter_explicit(args.calibration_a)
            if args.calibration_a is not None
            else _load_clutter(args.session_a)
        ),
        clutter_b=(
            _load_clutter_explicit(args.calibration_b)
            if args.calibration_b is not None
            else _load_clutter(args.session_b)
        ),
    )
    with args.output.open("w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
