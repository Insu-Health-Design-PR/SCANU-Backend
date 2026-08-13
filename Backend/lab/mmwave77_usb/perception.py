#!/usr/bin/env python3
"""Human-centric interpretation of sparse AWR1843 detected-point TLVs.

This module turns windowed points into one experimental human track and
auditable body-associated reflective anomalies.  Scores are engineering
evidence rankings, never material or firearm probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lab.mmwave77_usb.artifacts import sha256_file, ti_configuration_signature


PERCEPTION_SCHEMA = "scanu_lab_awr1843_human_perception_v1"
POINT_FIELDS = ("x", "y", "z", "doppler", "snr", "noise")


@dataclass(frozen=True)
class PerceptionSpec:
    near_field_m: float = 0.5
    maximum_person_range_m: float = 5.0
    person_cluster_radius_m: float = 0.42
    person_cluster_min_points: int = 10
    motion_threshold_mps: float = 0.12
    association_gate_m: float = 1.1
    maximum_missed_windows: int = 4
    minimum_track_evidence: float = 0.34
    body_margin_xy_m: float = 0.25
    body_margin_z_m: float = 0.3
    anomaly_cluster_radius_m: float = 0.18
    anomaly_cluster_min_points: int = 2
    anomaly_score_threshold: float = 0.68
    anomaly_min_persistence: int = 3
    initial_empty_windows: int = 0
    startup_clutter_occupancy_threshold: float = 0.6

    def validate(self) -> None:
        positive = (
            ("near_field_m", self.near_field_m),
            ("maximum_person_range_m", self.maximum_person_range_m),
            ("person_cluster_radius_m", self.person_cluster_radius_m),
            ("person_cluster_min_points", self.person_cluster_min_points),
            ("motion_threshold_mps", self.motion_threshold_mps),
            ("association_gate_m", self.association_gate_m),
            ("maximum_missed_windows", self.maximum_missed_windows),
            ("body_margin_xy_m", self.body_margin_xy_m),
            ("body_margin_z_m", self.body_margin_z_m),
            ("anomaly_cluster_radius_m", self.anomaly_cluster_radius_m),
            ("anomaly_cluster_min_points", self.anomaly_cluster_min_points),
            ("anomaly_min_persistence", self.anomaly_min_persistence),
        )
        for name, value in positive:
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name, value in (
            ("minimum_track_evidence", self.minimum_track_evidence),
            ("anomaly_score_threshold", self.anomaly_score_threshold),
            (
                "startup_clutter_occupancy_threshold",
                self.startup_clutter_occupancy_threshold,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.near_field_m >= self.maximum_person_range_m:
            raise ValueError("near field must be below maximum person range")
        if self.initial_empty_windows < 0:
            raise ValueError("initial_empty_windows cannot be negative")


@dataclass
class TrackState:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    extent: np.ndarray
    confidence: float
    hits: int
    misses: int
    last_time_s: float


@dataclass
class AnomalyTrack:
    anomaly_id: int
    relative_position: np.ndarray
    hits: int
    last_window: int


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value)))))


def _load_frames(path: Path) -> tuple[dict[int, dict[str, Any]], int]:
    frames: dict[int, dict[str, Any]] = {}
    rejected = 0
    with path.open() as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
            if not row.get("parse_ok") or not isinstance(row.get("points"), list):
                rejected += 1
                continue
            try:
                frame_number = int(row["frame_number"])
            except (KeyError, TypeError, ValueError):
                rejected += 1
                continue
            frames[frame_number] = row
    if not frames:
        raise ValueError(f"no parsed point frames in {path}")
    return frames, rejected


def _point_array(
    frames: dict[int, dict[str, Any]],
    frame_start: int,
    frame_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[list[float]] = []
    frame_identity: list[int] = []
    for frame_number in range(frame_start, frame_end + 1):
        frame = frames.get(frame_number)
        if frame is None:
            continue
        for point in frame["points"]:
            try:
                row = [float(point.get(field, 0.0)) for field in POINT_FIELDS]
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in row):
                values.append(row)
                frame_identity.append(frame_number)
    if not values:
        return (
            np.empty((0, len(POINT_FIELDS)), dtype=np.float32),
            np.empty(0, dtype=np.uint32),
        )
    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(frame_identity, dtype=np.uint32),
    )


def _window_time_s(
    frames: dict[int, dict[str, Any]],
    frame_start: int,
    frame_end: int,
    fallback: float,
) -> float:
    timestamps = []
    for frame_number in range(frame_start, frame_end + 1):
        row = frames.get(frame_number)
        if row is None:
            continue
        try:
            value = int(row["host_monotonic_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        if value >= 0:
            timestamps.append(value)
    return float(np.median(timestamps) / 1e9) if timestamps else fallback


def _dbscan_labels(
    xyz: np.ndarray,
    *,
    radius_m: float,
    min_points: int,
) -> np.ndarray:
    """Small dependency-free DBSCAN for bounded window point sets."""

    coordinates = np.asarray(xyz, dtype=np.float32)
    count = len(coordinates)
    labels = np.full(count, -1, dtype=np.int32)
    if count < min_points:
        return labels
    distance_sq = np.sum(
        np.square(coordinates[:, None, :] - coordinates[None, :, :]), axis=2
    )
    neighbours = distance_sq <= radius_m * radius_m
    core = neighbours.sum(axis=1) >= min_points
    visited = np.zeros(count, dtype=bool)
    cluster_id = 0
    for seed in range(count):
        if visited[seed] or not core[seed]:
            continue
        queue = [seed]
        visited[seed] = True
        members: set[int] = set()
        while queue:
            current = queue.pop()
            nearby = np.flatnonzero(neighbours[current])
            members.update(int(index) for index in nearby)
            if not core[current]:
                continue
            for neighbour in nearby:
                index = int(neighbour)
                if not visited[index] and core[index]:
                    visited[index] = True
                    queue.append(index)
        if len(members) >= min_points:
            labels[list(members)] = cluster_id
            cluster_id += 1
    return labels


def _cluster_features(
    points: np.ndarray,
    indexes: np.ndarray,
    spec: PerceptionSpec,
) -> dict[str, Any]:
    cluster = points[indexes]
    xyz = cluster[:, :3]
    lower = np.percentile(xyz, 5.0, axis=0)
    upper = np.percentile(xyz, 95.0, axis=0)
    extent = np.maximum(upper - lower, 0.01)
    centroid = np.median(xyz, axis=0)
    motion_fraction = float(
        np.mean(np.abs(cluster[:, 3]) >= spec.motion_threshold_mps)
    )
    count_score = _sigmoid((len(cluster) - spec.person_cluster_min_points) / 8.0)
    motion_score = _sigmoid((motion_fraction - 0.08) / 0.05)
    # Sparse elevation prevents reliable full-body height.  This term rewards
    # a distributed observed volume without asserting anatomical dimensions.
    spread_xy = float(math.hypot(extent[0], extent[1]))
    spread_score = math.exp(-abs(spread_xy - 0.65) / 0.9)
    evidence = float(
        np.clip(0.45 * count_score + 0.35 * motion_score + 0.20 * spread_score, 0, 1)
    )
    return {
        "indexes": indexes,
        "centroid": centroid.astype(np.float32),
        "extent": extent.astype(np.float32),
        "count": int(len(cluster)),
        "motion_fraction": motion_fraction,
        "median_abs_doppler_mps": float(np.median(np.abs(cluster[:, 3]))),
        "median_snr_db": float(np.median(cluster[:, 4])),
        "evidence": evidence,
    }


def _person_clusters(
    points: np.ndarray,
    spec: PerceptionSpec,
) -> list[dict[str, Any]]:
    if not len(points):
        return []
    ranges = np.linalg.norm(points[:, :3], axis=1)
    eligible = (
        (ranges >= spec.near_field_m)
        & (ranges <= spec.maximum_person_range_m)
        & (points[:, 1] > 0)
    )
    original_indexes = np.flatnonzero(eligible)
    if len(original_indexes) < spec.person_cluster_min_points:
        return []
    labels = _dbscan_labels(
        points[original_indexes, :3],
        radius_m=spec.person_cluster_radius_m,
        min_points=spec.person_cluster_min_points,
    )
    clusters = []
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        indexes = original_indexes[labels == label]
        clusters.append(_cluster_features(points, indexes, spec))
    return sorted(clusters, key=lambda row: row["evidence"], reverse=True)


def _update_track(
    track: TrackState | None,
    clusters: list[dict[str, Any]],
    *,
    time_s: float,
    spec: PerceptionSpec,
) -> tuple[TrackState | None, dict[str, Any] | None]:
    selected: dict[str, Any] | None = None
    if track is None:
        moving = [
            cluster
            for cluster in clusters
            if cluster["evidence"] >= spec.minimum_track_evidence
            and cluster["motion_fraction"] >= 0.05
        ]
        if moving:
            selected = moving[0]
            return (
                TrackState(
                    track_id=1,
                    position=selected["centroid"].copy(),
                    velocity=np.zeros(3, dtype=np.float32),
                    extent=selected["extent"].copy(),
                    confidence=float(selected["evidence"] * 0.65),
                    hits=1,
                    misses=0,
                    last_time_s=time_s,
                ),
                selected,
            )
        return None, None

    dt = max(time_s - track.last_time_s, 1e-3)
    predicted = track.position + track.velocity * dt
    associated = [
        (
            float(np.linalg.norm(cluster["centroid"] - predicted)),
            -float(cluster["evidence"]),
            cluster,
        )
        for cluster in clusters
        if np.linalg.norm(cluster["centroid"] - predicted) <= spec.association_gate_m
    ]
    if associated:
        _, _, selected = min(associated, key=lambda row: (row[0], row[1]))
        residual = selected["centroid"] - predicted
        alpha = 0.68
        beta = 0.16
        track.position = (predicted + alpha * residual).astype(np.float32)
        track.velocity = (
            track.velocity + (beta / dt) * residual
        ).astype(np.float32)
        track.extent = (
            0.7 * track.extent + 0.3 * selected["extent"]
        ).astype(np.float32)
        track.confidence = float(
            np.clip(0.72 * track.confidence + 0.28 * selected["evidence"] + 0.04, 0, 1)
        )
        track.hits += 1
        track.misses = 0
        track.last_time_s = time_s
        return track, selected

    track.position = predicted.astype(np.float32)
    track.confidence = max(0.0, track.confidence - 0.14)
    track.misses += 1
    track.last_time_s = time_s
    if track.misses > spec.maximum_missed_windows:
        return None, None
    return track, None


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    values = np.asarray(centers, dtype=np.float32).reshape(-1)
    if len(values) < 2:
        raise ValueError("at least two axis centers are required")
    mid = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        (
            [values[0] - (mid[0] - values[0])],
            mid,
            [values[-1] + (values[-1] - mid[-1])],
        )
    ).astype(np.float32)


def _clutter_membership(
    points: np.ndarray,
    clutter_source: dict[str, np.ndarray] | None,
) -> np.ndarray:
    if clutter_source is None or not len(points):
        return np.zeros(len(points), dtype=bool)
    mask = np.asarray(clutter_source["clutter_mask"], dtype=bool)
    if mask.ndim != 3 or not mask.size:
        return np.zeros(len(points), dtype=bool)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    range_m = np.sqrt(x * x + y * y + z * z)
    azimuth = np.degrees(np.arctan2(x, y))
    elevation = np.degrees(np.arctan2(z, np.hypot(x, y)))
    edges = (
        _centers_to_edges(clutter_source["range_centers_m"]),
        _centers_to_edges(clutter_source["azimuth_centers_deg"]),
        _centers_to_edges(clutter_source["elevation_centers_deg"]),
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


def _load_clutter(calibration_session: Path | None) -> dict[str, np.ndarray] | None:
    if calibration_session is None:
        return None
    path = calibration_session / "clutter_mask.npz"
    if not path.is_file():
        raise ValueError(f"calibration has no clutter_mask.npz: {calibration_session}")
    with np.load(path) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _startup_clutter(
    hit_count: np.ndarray,
    *,
    range_centers_m: np.ndarray,
    azimuth_centers_deg: np.ndarray,
    elevation_centers_deg: np.ndarray,
    empty_windows: int,
    occupancy_threshold: float,
    near_field_m: float,
) -> dict[str, np.ndarray] | None:
    """Build an explicitly requested, capture-local startup clutter mask."""

    hits = np.asarray(hit_count)
    if empty_windows <= 0:
        return None
    if hits.ndim != 4 or empty_windows > len(hits):
        raise ValueError(
            f"initial_empty_windows={empty_windows} is incompatible with "
            f"{len(hits) if hits.ndim else 0} cube windows"
        )
    occupancy = (hits[:empty_windows] > 0).mean(axis=0)
    mask = occupancy >= occupancy_threshold
    near = np.asarray(range_centers_m) < near_field_m
    mask |= np.broadcast_to(near[:, None, None], mask.shape)
    return {
        "clutter_mask": mask.astype(np.uint8),
        "range_centers_m": np.asarray(range_centers_m, dtype=np.float32),
        "azimuth_centers_deg": np.asarray(
            azimuth_centers_deg, dtype=np.float32
        ),
        "elevation_centers_deg": np.asarray(
            elevation_centers_deg, dtype=np.float32
        ),
    }


def _load_profile_context(
    session: Path,
    calibration_session: Path | None,
) -> dict[str, Any] | None:
    current_path = session / "range_profiles.npz"
    if calibration_session is None or not current_path.is_file():
        return None
    baseline_path = calibration_session / "empty_room_baseline.npz"
    if not baseline_path.is_file():
        raise ValueError(
            f"calibration has no empty_room_baseline.npz: {calibration_session}"
        )
    with np.load(current_path) as current:
        profiles = np.asarray(current["profiles"], dtype=np.float32)
        lengths = np.asarray(current["profile_lengths"], dtype=np.int64)
        frame_numbers = np.asarray(current["frame_number"], dtype=np.uint32)
    with np.load(baseline_path) as baseline:
        median = np.asarray(baseline["range_median"], dtype=np.float32)
        scale = np.asarray(baseline["range_robust_scale"], dtype=np.float32)
    if profiles.shape[1] != len(median) or len(scale) != len(median):
        raise ValueError("capture and calibration range-profile bins do not match")
    return {
        "profiles": profiles,
        "lengths": lengths,
        "frame_numbers": frame_numbers,
        "median": median,
        "scale": scale,
    }


def _calibration_compatibility(
    session: Path,
    calibration_session: Path | None,
) -> tuple[bool, str]:
    if calibration_session is None:
        return False, "no_external_calibration"
    manifest_path = calibration_session / "calibration_manifest.json"
    if not manifest_path.is_file():
        return False, "calibration_manifest_missing"
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid calibration manifest: {manifest_path}: {exc}") from exc
    expected = manifest.get("ti_configuration_signature")
    current = ti_configuration_signature(session)
    if not expected or not current:
        return False, "configuration_signature_unavailable"
    if str(expected) != str(current):
        raise ValueError(
            "participant and empty-room TI configurations are incompatible"
        )
    return True, "matching_ti_configuration_signature"


def _range_resolution_m(session: Path, profile_bins: int) -> tuple[float, str]:
    configuration_path = session / "configuration.json"
    if configuration_path.is_file():
        try:
            rows = json.loads(configuration_path.read_text())
        except json.JSONDecodeError:
            rows = []
        for row in rows if isinstance(rows, list) else []:
            command = str(row.get("command", ""))
            tokens = command.split()
            if tokens and tokens[0].lower() == "profilecfg" and len(tokens) >= 12:
                try:
                    slope_mhz_us = float(tokens[8])
                    samples = int(tokens[10])
                    sample_rate_ksps = float(tokens[11])
                except ValueError:
                    continue
                if slope_mhz_us > 0 and samples > 0 and sample_rate_ksps > 0:
                    resolution = (
                        299_792_458.0
                        * (sample_rate_ksps * 1e3)
                        / (2.0 * slope_mhz_us * 1e12 * samples)
                    )
                    return float(resolution), "profileCfg"
    return 8.64 / max(profile_bins, 1), "fallback_8.64m_extent"


def _profile_z_for_window(
    context: dict[str, Any] | None,
    frame_start: int,
    frame_end: int,
) -> np.ndarray | None:
    if context is None:
        return None
    mask = (
        (context["frame_numbers"] >= frame_start)
        & (context["frame_numbers"] <= frame_end)
        & (context["lengths"] == len(context["median"]))
    )
    if not np.any(mask):
        return None
    current = np.nanmedian(context["profiles"][mask], axis=0)
    return ((current - context["median"]) / context["scale"]).astype(np.float32)


def _body_region(relative: np.ndarray, extent: np.ndarray) -> str:
    vertical_scale = max(float(extent[2]) * 0.5, 0.1)
    vertical = float(relative[2] / vertical_scale)
    horizontal_scale = max(float(extent[0]) * 0.5, 0.1)
    horizontal = float(relative[0] / horizontal_scale)
    vertical_name = (
        "upper_observed_volume"
        if vertical > 0.35
        else "lower_observed_volume"
        if vertical < -0.35
        else "center_observed_volume"
    )
    side = "right" if horizontal > 0.45 else "left" if horizontal < -0.45 else "center"
    return f"{side}_{vertical_name}"


def _anomalies(
    points: np.ndarray,
    selected_cluster: dict[str, Any] | None,
    track: TrackState,
    *,
    profile_z: np.ndarray | None,
    calibration_compatible: bool,
    range_resolution_m: float,
    anomaly_tracks: list[AnomalyTrack],
    window_index: int,
    spec: PerceptionSpec,
) -> tuple[list[dict[str, Any]], list[AnomalyTrack]]:
    if selected_cluster is None:
        return [], anomaly_tracks
    centroid = track.position
    half_extent = np.maximum(
        0.5 * track.extent
        + np.asarray(
            [spec.body_margin_xy_m, spec.body_margin_xy_m, spec.body_margin_z_m],
            dtype=np.float32,
        ),
        np.asarray([0.3, 0.3, 0.35], dtype=np.float32),
    )
    body_mask = np.all(np.abs(points[:, :3] - centroid) <= half_extent, axis=1)
    body_indexes = np.flatnonzero(body_mask)
    if len(body_indexes) < spec.anomaly_cluster_min_points:
        return [], anomaly_tracks
    # Cluster only stronger-than-local-median body points.  This avoids
    # relabeling the entire torso as an anomaly.
    body_snr = points[body_indexes, 4]
    snr_median = float(np.median(body_snr))
    snr_mad = max(float(np.median(np.abs(body_snr - snr_median))), 1.0)
    strong = body_snr >= snr_median + 1.4826 * snr_mad
    strong_indexes = body_indexes[strong]
    if len(strong_indexes) < spec.anomaly_cluster_min_points:
        return [], anomaly_tracks
    labels = _dbscan_labels(
        points[strong_indexes, :3],
        radius_m=spec.anomaly_cluster_radius_m,
        min_points=spec.anomaly_cluster_min_points,
    )
    next_id = max((item.anomaly_id for item in anomaly_tracks), default=0) + 1
    results: list[dict[str, Any]] = []
    updated_tracks: list[AnomalyTrack] = [
        item for item in anomaly_tracks if window_index - item.last_window <= 3
    ]
    radial_direction = track.position / max(float(np.linalg.norm(track.position)), 1e-6)
    track_radial_velocity = float(np.dot(track.velocity, radial_direction))

    for label in sorted(set(int(value) for value in labels if value >= 0)):
        indexes = strong_indexes[labels == label]
        cluster = points[indexes]
        center = np.median(cluster[:, :3], axis=0)
        relative = center - centroid
        span = np.maximum(
            np.percentile(cluster[:, :3], 95.0, axis=0)
            - np.percentile(cluster[:, :3], 5.0, axis=0),
            0.01,
        )
        median_snr = float(np.median(cluster[:, 4]))
        snr_component = _sigmoid((median_snr - 18.0) / 4.0)
        point_range = float(np.linalg.norm(center))
        profile_bin = int(round(point_range / max(range_resolution_m, 1e-6)))
        if profile_z is not None and 0 <= profile_bin < len(profile_z):
            background_z = float(profile_z[profile_bin])
            background_component = _sigmoid((background_z - 4.0) / 2.0)
            background_available = calibration_compatible
        else:
            background_z = float("nan")
            background_component = 0.5
            background_available = False
        volume = float(np.prod(span))
        compactness_component = float(
            np.clip(math.exp(-volume / 0.025), 0.0, 1.0)
        )
        median_doppler = float(np.median(cluster[:, 3]))
        motion_coupling_component = float(
            math.exp(-abs(median_doppler - track_radial_velocity) / 0.5)
        )

        matching = [
            item
            for item in updated_tracks
            if np.linalg.norm(item.relative_position - relative) <= 0.25
        ]
        if matching:
            anomaly_track = min(
                matching,
                key=lambda item: np.linalg.norm(item.relative_position - relative),
            )
            anomaly_track.relative_position = (
                0.65 * anomaly_track.relative_position + 0.35 * relative
            ).astype(np.float32)
            anomaly_track.hits += 1
            anomaly_track.last_window = window_index
        else:
            anomaly_track = AnomalyTrack(
                anomaly_id=next_id,
                relative_position=relative.astype(np.float32),
                hits=1,
                last_window=window_index,
            )
            updated_tracks.append(anomaly_track)
            next_id += 1
        persistence_component = float(1.0 - math.exp(-anomaly_track.hits / 3.0))
        components = {
            "background_residual": background_component,
            "snr_above_body_context": snr_component,
            "compactness": compactness_component,
            "motion_coupling": motion_coupling_component,
            "temporal_persistence": persistence_component,
        }
        weights = {
            "background_residual": 0.30,
            "snr_above_body_context": 0.25,
            "compactness": 0.20,
            "motion_coupling": 0.15,
            "temporal_persistence": 0.10,
        }
        score = float(
            sum(components[name] * weights[name] for name in components)
        )
        persistent = anomaly_track.hits >= spec.anomaly_min_persistence
        qualifies = bool(
            background_available
            and persistent
            and track.confidence >= 0.5
            and score >= spec.anomaly_score_threshold
        )
        results.append(
            {
                "anomaly_id": anomaly_track.anomaly_id,
                "point_count": int(len(cluster)),
                "center_m": center.round(4).tolist(),
                "relative_to_track_m": relative.round(4).tolist(),
                "extent_m": span.round(4).tolist(),
                "body_region_estimate": _body_region(relative, track.extent),
                "body_region_calibrated": False,
                "median_snr_db": median_snr,
                "median_doppler_mps": median_doppler,
                "range_m": point_range,
                "profile_bin": profile_bin,
                "profile_robust_z": (
                    background_z if background_available else None
                ),
                "background_available": background_available,
                "persistence_windows": anomaly_track.hits,
                "components": components,
                "weights": weights,
                "reflective_anomaly_score": score,
                "persistent_body_associated_anomaly": qualifies,
                "material_confirmed": False,
                "weapon_classification": False,
            }
        )
    return sorted(
        results, key=lambda row: row["reflective_anomaly_score"], reverse=True
    ), updated_tracks


def _track_payload(
    track: TrackState | None,
    selected_cluster: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if track is None:
        return None
    return {
        "track_id": track.track_id,
        "position_m": track.position.round(4).tolist(),
        "velocity_mps": track.velocity.round(4).tolist(),
        "observed_extent_m": track.extent.round(4).tolist(),
        "track_confidence": track.confidence,
        "track_confidence_calibrated_probability": False,
        "hits": track.hits,
        "misses": track.misses,
        "observed_this_window": selected_cluster is not None,
        "cluster_point_count": (
            int(selected_cluster["count"]) if selected_cluster is not None else 0
        ),
        "cluster_motion_fraction": (
            float(selected_cluster["motion_fraction"])
            if selected_cluster is not None
            else 0.0
        ),
    }


def run_perception(
    session: Path,
    *,
    calibration_session: Path | None = None,
    frames_path: Path | None = None,
    cube_path: Path | None = None,
    output_path: Path | None = None,
    spec: PerceptionSpec = PerceptionSpec(),
    overwrite: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Build human tracks and body-associated reflective anomaly evidence."""

    spec.validate()
    session = session.expanduser().resolve()
    calibration_session = (
        calibration_session.expanduser().resolve()
        if calibration_session is not None
        else None
    )
    frames_path = (
        frames_path.expanduser().resolve()
        if frames_path is not None
        else session / "frames.jsonl"
    )
    cube_path = (
        cube_path.expanduser().resolve()
        if cube_path is not None
        else session / "rae_cube_tlv.npz"
    )
    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else session / "perception.jsonl"
    )
    summary_path = output_path.with_name("perception_summary.json")
    if not frames_path.is_file():
        raise ValueError(f"decoded frames do not exist: {frames_path}")
    if not cube_path.is_file():
        raise ValueError(f"sparse cube does not exist: {cube_path}")
    existing = [path for path in (output_path, summary_path) if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            f"perception output already exists: {existing[0]}; "
            "use --overwrite intentionally"
        )

    frames, rejected_rows = _load_frames(frames_path)
    with np.load(cube_path) as cube:
        frame_start = np.asarray(cube["frame_start"], dtype=np.uint32)
        frame_end = np.asarray(cube["frame_end"], dtype=np.uint32)
        startup_clutter = (
            _startup_clutter(
                np.asarray(cube["hit_count"]),
                range_centers_m=np.asarray(cube["range_centers_m"]),
                azimuth_centers_deg=np.asarray(cube["azimuth_centers_deg"]),
                elevation_centers_deg=np.asarray(cube["elevation_centers_deg"]),
                empty_windows=spec.initial_empty_windows,
                occupancy_threshold=spec.startup_clutter_occupancy_threshold,
                near_field_m=spec.near_field_m,
            )
            if spec.initial_empty_windows
            else None
        )
    if len(frame_start) != len(frame_end) or not len(frame_start):
        raise ValueError("cube has invalid temporal windows")

    clutter = _load_clutter(calibration_session)
    calibration_compatible, calibration_compatibility_reason = (
        _calibration_compatibility(session, calibration_session)
    )
    profile_context = _load_profile_context(session, calibration_session)
    profile_bins = (
        len(profile_context["median"]) if profile_context is not None else 256
    )
    range_resolution, range_resolution_source = _range_resolution_m(
        session, profile_bins
    )
    track: TrackState | None = None
    anomaly_tracks: list[AnomalyTrack] = []
    results: list[dict[str, Any]] = []
    track_history: list[list[float]] = []
    suppressed_clutter = 0
    input_points = 0
    persistent_anomaly_windows = 0
    first_time_s: float | None = None

    for window_index, (start, end) in enumerate(zip(frame_start, frame_end)):
        start_int, end_int = int(start), int(end)
        points, _ = _point_array(frames, start_int, end_int)
        input_points += len(points)
        window_time = _window_time_s(
            frames, start_int, end_int, fallback=window_index * 0.5
        )
        if first_time_s is None:
            first_time_s = window_time
        relative_time = window_time - first_time_s
        clutter_mask = _clutter_membership(points, clutter)
        clutter_mask |= _clutter_membership(points, startup_clutter)
        suppressed_clutter += int(np.count_nonzero(clutter_mask))
        perception_points = points[~clutter_mask]
        declared_startup_empty = window_index < spec.initial_empty_windows
        clusters = (
            [] if declared_startup_empty else _person_clusters(perception_points, spec)
        )
        if declared_startup_empty:
            track, selected = None, None
        else:
            track, selected = _update_track(
                track, clusters, time_s=window_time, spec=spec
            )
        profile_z = _profile_z_for_window(
            profile_context, start_int, end_int
        )
        if track is not None:
            track_history.append(track.position.round(4).tolist())
            anomalies, anomaly_tracks = _anomalies(
                perception_points,
                selected,
                track,
                profile_z=profile_z,
                calibration_compatible=calibration_compatible,
                range_resolution_m=range_resolution,
                anomaly_tracks=anomaly_tracks,
                window_index=window_index,
                spec=spec,
            )
        else:
            anomalies = []
            anomaly_tracks = [
                item
                for item in anomaly_tracks
                if window_index - item.last_window <= 3
            ]
        qualifying = [
            row for row in anomalies if row["persistent_body_associated_anomaly"]
        ]
        if qualifying:
            persistent_anomaly_windows += 1
        if declared_startup_empty:
            scene_state = "background"
            screening_state = "background"
        elif track is None:
            scene_state = "background"
            screening_state = (
                "background" if calibration_session is not None else "insufficient_signal"
            )
        else:
            scene_state = "person"
            if qualifying:
                screening_state = "suspicious_metal"
            elif not calibration_compatible or profile_z is None:
                screening_state = "insufficient_signal"
            else:
                screening_state = "person"
        results.append(
            {
                "schema_version": PERCEPTION_SCHEMA,
                "window_index": window_index,
                "frame_start": start_int,
                "frame_end": end_int,
                "time_s": relative_time,
                "input_point_count": int(len(points)),
                "clutter_suppressed_point_count": int(
                    np.count_nonzero(clutter_mask)
                ),
                "perception_point_count": int(len(perception_points)),
                "person_cluster_count": len(clusters),
                "track": _track_payload(track, selected),
                "track_history_m": track_history[-30:],
                "anomalies": anomalies,
                "scene_state": scene_state,
                "screening_state": screening_state,
                "material_confirmed": False,
                "weapon_classification": False,
                "quality": {
                    "background_calibrated": calibration_compatible,
                    "calibration_compatibility_reason": (
                        calibration_compatibility_reason
                    ),
                    "startup_empty_mask_applied": startup_clutter is not None,
                    "profile_residual_available": profile_z is not None,
                    "range_resolution_source": range_resolution_source,
                    "sparse_tlv_only": True,
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        for row in results:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    tracked_windows = sum(row["track"] is not None for row in results)
    observed_track_windows = sum(
        bool(row["track"] and row["track"]["observed_this_window"])
        for row in results
    )
    states = {
        state: sum(row["screening_state"] == state for row in results)
        for state in (
            "background",
            "person",
            "suspicious_metal",
            "insufficient_signal",
        )
    }
    summary = {
        "schema_version": PERCEPTION_SCHEMA,
        "experimental": True,
        "canonical_training_compatible": False,
        "material_confirmed": False,
        "weapon_classification": False,
        "source_session": str(session),
        "source_frames": str(frames_path),
        "source_frames_sha256": sha256_file(frames_path),
        "source_cube": str(cube_path),
        "source_cube_sha256": sha256_file(cube_path),
        "calibration_session": (
            str(calibration_session) if calibration_session is not None else None
        ),
        "background_calibrated": calibration_compatible,
        "calibration_compatibility_reason": calibration_compatibility_reason,
        "startup_empty_windows": spec.initial_empty_windows,
        "startup_empty_mask_applied": startup_clutter is not None,
        "spec": asdict(spec),
        "statistics": {
            "windows": len(results),
            "rejected_frame_rows": rejected_rows,
            "input_windowed_points": input_points,
            "clutter_suppressed_windowed_points": suppressed_clutter,
            "clutter_suppressed_fraction": (
                suppressed_clutter / input_points if input_points else 0.0
            ),
            "tracked_windows": tracked_windows,
            "observed_track_windows": observed_track_windows,
            "track_continuity_fraction": (
                observed_track_windows / tracked_windows
                if tracked_windows
                else 0.0
            ),
            "persistent_anomaly_windows": persistent_anomaly_windows,
            "screening_state_windows": states,
        },
        "range_profile_mapping": {
            "resolution_m": range_resolution,
            "source": range_resolution_source,
        },
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "track confidence and anomaly score are uncalibrated engineering rankings",
            "body region uses the observed radar volume, not calibrated anatomy",
            "sparse post-CFAR TLVs cannot recover dense complex radar imagery",
            "suspicious_metal denotes body-associated radar anomaly, not confirmed material or weapon",
            "person identity requires controlled validation and preferably synchronized camera evidence",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return output_path, summary_path, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track a person and score body-associated AWR1843 reflective anomalies"
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--calibration-session", type=Path)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--cube", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--near-field-m", type=float, default=0.5)
    parser.add_argument("--maximum-person-range-m", type=float, default=5.0)
    parser.add_argument("--person-cluster-radius-m", type=float, default=0.42)
    parser.add_argument("--person-cluster-min-points", type=int, default=10)
    parser.add_argument("--motion-threshold-mps", type=float, default=0.12)
    parser.add_argument("--association-gate-m", type=float, default=1.1)
    parser.add_argument("--anomaly-score-threshold", type=float, default=0.68)
    parser.add_argument(
        "--initial-empty-windows",
        type=int,
        default=0,
        help="explicitly declare leading empty-scene cube windows for local clutter suppression",
    )
    parser.add_argument(
        "--startup-clutter-occupancy-threshold", type=float, default=0.6
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    spec = PerceptionSpec(
        near_field_m=args.near_field_m,
        maximum_person_range_m=args.maximum_person_range_m,
        person_cluster_radius_m=args.person_cluster_radius_m,
        person_cluster_min_points=args.person_cluster_min_points,
        motion_threshold_mps=args.motion_threshold_mps,
        association_gate_m=args.association_gate_m,
        anomaly_score_threshold=args.anomaly_score_threshold,
        initial_empty_windows=args.initial_empty_windows,
        startup_clutter_occupancy_threshold=(
            args.startup_clutter_occupancy_threshold
        ),
    )
    try:
        output, summary_path, summary = run_perception(
            args.session,
            calibration_session=args.calibration_session,
            frames_path=args.frames,
            cube_path=args.cube,
            output_path=args.output,
            spec=spec,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "perception": str(output),
                "summary": str(summary_path),
                **summary["statistics"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
