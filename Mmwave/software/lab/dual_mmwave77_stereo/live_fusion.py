"""Incremental calibrated late fusion for two facing AWR1843 TLV streams."""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from lab.dual_mmwave77_stereo.global_tracks import (
    GlobalTrack,
    _make_observations,
    _track_row,
    _update_tracks,
    _view_clusters,
)
from lab.dual_mmwave77_stereo.point_cloud_fusion import transform_b_to_a
from lab.dual_server_77ghz.live_contracts import (
    FusedLiveFrame,
    GlobalTrackPayload,
    LiveQuality,
    ReflectiveCandidate,
)


@dataclass
class OnlineClutterCalibration:
    """Frame-occupancy voxel baseline built during an empty-room interval."""

    voxel_m: float = 0.10
    occupancy_fraction: float = 0.35
    minimum_frames: int = 40
    frame_count: int = 0
    _counts: Counter[tuple[int, int, int]] = field(default_factory=Counter)
    occupied: set[tuple[int, int, int]] = field(default_factory=set)

    def update(self, points: np.ndarray) -> None:
        self.frame_count += 1
        if len(points) == 0:
            return
        voxels = np.floor(points[:, :3] / self.voxel_m).astype(np.int32)
        self._counts.update({tuple(row.tolist()) for row in voxels})

    def finalize(self) -> None:
        threshold = max(2, int(np.ceil(self.frame_count * self.occupancy_fraction)))
        self.occupied = {voxel for voxel, count in self._counts.items() if count >= threshold}

    @property
    def valid(self) -> bool:
        # A genuinely clean room may contain no persistent post-CFAR voxels.
        # Frame coverage, rather than a non-empty mask, determines validity.
        return self.frame_count >= self.minimum_frames

    def suppress(self, points: np.ndarray) -> np.ndarray:
        if not self.occupied or len(points) == 0:
            return points
        voxels = np.floor(points[:, :3] / self.voxel_m).astype(np.int32)
        keep = np.asarray([tuple(row.tolist()) not in self.occupied for row in voxels], dtype=bool)
        return points[keep]


def _range_profile(points: np.ndarray, bins: int = 128, maximum_m: float = 6.0) -> list[float]:
    if len(points) == 0:
        return [0.0] * bins
    ranges = np.linalg.norm(points[:, :3], axis=1)
    weights = np.maximum(points[:, 3], 1.0) if points.shape[1] > 3 else None
    hist, _ = np.histogram(ranges, bins=bins, range=(0.0, maximum_m), weights=weights)
    return hist.astype(float).tolist()


class LiveFusionEngine:
    """Maintain global identities and calibrated reflective evidence over time."""

    def __init__(
        self,
        *,
        sensor_distance_m: float,
        calibration_a: OnlineClutterCalibration,
        calibration_b: OnlineClutterCalibration,
        calibration_id: str,
        min_cluster_points: int = 8,
        cross_view_gate_m: float = 0.8,
        track_gate_m: float = 1.0,
        max_missed_windows: int = 4,
    ) -> None:
        self.distance_m = float(sensor_distance_m)
        self.calibration_a = calibration_a
        self.calibration_b = calibration_b
        self.calibration_id = calibration_id
        self.min_cluster_points = int(min_cluster_points)
        self.cross_view_gate_m = float(cross_view_gate_m)
        self.track_gate_m = float(track_gate_m)
        self.max_missed_windows = int(max_missed_windows)
        self.tracks: list[GlobalTrack] = []
        self.next_id = 1
        self._candidate_hits: Counter[tuple[int, int, int]] = Counter()

    def _reflective_candidates(
        self,
        points_a: np.ndarray,
        points_b: np.ndarray,
        track_rows: list[dict[str, object]],
    ) -> list[ReflectiveCandidate]:
        source_points = (("A", points_a), ("B", points_b))
        candidates: list[ReflectiveCandidate] = []
        for source, points in source_points:
            if len(points) < 4 or points.shape[1] < 4:
                continue
            snr = points[:, 3]
            median = float(np.median(snr))
            mad = float(np.median(np.abs(snr - median)))
            threshold = median + max(3.0, 2.5 * mad)
            for point in points[snr >= threshold]:
                xyz = point[:3]
                nearest: dict[str, object] | None = None
                nearest_distance = float("inf")
                for track in track_rows:
                    center = np.asarray(track["centroid_m"], dtype=np.float32)
                    distance = float(np.linalg.norm(xyz - center))
                    if distance < nearest_distance:
                        nearest, nearest_distance = track, distance
                if nearest is None or nearest_distance > 1.0:
                    continue
                key = tuple(np.floor(xyz / 0.18).astype(int).tolist())
                self._candidate_hits[key] += 1
                score = min(1.0, 0.55 + 0.025 * max(0.0, float(point[3]) - median))
                candidates.append(
                    ReflectiveCandidate(
                        center_m=[float(value) for value in xyz],
                        score=round(score, 3),
                        source_views=[source],
                        global_track_id=str(nearest["global_track_id"]),
                        persistent=self._candidate_hits[key] >= 3,
                    )
                )
        # Bound the persistence map during indefinite operation.
        if len(self._candidate_hits) > 2048:
            self._candidate_hits = Counter(dict(self._candidate_hits.most_common(1024)))
        return candidates[:32]

    def update(
        self,
        points_a_raw: np.ndarray,
        points_b_raw: np.ndarray,
        *,
        timestamp_ns: int | None = None,
        timestamp_a_ns: int | None = None,
        timestamp_b_ns: int | None = None,
        frames_a: int = 0,
        frames_b: int = 0,
        dropped_frames_a: int = 0,
        dropped_frames_b: int = 0,
    ) -> FusedLiveFrame:
        now_ns = int(timestamp_ns or time.monotonic_ns())
        points_a = self.calibration_a.suppress(points_a_raw)
        points_b_local = self.calibration_b.suppress(points_b_raw)
        points_b = transform_b_to_a(points_b_local, self.distance_m)
        clusters_a = _view_clusters(points_a, self.distance_m, self.min_cluster_points)
        clusters_b = _view_clusters(points_b, self.distance_m, self.min_cluster_points)
        observations = _make_observations(clusters_a, clusters_b, self.cross_view_gate_m)
        self.tracks, observations, self.next_id = _update_tracks(
            self.tracks,
            observations,
            now_ns / 1_000_000_000.0,
            self.track_gate_m,
            self.max_missed_windows,
            self.next_id,
        )
        observation_by_id = {
            str(observation.get("global_track_id")): observation for observation in observations
        }
        track_rows = [_track_row(track) for track in self.tracks]
        payload_tracks: list[GlobalTrackPayload] = []
        for row in track_rows:
            observation = observation_by_id.get(str(row["global_track_id"]), {})
            payload_tracks.append(
                GlobalTrackPayload(
                    global_track_id=str(row["global_track_id"]),
                    centroid_m=[float(v) for v in row["centroid_m"]],
                    velocity_mps=[float(v) for v in row["velocity_mps"]],
                    extent_m=[float(v) for v in observation.get("extent_m", [0.0, 0.0, 0.0])],
                    point_count=int(observation.get("point_count", 0)),
                    source_views=[str(v) for v in row["last_source_views"]],
                    age_windows=int(row["age_windows"]),
                    missed_windows=int(row["missed_windows"]),
                )
            )
        fused = np.concatenate([points_a, points_b], axis=0) if len(points_b) else points_a.copy()
        alignment_ms = None
        if timestamp_a_ns and timestamp_b_ns:
            alignment_ms = abs(timestamp_a_ns - timestamp_b_ns) / 1_000_000.0
        quality = LiveQuality(
            radar_a_ok=frames_a > 0,
            radar_b_ok=frames_b > 0,
            frames_a=frames_a,
            frames_b=frames_b,
            alignment_error_ms=round(alignment_ms, 3) if alignment_ms is not None else None,
            calibration_valid=self.calibration_a.valid and self.calibration_b.valid,
            dropped_frames_a=dropped_frames_a,
            dropped_frames_b=dropped_frames_b,
        )
        candidates = self._reflective_candidates(points_a, points_b, track_rows)
        return FusedLiveFrame(
            timestamp_ns=now_ns,
            points_a=points_a.astype(float).tolist(),
            points_b=points_b.astype(float).tolist(),
            fused_points=fused.astype(float).tolist(),
            tracks=payload_tracks,
            reflective_candidates=candidates,
            quality=quality,
            range_profile_a=_range_profile(points_a_raw),
            range_profile_b=_range_profile(points_b_raw),
            calibration_id=self.calibration_id,
        )
