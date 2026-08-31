"""Synthetic checks for global, deduplicated dual-radar tracks."""
from __future__ import annotations

import numpy as np

from lab.dual_mmwave77_stereo.global_tracks import run_global_tracking
from lab.dual_mmwave77_stereo.point_cloud_fusion import SensorSession


D = 3.6576


def _cloud(centers: list[tuple[float, float, float]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = []
    for x, y, z in centers:
        point = np.zeros((28, 6), dtype=np.float32)
        point[:, :3] = rng.normal((x, y, z), (0.06, 0.08, 0.15), (28, 3))
        point[:, 4] = 16.0
        groups.append(point)
    return np.concatenate(groups) if groups else np.empty((0, 6), dtype=np.float32)


def _session(windows: list[np.ndarray], anomalies: list[list[dict]] | None = None) -> SensorSession:
    n = len(windows)
    return SensorSession(
        frames_utc_ns=[1_700_000_000_000_000_000 + index * 100_000_000 for index in range(n)],
        frame_numbers=list(range(1, n + 1)),
        frames_monotonic_ns=list(range(n)),
        points=windows,
        window_index=list(range(n)),
        window_frame_start=list(range(1, n + 1)),
        window_time_s=[index * 0.25 for index in range(n)],
        window_present=[True] * n,
        window_cluster_points=[28] * n,
        window_input_points=[28] * n,
        window_state=["person"] * n,
        window_anomalies=anomalies or [[] for _ in range(n)],
    )


def test_same_person_from_facing_views_has_one_global_id() -> None:
    # B local coordinates invert x/y before mapping to A's world frame.
    a = _session([_cloud([(0.20, 1.80, 0.0)], 1), _cloud([(0.25, 1.82, 0.0)], 2)])
    b = _session([_cloud([(-0.20, D - 1.80, 0.0)], 3), _cloud([(-0.25, D - 1.82, 0.0)], 4)])
    report = run_global_tracking(a, b, distance_m=D, min_cluster_points=10)
    assert report["maximum_simultaneous_global_person_count"] == 1
    assert report["unique_global_track_ids"] == ["G001"]
    assert report["windows"][0]["observations"][0]["source_views"] == ("A", "B")


def test_two_people_are_not_collapsed_into_one_track() -> None:
    world = [(-0.7, 1.4, 0.0), (0.75, 2.2, 0.0)]
    a = _session([_cloud(world, 10)])
    b = _session([_cloud([(-x, D - y, z) for x, y, z in world], 11)])
    report = run_global_tracking(a, b, distance_m=D, min_cluster_points=10)
    assert report["maximum_simultaneous_global_person_count"] == 2
    assert report["unique_global_track_ids"] == ["G001", "G002"]


def test_single_view_reflective_event_remains_unverified() -> None:
    a = _session([_cloud([(0.0, 1.8, 0.0)], 20)], [[{"center_m": [0.1, 1.7, 0.1]}]])
    b = _session([_cloud([(0.0, D - 1.8, 0.0)], 21)])
    report = run_global_tracking(a, b, distance_m=D, min_cluster_points=10)
    evidence = report["windows"][0]["reflective_evidence"]
    assert evidence[0]["classification"] == "single_view_reflective_anomaly"
    assert evidence[0]["material_confirmed"] is False
    assert evidence[0]["global_track_id"] == "G001"


def test_matching_reflective_events_become_multiview_evidence_only() -> None:
    a = _session([_cloud([(0.0, 1.8, 0.0)], 30)], [[{"center_m": [0.1, 1.7, 0.1]}]])
    b = _session([_cloud([(0.0, D - 1.8, 0.0)], 31)], [[{"center_m": [-0.1, D - 1.7, 0.1]}]])
    report = run_global_tracking(a, b, distance_m=D, min_cluster_points=10)
    evidence = report["windows"][0]["reflective_evidence"]
    assert evidence[0]["classification"] == "multiview_reflective_anomaly"
    assert evidence[0]["source_views"] == ["A", "B"]
    assert evidence[0]["material_confirmed"] is False
