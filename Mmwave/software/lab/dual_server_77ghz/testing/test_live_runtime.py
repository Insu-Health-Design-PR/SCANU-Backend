from __future__ import annotations

import numpy as np

from lab.dual_mmwave77_stereo.live_fusion import LiveFusionEngine, OnlineClutterCalibration
from lab.dual_server_77ghz.live_acquisition import points_from_rows


def _calibration() -> OnlineClutterCalibration:
    calibration = OnlineClutterCalibration(minimum_frames=2)
    calibration.update(np.zeros((0, 4), dtype=np.float32))
    calibration.update(np.zeros((0, 4), dtype=np.float32))
    calibration.finalize()
    assert calibration.valid
    return calibration


def test_points_from_rows_preserves_measured_xyz_snr() -> None:
    rows = [{"points": [{"x": 1, "y": 2, "z": 3, "snr": 17.5}]}]
    points = points_from_rows(rows)
    assert points.shape == (1, 4)
    assert points.tolist() == [[1.0, 2.0, 3.0, 17.5]]


def test_dual_views_create_one_global_track() -> None:
    rng = np.random.default_rng(4)
    distance_m = 3.6576
    world = np.array([0.2, 1.8, 0.0], dtype=np.float32)
    xyz_a = world + rng.normal(0, 0.04, size=(20, 3)).astype(np.float32)
    xyz_b_world = world + rng.normal(0, 0.04, size=(20, 3)).astype(np.float32)
    xyz_b_local = xyz_b_world.copy()
    xyz_b_local[:, 0] *= -1
    xyz_b_local[:, 1] = distance_m - xyz_b_local[:, 1]
    points_a = np.column_stack([xyz_a, np.full(20, 14.0, dtype=np.float32)])
    points_b = np.column_stack([xyz_b_local, np.full(20, 15.0, dtype=np.float32)])
    engine = LiveFusionEngine(
        sensor_distance_m=distance_m,
        calibration_a=_calibration(),
        calibration_b=_calibration(),
        calibration_id="test-calibration",
    )
    frame = engine.update(points_a, points_b, frames_a=20, frames_b=20)
    active = [track for track in frame.tracks if track.missed_windows == 0]
    assert len(active) == 1
    assert active[0].source_views == ["A", "B"]
    assert len(frame.fused_points) == 40
    assert frame.quality.calibration_valid
