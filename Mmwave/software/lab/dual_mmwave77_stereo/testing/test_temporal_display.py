import numpy as np

from lab.dual_mmwave77_stereo.dual_fusion_video import _track_aligned_temporal_voxels
from lab.dual_mmwave77_stereo.unified_fusion_video import (
    _off_core_anomalies,
    _source_track_mask,
    _transform_track_b_to_a,
)


def _window(center, points):
    return {
        "row": {"fused_centroid_m": center, "fused_extent_m": [1.0, 1.0, 1.0]},
        "fused": np.asarray(points, dtype=np.float32),
    }


def test_temporal_voxels_translate_previous_track_points_to_current_centroid():
    # The person translated 1 m along y.  Both observations are body-relative
    # point returns and must accumulate near the current position, not smear.
    windows = [
        _window([0.0, 1.0, 0.0], [[0.0, 1.0, 0.0, 0, 0, 0]]),
        _window([0.0, 2.0, 0.0], [[0.0, 2.0, 0.0, 0, 0, 0]]),
    ]
    xyz, weights = _track_aligned_temporal_voxels(
        windows, 1, history_windows=1, voxel_m=0.1
    )
    assert len(xyz) == 1
    assert np.allclose(xyz[0], [0.05, 2.05, 0.05])
    assert weights[0] == 1.5


def test_temporal_voxels_fail_closed_without_current_track():
    xyz, weights = _track_aligned_temporal_voxels(
        [{"row": {"fused_centroid_m": None, "fused_extent_m": None}, "fused": np.empty((0, 6))}],
        0,
        history_windows=1,
        voxel_m=0.1,
    )
    assert xyz.shape == (0, 3)
    assert weights.shape == (0,)


def test_source_track_mask_excludes_scene_returns():
    points = np.asarray(
        [
            [0.0, 1.5, 0.0, 0.0, 10.0, 2.0],
            [1.8, 3.2, 1.4, 0.0, 10.0, 2.0],
        ],
        dtype=np.float32,
    )
    track = {
        "position_m": [0.0, 1.5, 0.0],
        "observed_extent_m": [0.6, 0.5, 1.0],
    }
    assert _source_track_mask(points, track).tolist() == [True, False]


def test_sensor_b_track_uses_same_facing_transform_as_points():
    track = {
        "position_m": [0.2, 2.1, -0.1],
        "observed_extent_m": [0.7, 0.4, 0.9],
    }
    transformed = _transform_track_b_to_a(track, 3.6576)
    assert transformed is not None
    assert np.allclose(transformed["position_m"], [-0.2, 1.5576, -0.1])
    assert transformed["observed_extent_m"] == [0.7, 0.4, 0.9]


def test_persistent_torso_core_reflection_is_not_colored():
    centers, anomalies = _off_core_anomalies(
        [[0.0, 1.6, -0.1]],
        [{
            "persistent_body_associated_anomaly": True,
            "reflective_anomaly_score": 0.92,
            "relative_to_track_m": [0.04, -0.03, 0.06],
        }],
    )
    assert centers == []
    assert anomalies == []


def test_strong_transient_raised_hand_reflection_remains_visible():
    anomaly = {
        "persistent_body_associated_anomaly": False,
        "reflective_anomaly_score": 0.80,
        "relative_to_track_m": [0.48, 0.02, 0.42],
    }
    centers, anomalies = _off_core_anomalies(
        [[0.5, 1.6, 0.35]], [anomaly]
    )
    assert centers == [[0.5, 1.6, 0.35]]
    assert anomalies == [anomaly]


def test_weak_transient_off_core_reflection_is_suppressed():
    centers, anomalies = _off_core_anomalies(
        [[0.5, 1.6, 0.35]],
        [{
            "persistent_body_associated_anomaly": False,
            "reflective_anomaly_score": 0.60,
            "relative_to_track_m": [0.48, 0.02, 0.42],
        }],
    )
    assert centers == []
    assert anomalies == []
