"""Unit tests for the optional person-scale display profile."""

import numpy as np

from lab.dual_mmwave77_stereo.unified_fusion_video import (
    _balanced_density_weights,
    _classic_anomaly_markers,
    _classic_anomaly_scores,
    _classic_track_geometry,
    _human_focus_limits,
    _human_focus_sizes,
    _recent_track_returns,
)


def test_classic_track_geometry_uses_observed_points_without_voxelizing() -> None:
    observed_a = np.asarray(
        [
            [-0.2, 1.4, -0.6, 0, 10, 2],
            [0.2, 1.6, 0.7, 0, 20, 2],
        ],
        dtype=np.float32,
    )
    observed_b = np.asarray(
        [
            [-0.1, 1.5, -0.5, 0, 12, 2],
            [0.1, 1.7, 0.6, 0, 18, 2],
        ],
        dtype=np.float32,
    )

    center, extent = _classic_track_geometry(
        observed_a, observed_b, None, None
    )

    assert center is not None and extent is not None
    assert np.allclose(center, [0.0, 1.55, 0.05], atol=1e-6)
    assert np.all(extent > 0)


def test_classic_anomaly_markers_preserve_transformed_centers_and_persistence() -> None:
    window = {
        "row": {
            "a_anomaly_centers": [[0.2, 1.4, 0.3]],
            "b_anomaly_centers": [[-0.1, 1.6, 0.4]],
        },
        "anomalies_a": [{
            "reflective_anomaly_score": 0.7,
            "persistent_body_associated_anomaly": False,
        }],
        "anomalies_b": [{
            "reflective_anomaly_score": 0.9,
            "persistent_body_associated_anomaly": True,
        }],
    }

    xyz, colors, sizes = _classic_anomaly_markers(window)

    assert np.allclose(xyz, [[0.2, 1.4, 0.3], [-0.1, 1.6, 0.4]])
    assert colors.tolist() == ["#ffc247", "#ff4057"]
    assert sizes[1] > sizes[0]
    assert _classic_anomaly_scores(window) == (0.7, 0.9, 0.9)


def test_human_focus_limits_are_a_fixed_person_scale_crop() -> None:
    limits = _human_focus_limits(np.asarray([1.0, 2.0, 0.0], dtype=np.float32))

    assert np.allclose(limits, ((0.3, 1.7), (1.25, 2.75), (-1.15, 1.15)))


def test_human_focus_sizes_are_bounded_and_snr_ordered() -> None:
    points = np.asarray([[0, 0, 0, 0, 0], [0, 0, 0, 0, 20], [0, 0, 0, 0, 80]], dtype=np.float32)

    sizes = _human_focus_sizes(points)

    assert np.all(sizes >= 12.0)
    assert np.all(sizes <= 28.0)
    assert list(sizes) == [12.0, 20.0, 28.0]


def test_human_focus_density_weight_reduces_the_denser_source_visual_mass() -> None:
    weight_a, weight_b = _balanced_density_weights(20, 80)

    assert weight_a == 1.0
    assert 0.0 < weight_b < weight_a


def test_recent_track_returns_preserves_original_coordinates_without_voxels() -> None:
    a = np.asarray([[0.1, 1.0, 0.2, 0.0, 15.0]], dtype=np.float32)
    b = np.asarray([[0.2, 1.1, 0.3, 0.0, 16.0]], dtype=np.float32)
    windows = [
        {"pts_a": a, "pts_b": b, "track_a": {"position_m": [0.1, 1.0, 0.2], "observed_extent_m": [0.1, 0.1, 0.1]}, "track_b": {"position_m": [0.2, 1.1, 0.3], "observed_extent_m": [0.1, 0.1, 0.1]}},
        {"pts_a": a + np.asarray([0.1, 0, 0, 0, 0], dtype=np.float32), "pts_b": b, "track_a": {"position_m": [0.2, 1.0, 0.2], "observed_extent_m": [0.1, 0.1, 0.1]}, "track_b": {"position_m": [0.2, 1.1, 0.3], "observed_extent_m": [0.1, 0.1, 0.1]}},
    ]

    xyz, age = _recent_track_returns(windows, 1, history_windows=2)

    assert len(xyz) == 4
    assert set(age.tolist()) == {0, 1}
    assert any(np.allclose(point, [0.1, 1.0, 0.2]) for point in xyz)
