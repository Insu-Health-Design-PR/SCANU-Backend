import numpy as np

from lab.dual_mmwave77_stereo.dual_fusion_video import (
    _candidate_point_mask,
    _corroborated_reflective_candidates,
    _reported_point_sizes,
)


def _persistent(score: float = 0.8) -> dict:
    return {
        "persistent_body_associated_anomaly": True,
        "reflective_anomaly_score": score,
    }


def test_dual_view_candidate_requires_two_persistent_nearby_returns():
    candidates = _corroborated_reflective_candidates(
        [[0.2, 1.5, -0.1]],
        [_persistent(0.81)],
        [[0.25, 1.52, -0.08]],
        [_persistent(0.77)],
        gate_m=0.10,
    )

    assert len(candidates) == 1
    assert candidates[0]["association_distance_m"] < 0.1
    assert np.allclose(candidates[0]["center_m"], [0.225, 1.51, -0.09])


def test_single_view_or_nonpersistent_candidate_is_not_colored_as_dual_view():
    assert not _corroborated_reflective_candidates(
        [[0.2, 1.5, 0.0]], [_persistent()], [], [], gate_m=0.35
    )
    assert not _corroborated_reflective_candidates(
        [[0.2, 1.5, 0.0]],
        [{"persistent_body_associated_anomaly": False}],
        [[0.2, 1.5, 0.0]],
        [_persistent()],
        gate_m=0.35,
    )


def test_candidate_mask_only_marks_existing_returns_close_to_candidate():
    points = np.asarray(
        [[0.20, 1.50, 0.0, 0, 0, 0], [1.0, 1.0, 0.0, 0, 0, 0]], dtype=np.float32
    )
    mask = _candidate_point_mask(points, [{"center_m": [0.21, 1.50, 0.0]}], radius_m=0.05)
    assert mask.tolist() == [True, False]


def test_reported_point_sizes_remain_small_and_follow_relative_snr():
    points = np.asarray(
        [[0, 0, 0, 0, 5, 0], [0, 0, 0, 0, 15, 0], [0, 0, 0, 0, 35, 0]],
        dtype=np.float32,
    )
    sizes = _reported_point_sizes(points)
    assert sizes.shape == (3,)
    assert np.all(sizes >= 2.0)
    assert np.all(sizes <= 6.5)
    assert sizes[0] < sizes[-1]
