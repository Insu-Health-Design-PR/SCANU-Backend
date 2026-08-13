from __future__ import annotations

import numpy as np
import pytest

from lab.mmwave77_usb.heatmap_video import sparse_heatmaps
from lab.mmwave77_usb.metal_like import MetalLikeSpec, score_metal_like
from lab.mmwave77_usb.metal_video import _raw_window_points


def test_high_snr_persistent_voxel_scores_above_weak_voxel():
    hits = np.zeros((1, 1, 2, 2), dtype=np.uint16)
    snr = np.zeros_like(hits, dtype=np.float32)
    hits[0, 0, 0, 0] = 7
    snr[0, 0, 0, 0] = 25.0
    hits[0, 0, 1, 1] = 2
    snr[0, 0, 1, 1] = 13.0

    score, candidate = score_metal_like(hits, snr, MetalLikeSpec())

    assert score[0, 0, 0, 0] > score[0, 0, 1, 1]
    assert candidate[0, 0, 0, 0]
    assert not candidate[0, 0, 1, 1]


def test_ineligible_voxels_are_zero_even_with_high_snr():
    hits = np.array([[[[0, 1]]]], dtype=np.uint16)
    snr = np.array([[[[40.0, 40.0]]]], dtype=np.float32)

    score, candidate = score_metal_like(hits, snr, MetalLikeSpec(min_hits=2))

    assert np.all(score == 0)
    assert not np.any(candidate)


def test_raw_points_map_back_to_scored_voxels():
    frames = {
        4: [{"x": 0.0, "y": 1.0, "z": 0.0}],
        5: [{"x": 1.0, "y": 1.0, "z": 0.0}],
    }
    score = np.zeros((2, 2, 2), dtype=np.float32)
    candidate = np.zeros((2, 2, 2), dtype=bool)
    score[0, 1, 1] = 0.9
    candidate[0, 1, 1] = True

    points = _raw_window_points(
        frames,
        4,
        5,
        np.array([0.0, 1.5, 3.0]),
        np.array([-90.0, 0.0, 90.0]),
        np.array([-45.0, 0.0, 45.0]),
        score,
        candidate,
    )

    assert len(points["x"]) == 2
    assert points["candidate"].tolist() == [True, True]
    assert points["score"].tolist() == pytest.approx([0.9, 0.9])


def test_sparse_heatmaps_are_normalized_and_keep_axis_order():
    hits = np.zeros((4, 5, 3), dtype=np.uint16)
    snr = np.zeros_like(hits, dtype=np.float32)
    candidates = np.zeros_like(hits, dtype=bool)
    hits[2, 3, 1] = 4
    snr[2, 3, 1] = 18.0
    candidates[2, 3, 1] = True

    ra, ae, ra_candidates, ae_candidates = sparse_heatmaps(
        hits, snr, candidates, sigma_bins=0.75
    )

    assert ra.shape == (4, 5)
    assert ae.shape == (3, 5)
    assert ra_candidates.shape == ra.shape
    assert ae_candidates.shape == ae.shape
    assert float(ra.max()) == 1.0
    assert float(ae.max()) == 1.0
    assert np.unravel_index(int(np.argmax(ra)), ra.shape) == (2, 3)
    assert np.unravel_index(int(np.argmax(ae)), ae.shape) == (1, 3)


def test_sparse_heatmaps_remain_zero_without_detections():
    hits = np.zeros((2, 3, 4), dtype=np.uint16)
    maps = sparse_heatmaps(
        hits,
        np.zeros_like(hits, dtype=np.float32),
        np.zeros_like(hits, dtype=bool),
    )
    assert all(not np.any(values) for values in maps)
