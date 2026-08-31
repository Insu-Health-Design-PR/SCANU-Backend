from __future__ import annotations

import numpy as np

from lab.mmwave77_usb.handheld_point_cloud_video import (
    RaisedHandSpec,
    _persistent_mask,
    person_and_raised_hand_masks,
)


def _track() -> dict:
    return {
        "position_m": [0.0, 2.0, 0.0],
        "observed_extent_m": [0.6, 0.5, 1.2],
    }


def _points() -> np.ndarray:
    torso = [[0.05 * (index % 3), 2.0, -0.2 + 0.1 * (index % 4), 0.0, 14.0, 10.0] for index in range(12)]
    raised_hand = [[0.34, 2.03, 0.78, 0.0, 31.0, 10.0], [0.39, 2.00, 0.80, 0.0, 30.0, 10.0]]
    return np.asarray(torso + raised_hand, dtype=np.float32)


def test_raised_hand_roi_exposes_compact_high_reflectivity_returns():
    person, candidate = person_and_raised_hand_masks(_points(), _track(), RaisedHandSpec())

    assert person[-2:].all()
    assert candidate[-2:].all()
    assert not candidate[:-2].any()


def test_red_requires_temporal_persistence_not_one_high_snr_window():
    points = _points()
    spec = RaisedHandSpec(persistence_windows=2)
    _, candidate = person_and_raised_hand_masks(points, _track(), spec)
    first, tracks = _persistent_mask(points, candidate, _track(), [], 0, spec)
    second, _ = _persistent_mask(points, candidate, _track(), tracks, 1, spec)

    assert not first.any()
    assert second[-2:].all()
