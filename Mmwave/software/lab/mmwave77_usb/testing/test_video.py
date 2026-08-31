from __future__ import annotations

import numpy as np

from lab.mmwave77_usb.video import _occupied_cartesian


def test_occupied_cartesian_converts_cube_centers():
    hits = np.zeros((2, 3, 3), dtype=np.uint16)
    hits[1, 2, 1] = 4
    points = _occupied_cartesian(
        hits,
        np.array([1.0, 2.0]),
        np.array([-45.0, 0.0, 90.0]),
        np.array([-30.0, 0.0, 30.0]),
    )

    assert points["count"].tolist() == [4.0]
    assert points["elevation"].tolist() == [0.0]
    assert points["x"][0] == np.float32(2.0)
    assert abs(float(points["y"][0])) < 1e-6
    assert points["z"][0] == np.float32(0.0)


def test_occupied_cartesian_handles_empty_window():
    points = _occupied_cartesian(
        np.zeros((1, 1, 1), dtype=np.uint16),
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
    )
    assert all(value.size == 0 for value in points.values())
