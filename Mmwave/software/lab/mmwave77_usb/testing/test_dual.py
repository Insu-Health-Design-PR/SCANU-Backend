from __future__ import annotations

import json

import numpy as np

from lab.mmwave77_usb.dual import _build_overlay


def _cube(path, count: int, frame: int) -> None:
    np.savez_compressed(
        path,
        hit_count=np.full((1, 2, 2, 2), count, dtype=np.uint16),
        range_edges_m=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        range_centers_m=np.array([0.5, 1.5], dtype=np.float32),
        azimuth_edges_deg=np.array([-10.0, 0.0, 10.0], dtype=np.float32),
        azimuth_centers_deg=np.array([-5.0, 5.0], dtype=np.float32),
        elevation_edges_deg=np.array([-4.0, 0.0, 4.0], dtype=np.float32),
        elevation_centers_deg=np.array([-2.0, 2.0], dtype=np.float32),
        frame_start=np.array([frame], dtype=np.uint32),
        frame_end=np.array([frame + 9], dtype=np.uint32),
    )


def test_build_overlay_preserves_both_sensors_and_marks_uncalibrated(tmp_path):
    cube_a = tmp_path / "a.npz"
    cube_b = tmp_path / "b.npz"
    output = tmp_path / "dual.npz"
    _cube(cube_a, 2, 1)
    _cube(cube_b, 3, 4)

    overlay, metadata_path, metadata = _build_overlay(cube_a, cube_b, output)

    with np.load(overlay) as result:
        assert np.all(result["hit_count_a"] == 2)
        assert np.all(result["hit_count_b"] == 3)
        assert np.all(result["hit_count_overlay"] == 5)
        assert result["frame_start_a"].tolist() == [1]
        assert result["frame_start_b"].tolist() == [4]
    assert metadata["extrinsic_calibrated"] is False
    assert metadata["fusion_mode"] == "identity_overlay_comparison_only"
    assert json.loads(metadata_path.read_text())["windows"] == 1
