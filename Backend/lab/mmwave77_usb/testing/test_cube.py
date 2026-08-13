from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from lab.mmwave77_usb.cube import CubeSpec, build_session_cube, build_sparse_cube


def _frame(number: int, points: list[dict]) -> dict:
    return {"parse_ok": True, "frame_number": number, "points": points}


def test_sparse_cube_maps_range_azimuth_and_elevation():
    frames = [
        _frame(
            10,
            [
                {"x": 1.0, "y": 1.0, "z": 1.0, "doppler": -0.5, "snr": 12.0},
                {"x": 0.0, "y": 2.0, "z": 0.0, "doppler": 0.25, "snr": 8.0},
            ],
        ),
        _frame(
            11,
            [{"x": 1.0, "y": 1.0, "z": 1.0, "doppler": 0.5, "snr": 16.0}],
        ),
    ]
    spec = CubeSpec(
        range_min_m=0.0,
        range_max_m=4.0,
        azimuth_min_deg=-90.0,
        azimuth_max_deg=90.0,
        elevation_min_deg=-45.0,
        elevation_max_deg=45.0,
        range_bins=4,
        azimuth_bins=4,
        elevation_bins=3,
        window_frames=2,
        stride_frames=1,
    )

    arrays, stats = build_sparse_cube(frames, spec)

    assert arrays["hit_count"].shape == (1, 4, 4, 3)
    assert int(arrays["hit_count"].sum()) == 3
    assert stats["accepted_point_observations"] == 3
    repeated = np.argwhere(arrays["hit_count"] == 2)
    assert repeated.shape == (1, 4)
    voxel = tuple(repeated[0])
    assert arrays["snr_mean_db"][voxel] == pytest.approx(14.0)
    assert arrays["doppler_mean_mps"][voxel] == pytest.approx(0.0)
    assert arrays["doppler_abs_max_mps"][voxel] == pytest.approx(0.5)
    assert arrays["frame_start"].tolist() == [10]
    assert arrays["frame_end"].tolist() == [11]


def test_sparse_cube_rejects_invalid_and_out_of_bounds_points():
    frames = [
        _frame(
            1,
            [
                {"x": "bad", "y": 1, "z": 0},
                {"x": 20, "y": 0, "z": 0},
                {"x": 0, "y": 1, "z": 0, "doppler": 0, "snr": 5},
            ],
        )
    ]
    arrays, stats = build_sparse_cube(frames, CubeSpec())

    assert int(arrays["hit_count"].sum()) == 1
    assert stats["dropped_nonfinite_points"] == 1
    assert stats["dropped_out_of_bounds_points"] == 1


def test_session_cube_writes_auditable_npz_and_metadata(tmp_path):
    session = tmp_path / "capture"
    session.mkdir()
    frames_path = session / "frames.jsonl"
    valid = _frame(
        4, [{"x": 0.0, "y": 1.0, "z": 0.25, "doppler": 0.1, "snr": 9.0}]
    )
    frames_path.write_text(
        json.dumps(valid) + "\n" + json.dumps({"parse_ok": False}) + "\n"
    )

    output, metadata_path, metadata = build_session_cube(
        session, None, CubeSpec(window_frames=1)
    )

    assert output.is_file()
    assert metadata_path.is_file()
    assert metadata["dense_adc_cube"] is False
    assert metadata["canonical_training_compatible"] is False
    assert metadata["statistics"]["rejected_frame_rows"] == 1
    assert metadata["source_frames_sha256"] == hashlib.sha256(
        frames_path.read_bytes()
    ).hexdigest()
    with np.load(output) as cube:
        assert cube["hit_count"].shape == (1, 64, 48, 24)
        assert int(cube["hit_count"].sum()) == 1

    with pytest.raises(ValueError, match="already exists"):
        build_session_cube(session, None, CubeSpec(window_frames=1))
