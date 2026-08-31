from __future__ import annotations

import json

import numpy as np
import pytest

from lab.mmwave77_usb.artifacts import SignalFrameRecord, write_signal_artifacts
from lab.mmwave77_usb.background import (
    BackgroundSpec,
    build_empty_room_baseline,
    range_profile_residual,
    robust_profile_statistics,
)


def _write_profiles(session, count: int = 50) -> None:
    (session / "raw_uart.bin").write_bytes(b"test")
    (session / "configuration.json").write_text(
        json.dumps(
            [
                {"command": "version", "response": "xWR18xx"},
                {
                    "command": "profileCfg 0 77 7 3 39 0 0 100 1 4 7200 0 0 30",
                    "response": "Done",
                },
            ]
        )
    )
    records = []
    for index in range(count):
        variation = float((index % 5) - 2)
        records.append(
            SignalFrameRecord(
                frame_number=index + 1,
                sensor_cycles=index,
                host_monotonic_ns=index,
                host_utc="2026-07-30T00:00:00+00:00",
                range_profile=np.asarray(
                    [100.0 + variation, 200.0, 300.0, 400.0],
                    dtype=np.float32,
                ),
                noise_profile=np.asarray([10, 11, 12, 13], dtype=np.float32),
            )
        )
    write_signal_artifacts(session, records, source="test")


def _write_cube(session) -> None:
    hits = np.zeros((5, 4, 2, 2), dtype=np.uint16)
    snr = np.zeros_like(hits, dtype=np.float32)
    hits[:, 3, 1, 1] = 3
    snr[:, 3, 1, 1] = 18.0
    hits[0, 1, 0, 0] = 2
    snr[0, 1, 0, 0] = 12.0
    np.savez_compressed(
        session / "rae_cube_tlv.npz",
        hit_count=hits,
        snr_mean_db=snr,
        range_centers_m=np.asarray([0.3, 1.0, 2.0, 5.8], dtype=np.float32),
        azimuth_centers_deg=np.asarray([-10.0, 10.0], dtype=np.float32),
        elevation_centers_deg=np.asarray([-5.0, 5.0], dtype=np.float32),
    )


def test_robust_profile_statistics_and_residual_are_range_conditioned():
    profiles = np.asarray(
        [[10.0, 20.0], [10.0, 22.0], [10.0, 18.0]], dtype=np.float32
    )
    stats = robust_profile_statistics(profiles, minimum_scale=1.0)
    residual, zscore = range_profile_residual(
        np.asarray([[12.0, 24.0]], dtype=np.float32),
        stats["median"],
        stats["robust_scale"],
    )

    assert stats["median"].tolist() == [10.0, 20.0]
    assert residual.tolist() == [[2.0, 4.0]]
    assert zscore[0, 0] == pytest.approx(2.0)
    assert zscore[0, 1] > 1.0


def test_empty_room_baseline_writes_profile_and_sparse_clutter_artifacts(tmp_path):
    _write_profiles(tmp_path)
    _write_cube(tmp_path)

    baseline, clutter, quality_path, manifest_path, manifest = (
        build_empty_room_baseline(
            tmp_path,
            condition="empty_room",
            spec=BackgroundSpec(
                min_frames=40,
                near_field_m=0.5,
                static_occupancy_threshold=0.8,
            ),
        )
    )

    assert baseline.is_file()
    assert clutter.is_file()
    assert quality_path.is_file()
    assert manifest_path.is_file()
    assert manifest["condition"] == "empty_room"
    with np.load(baseline) as source:
        assert source["range_median"].tolist() == [100.0, 200.0, 300.0, 400.0]
        assert np.all(source["range_robust_scale"] > 0)
    with np.load(clutter) as source:
        assert bool(source["near_field_mask"][0].all())
        assert bool(source["static_voxel_mask"][3, 1, 1])
        assert not bool(source["static_voxel_mask"][1, 0, 0])
        assert bool(source["persistent_range_mask"][3])
    quality = json.loads(quality_path.read_text())
    assert quality["status"] == "ready"
    assert quality["checks"]["enough_frames"] is True


def test_empty_room_baseline_refuses_uncontrolled_condition(tmp_path):
    _write_profiles(tmp_path)
    with pytest.raises(ValueError, match="empty_room"):
        build_empty_room_baseline(
            tmp_path,
            condition="person_present",
            spec=BackgroundSpec(min_frames=40),
        )
