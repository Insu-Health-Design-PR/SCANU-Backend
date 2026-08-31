from __future__ import annotations

import json

import numpy as np
import pytest

from lab.mmwave77_usb.artifacts import SignalFrameRecord, write_signal_artifacts
from lab.mmwave77_usb.background import BackgroundSpec, build_empty_room_baseline
from lab.mmwave77_usb.perception import (
    PerceptionSpec,
    _dbscan_labels,
    _startup_clutter,
    run_perception,
)
from lab.mmwave77_usb.perception_video import _display_anomalies


def _point(x, y, z, *, doppler=0.2, snr=14.0):
    return {
        "x": x,
        "y": y,
        "z": z,
        "doppler": doppler,
        "snr": snr,
        "noise": 10.0,
    }


def _write_person_session(
    session, windows: int = 7, *, profile_peak: float = 120.0
) -> None:
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
    rows = []
    profile_records = []
    for index in range(windows):
        center_x = -0.1 + 0.02 * index
        points = [
            _point(
                center_x + 0.06 * ((point_index % 4) - 1.5),
                1.5 + 0.04 * ((point_index // 4) - 1),
                -0.2 + 0.12 * (point_index % 3),
            )
            for point_index in range(12)
        ]
        points.extend(
            [
                _point(center_x - 0.02, 1.48, -0.05, snr=40.0),
                _point(center_x + 0.02, 1.49, -0.04, snr=39.0),
            ]
        )
        rows.append(
            {
                "parse_ok": True,
                "frame_number": index + 1,
                "host_monotonic_ns": index * 100_000_000,
                "host_utc": "2026-07-30T00:00:00+00:00",
                "points": points,
            }
        )
        profile_records.append(
            SignalFrameRecord(
                frame_number=index + 1,
                sensor_cycles=index,
                host_monotonic_ns=index * 100_000_000,
                host_utc="2026-07-30T00:00:00+00:00",
                range_profile=np.asarray(
                    [100, profile_peak, 100, 100], dtype=np.float32
                ),
                noise_profile=np.asarray([10, 10, 10, 10], dtype=np.float32),
            )
        )
    (session / "frames.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (session / "raw_uart.bin").write_bytes(b"person")
    write_signal_artifacts(session, profile_records, source="test")
    np.savez_compressed(
        session / "rae_cube_tlv.npz",
        frame_start=np.arange(1, windows + 1, dtype=np.uint32),
        frame_end=np.arange(1, windows + 1, dtype=np.uint32),
    )


def _write_calibration(session) -> None:
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
    (session / "raw_uart.bin").write_bytes(b"calibration")
    records = [
        SignalFrameRecord(
            frame_number=index + 1,
            sensor_cycles=index,
            host_monotonic_ns=index,
            host_utc="2026-07-30T00:00:00+00:00",
            range_profile=np.asarray([100, 100, 100, 100], dtype=np.float32),
            noise_profile=np.asarray([10, 10, 10, 10], dtype=np.float32),
        )
        for index in range(40)
    ]
    write_signal_artifacts(session, records, source="test")
    build_empty_room_baseline(
        session,
        condition="empty_room",
        spec=BackgroundSpec(min_frames=40),
    )


def test_dependency_free_dbscan_separates_spatial_groups():
    xyz = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.05, 1.0, 0.0],
            [0.0, 1.05, 0.0],
            [2.0, 2.0, 0.0],
            [2.05, 2.0, 0.0],
            [2.0, 2.05, 0.0],
        ],
        dtype=np.float32,
    )
    labels = _dbscan_labels(xyz, radius_m=0.1, min_points=3)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1]


def test_explicit_startup_empty_mask_marks_persistent_and_near_field_voxels():
    hits = np.zeros((4, 3, 2, 2), dtype=np.uint16)
    hits[:3, 2, 1, 1] = 1
    hits[0, 1, 0, 0] = 1
    source = _startup_clutter(
        hits,
        range_centers_m=np.asarray([0.3, 1.0, 4.5], dtype=np.float32),
        azimuth_centers_deg=np.asarray([-5.0, 5.0], dtype=np.float32),
        elevation_centers_deg=np.asarray([-5.0, 5.0], dtype=np.float32),
        empty_windows=3,
        occupancy_threshold=0.8,
        near_field_m=0.5,
    )

    assert source is not None
    mask = source["clutter_mask"].astype(bool)
    assert mask[0].all()
    assert mask[2, 1, 1]
    assert not mask[1, 0, 0]


def test_perception_tracks_person_and_requires_baseline_for_screening(tmp_path):
    person_session = tmp_path / "person"
    person_session.mkdir()
    _write_person_session(person_session)

    output, summary_path, summary = run_perception(
        person_session,
        spec=PerceptionSpec(person_cluster_min_points=6),
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary_path.is_file()
    assert summary["statistics"]["tracked_windows"] >= 6
    assert summary["background_calibrated"] is False
    assert any(row["track"] is not None for row in rows)
    assert all(
        row["screening_state"] != "suspicious_metal" for row in rows
    )
    assert any(row["screening_state"] == "insufficient_signal" for row in rows)


def test_calibrated_persistent_body_anomaly_is_auditable_not_weapon(tmp_path):
    person_session = tmp_path / "person"
    calibration_session = tmp_path / "empty"
    person_session.mkdir()
    calibration_session.mkdir()
    _write_person_session(person_session)
    _write_calibration(calibration_session)

    output, _, summary = run_perception(
        person_session,
        calibration_session=calibration_session,
        spec=PerceptionSpec(
            person_cluster_min_points=6,
            anomaly_min_persistence=3,
        ),
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    qualifying = [
        anomaly
        for row in rows
        for anomaly in row["anomalies"]
        if anomaly["persistent_body_associated_anomaly"]
    ]
    assert qualifying
    assert any(row["screening_state"] == "suspicious_metal" for row in rows)
    assert all(anomaly["material_confirmed"] is False for anomaly in qualifying)
    assert all(anomaly["weapon_classification"] is False for anomaly in qualifying)
    assert summary["weapon_classification"] is False


def test_perception_rejects_incompatible_empty_room_configuration(tmp_path):
    person_session = tmp_path / "person"
    calibration_session = tmp_path / "empty"
    person_session.mkdir()
    calibration_session.mkdir()
    _write_person_session(person_session)
    _write_calibration(calibration_session)
    (person_session / "configuration.json").write_text(
        json.dumps(
            [
                {"command": "version", "response": "xWR18xx"},
                {
                    "command": "profileCfg 0 77 7 3 39 0 0 80 1 4 7200 0 0 30",
                    "response": "Done",
                },
            ]
        )
    )

    with pytest.raises(ValueError, match="configurations are incompatible"):
        run_perception(
            person_session,
            calibration_session=calibration_session,
            spec=PerceptionSpec(person_cluster_min_points=6),
        )


def test_matched_person_reference_suppresses_body_only_reflections(tmp_path):
    calibration_session = tmp_path / "empty"
    person_reference_session = tmp_path / "person_reference"
    participant_session = tmp_path / "participant"
    for path in (
        calibration_session,
        person_reference_session,
        participant_session,
    ):
        path.mkdir()
    _write_calibration(calibration_session)
    _write_person_session(person_reference_session, windows=40)
    _write_person_session(participant_session)

    output, _, summary = run_perception(
        participant_session,
        calibration_session=calibration_session,
        person_reference_session=person_reference_session,
        spec=PerceptionSpec(person_cluster_min_points=6),
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["person_reference_calibrated"] is True
    assert summary["person_reference_profile_frames"] == 40
    assert all(row["screening_state"] != "suspicious_metal" for row in rows)
    assert all(
        not anomaly["person_reference_excess"]
        for row in rows
        for anomaly in row["anomalies"]
    )


def test_matched_person_reference_allows_stronger_profile_excess(tmp_path):
    calibration_session = tmp_path / "empty"
    person_reference_session = tmp_path / "person_reference"
    participant_session = tmp_path / "participant"
    for path in (
        calibration_session,
        person_reference_session,
        participant_session,
    ):
        path.mkdir()
    _write_calibration(calibration_session)
    _write_person_session(person_reference_session, windows=40)
    _write_person_session(participant_session, profile_peak=180.0)

    output, _, _ = run_perception(
        participant_session,
        calibration_session=calibration_session,
        person_reference_session=person_reference_session,
        spec=PerceptionSpec(person_cluster_min_points=6),
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    qualifying = [
        anomaly
        for row in rows
        for anomaly in row["anomalies"]
        if anomaly["persistent_body_associated_anomaly"]
    ]
    assert qualifying
    assert all(anomaly["person_reference_excess"] for anomaly in qualifying)
    assert any(row["screening_state"] == "suspicious_metal" for row in rows)


def test_video_hides_candidates_that_do_not_exceed_person_reference():
    row = {
        "quality": {"person_reference_calibrated": True},
        "anomalies": [
            {"anomaly_id": 1, "person_reference_excess": False},
            {"anomaly_id": 2, "person_reference_excess": True},
        ],
    }

    assert [item["anomaly_id"] for item in _display_anomalies(row)] == [2]
