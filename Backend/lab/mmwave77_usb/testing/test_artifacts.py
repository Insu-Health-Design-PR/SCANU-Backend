from __future__ import annotations

import json
import struct

import numpy as np

from lab.mmwave77_usb.artifacts import SignalFrameRecord, write_signal_artifacts
from lab.mmwave77_usb.reprocess import reprocess_session
from layer1_sensor_hub.radar.radar_constants import MAGIC_WORD


def _tlv_frame(frame_number: int, profile_offset: int = 0) -> bytes:
    point = struct.pack("<4f", 0.25, 1.5, -0.1, 0.2)
    range_profile = struct.pack(
        "<4H", *(100 + profile_offset + index for index in range(4))
    )
    noise_profile = struct.pack(
        "<4H", *(10 + profile_offset + index for index in range(4))
    )
    side_info = struct.pack("<2H", 230, 17)
    stats = struct.pack("<6I", 1, 2, 3, 4, 5, 6)
    tlvs = b"".join(
        struct.pack("<II", tlv_type, len(payload)) + payload
        for tlv_type, payload in (
            (1, point),
            (2, range_profile),
            (3, noise_profile),
            (7, side_info),
            (6, stats),
        )
    )
    total_length = 40 + len(tlvs)
    header = MAGIC_WORD + struct.pack(
        "<8I",
        0x03040003,
        total_length,
        0xA1843,
        frame_number,
        123456 + frame_number,
        1,
        5,
        0,
    )
    return header + tlvs


def test_signal_artifacts_preserve_complete_profiles_and_quality(tmp_path):
    raw_path = tmp_path / "raw_uart.bin"
    raw_path.write_bytes(b"raw-evidence")
    records = [
        SignalFrameRecord(
            frame_number=10 + index,
            sensor_cycles=1000 + index,
            host_monotonic_ns=2000 + index,
            host_utc=f"2026-07-30T00:00:0{index}+00:00",
            range_profile=np.asarray([1, 2, 3, 4], dtype=np.float32) + index,
            noise_profile=np.asarray([5, 6, 7, 8], dtype=np.float32) + index,
            tlv_types=(1, 2, 3, 6, 7),
            stats={"active_frame_cpu_load": 5},
        )
        for index in range(2)
    ]

    outputs = write_signal_artifacts(
        tmp_path,
        records,
        source="test",
        transport_quality={"tlv_parse_errors": 0},
    )

    with np.load(outputs["range_profiles"]) as artifact:
        assert artifact["profiles"].tolist() == [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
        ]
        assert artifact["profile_lengths"].tolist() == [4, 4]
        assert artifact["frame_number"].tolist() == [10, 11]
    quality = json.loads(outputs["capture_quality"].read_text())
    assert quality["sequence"]["monotonic"] is True
    assert quality["sequence"]["missing_frames"] == 0
    assert quality["range_profiles"]["consistent_length"] is True


def test_raw_uart_reprocessing_recovers_profiles_points_and_timestamps(tmp_path):
    frames = [_tlv_frame(7), _tlv_frame(8, profile_offset=2)]
    (tmp_path / "raw_uart.bin").write_bytes(b"noise" + b"".join(frames))
    original_rows = [
        {
            "frame_number": 7 + index,
            "host_monotonic_ns": 9000 + index,
            "host_utc": f"2026-07-30T00:00:0{index}+00:00",
        }
        for index in range(2)
    ]
    (tmp_path / "frames.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in original_rows)
    )

    output_frames, manifest_path, manifest = reprocess_session(
        tmp_path, chunk_bytes=67
    )

    assert output_frames.is_file()
    assert manifest_path.is_file()
    assert manifest["statistics"]["recovered_frames"] == 2
    assert manifest["statistics"]["discarded_bytes"] == 5
    rows = [json.loads(line) for line in output_frames.read_text().splitlines()]
    assert rows[0]["host_monotonic_ns"] == 9000
    assert rows[0]["range_profile_bins"] == 4
    assert rows[0]["noise_profile_bins"] == 4
    assert rows[0]["points"][0]["snr"] == 23.0
    assert rows[0]["stats"]["active_frame_cpu_load"] == 5
    with np.load(tmp_path / "range_profiles.npz") as profiles:
        assert profiles["profiles"].tolist() == [
            [100.0, 101.0, 102.0, 103.0],
            [102.0, 103.0, 104.0, 105.0],
        ]
