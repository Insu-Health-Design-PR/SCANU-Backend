import json
from pathlib import Path

from lab.mmwave77_usb.camera_capture import (
    radar_window_center_camera_offset_s,
)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_window_center_offset_uses_median_radar_timestamp(tmp_path: Path) -> None:
    camera = tmp_path / "camera_frames.jsonl"
    radar = tmp_path / "frames.jsonl"
    perception = tmp_path / "perception.jsonl"
    _jsonl(camera, [{"index": 0, "host_monotonic_ns": 1_000_000_000}])
    _jsonl(
        radar,
        [
            {
                "frame_number": frame_number,
                "parse_ok": True,
                "host_monotonic_ns": 2_000_000_000 + frame_number * 100_000_000,
            }
            for frame_number in range(1, 11)
        ],
    )
    _jsonl(perception, [{"frame_start": 1, "frame_end": 10}])

    assert radar_window_center_camera_offset_s(
        camera, radar, perception
    ) == 1.55


def test_window_center_offset_ignores_invalid_radar_rows(tmp_path: Path) -> None:
    camera = tmp_path / "camera_frames.jsonl"
    radar = tmp_path / "frames.jsonl"
    perception = tmp_path / "perception.jsonl"
    _jsonl(camera, [{"index": 0, "host_monotonic_ns": 100}])
    _jsonl(
        radar,
        [
            {"frame_number": 1, "parse_ok": False, "host_monotonic_ns": 150},
            {"frame_number": 2, "parse_ok": True, "host_monotonic_ns": 300},
            {"frame_number": 3, "parse_ok": True, "host_monotonic_ns": 500},
        ],
    )
    _jsonl(perception, [{"frame_start": 1, "frame_end": 3}])

    assert radar_window_center_camera_offset_s(
        camera, radar, perception
    ) == 0.0000003
