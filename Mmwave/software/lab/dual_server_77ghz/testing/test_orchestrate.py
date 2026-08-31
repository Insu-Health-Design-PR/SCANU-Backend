from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lab.dual_server_77ghz.orchestrate import (
    CONFIG_SCHEMA,
    discover_camera_devices,
    load_config,
    select_camera_devices,
    select_radar_pairs,
)
from lab.mmwave77_usb.runner import Awr1843PortPair


def test_finish_accepts_successful_camera_contract_without_ok(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import json; print(json.dumps({'video':'v.mp4','frames':'f.jsonl','metadata':'m.json'}))",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    from lab.dual_server_77ghz.orchestrate import _finish

    result = _finish(process, "camera", tmp_path, require_ok=False)
    assert result["video"] == "v.mp4"


def _pair(location: str, index: int) -> Awr1843PortPair:
    return Awr1843PortPair(
        serial_number="R2091049",
        usb_location=location,
        cli_port=f"/dev/ttyACM{index}",
        data_port=f"/dev/ttyACM{index + 1}",
        assignment="test",
    )


def test_select_radar_pairs_is_stable_by_usb_location() -> None:
    a, b = select_radar_pairs([_pair("1-4.2", 2), _pair("1-4.1", 0)])
    assert a.usb_location == "1-4.1"
    assert b.usb_location == "1-4.2"
    explicit_a, explicit_b = select_radar_pairs(
        [_pair("1-4.2", 2), _pair("1-4.1", 0)], "1-4.2", "1-4.1"
    )
    assert explicit_a.usb_location == "1-4.2"
    assert explicit_b.usb_location == "1-4.1"


def test_select_radar_pairs_requires_two_devices() -> None:
    with pytest.raises(RuntimeError, match="two complete"):
        select_radar_pairs([_pair("1-4.1", 0)])


def test_camera_discovery_prefers_stable_index_zero_links(tmp_path: Path) -> None:
    video0 = tmp_path / "video0"
    video1 = tmp_path / "video1"
    video0.touch()
    video1.touch()
    by_id = tmp_path / "v4l" / "by-id"
    by_id.mkdir(parents=True)
    (by_id / "camera-a-video-index0").symlink_to(video0)
    (by_id / "camera-b-video-index0").symlink_to(video1)
    (by_id / "camera-a-video-index1").symlink_to(video0)

    devices = discover_camera_devices(tmp_path)
    assert devices == [
        str(by_id / "camera-a-video-index0"),
        str(by_id / "camera-b-video-index0"),
    ]
    a, b = select_camera_devices(devices, "auto", "auto")
    assert a != b


def test_camera_discovery_does_not_count_auxiliary_node_as_second_camera(
    tmp_path: Path,
) -> None:
    video0 = tmp_path / "video0"
    video1 = tmp_path / "video1"
    video0.touch()
    video1.touch()
    by_id = tmp_path / "v4l" / "by-id"
    by_id.mkdir(parents=True)
    (by_id / "camera-a-video-index0").symlink_to(video0)
    (by_id / "camera-a-video-index1").symlink_to(video1)

    devices = discover_camera_devices(tmp_path)
    assert devices == [str(by_id / "camera-a-video-index0")]
    with pytest.raises(RuntimeError, match="two distinct camera"):
        select_camera_devices(devices, "auto", "auto")


def test_load_config_rejects_invalid_duration(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "schema_version": CONFIG_SCHEMA,
        "output_root": str(tmp_path),
        "sensor_distance_m": 3.0,
        "radar_profile": "profile.cfg",
        "camera_a": {},
        "camera_b": {},
        "calibration_seconds": 20,
        "entry_delay_seconds": 10,
        "capture_seconds": 0,
    }))
    with pytest.raises(ValueError, match="capture_seconds"):
        load_config(path)


def test_server_profile_enables_classic_fused_dashboard() -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "server_local.json"
    )

    assert config["classic_human_centric_dashboard"] is True
