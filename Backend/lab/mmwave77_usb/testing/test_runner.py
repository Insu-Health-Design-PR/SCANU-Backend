from __future__ import annotations

import struct
from pathlib import Path

import pytest

from lab.mmwave77_usb.runner import (
    TiTlvFramer,
    UsbSerialDevice,
    _validate_awr1843_config,
    discover_awr1843_pairs,
    select_awr1843_pair,
    select_data_port,
)
from layer1_sensor_hub.radar.radar_constants import MAGIC_WORD


def _device(device: str, serial_number: str) -> UsbSerialDevice:
    return UsbSerialDevice(
        device=device,
        description="test",
        hwid="",
        manufacturer="test",
        product="test",
        serial_number=serial_number,
        interface="",
        location="",
        vid=0x1234,
        pid=0x5678,
    )


def _frame(frame_number: int = 7) -> bytes:
    total_length = 48
    header = MAGIC_WORD + struct.pack(
        "<8I",
        0x01020304,
        total_length,
        0xA6843,
        frame_number,
        123456,
        0,
        1,
        0,
    )
    return header + struct.pack("<II", 6, 0)


def test_ti_tlv_framer_handles_split_input_and_leading_noise():
    expected = _frame()
    framer = TiTlvFramer()

    assert framer.feed(b"noise" + expected[:13]) == []
    assert framer.feed(expected[13:]) == [expected]
    assert framer.frames == 1
    assert framer.discarded_bytes == 5
    assert framer.invalid_headers == 0


def test_auto_selection_excludes_canonical_scanu_radars():
    devices = [
        _device("/dev/ttyUSB0", "011D1A12"),
        _device("/dev/ttyACM0", "NEW77"),
    ]
    assert select_data_port(devices) == "/dev/ttyACM0"


def test_auto_selection_refuses_ambiguous_devices():
    devices = [
        _device("/dev/ttyACM0", "NEW77A"),
        _device("/dev/ttyACM1", "NEW77B"),
    ]
    with pytest.raises(RuntimeError, match="multiple non-SCAN-U"):
        select_data_port(devices)


def test_explicit_missing_port_is_rejected():
    with pytest.raises(RuntimeError, match="not present"):
        select_data_port([], "/dev/ttyACM9")


def test_awr1843_xds110_pair_uses_interface_metadata():
    cli = _device("/dev/ttyACM2", "XDS110-A")
    data = _device("/dev/ttyACM3", "XDS110-A")
    cli = UsbSerialDevice(
        **{
            **cli.__dict__,
            "description": "XDS110 Class Application/User UART",
            "interface": "XDS110 Application/User UART",
        }
    )
    data = UsbSerialDevice(
        **{
            **data.__dict__,
            "description": "XDS110 Class Auxiliary Data Port",
            "interface": "XDS110 Auxiliary Data Port",
        }
    )

    pairs = discover_awr1843_pairs([data, cli])
    assert len(pairs) == 1
    assert pairs[0].cli_port == "/dev/ttyACM2"
    assert pairs[0].data_port == "/dev/ttyACM3"
    assert pairs[0].usb_location == "device:/dev/ttyACM"
    assert pairs[0].assignment == "xds110_interface_metadata"


def test_duplicate_xds110_serials_are_grouped_by_usb_location():
    devices = []
    for device, location in (
        ("/dev/ttyACM0", "1-4.2:1.0"),
        ("/dev/ttyACM1", "1-4.2:1.3"),
        ("/dev/ttyACM2", "1-4.1:1.0"),
        ("/dev/ttyACM3", "1-4.1:1.3"),
    ):
        row = _device(device, "DUPLICATE")
        devices.append(
            UsbSerialDevice(
                **{
                    **row.__dict__,
                    "description": "XDS110 Embed with CMSIS-DAP",
                    "location": location,
                }
            )
        )

    pairs = discover_awr1843_pairs(devices)

    assert [(pair.usb_location, pair.cli_port, pair.data_port) for pair in pairs] == [
        ("1-4.1", "/dev/ttyACM2", "/dev/ttyACM3"),
        ("1-4.2", "/dev/ttyACM0", "/dev/ttyACM1"),
    ]
    assert select_awr1843_pair(devices, "1-4.2").data_port == "/dev/ttyACM1"
    with pytest.raises(RuntimeError, match="multiple XDS110"):
        select_awr1843_pair(devices)


def test_awr1843_config_rejects_60_ghz_profile(tmp_path):
    config = tmp_path / "wrong.cfg"
    config.write_text(
        "sensorStop\nflushCfg\nprofileCfg 0 60.75 7 7 57 0 0 70 1 256 5000 0 0 30\nsensorStart\n"
    )
    with pytest.raises(RuntimeError, match="76-81 GHz"):
        _validate_awr1843_config(config)


def test_awr1843_config_accepts_77_ghz_profile(tmp_path):
    config = tmp_path / "profile.cfg"
    config.write_text(
        "sensorStop\nflushCfg\nprofileCfg 0 77 7 2.54 140.3 0 0 26.74 1 256 7200 0 0 30\nsensorStart\n"
    )
    _validate_awr1843_config(config)


def test_repository_awr1843_profile_is_77_ghz_compatible():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    for name in (
        "awr1843boost_sdk_3_4_profile_2d.cfg",
        "awr1843boost_sdk_3_4_profile_3d.cfg",
    ):
        _validate_awr1843_config(config_dir / name)
