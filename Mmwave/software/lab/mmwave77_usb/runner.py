#!/usr/bin/env python3
"""Discover and record an experimental 77 GHz USB mmWave sensor.

The default ``raw`` protocol never writes to the device.  ``ti-tlv`` additionally
frames and decodes the common TI mmWave UART stream.  A CLI configuration is
sent only when both ``--cli-port`` and ``--config`` are explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from lab.mmwave77_usb.artifacts import SignalFrameRecord, write_signal_artifacts
from layer1_sensor_hub.hardware_registry import RADAR_SERIAL_ROLES
from layer1_sensor_hub.radar.radar_constants import FRAME_HEADER_SIZE, MAGIC_WORD
from layer1_sensor_hub.radar.tlv_parser import TLVParser
from layer1_sensor_hub.radar.uart_source import FrameHeader


LAB_SCHEMA_VERSION = "scanu_lab_mmwave77_usb_v1"
DEFAULT_BAUD = 921_600
DEFAULT_OUTPUT_ROOT = Path("data/lab/mmwave77_usb")
MAX_TLV_FRAME_BYTES = 131_072


@dataclass(frozen=True)
class UsbSerialDevice:
    device: str
    description: str
    hwid: str
    manufacturer: str
    product: str
    serial_number: str
    interface: str
    location: str
    vid: int | None
    pid: int | None

    @property
    def is_canonical_scanu_radar(self) -> bool:
        return self.serial_number in RADAR_SERIAL_ROLES

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["vid_hex"] = f"0x{self.vid:04x}" if self.vid is not None else None
        row["pid_hex"] = f"0x{self.pid:04x}" if self.pid is not None else None
        row["is_canonical_scanu_radar"] = self.is_canonical_scanu_radar
        return row


@dataclass(frozen=True)
class Awr1843PortPair:
    serial_number: str
    usb_location: str
    cli_port: str
    data_port: str
    assignment: str


class TiTlvFramer:
    """Incrementally assemble common TI mmWave UART TLV packets."""

    def __init__(self, *, max_frame_bytes: int = MAX_TLV_FRAME_BYTES):
        self.max_frame_bytes = int(max_frame_bytes)
        self.buffer = bytearray()
        self.discarded_bytes = 0
        self.invalid_headers = 0
        self.frames = 0

    def feed(self, chunk: bytes) -> list[bytes]:
        if chunk:
            self.buffer.extend(chunk)
        frames: list[bytes] = []

        while True:
            magic_index = self.buffer.find(MAGIC_WORD)
            if magic_index < 0:
                keep = min(len(self.buffer), len(MAGIC_WORD) - 1)
                self.discarded_bytes += len(self.buffer) - keep
                if keep:
                    self.buffer[:] = self.buffer[-keep:]
                else:
                    self.buffer.clear()
                break

            if magic_index:
                self.discarded_bytes += magic_index
                del self.buffer[:magic_index]

            if len(self.buffer) < FRAME_HEADER_SIZE:
                break

            try:
                header = FrameHeader.from_bytes(bytes(self.buffer[:FRAME_HEADER_SIZE]))
            except ValueError:
                self.invalid_headers += 1
                del self.buffer[0]
                continue

            frame_bytes = int(header.total_packet_length)
            if frame_bytes < FRAME_HEADER_SIZE or frame_bytes > self.max_frame_bytes:
                self.invalid_headers += 1
                del self.buffer[0]
                continue
            if len(self.buffer) < frame_bytes:
                break

            frames.append(bytes(self.buffer[:frame_bytes]))
            del self.buffer[:frame_bytes]
            self.frames += 1

        if len(self.buffer) > self.max_frame_bytes:
            excess = len(self.buffer) - (len(MAGIC_WORD) - 1)
            self.discarded_bytes += excess
            self.buffer[:] = self.buffer[-(len(MAGIC_WORD) - 1) :]
        return frames


def list_serial_devices() -> list[UsbSerialDevice]:
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("pyserial is required; install software/requirements.txt") from exc

    devices: list[UsbSerialDevice] = []
    for port in list_ports.comports():
        devices.append(
            UsbSerialDevice(
                device=str(port.device),
                description=str(getattr(port, "description", "") or ""),
                hwid=str(getattr(port, "hwid", "") or ""),
                manufacturer=str(getattr(port, "manufacturer", "") or ""),
                product=str(getattr(port, "product", "") or ""),
                serial_number=str(getattr(port, "serial_number", "") or ""),
                interface=str(getattr(port, "interface", "") or ""),
                location=str(getattr(port, "location", "") or ""),
                vid=getattr(port, "vid", None),
                pid=getattr(port, "pid", None),
            )
        )
    return sorted(devices, key=lambda row: row.device)


def discover_awr1843_pairs(
    devices: Sequence[UsbSerialDevice],
) -> list[Awr1843PortPair]:
    """Resolve AWR1843BOOST XDS110 Application and Auxiliary UARTs."""

    xds_devices = [
        row
        for row in devices
        if "xds110"
        in " ".join(
            (
                row.description,
                row.hwid,
                row.manufacturer,
                row.product,
                row.interface,
            )
        ).lower()
    ]
    grouped: dict[tuple[str, str], list[UsbSerialDevice]] = {}
    for row in xds_devices:
        usb_location = row.location.split(":", 1)[0]
        key = (
            row.serial_number or "xds110-without-usb-serial",
            usb_location or f"device:{_device_suffix(row.device)[0]}",
        )
        grouped.setdefault(key, []).append(row)

    pairs: list[Awr1843PortPair] = []
    for (serial_number, usb_location), rows in grouped.items():
        if len(rows) != 2:
            continue

        def text(row: UsbSerialDevice) -> str:
            return " ".join((row.description, row.product, row.interface)).lower()

        cli = next(
            (
                row
                for row in rows
                if "application/user" in text(row)
                or "application user" in text(row)
                or "user uart" in text(row)
            ),
            None,
        )
        data = next(
            (
                row
                for row in rows
                if "auxiliary" in text(row) or "aux " in f"{text(row)} "
            ),
            None,
        )
        assignment = "xds110_interface_metadata"
        if cli is None or data is None or cli.device == data.device:
            ordered = sorted(rows, key=lambda row: _device_suffix(row.device))
            cli, data = ordered[0], ordered[1]
            assignment = "xds110_device_order_fallback"

        pairs.append(
            Awr1843PortPair(
                serial_number=serial_number,
                usb_location=usb_location,
                cli_port=cli.device,
                data_port=data.device,
                assignment=assignment,
            )
        )
    return sorted(pairs, key=lambda pair: (pair.usb_location, pair.serial_number))


def select_awr1843_pair(
    devices: Sequence[UsbSerialDevice],
    usb_location: str = "",
) -> Awr1843PortPair:
    pairs = discover_awr1843_pairs(devices)
    if usb_location:
        pairs = [pair for pair in pairs if pair.usb_location == usb_location]
        if not pairs:
            raise RuntimeError(
                f"no AWR1843BOOST XDS110 pair at USB location {usb_location!r}"
            )
    if not pairs:
        raise RuntimeError(
            "no complete AWR1843BOOST XDS110 UART pair found; verify 5 V power, "
            "the micro-USB cable, XDS110 enumeration, and Linux permissions"
        )
    if len(pairs) > 1:
        identities = ", ".join(
            f"{pair.serial_number}@{pair.usb_location}" for pair in pairs
        )
        raise RuntimeError(
            f"multiple XDS110 pairs found ({identities}); use "
            "--awr-usb-location or explicit --cli-port and --data-port"
        )
    return pairs[0]


def _device_suffix(device: str) -> tuple[str, int]:
    digits = ""
    for character in reversed(device):
        if not character.isdigit():
            break
        digits = character + digits
    return (device[: -len(digits)] if digits else device, int(digits or -1))


def select_data_port(devices: Sequence[UsbSerialDevice], explicit: str = "") -> str:
    if explicit:
        matches = [row for row in devices if row.device == explicit]
        if not matches:
            raise RuntimeError(
                f"requested port {explicit!r} is not present; run the 'list' command"
            )
        return explicit

    candidates = [row for row in devices if not row.is_canonical_scanu_radar]
    if len(candidates) == 1:
        return candidates[0].device
    if not candidates:
        raise RuntimeError(
            "no non-SCAN-U USB serial device found; connect the 77 GHz sensor "
            "or pass --data-port explicitly"
        )
    names = ", ".join(row.device for row in candidates)
    raise RuntimeError(
        f"multiple non-SCAN-U serial devices found ({names}); pass --data-port explicitly"
    )


def _validate_awr1843_config(config_path: Path) -> None:
    profile_frequencies: list[float] = []
    commands: set[str] = set()
    for raw_line in config_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("%", "#")):
            continue
        tokens = line.split()
        commands.add(tokens[0].lower())
        if tokens[0].lower() == "profilecfg" and len(tokens) >= 3:
            try:
                profile_frequencies.append(float(tokens[2]))
            except ValueError as exc:
                raise RuntimeError(f"invalid profileCfg start frequency: {line}") from exc

    if not profile_frequencies:
        raise RuntimeError("AWR1843 configuration has no profileCfg command")
    invalid = [value for value in profile_frequencies if not 76.0 <= value <= 81.0]
    if invalid:
        raise RuntimeError(
            f"AWR1843 requires a 76-81 GHz profile; found start frequency {invalid}"
        )
    missing = {"sensorstop", "flushcfg", "sensorstart"} - commands
    if missing:
        raise RuntimeError(f"AWR1843 configuration is missing commands: {sorted(missing)}")


def _configure_ti_sensor(
    cli_port: str,
    config_path: Path,
    *,
    validate_awr1843: bool,
    audit_path: Path | None = None,
) -> list[dict[str, str]]:
    from layer1_sensor_hub.radar.radar_cli import (
        RadarCliConfig,
        load_cli_commands,
        send_cli_commands,
    )

    if not config_path.is_file():
        raise RuntimeError(f"configuration file does not exist: {config_path}")
    if validate_awr1843:
        _validate_awr1843_config(config_path)
    cli = RadarCliConfig(
        port=cli_port,
        baud=115_200,
        timeout_s=1.0,
        command_delay_s=0.1,
    )
    commands = load_cli_commands(config_path)
    failure_tokens = ("error", "exception", "not recognized", "failed")
    rows: list[dict[str, str]] = []
    # Consume any stale boot/ModemManager response and verify the selected role
    # before the first configuration command. This is particularly important
    # when two identical XDS110 devices enumerate simultaneously.
    preflight_responses = send_cli_commands(cli, ["version"])
    preflight_response = preflight_responses[0] if preflight_responses else ""
    rows.append({"command": "version", "response": preflight_response})
    if audit_path is not None:
        audit_path.write_text(json.dumps(rows, indent=2) + "\n")
    if "xwr18" not in preflight_response.lower():
        raise RuntimeError(
            f"TI CLI preflight failed on {cli_port}: "
            f"{preflight_response.strip()[-300:] or 'empty response'}"
        )
    for command in commands:
        responses = send_cli_commands(cli, [command])
        response = responses[0] if responses else ""
        row = {"command": command, "response": response}
        rows.append(row)
        if audit_path is not None:
            audit_path.write_text(json.dumps(rows, indent=2) + "\n")
        if not response.strip() or any(
            token in response.lower() for token in failure_tokens
        ):
            command_name = command.split()[0]
            detail = response.strip().replace("\r", " ").replace("\n", " ")[-300:]
            raise RuntimeError(
                f"TI CLI rejected {command_name}: {detail or 'empty response'}"
            )
    return rows


def _stop_ti_sensor(cli_port: str) -> None:
    try:
        from layer1_sensor_hub.radar.radar_cli import RadarCliConfig, send_sensor_stop

        send_sensor_stop(
            RadarCliConfig(
                port=cli_port,
                baud=115_200,
                timeout_s=1.0,
                command_delay_s=0.1,
            )
        )
    except Exception as exc:
        print(f"warning: could not stop TI sensor: {type(exc).__name__}: {exc}", file=sys.stderr)


def _decode_frame(
    frame: bytes,
    parser: TLVParser,
    *,
    host_monotonic_ns: int | None = None,
    host_utc: str | None = None,
) -> tuple[dict[str, Any], SignalFrameRecord | None]:
    header = FrameHeader.from_bytes(frame[:FRAME_HEADER_SIZE])
    receipt_monotonic_ns = (
        int(host_monotonic_ns)
        if host_monotonic_ns is not None
        else time.monotonic_ns()
    )
    receipt_utc = host_utc or datetime.now(timezone.utc).isoformat()
    row: dict[str, Any] = {
        "host_monotonic_ns": receipt_monotonic_ns,
        "host_utc": receipt_utc,
        "frame_number": header.frame_number,
        "sensor_cycles": header.time_cpu_cycles,
        "packet_bytes": header.total_packet_length,
        "declared_objects": header.num_detected_obj,
        "declared_tlvs": header.num_tlvs,
        "subframe_number": header.subframe_number,
    }
    try:
        parsed = parser.parse(frame)
        row["parse_ok"] = True
        row["points"] = [point.to_dict() for point in parsed.points]
        row["range_profile_bins"] = (
            int(len(parsed.range_profile)) if parsed.range_profile is not None else 0
        )
        row["noise_profile_bins"] = (
            int(len(parsed.noise_profile)) if parsed.noise_profile is not None else 0
        )
        row["tlv_types"] = sorted(int(value) for value in parsed.raw_tlvs)
        row["stats"] = dict(parsed.stats)
        signal_record = SignalFrameRecord(
            frame_number=int(header.frame_number),
            sensor_cycles=int(header.time_cpu_cycles),
            host_monotonic_ns=receipt_monotonic_ns,
            host_utc=receipt_utc,
            range_profile=(
                np.asarray(parsed.range_profile, dtype=np.float32).copy()
                if parsed.range_profile is not None
                else None
            ),
            noise_profile=(
                np.asarray(parsed.noise_profile, dtype=np.float32).copy()
                if parsed.noise_profile is not None
                else None
            ),
            tlv_types=tuple(sorted(int(value) for value in parsed.raw_tlvs)),
            stats=dict(parsed.stats),
        )
    except Exception as exc:
        row["parse_ok"] = False
        row["parse_error"] = f"{type(exc).__name__}: {exc}"
        signal_record = None
    return row, signal_record


def _frame_row(frame: bytes, parser: TLVParser) -> dict[str, Any]:
    """Backward-compatible row-only decoder used by older lab tests."""

    row, _ = _decode_frame(frame, parser)
    return row


def capture(args: argparse.Namespace) -> int:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required; install software/requirements.txt") from exc

    devices = list_serial_devices()
    detected_pair: Awr1843PortPair | None = None
    if args.auto_awr1843:
        detected_pair = select_awr1843_pair(devices, args.awr_usb_location)
        if not args.data_port:
            args.data_port = detected_pair.data_port
        if args.config and not args.cli_port:
            args.cli_port = detected_pair.cli_port
        if args.sensor_model == "unknown-77ghz":
            args.sensor_model = "Texas Instruments AWR1843BOOST"

    data_port = select_data_port(devices, args.data_port)
    selected = next(row for row in devices if row.device == data_port)
    if selected.is_canonical_scanu_radar and not args.allow_scanu_port:
        raise RuntimeError(
            f"{data_port} belongs to canonical SCAN-U radar serial "
            f"{selected.serial_number}; use --allow-scanu-port only for an intentional lab test"
        )

    config_path = Path(args.config).expanduser() if args.config else None
    if bool(args.cli_port) != bool(config_path):
        raise RuntimeError("--cli-port and --config must be supplied together")
    if args.cli_port == data_port:
        raise RuntimeError("CLI and data ports must be different")

    output_root = Path(args.output_root).expanduser()
    session_name = datetime.now().strftime("capture_%Y%m%d_%H%M%S")
    session_dir = output_root / session_name
    session_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "schema_version": LAB_SCHEMA_VERSION,
        "experimental": True,
        "canonical_training_compatible": False,
        "sensor_model": args.sensor_model,
        "frequency_ghz": float(args.frequency_ghz),
        "protocol": args.protocol,
        "data_port": data_port,
        "cli_port": args.cli_port or None,
        "detected_xds110_pair": (
            asdict(detected_pair) if detected_pair is not None else None
        ),
        "baud": int(args.baud),
        "duration_s": float(args.duration_s),
        "device": selected.to_dict(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
        },
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    configured = False
    bytes_read = 0
    read_calls = 0
    parsed_rows = 0
    parse_errors = 0
    digest = hashlib.sha256()
    framer = TiTlvFramer()
    parser = TLVParser()
    started = time.monotonic()
    configuration_rows: list[dict[str, str]] = []
    signal_records: list[SignalFrameRecord] = []

    try:
        if config_path is not None:
            # Configuration may partially start or alter the device before a
            # later command fails. Always attempt sensorStop in ``finally``.
            configured = True
            configuration_rows = _configure_ti_sensor(
                args.cli_port,
                config_path,
                validate_awr1843=bool(
                    args.auto_awr1843
                    or "awr1843" in str(args.sensor_model).lower()
                ),
                audit_path=session_dir / "configuration.json",
            )

        started = time.monotonic()
        with serial.Serial(
            data_port,
            baudrate=int(args.baud),
            timeout=float(args.read_timeout_s),
        ) as stream, (session_dir / "raw_uart.bin").open("wb") as raw_file, (
            session_dir / "frames.jsonl"
        ).open("w") as frame_file:
            stream.reset_input_buffer()
            deadline = started + float(args.duration_s)
            while time.monotonic() < deadline:
                waiting = int(stream.in_waiting)
                chunk = stream.read(min(max(waiting, 1), int(args.chunk_bytes)))
                read_calls += 1
                if not chunk:
                    continue
                raw_file.write(chunk)
                digest.update(chunk)
                bytes_read += len(chunk)

                if args.protocol == "ti-tlv":
                    for frame in framer.feed(chunk):
                        row, signal_record = _decode_frame(frame, parser)
                        parsed_rows += 1
                        if not row["parse_ok"]:
                            parse_errors += 1
                        elif signal_record is not None:
                            signal_records.append(signal_record)
                        frame_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    finally:
        if configured and not args.leave_running:
            _stop_ti_sensor(args.cli_port)

    elapsed = time.monotonic() - started
    ok = bytes_read > 0 and (args.protocol == "raw" or parsed_rows > 0)
    summary = {
        "schema_version": LAB_SCHEMA_VERSION,
        "ok": ok,
        "bytes_read": bytes_read,
        "sha256": digest.hexdigest(),
        "read_calls": read_calls,
        "elapsed_s": elapsed,
        "average_bytes_per_s": bytes_read / elapsed if elapsed > 0 else 0.0,
        "tlv_frames": parsed_rows,
        "tlv_parse_errors": parse_errors,
        "tlv_invalid_headers": framer.invalid_headers,
        "tlv_discarded_bytes": framer.discarded_bytes,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "experimental lab capture; not a canonical SCAN-U training artifact",
            "USB arrival time is not a hardware acquisition timestamp",
            "USB provides processed xWR18xx demo TLVs, not raw ADC samples",
        ],
    }
    if args.protocol == "ti-tlv" and signal_records:
        signal_outputs = write_signal_artifacts(
            session_dir,
            signal_records,
            source="live_usb_capture",
            transport_quality={
                "bytes_read": bytes_read,
                "sha256": digest.hexdigest(),
                "tlv_frames": parsed_rows,
                "tlv_parse_errors": parse_errors,
                "tlv_invalid_headers": framer.invalid_headers,
                "tlv_discarded_bytes": framer.discarded_bytes,
            },
        )
        summary["signal_artifacts"] = {
            "range_profiles": str(signal_outputs["range_profiles"]),
            "noise_profiles": str(signal_outputs["noise_profiles"]),
            "capture_quality": str(signal_outputs["capture_quality"]),
        }
    (session_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"ok": ok, "session": str(session_dir), **summary}, indent=2))
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimental USB runner for an additional 77 GHz mmWave sensor"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list USB serial devices without opening them")
    list_parser.add_argument("--json", action="store_true", help="print machine-readable JSON")

    subparsers.add_parser(
        "detect-awr1843",
        help="resolve AWR1843BOOST XDS110 CLI and data ports without opening them",
    )

    capture_parser = subparsers.add_parser("capture", help="record a bounded diagnostic session")
    capture_parser.add_argument("--data-port", default="", help="for example /dev/ttyUSB10")
    capture_parser.add_argument("--cli-port", default="", help="optional TI CLI/control port")
    capture_parser.add_argument("--config", default="", help="TI .cfg file; requires --cli-port")
    capture_parser.add_argument(
        "--protocol",
        choices=("raw", "ti-tlv"),
        default="raw",
        help="raw is non-mutating and works with unknown serial protocols",
    )
    capture_parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    capture_parser.add_argument("--duration-s", type=float, default=10.0)
    capture_parser.add_argument("--read-timeout-s", type=float, default=0.1)
    capture_parser.add_argument("--chunk-bytes", type=int, default=65_536)
    capture_parser.add_argument("--sensor-model", default="unknown-77ghz")
    capture_parser.add_argument("--frequency-ghz", type=float, default=77.0)
    capture_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    capture_parser.add_argument(
        "--auto-awr1843",
        action="store_true",
        help="auto-resolve the AWR1843BOOST XDS110 Application and Auxiliary UARTs",
    )
    capture_parser.add_argument(
        "--awr-usb-location",
        default="",
        help="select one XDS110 topology path, for example 1-4.2",
    )
    capture_parser.add_argument(
        "--allow-scanu-port",
        action="store_true",
        help="allow an existing canonical SCAN-U radar port (normally refused)",
    )
    capture_parser.add_argument(
        "--leave-running",
        action="store_true",
        help="do not send sensorStop after an explicitly configured TI session",
    )
    return parser


def _validate_capture_args(args: argparse.Namespace) -> None:
    if args.duration_s <= 0:
        raise RuntimeError("--duration-s must be greater than zero")
    if args.baud <= 0:
        raise RuntimeError("--baud must be greater than zero")
    if args.read_timeout_s <= 0:
        raise RuntimeError("--read-timeout-s must be greater than zero")
    if args.chunk_bytes < 64:
        raise RuntimeError("--chunk-bytes must be at least 64")
    if not 70.0 <= args.frequency_ghz <= 90.0:
        raise RuntimeError("--frequency-ghz must be between 70 and 90 for this lab runner")


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "list":
            rows = [device.to_dict() for device in list_serial_devices()]
            if args.json:
                print(json.dumps(rows, indent=2))
            elif not rows:
                print("No USB serial devices found.")
            else:
                for row in rows:
                    marker = " [canonical SCAN-U]" if row["is_canonical_scanu_radar"] else ""
                    print(
                        f"{row['device']}: {row['description']} "
                        f"VID={row['vid_hex']} PID={row['pid_hex']} "
                        f"serial={row['serial_number'] or '-'}{marker}"
                    )
            return 0
        if args.command == "detect-awr1843":
            pairs = [
                asdict(pair)
                for pair in discover_awr1843_pairs(list_serial_devices())
            ]
            print(json.dumps(pairs, indent=2))
            return 0 if pairs else 2

        _validate_capture_args(args)
        return capture(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
