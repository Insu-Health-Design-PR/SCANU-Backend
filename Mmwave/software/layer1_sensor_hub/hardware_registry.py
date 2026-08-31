"""Stable SCANU hardware discovery and preflight checks.

The three CP2105 bridges are assigned by their USB serial number, never by the
volatile ``/dev/ttyUSBN`` enumeration.  Capture entrypoints may still provide
explicit overrides, but empty values resolve through this registry.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


RADAR_SERIAL_ROLES = {
    # Physical topology verified on the Jetson on 2026-07-21.  The DCA1000
    # LVDS ribbon is attached to 011D1A12; assigning any other CP2105 as center
    # yields valid control replies but zero UDP/4098 data packets.
    "011D1A12": "center",
    "011D1948": "right",
    "010BD089": "left",
}


@dataclass(frozen=True)
class RadarPortPair:
    role: str
    serial_number: str
    cli: str
    data: str
    cli_responsive: bool = False
    cli_detail: str = "not probed"
    assignment: str = "unknown"


@dataclass(frozen=True)
class HardwarePorts:
    left_cli: str
    left_data: str
    center_cli: str
    center_data: str
    right_cli: str
    right_data: str
    camera_index: int = -1


@dataclass
class PreflightCheck:
    name: str
    ok: bool
    detail: str


@dataclass
class PreflightReport:
    ok: bool
    checks: list[PreflightCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [asdict(check) for check in self.checks]}


def discover_radar_pairs() -> dict[str, RadarPortPair]:
    """Discover all known CP2105 pairs, even when radar firmware is silent.

    USB enumeration and radar CLI health are separate facts.  Earlier code
    discarded a bridge unless ``version`` answered, causing a powered CP2105
    with a stopped/reset radar to be reported as physically absent.  Interface
    metadata is now the primary CLI/data assignment, with an active probe as a
    fallback and as a separate health signal.
    """
    try:
        import serial.tools.list_ports
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("pyserial is required for radar discovery") from exc

    grouped: dict[str, list[Any]] = {}
    for port in serial.tools.list_ports.comports():
        serial_number = str(getattr(port, "serial_number", "") or "")
        if serial_number in RADAR_SERIAL_ROLES:
            grouped.setdefault(serial_number, []).append(port)

    result: dict[str, RadarPortPair] = {}
    for serial_number, ports in grouped.items():
        role = RADAR_SERIAL_ROLES[serial_number]
        if len(ports) < 2:
            continue

        ordered_ports = sorted(ports, key=lambda port: _device_number(str(port.device)))
        cli_port = next((port for port in ordered_ports if _cp2105_interface_kind(port) == "cli"), None)
        data_port = next((port for port in ordered_ports if _cp2105_interface_kind(port) == "data"), None)
        assignment = "usb_interface_metadata" if cli_port is not None and data_port is not None else ""

        probe_results: dict[str, tuple[bool, str]] = {}
        if cli_port is None or data_port is None:
            for port in ordered_ports:
                device = str(port.device)
                probe_results[device] = probe_radar_cli(device, timeout_s=0.45)
            cli_port = next(
                (port for port in ordered_ports if probe_results[str(port.device)][0]),
                None,
            )
            if cli_port is not None:
                data_port = next((port for port in ordered_ports if port is not cli_port), None)
                assignment = "active_cli_probe"

        # Last-resort deterministic mapping for a bridge whose metadata is
        # incomplete and whose radar firmware is not answering.  The health
        # flag remains false, so preflight/side diagnostics still fail clearly.
        if cli_port is None or data_port is None:
            cli_port, data_port = ordered_ports[0], ordered_ports[1]
            assignment = "device_order_fallback"

        cli = str(cli_port.device)
        data = str(data_port.device)
        if cli in probe_results:
            cli_ok, cli_detail = probe_results[cli]
        else:
            cli_ok, cli_detail = probe_radar_cli(cli, timeout_s=0.45)
        result[role] = RadarPortPair(
            role=role,
            serial_number=serial_number,
            cli=cli,
            data=data,
            cli_responsive=cli_ok,
            cli_detail=cli_detail,
            assignment=assignment,
        )
    return result


def resolve_hardware_ports(
    *,
    left_cli: str = "",
    left_data: str = "",
    center_cli: str = "",
    center_data: str = "",
    right_cli: str = "",
    right_data: str = "",
    camera_index: int = -1,
) -> HardwarePorts:
    """Resolve explicit overrides against stable USB serial discovery."""
    explicit_values = (left_cli, left_data, center_cli, center_data, right_cli, right_data)
    discovered = {} if all(explicit_values) else discover_radar_pairs()

    def value(explicit: str, role: str, attr: str) -> str:
        if explicit:
            return explicit
        pair = discovered.get(role)
        if pair is None:
            serial_number = next(
                (serial for serial, mapped_role in RADAR_SERIAL_ROLES.items() if mapped_role == role),
                "unknown",
            )
            raise RuntimeError(
                f"{role} CP2105 USB bridge ({serial_number}) is absent or does not expose two UART ports"
            )
        return str(getattr(pair, attr))

    return HardwarePorts(
        left_cli=value(left_cli, "left", "cli"),
        left_data=value(left_data, "left", "data"),
        center_cli=value(center_cli, "center", "cli"),
        center_data=value(center_data, "center", "data"),
        right_cli=value(right_cli, "right", "cli"),
        right_data=value(right_data, "right", "data"),
        camera_index=discover_rgb_camera_index() if int(camera_index) == -2 else int(camera_index),
    )


def discover_rgb_camera_index(*, max_index: int = 6) -> int:
    """Return a stable RGB camera index without opening every video node.

    Opening the PureThermal Y16 node as if it were RGB can block the thermal
    source and has caused OpenCV driver crashes on the live host.  Linux udev
    links and sysfs metadata let us select an RGB device without consuming it.
    """

    def video_index(path: Path) -> int | None:
        match = re.search(r"video(\d+)$", str(path.resolve(strict=False)))
        if match is None:
            return None
        value = int(match.group(1))
        return value if value < max_index else None

    preferred_tokens = ("c920", "hd_pro", "hd-pro", "nexigo", "webcam", "camera")
    candidates: list[tuple[int, int]] = []
    by_id = Path("/dev/v4l/by-id")
    if by_id.exists():
        for link in by_id.glob("*-video-index0"):
            label = link.name.lower()
            if any(token in label for token in ("thermal", "purethermal", "flir")):
                continue
            index = video_index(link)
            if index is None:
                continue
            rank = 0 if any(token in label for token in preferred_tokens) else 1
            candidates.append((rank, index))
    if candidates:
        return min(candidates)[1]

    # Fallback for systems without persistent udev links.  Reading sysfs names
    # is non-invasive and still prevents a thermal node from being opened.
    sysfs_candidates: list[tuple[int, int]] = []
    for name_path in Path("/sys/class/video4linux").glob("video*/name"):
        match = re.search(r"video(\d+)", name_path.parent.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index >= max_index:
            continue
        try:
            label = name_path.read_text(errors="replace").strip().lower()
        except OSError:
            continue
        if any(token in label for token in ("thermal", "purethermal", "flir")):
            continue
        rank = 0 if any(token.replace("_", " ") in label for token in preferred_tokens) else 1
        sysfs_candidates.append((rank, index))
    return min(sysfs_candidates)[1] if sysfs_candidates else -1


def probe_radar_cli(port: str, *, timeout_s: float = 1.2) -> tuple[bool, str]:
    """Send a non-mutating ``version`` command and verify a mmWave prompt."""
    try:
        import serial
        with serial.Serial(port, 115200, timeout=0.2) as device:
            device.reset_input_buffer()
            device.write(b"\r\nversion\r\n")
            deadline = time.monotonic() + timeout_s
            response = bytearray()
            while time.monotonic() < deadline:
                waiting = int(device.in_waiting)
                if waiting:
                    response.extend(device.read(waiting))
                text = response.decode("utf-8", errors="ignore")
                if any(token in text for token in ("mmwDemo", "xWR68", "IWR68", "mmWave")):
                    return True, text[-300:].strip()
                time.sleep(0.05)
        return False, "no mmWave version response"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run_hardware_preflight(
    ports: HardwarePorts,
    *,
    pc_ip: str,
    dca_ip: str,
    output_dir: str | Path,
    camera_required: bool,
    min_free_gb: float = 2.0,
) -> PreflightReport:
    """Verify required devices without starting a capture."""
    checks: list[PreflightCheck] = []
    for role, port in (("left_cli", ports.left_cli), ("center_cli", ports.center_cli), ("right_cli", ports.right_cli)):
        ok, detail = probe_radar_cli(port)
        checks.append(PreflightCheck(role, ok, f"{port}: {detail}"))

    try:
        from .radar import Dca1000NativeClient, Dca1000NetworkConfig
        network = Dca1000NetworkConfig(pc_ip=pc_ip, dca_ip=dca_ip)
        response = Dca1000NativeClient(network=network, timeout_s=1.0, retries=1).send_command("version")
        checks.append(PreflightCheck("dca1000", bool(response.ok), response.response_hex or "no response"))
    except Exception as exc:
        checks.append(PreflightCheck("dca1000", False, f"{type(exc).__name__}: {exc}"))

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / (1024 ** 3)
    checks.append(PreflightCheck("disk", free_gb >= min_free_gb, f"{free_gb:.2f} GiB free"))

    if camera_required:
        try:
            import cv2
            camera = cv2.VideoCapture(ports.camera_index)
            ok, frame = camera.read()
            camera.release()
            checks.append(PreflightCheck("camera", bool(ok and frame is not None), f"index={ports.camera_index}"))
        except Exception as exc:
            checks.append(PreflightCheck("camera", False, f"{type(exc).__name__}: {exc}"))

    return PreflightReport(ok=all(check.ok for check in checks), checks=checks)


def save_preflight_report(path: str | Path, report: PreflightReport) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return output


def _device_number(value: str) -> tuple[str, int]:
    head = value.rstrip("0123456789")
    tail = value[len(head):]
    return head, int(tail) if tail else -1


def _cp2105_interface_kind(port: Any) -> str:
    """Return ``cli``/``data`` from CP2105 interface metadata when available."""
    metadata = " ".join(
        str(getattr(port, attr, "") or "")
        for attr in ("interface", "description", "hwid", "location")
    ).lower()
    if "enhanced" in metadata or "if00" in metadata or ":1.0" in metadata:
        return "cli"
    if "standard" in metadata or "if01" in metadata or ":1.1" in metadata:
        return "data"
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SCAN-U non-mutating hardware preflight")
    parser.add_argument("--pc-ip", default="192.168.33.30")
    parser.add_argument("--dca-ip", default="192.168.33.180")
    parser.add_argument("--output", default="preflight.json")
    parser.add_argument("--camera", type=int, default=-2, help="-2 auto, -1 disabled")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    args = parser.parse_args(argv)
    try:
        ports = resolve_hardware_ports(camera_index=-1 if args.no_camera else args.camera)
        report = run_hardware_preflight(
            ports, pc_ip=args.pc_ip, dca_ip=args.dca_ip,
            output_dir=Path(args.output).parent,
            camera_required=not args.no_camera,
            min_free_gb=args.min_free_gb,
        )
    except Exception as exc:
        report = PreflightReport(False, [
            PreflightCheck("hardware_resolution", False, f"{type(exc).__name__}: {exc}")
        ])
    save_preflight_report(args.output, report)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 2
