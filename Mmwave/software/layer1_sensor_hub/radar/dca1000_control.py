"""Jetson-native UDP control for TI DCA1000EVM.

This module replaces the Windows-only ``DCA1000EVM_CLI_Control.exe`` for the
basic lab flow used by SCAN-U:

1. connect/reset/configure DCA1000 over UDP
2. arm DCA1000 recording
3. start the radar over UART
4. receive ADC packets on the Jetson

The packet framing follows the public DCA1000 command protocol used by TI's
CLI: 0xA55A header, command code, payload length, payload, 0xEEAA footer.
The payload values are intentionally configurable from JSON because TI has
shipped slightly different CLI examples across mmWave Studio releases.

Some DCA1000 firmware versions require the FPGA bitstream to be loaded
before they accept any commands. The ``load_fpga_bitstream`` helper can
send the FPGA config file over UDP if needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

DCA1000_HOST_IP = "192.168.33.30"
DCA1000_DEVICE_IP = "192.168.33.180"  # was: from software.settings
from .dca1000_udp import Dca1000NetworkConfig

logger = logging.getLogger(__name__)

HEADER = 0xA55A
FOOTER = 0xEEAA


class Dca1000Command:
    """DCA1000 command codes used by TI's CLI control utility."""

    RESET_FPGA = 0x01
    RESET_AR_DEV = 0x02
    CONFIG_FPGA_GEN = 0x03
    CONFIG_EEPROM = 0x04
    RECORD_START = 0x05
    RECORD_STOP = 0x06
    PLAYBACK_START = 0x07
    PLAYBACK_STOP = 0x08
    SYSTEM_CONNECT = 0x09
    SYSTEM_ERROR = 0x0A
    CONFIG_PACKET_DATA = 0x0B
    READ_FPGA_VERSION = 0x0E


COMMAND_NAMES: dict[str, int] = {
    "connect": Dca1000Command.SYSTEM_CONNECT,
    "reset_fpga": Dca1000Command.RESET_FPGA,
    "reset_radar": Dca1000Command.RESET_AR_DEV,
    "fpga": Dca1000Command.CONFIG_FPGA_GEN,
    "packet": Dca1000Command.CONFIG_PACKET_DATA,
    "start": Dca1000Command.RECORD_START,
    "start_record": Dca1000Command.RECORD_START,
    "stop": Dca1000Command.RECORD_STOP,
    "stop_record": Dca1000Command.RECORD_STOP,
    "version": Dca1000Command.READ_FPGA_VERSION,
}


@dataclass(frozen=True)
class Dca1000CommandResult:
    """Result of a UDP command sent to DCA1000."""

    command: str
    command_code: int
    ok: bool
    response_hex: str
    elapsed_s: float


def _config_root(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("DCA1000Config")
    return section if isinstance(section, dict) else config


def network_from_config(config: dict[str, Any]) -> Dca1000NetworkConfig:
    """Build network config from TI-style DCA1000 JSON."""

    root = _config_root(config)
    eth = root.get("ethernetConfig") or {}
    update = root.get("ethernetConfigUpdate") or {}
    pc_ip = str(update.get("systemIPAddress") or root.get("pc_ip") or DCA1000_HOST_IP)
    dca_ip = str(eth.get("DCA1000IPAddress") or update.get("DCA1000IPAddress") or root.get("dca_ip") or DCA1000_DEVICE_IP)
    config_port = int(eth.get("DCA1000ConfigPort") or update.get("DCA1000ConfigPort") or root.get("config_port") or 4096)
    data_port = int(eth.get("DCA1000DataPort") or update.get("DCA1000DataPort") or root.get("data_port") or 4098)
    return Dca1000NetworkConfig(pc_ip=pc_ip, dca_ip=dca_ip, config_port=config_port, data_port=data_port)


def load_dca_config(path: str | Path) -> dict[str, Any]:
    """Load a TI-style DCA1000 JSON config file."""

    return json.loads(Path(path).read_text())


def _mode_value(value: Any, mapping: dict[str, int], default: int) -> int:
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if text.isdigit():
        return int(text)
    return mapping.get(text, default)


def _payload_from_hex(value: Any) -> bytes:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.replace("0x", "").replace(",", " ").replace(":", " ")
    return bytes.fromhex("".join(cleaned.split()))


def default_fpga_payload(config: dict[str, Any]) -> bytes:
    """Build a compact FPGA config payload from JSON values.

    The default values match the normal SCAN-U raw ADC flow:
    raw logging, LVDS capture, ethernet stream, 4-lane/complex ADC.
    If your TI package expects a different byte order, put a hex override under:

    ``DCA1000Config.nativeCommandPayloads.fpga``
    """

    root = _config_root(config)
    override = _payload_from_hex((root.get("nativeCommandPayloads") or {}).get("fpga"))
    if override is not None:
        return override

    data_logging_mode = _mode_value(root.get("dataLoggingMode"), {"raw": 1, "multi": 2}, 1)
    lvds_mode = int(root.get("lvdsMode", 2))
    data_transfer_mode = _mode_value(root.get("dataTransferMode"), {"LVDSCapture": 1, "playback": 2}, 1)
    data_capture_mode = _mode_value(root.get("dataCaptureMode"), {"ethernetStream": 2, "sdCard": 1}, 2)
    data_format_mode = int(root.get("dataFormatMode", 3))
    timeout_s = int(root.get("timeout_s", 30))
    return bytes(
        [
            data_logging_mode & 0xFF,
            lvds_mode & 0xFF,
            data_transfer_mode & 0xFF,
            data_capture_mode & 0xFF,
            data_format_mode & 0xFF,
            timeout_s & 0xFF,
        ]
    )


def default_packet_payload(config: dict[str, Any]) -> bytes:
    """Build packet-delay payload for DCA1000 CONFIG_PACKET_DATA.

    TI stores the delay in 8 ns clock ticks, not directly in microseconds.
    Therefore a configured delay of 25 us must be sent as 3125 ticks.
    """

    root = _config_root(config)
    override = _payload_from_hex((root.get("nativeCommandPayloads") or {}).get("packet"))
    if override is not None:
        return override
    packet_size = int(root.get("packetSize", 1472))
    delay_us = int(root.get("packetDelay_us", 25))
    delay_ticks = round(max(0, delay_us) * 1000 / 8)
    return struct.pack(
        "<HHH",
        max(48, min(packet_size, 1472)),
        max(0, min(delay_ticks, 65535)),
        0,
    )


class Dca1000NativeClient:
    """UDP command client for DCA1000EVM with retry support."""

    def __init__(
        self,
        network: Dca1000NetworkConfig = None,
        *,
        timeout_s: float = 2.0,
        retries: int = 2,
    ) -> None:
        self.network = network or Dca1000NetworkConfig()
        self.timeout_s = timeout_s
        self.retries = retries

    def send_command(
        self,
        command: str | int,
        payload: bytes = b"",
        *,
        retries: Optional[int] = None,
    ) -> Dca1000CommandResult:
        command_name = str(command)
        command_code = COMMAND_NAMES.get(command_name, command if isinstance(command, int) else -1)
        if not isinstance(command_code, int) or command_code < 0:
            raise ValueError(f"unknown DCA1000 command: {command}")

        packet = self._build_packet(command_code, payload)
        max_attempts = (retries if retries is not None else self.retries) + 1
        last_response = b""
        last_ok = False

        start = time.monotonic()
        for attempt in range(max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((self.network.pc_ip, self.network.config_port))
                    sock.settimeout(self.timeout_s)
                    sock.sendto(packet, (self.network.dca_ip, self.network.config_port))
                    response, _addr = sock.recvfrom(2048)
                    last_response = response
                    last_ok = self._response_ok(response, command_code)
                    if last_ok:
                        break
            except socket.timeout:
                last_ok = False
                if attempt < max_attempts - 1:
                    logger.warning(
                        "DCA1000 %r timeout (attempt %d/%d), retrying...",
                        command_name,
                        attempt + 1,
                        max_attempts,
                    )
                    time.sleep(0.2)
                continue
            except OSError as e:
                last_ok = False
                logger.warning("DCA1000 %r socket error: %s", command_name, e)
                if attempt < max_attempts - 1:
                    time.sleep(0.5)
                continue

        return Dca1000CommandResult(
            command=command_name,
            command_code=command_code,
            ok=last_ok,
            response_hex=last_response.hex(" "),
            elapsed_s=time.monotonic() - start,
        )

    def configure_from_json(
        self,
        config: dict[str, Any],
        *,
        reset: bool = True,
        connect_first: bool = False,
    ) -> list[Dca1000CommandResult]:
        """Run the standard SCAN-U DCA1000 setup sequence.

        The typical DCA1000 initialization sequence is:

        1. **connect** — establish communication with the board
        2. **reset_fpga** — reset the FPGA to a known state
        3. **fpga** — configure FPGA (LVDS mode, data format, etc.)
        4. **packet** — set packet delay/us parameters

        Some firmware versions need a short delay between commands.
        """

        sequence: list[tuple[str, bytes]] = []
        if connect_first:
            sequence.append(("connect", b""))
        if reset:
            sequence.append(("reset_fpga", b""))
        sequence.extend(
            [
                ("fpga", default_fpga_payload(config)),
                ("packet", default_packet_payload(config)),
            ]
        )
        results: list[Dca1000CommandResult] = []
        for name, payload in sequence:
            results.append(self.send_command(name, payload))
            if not results[-1].ok:
                logger.warning("DCA1000 %r failed; aborting configuration sequence", name)
                break
            time.sleep(0.05)
        return results

    @staticmethod
    def _build_packet(command_code: int, payload: bytes) -> bytes:
        return struct.pack("<HHH", HEADER, command_code, len(payload)) + payload + struct.pack("<H", FOOTER)

    @staticmethod
    def _response_ok(response: bytes, expected_command: int) -> bool:
        if len(response) < 8:
            return False
        if response[:2] != HEADER.to_bytes(2, "little"):
            return False
        if int.from_bytes(response[2:4], "little") != expected_command:
            return False
        if response[-2:] != FOOTER.to_bytes(2, "little"):
            return False
        if expected_command == Dca1000Command.READ_FPGA_VERSION:
            return True
        return int.from_bytes(response[4:6], "little") == 0


def _print_results(results: Iterable[Dca1000CommandResult]) -> bool:
    ok_all = True
    for result in results:
        ok_all = ok_all and result.ok
        status = "OK" if result.ok else "FAIL"
        print(
            f"{status} command={result.command} code=0x{result.command_code:02x} "
            f"elapsed_s={result.elapsed_s:.3f} response={result.response_hex or '<none>'}"
        )
    return ok_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jetson-native DCA1000EVM UDP control")
    parser.add_argument("command", choices=[*COMMAND_NAMES.keys(), "configure"], help="DCA1000 command to send")
    parser.add_argument("--config", default="", help="TI-style DCA1000 JSON config")
    parser.add_argument("--pc-ip", default="", help="Jetson Ethernet IP")
    parser.add_argument("--dca-ip", default="", help="DCA1000 board IP")
    parser.add_argument("--config-port", type=int, default=0)
    parser.add_argument("--data-port", type=int, default=0)
    parser.add_argument("--payload-hex", default="", help="Optional raw payload override for one command")
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=2, help="Retry count for each command")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--connect-first", action="store_true",
                        help="Send SYSTEM_CONNECT before reset (disabled by default for SCANU firmware)")
    parser.add_argument("--no-connect", action="store_true",
                        help="Deprecated compatibility flag; connect is already disabled by default")
    parser.add_argument("--inter-delay", type=float, default=0.05, help="Delay (s) between configure steps")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    config = load_dca_config(args.config) if args.config else {}
    network = network_from_config(config) if config else Dca1000NetworkConfig()
    network = Dca1000NetworkConfig(
        pc_ip=args.pc_ip or network.pc_ip,
        dca_ip=args.dca_ip or network.dca_ip,
        config_port=args.config_port or network.config_port,
        data_port=args.data_port or network.data_port,
    )
    client = Dca1000NativeClient(network, timeout_s=args.timeout_s, retries=args.retries)

    if args.command == "configure":
        results = client.configure_from_json(
            config, reset=not args.no_reset,
            connect_first=bool(args.connect_first and not args.no_connect),
        )
        return 0 if _print_results(results) else 2

    payload = _payload_from_hex(args.payload_hex) or b""
    if args.command == "fpga" and not payload:
        payload = default_fpga_payload(config)
    elif args.command == "packet" and not payload:
        payload = default_packet_payload(config)
    result = client.send_command(args.command, payload)
    return 0 if _print_results([result]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
