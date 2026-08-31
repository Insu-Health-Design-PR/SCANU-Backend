"""Short center-radar/DCA1000 stream diagnostic with guaranteed cleanup."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

from layer1_sensor_hub.hardware_registry import discover_radar_pairs

from .dca1000_control import Dca1000NativeClient, load_dca_config, network_from_config
from .radar_cli import RadarCliConfig, configure_radar_from_file, send_sensor_start, send_sensor_stop


def _require(responses: list[str], operation: str) -> None:
    failures = [value for value in responses if not value or "error" in value.lower()]
    if failures:
        raise RuntimeError(f"{operation} failed: {failures[:3]}")


def run_dca1000_smoke(
    *,
    duration_s: float = 3.0,
    cli_port: str = "",
    radar_config: str | Path,
    dca_config: str | Path,
) -> dict[str, object]:
    """Configure, stream and stop the center radar, returning packet integrity."""

    if not cli_port:
        center = discover_radar_pairs().get("center")
        if center is None:
            raise RuntimeError("center radar not found by serial/probe")
        cli_port = center.cli
    loaded_dca_config = load_dca_config(dca_config)
    network = network_from_config(loaded_dca_config)
    client = Dca1000NativeClient(network)
    radar = RadarCliConfig(port=cli_port)
    data_socket: socket.socket | None = None
    packets = 0
    payload_bytes = 0
    sequences: list[int] = []
    error = ""
    try:
        results = client.configure_from_json(loaded_dca_config)
        failed = [row.command for row in results if not row.ok]
        if failed:
            raise RuntimeError(f"DCA configuration failed: {failed}")
        _require(configure_radar_from_file(radar, Path(radar_config), defer_sensor_start=True), "radar config")
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        data_socket.bind((network.pc_ip, network.data_port))
        data_socket.settimeout(0.25)
        if not client.send_command("start_record").ok:
            raise RuntimeError("DCA start_record failed")
        _require(send_sensor_start(radar), "sensorStart")
        deadline = time.monotonic() + max(duration_s, 0.1)
        while time.monotonic() < deadline:
            try:
                packet, _ = data_socket.recvfrom(4096)
            except socket.timeout:
                continue
            if len(packet) <= 10:
                continue
            packets += 1
            payload_bytes += len(packet) - 10
            sequences.append(int.from_bytes(packet[:4], "little"))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            send_sensor_stop(radar)
        except Exception:
            pass
        try:
            client.send_command("stop_record")
        except Exception:
            pass
        if data_socket is not None:
            data_socket.close()

    missing = sum(max(0, b - a - 1) for a, b in zip(sequences, sequences[1:]))
    if not error and packets == 0:
        error = (
            "No DCA1000 UDP data packets received on port "
            f"{network.data_port}; verify LVDS cabling and DCA1000 SW2.5 software-config mode"
        )
    report = {
        "ok": not error and packets > 0 and missing == 0,
        "error": error,
        "cli_port": cli_port,
        "duration_s": duration_s,
        "packets": packets,
        "payload_bytes": payload_bytes,
        "first_sequence": sequences[0] if sequences else None,
        "last_sequence": sequences[-1] if sequences else None,
        "missing_packets": missing,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="SCAN-U center DCA1000 packet smoke test")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--cli-port", default="")
    parser.add_argument("--radar-config", default=str(root / "configs" / "weapon_detection_dca1000.cfg"))
    parser.add_argument("--dca-config", default=str(root / "configs" / "dca1000_config.json"))
    parser.add_argument("--output", default="dca1000_smoke.json")
    args = parser.parse_args(argv)
    report = run_dca1000_smoke(
        duration_s=args.duration_s,
        cli_port=args.cli_port,
        radar_config=args.radar_config,
        dca_config=args.dca_config,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
