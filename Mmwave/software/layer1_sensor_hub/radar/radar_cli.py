"""Configure a TI mmWave radar over the CLI/control COM port."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import serial


@dataclass(frozen=True)
class RadarCliConfig:
    """Serial settings for the mmWave CLI/control port."""

    port: str
    baud: int = 115200
    timeout_s: float = 0.5
    # TI CLI firmware can still be draining its response when the next
    # command arrives, especially immediately after a power cycle.  A
    # conservative 100 ms inter-command gap avoids silently losing a response
    # (which would otherwise make the DCA smoke test abort before streaming).
    command_delay_s: float = 0.10


def _find_cli_port() -> str:
    """Auto-detect the radar CLI serial port.

    Scans all available serial ports, sends ``version``, and returns the
    first port whose response contains ``xWR68``, ``IWR68``, or ``mmWave``.
    Returns ``None`` if no radar is found.
    """
    import serial.tools.list_ports

    for p in serial.tools.list_ports.comports():
        port = p.device
        try:
            with serial.Serial(port, 115200, timeout=1.0) as ser:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                ser.write(b"version\r\n")
                time.sleep(0.8)
                waiting = ser.in_waiting
                if waiting:
                    resp = ser.read(waiting).decode("utf-8", errors="ignore")
                    if "xWR68" in resp or "IWR68" in resp or "mmWave" in resp or "mmwDemo" in resp:
                        return port
        except (OSError, serial.SerialException):
            continue
    return None


def load_cli_commands(config_path: str | Path) -> List[str]:
    """Load non-empty, non-comment CLI commands from a TI ``.cfg`` file."""

    commands: List[str] = []
    for line in Path(config_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue
        commands.append(line)
    return commands


def send_cli_commands(config: RadarCliConfig, commands: Iterable[str]) -> List[str]:
    """Send CLI commands and return raw text responses."""

    responses: List[str] = []
    with serial.Serial(config.port, config.baud, timeout=config.timeout_s) as ser:
        for command in commands:
            cmd = command.strip()
            if not cmd:
                continue
            ser.write((cmd + "\r\n").encode("utf-8"))
            time.sleep(config.command_delay_s)
            waiting = ser.in_waiting
            response = ser.read(waiting).decode("utf-8", errors="ignore") if waiting else ""
            responses.append(response)
    return responses


def configure_radar_from_file(
    config: RadarCliConfig,
    config_path: str | Path,
    *,
    defer_sensor_start: bool = False,
) -> List[str]:
    """Send a radar ``.cfg`` file to the sensor.

    When recording through DCA1000, pass ``defer_sensor_start=True`` so a
    ``sensorStart`` line inside the config does not fire before Ethernet
    recording is armed.
    """

    commands = load_cli_commands(config_path)
    if defer_sensor_start:
        commands = [cmd for cmd in commands if cmd.lower() != "sensorstart"]
    return send_cli_commands(config, commands)


def send_sensor_start(config: RadarCliConfig) -> List[str]:
    """Start frame transmission after DCA1000 recording is armed."""

    return send_cli_commands(config, ["sensorStart"])


def send_sensor_stop(config: RadarCliConfig) -> List[str]:
    """Stop the radar sensor."""

    return send_cli_commands(config, ["sensorStop"])
