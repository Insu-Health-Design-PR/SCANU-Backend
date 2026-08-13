from __future__ import annotations
"""
Radar configuration via CLI commands.

Sends configuration commands to the IWR6843 over the config UART port.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .radar_constants import SerialConfig
from .serial_manager import SerialManager

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = """
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60.75 7 7 57.14 0 0 70 1 256 5000 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
chirpCfg 2 2 0 0 0 0 0 2
frameCfg 0 2 16 0 100 1 0
lowPower 0 0
guiMonitor -1 1 1 0 0 0 1
cfarCfg -1 0 2 8 4 3 0 15.0 0
cfarCfg -1 1 0 4 2 3 1 15.0 0
multiObjBeamForming -1 1 0.5
clutterRemoval -1 0
calibDcRangeSig -1 0 -5 8 256
extendedMaxVelocity -1 0
compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
measureRangeBiasAndRxChanPhase 0 1.5 0.2
CQRxSatMonitor 0 3 5 121 0
CQSigImgMonitor 0 127 4
analogMonitor 0 0
aoaFovCfg -1 -90 90 -90 90
cfarFovCfg -1 0 0.25 9.0
cfarFovCfg -1 1 -1 1.0
sensorStart
""".strip()

WPN_CONFIG = """
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60.75 7 7 57.14 0 0 70 1 256 5000 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
chirpCfg 2 2 0 0 0 0 0 2
frameCfg 0 2 64 0 80 1 0
lowPower 0 0
guiMonitor -1 1 1 1 1 0 1
cfarCfg -1 0 2 8 4 3 0 6.0 0
cfarCfg -1 1 0 4 2 3 1 8.0 0
multiObjBeamForming -1 1 0.5
clutterRemoval -1 1
calibDcRangeSig -1 0 -5 8 256
extendedMaxVelocity -1 0
compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
measureRangeBiasAndRxChanPhase 0 1.5 0.2
CQRxSatMonitor 0 3 5 121 0
CQSigImgMonitor 0 127 4
analogMonitor 0 0
aoaFovCfg -1 -90 90 -90 90
cfarFovCfg -1 0 0.15 6.0
cfarFovCfg -1 1 -5 5.0
sensorStart
""".strip()

# Minimal commands to get sensor running (first start)
FIRST_START_CONFIG = """
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60.75 7 7 57.14 0 0 70 1 256 5000 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
chirpCfg 2 2 0 0 0 0 0 2
frameCfg 0 2 16 0 100 1 0
lowPower 0 0
sensorStart
""".strip()

WPN_FIRST_START = """
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60.75 7 7 57.14 0 0 70 1 256 5000 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
chirpCfg 2 2 0 0 0 0 0 2
frameCfg 0 2 64 0 80 1 0
lowPower 0 0
clutterRemoval -1 1
sensorStart
""".strip()

# Optional commands applied after start
POST_START_CONFIG = """
guiMonitor -1 1 1 0 0 0 1
cfarCfg -1 0 2 8 4 3 0 15.0 0
cfarCfg -1 1 0 4 2 3 1 15.0 0
multiObjBeamForming -1 1 0.5
clutterRemoval -1 0
calibDcRangeSig -1 0 -5 8 256
extendedMaxVelocity -1 0
compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
measureRangeBiasAndRxChanPhase 0 1.5 0.2
CQRxSatMonitor 0 3 5 121 0
CQSigImgMonitor 0 127 4
analogMonitor 0 0
aoaFovCfg -1 -90 90 -90 90
cfarFovCfg -1 0 0.25 9.0
cfarFovCfg -1 1 -1 1.0
""".strip()

WPN_POST_START = """
guiMonitor -1 1 1 1 1 0 1
cfarCfg -1 0 2 8 4 3 0 6.0 0
cfarCfg -1 1 0 4 2 3 1 8.0 0
multiObjBeamForming -1 1 0.5
clutterRemoval -1 1
calibDcRangeSig -1 0 -5 8 256
extendedMaxVelocity -1 0
compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
measureRangeBiasAndRxChanPhase 0 1.5 0.2
CQRxSatMonitor 0 3 5 121 0
CQSigImgMonitor 0 127 4
analogMonitor 0 0
aoaFovCfg -1 -90 90 -90 90
cfarFovCfg -1 0 0.15 6.0
cfarFovCfg -1 1 -5 5.0
""".strip()


@dataclass
class ConfigResult:
    """Result of configuration attempt."""

    success: bool
    commands_sent: int
    errors: List[str]
    responses: List[str]


class RadarConfigurator:
    """
    Configures the IWR6843 radar via CLI commands.

    Usage:
        configurator = RadarConfigurator(serial_manager)
        result = configurator.configure_first_start()  # Handles first start properly
        result = configurator.configure_post_start()   # Apply optional commands
        configurator.stop()
    """

    def __init__(self, serial_manager: SerialManager):
        self.serial = serial_manager
        self._is_running = False

    def send_command(
        self,
        command: str,
        delay: float = SerialConfig.COMMAND_DELAY,
        response_timeout: float = 3.0,
    ) -> str:
        if not self.serial.config_port or not self.serial.config_port.is_open:
            raise RuntimeError("Config port not connected")

        cmd = command.strip()
        if not cmd or cmd.startswith("%"):
            return ""

        try:
            if self.serial.config_port.in_waiting:
                self.serial.config_port.read(self.serial.config_port.in_waiting)
        except Exception:
            pass

        self.serial.config_port.write((cmd + "\r\n").encode("utf-8"))
        if delay > 0:
            time.sleep(delay)

        response_chunks: List[str] = []
        start = time.time()
        while (time.time() - start) < response_timeout:
            try:
                waiting = self.serial.config_port.in_waiting
            except Exception:
                waiting = 0

            if waiting > 0:
                chunk = self.serial.config_port.read(waiting).decode("utf-8", errors="ignore")
                if chunk:
                    response_chunks.append(chunk)
                    combined = "".join(response_chunks)
                    if "mmwDemo:/>" in combined or combined.strip().endswith(">"):
                        break
            else:
                # If the device is wedged and never responds, waiting the full
                # timeout for every command can look like a "hang" during config.
                # Exit early when *nothing* has been received for a short grace period.
                if not response_chunks and (time.time() - start) > 0.35:
                    break
                time.sleep(0.01)

        response = "".join(response_chunks)
        logger.debug(f"CMD: {cmd}")
        if response.strip():
            logger.debug(f"RSP: {response.strip()}")
        return response

    def _parse_config_string(self, config: str) -> List[str]:
        commands = []
        for line in config.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("%") and not line.startswith("#"):
                commands.append(line)
        return commands

    def configure(
        self,
        config: Optional[str] = None,
        *,
        max_consecutive_no_response: int = 2,
    ) -> ConfigResult:
        """
        Configure radar from a config string.

        When `config` is None, uses `DEFAULT_CONFIG`.
        """

        config_text = DEFAULT_CONFIG if config is None else config
        commands = self._parse_config_string(config_text)
        logger.info(f"Sending {len(commands)} commands...")

        errors: List[str] = []
        responses: List[str] = []
        consecutive_no_response = 0

        for i, cmd in enumerate(commands):
            response = self.send_command(cmd)
            responses.append(response)

            if not response.strip():
                # Empty response is treated as a transport/config failure because
                # CLI commands should normally emit prompt/ack/error text.
                error_msg = f"Command {i+1} '{cmd}': no response from sensor"
                errors.append(error_msg)
                logger.error(error_msg)
                consecutive_no_response += 1
                if consecutive_no_response >= max(1, int(max_consecutive_no_response)):
                    logger.error(
                        "Aborting radar configuration after %d consecutive empty CLI responses; "
                        "this is a UART/boot-state failure, not a rejected cfg command",
                        consecutive_no_response,
                    )
                    break
            elif "Error" in response or "error" in response:
                error_msg = f"Command {i+1} '{cmd}': {response.strip()}"
                errors.append(error_msg)
                logger.error(error_msg)
                consecutive_no_response = 0
            else:
                consecutive_no_response = 0

            if cmd.lower() == "sensorstart" and response.strip() and "error" not in response.lower():
                self._is_running = True
            elif cmd.lower() == "sensorstop" and response.strip() and "error" not in response.lower():
                self._is_running = False

        success = len(errors) == 0
        if success:
            logger.info(f"Configuration complete: {len(commands)} commands sent")
        else:
            logger.error(f"Configuration had {len(errors)} errors")

        return ConfigResult(
            success=success,
            commands_sent=len(responses),
            errors=errors,
            responses=responses,
        )

    def configure_first_start(self) -> ConfigResult:
        """Send first-start minimal configuration to radar."""

        return self.configure(FIRST_START_CONFIG)

    def configure_post_start(self) -> ConfigResult:
        """Apply optional commands after sensor is running."""

        return self.configure(POST_START_CONFIG)

    def configure_weapon(self) -> ConfigResult:
        """Apply weapon-optimised configuration (192 chirps, low CFAR, clutter removal)."""

        logger.info("Applying weapon-optimised radar config")
        first = self.configure(WPN_FIRST_START)
        if not first.success:
            return first
        return self.configure(WPN_POST_START)

    def configure_from_file(self, config_path: Path) -> ConfigResult:
        commands = self._parse_config_string(Path(config_path).read_text())
        return self.configure("\n".join(commands))

    def stop(self) -> str:
        response = self.send_command("sensorStop")
        self._is_running = False
        logger.info("Radar stopped")
        return response

    def start(self) -> str:
        response = self.send_command("sensorStart")
        self._is_running = True
        logger.info("Radar started")
        return response

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_version(self) -> str:
        return self.send_command("version")
