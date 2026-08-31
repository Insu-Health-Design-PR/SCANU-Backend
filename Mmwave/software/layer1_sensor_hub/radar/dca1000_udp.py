"""UDP recorder for DCA1000EVM ADC data packets."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Optional

import numpy as np

DCA1000_HOST_IP = "192.168.33.30"
DCA1000_DEVICE_IP = "192.168.33.180"
DCA1000_CONFIG_PORT = 4096
DCA1000_DATA_PORT = 4098  # was: from software.settings


@dataclass(frozen=True)
class Dca1000NetworkConfig:
    pc_ip: str = DCA1000_HOST_IP
    dca_ip: str = DCA1000_DEVICE_IP
    config_port: int = DCA1000_CONFIG_PORT
    data_port: int = DCA1000_DATA_PORT
    packet_size: int = 4096
    socket_timeout_s: float = 1.0


@dataclass(frozen=True)
class UdpCaptureResult:
    """Summary of one UDP ADC recording."""

    output_path: Path
    packets: int
    payload_bytes: int
    elapsed_s: float
    timed_out: bool


class UdpDca1000Recorder:
    """Listen for DCA1000 data UDP packets and write their payload to disk.

    This class records the data stream. The DCA1000 board still needs to be
    configured/started by mmWave Studio, TI's DCA1000 CLI, or an equivalent
    board-control tool before packets will arrive.
    """

    def __init__(self, network: Dca1000NetworkConfig = None) -> None:
        self.network = network or Dca1000NetworkConfig()

    def capture(
        self,
        output_path: str | Path,
        *,
        duration_s: Optional[float] = None,
        max_packets: int = 0,
        strip_packet_header: bool = True,
        on_ready: Optional[Callable[[], None]] = None,
    ) -> UdpCaptureResult:
        """Capture UDP data packets to ``output_path``.

        Args:
            duration_s: Stop after this many seconds. ``None`` waits until the
                socket times out.
            max_packets: Optional packet limit. ``0`` means no packet limit.
            strip_packet_header: DCA1000 data packets commonly include a
                10-byte sequence/byte-count header before ADC payload. Keep
                this enabled for ``adc_data.bin`` compatible output.
        """

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        packets = 0
        payload_bytes = 0
        timed_out = False
        start = time.monotonic()

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.network.pc_ip, self.network.data_port))
            sock.settimeout(self.network.socket_timeout_s)

            with out.open("wb") as fh:
                if on_ready is not None:
                    on_ready()

                while True:
                    if duration_s is not None and (time.monotonic() - start) >= duration_s:
                        break
                    if max_packets and packets >= max_packets:
                        break
                    try:
                        packet, _addr = sock.recvfrom(self.network.packet_size + 64)
                    except socket.timeout:
                        timed_out = True
                        break

                    payload = self._extract_payload(packet, strip_packet_header=strip_packet_header)
                    self._write_payload(fh, payload)
                    packets += 1
                    payload_bytes += len(payload)

        return UdpCaptureResult(
            output_path=out,
            packets=packets,
            payload_bytes=payload_bytes,
            elapsed_s=time.monotonic() - start,
            timed_out=timed_out,
        )

    @staticmethod
    def _extract_payload(packet: bytes, *, strip_packet_header: bool) -> bytes:
        if strip_packet_header and len(packet) > 10:
            return packet[10:]
        return packet

    @staticmethod
    def _write_payload(fh: BinaryIO, payload: bytes) -> None:
        fh.write(payload)
