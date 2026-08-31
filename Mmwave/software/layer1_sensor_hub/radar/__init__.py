"""Layer 1 Radar — TI mmWave sensor sources and hardware communication.

Self-contained hardware I/O for DCA1000 (Ethernet UDP) and
USB/TLV radars (UART serial). No re-exports — all files live here.

Sources:
    Dca1000NativeClient  — UDP command client for DCA1000EVM
    Dca1000NetworkConfig — Network configuration (IPs, ports)
    UdpDca1000Recorder   — UDP data packet recorder
    Dca1000StreamProcessor — Real-time UDP stream processor
    SerialManager        — USB serial port management
    RadarConfigurator    — .cfg file parser and radar programmer
    TLVParser            — Binary TLV protocol parser
    UARTSource           — UART frame reader
"""

from __future__ import annotations

from .dca1000_control import Dca1000NativeClient, load_dca_config, network_from_config
from .dca1000_udp import Dca1000NetworkConfig, UdpCaptureResult, UdpDca1000Recorder
from .radar_cli import (
    RadarCliConfig,
    _find_cli_port,
    configure_radar_from_file,
    send_sensor_start,
    send_sensor_stop,
)
from .radar_config import DEFAULT_CONFIG, RadarConfigurator
from .radar_constants import (
    FRAME_HEADER_SIZE,
    MAGIC_WORD,
    POINT_SIZE,
    POINT_SIDE_INFO_SIZE,
    TLVType,
    SerialConfig,
)
from .serial_manager import SerialManager
from .tlv_parser import DetectedPoint, ParsedFrame, TLVParser
from .uart_source import FrameHeader, UARTSource
