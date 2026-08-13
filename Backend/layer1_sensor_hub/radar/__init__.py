"""Layer 1 Radar — UART/TLV subset used by ``lab.mmwave77_usb`` (no DCA1000).

Import submodules directly when possible::

    from layer1_sensor_hub.radar.radar_constants import MAGIC_WORD
    from layer1_sensor_hub.radar.tlv_parser import TLVParser
"""

from __future__ import annotations

from .radar_constants import (
    FRAME_HEADER_SIZE,
    MAGIC_WORD,
    POINT_SIZE,
    POINT_SIDE_INFO_SIZE,
    TLVType,
    SerialConfig,
)
from .tlv_parser import DetectedPoint, ParsedFrame, TLVParser
from .uart_source import FrameHeader

__all__ = [
    "FRAME_HEADER_SIZE",
    "MAGIC_WORD",
    "POINT_SIZE",
    "POINT_SIDE_INFO_SIZE",
    "SerialConfig",
    "TLVParser",
    "TLVType",
    "DetectedPoint",
    "ParsedFrame",
    "FrameHeader",
]
