from __future__ import annotations
"""
TLV (Type-Length-Value) parser for radar frames.

Parses the TLV data structures from IWR6843 frames to extract:
- Detected points (x, y, z, doppler)
- Range profile
- Noise profile
- Azimuth static heatmap (range-azimuth 2D)
- Range-doppler heatmap (range-doppler 2D)
- Azimuth-elevation heatmap (3D)
- Statistics
- Side info (SNR, noise per point)
"""

import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .radar_constants import FRAME_HEADER_SIZE, POINT_SIDE_INFO_SIZE, POINT_SIZE, TLVType
from .uart_source import FrameHeader

logger = logging.getLogger(__name__)


@dataclass
class DetectedPoint:
    """A single detected point from the radar."""

    x: float  # X position in meters
    y: float  # Y position in meters (range direction)
    z: float  # Z position in meters (elevation)
    doppler: float  # Velocity in m/s (positive = approaching)
    snr: float = 0.0  # Signal-to-noise ratio (dB)
    noise: float = 0.0  # Noise level

    @property
    def range(self) -> float:
        """Calculate range (distance) from radar."""

        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def azimuth_deg(self) -> float:
        """Calculate azimuth angle in degrees."""

        return np.degrees(np.arctan2(self.x, self.y))

    @property
    def elevation_deg(self) -> float:
        """Calculate elevation angle in degrees."""

        r_xy = np.sqrt(self.x**2 + self.y**2)
        return np.degrees(np.arctan2(self.z, r_xy))

    def to_dict(self) -> dict:
        """Convert to dictionary."""

        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "doppler": self.doppler,
            "range": self.range,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "snr": self.snr,
            "noise": self.noise,
        }


@dataclass
class ParsedFrame:
    """Completely parsed radar frame with all extracted data."""

    frame_number: int
    num_detected_obj: int
    num_tlvs: int
    timestamp_cycles: int

    points: List[DetectedPoint] = field(default_factory=list)
    range_profile: Optional[np.ndarray] = None
    noise_profile: Optional[np.ndarray] = None
    azimuth_static_heatmap: Optional[np.ndarray] = None
    range_doppler_heatmap: Optional[np.ndarray] = None
    azimuth_elevation_heatmap: Optional[np.ndarray] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    raw_tlvs: Dict[int, bytes] = field(default_factory=dict)

    def get_point_cloud(self) -> np.ndarray:
        """Get points as Nx4 numpy array [x, y, z, doppler]."""

        if not self.points:
            return np.zeros((0, 4), dtype=np.float32)

        return np.array([[p.x, p.y, p.z, p.doppler] for p in self.points], dtype=np.float32)

    def get_point_cloud_with_snr(self) -> np.ndarray:
        """Get points as Nx6 numpy array [x, y, z, doppler, snr, noise]."""

        if not self.points:
            return np.zeros((0, 6), dtype=np.float32)

        return np.array([[p.x, p.y, p.z, p.doppler, p.snr, p.noise] for p in self.points], dtype=np.float32)

    def __str__(self) -> str:
        return (
            f"Frame #{self.frame_number}: "
            f"{len(self.points)} points, "
            f"range_profile={'yes' if self.range_profile is not None else 'no'}, "
            f"range_doppler={'yes' if self.range_doppler_heatmap is not None else 'no'}, "
            f"azimuth_static={'yes' if self.azimuth_static_heatmap is not None else 'no'}"
        )


class TLVParser:
    """Parses TLV frames from IWR6843 radar."""

    def __init__(self):
        self._frames_parsed = 0
        self._parse_errors = 0

    def parse(self, frame: bytes) -> ParsedFrame:
        """Parse a complete frame."""

        if len(frame) < FRAME_HEADER_SIZE:
            raise ValueError(f"Frame too short: {len(frame)} bytes")

        header = FrameHeader.from_bytes(frame[:FRAME_HEADER_SIZE])

        # Validate header — reject corrupted frames
        if header.num_tlvs < 1 or header.num_tlvs > 20:
            raise ValueError(f"Invalid num_tlvs: {header.num_tlvs}")
        if header.num_detected_obj < 0 or header.num_detected_obj > 500:
            raise ValueError(f"Invalid num_detected_obj: {header.num_detected_obj}")

        result = ParsedFrame(
            frame_number=header.frame_number,
            num_detected_obj=header.num_detected_obj,
            num_tlvs=header.num_tlvs,
            timestamp_cycles=header.time_cpu_cycles,
        )

        offset = FRAME_HEADER_SIZE

        for _ in range(header.num_tlvs):
            if offset + 8 > len(frame):
                break

            tlv_type, tlv_length = struct.unpack("<II", frame[offset : offset + 8])

            # Validate TLV type — skip garbage
            if tlv_type not in (1, 2, 3, 4, 5, 6, 7, 8):
                offset += 1
                continue

            offset += 8

            if offset + tlv_length > len(frame):
                break

            tlv_data = frame[offset : offset + tlv_length]
            offset += tlv_length

            result.raw_tlvs[tlv_type] = tlv_data

            try:
                self._parse_tlv(tlv_type, tlv_data, result)
            except Exception as e:
                self._parse_errors += 1

        self._frames_parsed += 1
        return result

    def _parse_tlv(self, tlv_type: int, data: bytes, result: ParsedFrame) -> None:
        if tlv_type == TLVType.DETECTED_POINTS:
            self._parse_detected_points(data, result)
        elif tlv_type == TLVType.RANGE_PROFILE:
            self._parse_range_profile(data, result)
        elif tlv_type == TLVType.NOISE_PROFILE:
            self._parse_noise_profile(data, result)
        elif tlv_type == TLVType.DETECTED_POINTS_SIDE_INFO:
            self._parse_side_info(data, result)
        elif tlv_type == TLVType.STATS:
            self._parse_stats(data, result)
        elif tlv_type == TLVType.AZIMUTH_STATIC_HEATMAP:
            self._parse_azimuth_static_heatmap(data, result)
        elif tlv_type == TLVType.RANGE_DOPPLER_HEATMAP:
            self._parse_range_doppler_heatmap(data, result)
        elif tlv_type == TLVType.AZIMUTH_ELEVATION_HEATMAP:
            self._parse_azimuth_elevation_heatmap(data, result)
        else:
            logger.debug(f"Unhandled TLV type {tlv_type} ({len(data)} bytes)")

    def _parse_detected_points(self, data: bytes, result: ParsedFrame) -> None:
        num_points = len(data) // POINT_SIZE

        for i in range(num_points):
            offset = i * POINT_SIZE
            x, y, z, doppler = struct.unpack("<4f", data[offset : offset + POINT_SIZE])
            result.points.append(DetectedPoint(x=x, y=y, z=z, doppler=doppler))

        logger.debug(f"Parsed {num_points} detected points")

    def _parse_range_profile(self, data: bytes, result: ParsedFrame) -> None:
        result.range_profile = np.frombuffer(data, dtype=np.uint16).astype(np.float32)

    def _parse_noise_profile(self, data: bytes, result: ParsedFrame) -> None:
        result.noise_profile = np.frombuffer(data, dtype=np.uint16).astype(np.float32)

    def _parse_side_info(self, data: bytes, result: ParsedFrame) -> None:
        num_entries = len(data) // POINT_SIDE_INFO_SIZE

        for i in range(min(num_entries, len(result.points))):
            offset = i * POINT_SIDE_INFO_SIZE
            snr, noise = struct.unpack("<2H", data[offset : offset + POINT_SIDE_INFO_SIZE])
            result.points[i].snr = snr / 10.0
            result.points[i].noise = float(noise)

    def _parse_stats(self, data: bytes, result: ParsedFrame) -> None:
        if len(data) < 24:
            return
        inter_frame_proc_time, transmit_out_time, inter_frame_proc_margin, inter_chirp_proc_margin, active_frame_cpu_load, inter_frame_cpu_load = struct.unpack(
            "<6I", data[:24]
        )
        result.stats.update(
            {
                "inter_frame_proc_time": inter_frame_proc_time,
                "transmit_out_time": transmit_out_time,
                "inter_frame_proc_margin": inter_frame_proc_margin,
                "inter_chirp_proc_margin": inter_chirp_proc_margin,
                "active_frame_cpu_load": active_frame_cpu_load,
                "inter_frame_cpu_load": inter_frame_cpu_load,
            }
        )

    def _parse_azimuth_static_heatmap(self, data: bytes, result: ParsedFrame) -> None:
        heatmap = np.frombuffer(data, dtype=np.uint16).astype(np.float32)
        heatmap = heatmap / 2.0
        result.azimuth_static_heatmap = heatmap
        logger.debug(f"Parsed azimuth static heatmap: {heatmap.shape}")

    def _parse_range_doppler_heatmap(self, data: bytes, result: ParsedFrame) -> None:
        heatmap = np.frombuffer(data, dtype=np.uint16).astype(np.float32)
        heatmap = heatmap / 2.0
        result.range_doppler_heatmap = heatmap
        logger.debug(f"Parsed range-doppler heatmap: {heatmap.shape}")

    def _parse_azimuth_elevation_heatmap(self, data: bytes, result: ParsedFrame) -> None:
        heatmap = np.frombuffer(data, dtype=np.uint16).astype(np.float32)
        heatmap = heatmap / 2.0
        result.azimuth_elevation_heatmap = heatmap
        logger.debug(f"Parsed azimuth-elevation heatmap: {heatmap.shape}")


def parse_frame(frame: bytes) -> ParsedFrame:
    """Parse a single frame (convenience function)."""

    return TLVParser().parse(frame)

