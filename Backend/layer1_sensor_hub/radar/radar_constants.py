from __future__ import annotations
"""
Constants for IWR6843 radar communication.

Reference: TI mmWave SDK documentation
"""

# Magic word that starts every TLV frame (8 bytes)
# This is how we find the start of each frame in the UART stream
MAGIC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

# Frame header size in bytes (after magic word)
FRAME_HEADER_SIZE = 40  # Total header including magic word


class TLVType:
    """TLV type identifiers for different data outputs."""

    DETECTED_POINTS = 1  # List of detected objects (x, y, z, doppler)
    RANGE_PROFILE = 2  # 1D range profile (energy vs range)
    NOISE_PROFILE = 3  # Noise floor profile
    AZIMUTH_STATIC_HEATMAP = 4  # Range-azimuth heatmap
    RANGE_DOPPLER_HEATMAP = 5  # Range-doppler heatmap
    STATS = 6  # Processing statistics
    DETECTED_POINTS_SIDE_INFO = 7  # SNR and noise for each point
    AZIMUTH_ELEVATION_HEATMAP = 8  # 3D heatmap
    TEMPERATURE_STATS = 9  # Chip temperature


class SerialConfig:
    """Serial port configuration."""

    # Configuration/CLI port
    CONFIG_BAUD = 115200
    CONFIG_TIMEOUT = 1.0  # seconds

    # Data port
    DATA_BAUD = 921600
    DATA_TIMEOUT = 0.1  # seconds

    # Command timing
    COMMAND_DELAY = 0.03  # seconds between commands


class RadarPlatform:
    """Platform identifiers."""

    IWR6843 = 0x6843
    AWR1843 = 0x1843


# Point structure sizes (bytes)
POINT_SIZE = 16  # x(4) + y(4) + z(4) + doppler(4)
POINT_SIDE_INFO_SIZE = 4  # snr(2) + noise(2)

