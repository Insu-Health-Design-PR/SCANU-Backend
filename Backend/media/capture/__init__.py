"""Capture backends."""

from media.capture.gstreamer_webcam import GStreamerWebcamCapture, gstreamer_available
from media.capture.live_webcam_capture import LiveWebcamCapture
from media.capture.v4l2 import V4L2ThermalCapture

__all__ = [
    "GStreamerWebcamCapture",
    "LiveWebcamCapture",
    "V4L2ThermalCapture",
    "gstreamer_available",
]
