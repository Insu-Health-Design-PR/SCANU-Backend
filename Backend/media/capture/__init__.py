"""Capture backends."""

from media.capture.ffmpeg_cuda_webcam import (
    FFmpegCudaWebcamCapture,
    ffmpeg_cuda_available,
    open_ffmpeg_cuda_webcam,
)
from media.capture.gstreamer_webcam import GStreamerWebcamCapture, gstreamer_available
from media.capture.live_webcam_capture import LiveWebcamCapture
from media.capture.v4l2 import V4L2ThermalCapture

__all__ = [
    "FFmpegCudaWebcamCapture",
    "GStreamerWebcamCapture",
    "LiveWebcamCapture",
    "V4L2ThermalCapture",
    "ffmpeg_cuda_available",
    "gstreamer_available",
    "open_ffmpeg_cuda_webcam",
]
