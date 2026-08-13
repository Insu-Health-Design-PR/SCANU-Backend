"""GStreamer UVC/MJPEG capture via ``gst-launch-1.0`` → raw BGR pipe.

OpenCV in this environment is built without GStreamer, and the project venv cannot
import system ``gi`` (Python 3.13 vs system 3.14). Spawning ``gst-launch`` keeps
decode off the OpenCV path and matches ``LiveWebcamCapture``'s latest-frame API.

Hardware note: NVIDIA DeepStream elements (``nvjpegdec`` / ``nvv4l2decoder``) are
optional. When missing, MJPEG uses CPU ``jpegdec`` — still useful for low-latency
``appsink``-style drop behavior via the pipe reader.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any

import numpy as np

_GST_LAUNCH = "gst-launch-1.0"


def gstreamer_available() -> bool:
    return shutil.which(_GST_LAUNCH) is not None


def nvidia_gst_jpeg_available() -> bool:
    """True when DeepStream-style NVJPEG decode is inspectable."""
    inspect = shutil.which("gst-inspect-1.0")
    if not inspect:
        return False
    try:
        p = subprocess.run(
            [inspect, "nvjpegdec"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return p.returncode == 0 and "Factory Details" in (p.stdout or "")
    except Exception:
        return False


def build_mjpeg_pipeline(
    device: str,
    *,
    width: int,
    height: int,
    fps: float,
    use_nvjpeg: bool | None = None,
) -> list[str]:
    """Return ``gst-launch-1.0`` argv that writes contiguous BGR frames to stdout."""
    if use_nvjpeg is None:
        use_nvjpeg = nvidia_gst_jpeg_available()
    w = max(16, int(width))
    h = max(16, int(height))
    # UVC MJPEG is typically ≤30 fps at 1080p+; requesting 60 often fails negotiation.
    req_fps = float(fps)
    if w >= 1920 and h >= 1080 and req_fps > 30.0:
        req_fps = 30.0
    fps_i = max(1, int(round(req_fps)))
    jpeg_dec = "nvjpegdec" if use_nvjpeg else "jpegdec"
    # Force output size so the reader knows nbytes even if the driver scales.
    return [
        _GST_LAUNCH,
        "-q",
        "v4l2src",
        f"device={device}",
        "do-timestamp=true",
        "!",
        f"image/jpeg,width={w},height={h},framerate={fps_i}/1",
        "!",
        jpeg_dec,
        "!",
        "videoconvert",
        "!",
        "videoscale",
        "!",
        f"video/x-raw,format=BGR,width={w},height={h}",
        "!",
        "fdsink",
        "fd=1",
        "sync=false",
    ]


def build_test_pipeline(*, width: int = 320, height: int = 240, fps: float = 30.0) -> list[str]:
    """``videotestsrc`` pipeline for smoke tests (no camera)."""
    w = max(16, int(width))
    h = max(16, int(height))
    fps_i = max(1, int(round(float(fps))))
    return [
        _GST_LAUNCH,
        "-q",
        "videotestsrc",
        "is-live=true",
        "pattern=ball",
        "!",
        f"video/x-raw,format=BGR,width={w},height={h},framerate={fps_i}/1",
        "!",
        "fdsink",
        "fd=1",
        "sync=false",
    ]


class GStreamerWebcamCapture:
    """Background reader: GStreamer MJPEG decode → latest BGR frame."""

    def __init__(
        self,
        device: str,
        *,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        use_nvjpeg: bool | None = None,
        pipeline_argv: list[str] | None = None,
    ) -> None:
        if not gstreamer_available():
            raise RuntimeError(f"{_GST_LAUNCH} not found on PATH")
        self.device = str(device)
        self.width = max(16, int(width))
        self.height = max(16, int(height))
        self.fps = float(fps)
        self._frame_nbytes = self.width * self.height * 3
        self._use_nvjpeg = nvidia_gst_jpeg_available() if use_nvjpeg is None else bool(use_nvjpeg)
        self._pipeline_argv = (
            list(pipeline_argv)
            if pipeline_argv is not None
            else build_mjpeg_pipeline(
                self.device,
                width=self.width,
                height=self.height,
                fps=self.fps,
                use_nvjpeg=self._use_nvjpeg,
            )
        )
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._last_error: str | None = None
        self._decoder = "nvjpegdec" if self._use_nvjpeg else "jpegdec"

    @property
    def decoder_name(self) -> str:
        return self._decoder

    @property
    def pipeline_description(self) -> str:
        return " ".join(self._pipeline_argv)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._last_error = None
        self._thread = threading.Thread(target=self._run, name="gst-webcam-capture", daemon=True)
        self._thread.start()

    def _spawn(self) -> subprocess.Popen[bytes]:
        env = os.environ.copy()
        # Keep gst chatter off stdout (frame bytes only).
        env.setdefault("GST_DEBUG", "0")
        return subprocess.Popen(
            self._pipeline_argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=self._frame_nbytes * 2,
        )

    def _read_exact(self, stdout: Any, nbytes: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < nbytes and not self._stop.is_set():
            chunk = stdout.read(nbytes - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf) if len(buf) == nbytes else None

    def _run(self) -> None:
        try:
            self._proc = self._spawn()
        except Exception as exc:
            self._last_error = f"gst spawn failed: {exc}"
            return
        assert self._proc.stdout is not None
        stdout = self._proc.stdout
        scratch = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frames_ok = 0
        try:
            while not self._stop.is_set():
                raw = self._read_exact(stdout, self._frame_nbytes)
                if raw is None:
                    break
                np.copyto(scratch, np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3))
                with self._lock:
                    if self._frame is None or self._frame.shape != scratch.shape:
                        self._frame = np.empty_like(scratch)
                    np.copyto(self._frame, scratch)
                    self._seq += 1
                    frames_ok = self._seq
        finally:
            if frames_ok == 0 and self._proc is not None:
                err = ""
                try:
                    if self._proc.stderr is not None:
                        err = self._proc.stderr.read(4000).decode("utf-8", errors="replace")
                except Exception:
                    pass
                rc = self._proc.poll()
                self._last_error = (
                    f"gst pipeline produced no frames (rc={rc}). {err.strip() or self.pipeline_description}"
                )
            self._terminate_proc()

    def wait_first_frame(self, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None and self._seq > 0:
                    return True
            if self._last_error:
                return False
            t = self._thread
            if t is not None and not t.is_alive() and self._seq == 0:
                return False
            time.sleep(0.02)
        return False

    def copy_latest_into(self, dst: np.ndarray) -> int:
        with self._lock:
            if self._frame is None:
                return -1
            if dst.shape != self._frame.shape:
                raise ValueError(f"dst shape {dst.shape} != capture {self._frame.shape}")
            np.copyto(dst, self._frame)
            return int(self._seq)

    def snapshot(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def shape(self) -> tuple[int, ...] | None:
        with self._lock:
            return None if self._frame is None else tuple(self._frame.shape)

    def _terminate_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
        except Exception:
            pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    def stop(self, join_timeout_s: float = 3.0) -> None:
        self._stop.set()
        self._terminate_proc()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=float(join_timeout_s))

    def __enter__(self) -> GStreamerWebcamCapture:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()
