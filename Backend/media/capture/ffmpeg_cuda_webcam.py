"""UVC capture via FFmpeg NVIDIA CUVID (mjpeg_cuvid / h264_cuvid) → latest BGR frame.

GStreamer on this host has no NVDEC/NVJPEG plugins. FFmpeg does (``mjpeg_cuvid``,
``h264_cuvid``), so 4K60 MJPEG can be decoded on the GPU instead of CPU ``jpegdec``.

4K BGR @ 60 is ~1.4 GiB/s over a pipe and will not hold 60 FPS in Python overlay.
When ``out_width`` is set (default 1920 for 4K60), ``scale_cuda`` shrinks on GPU
before download so the infer/overlay loop can run at 60.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

import numpy as np

_FFMPEG = "ffmpeg"


def ffmpeg_available() -> bool:
    return shutil.which(_FFMPEG) is not None


def ffmpeg_cuvid_decoders() -> set[str]:
    """Return CUVID decoder names advertised by this FFmpeg build."""
    if not ffmpeg_available():
        return set()
    try:
        p = subprocess.run(
            [_FFMPEG, "-hide_banner", "-decoders"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return set()
    found: set[str] = set()
    blob = (p.stdout or "") + (p.stderr or "")
    for name in ("mjpeg_cuvid", "h264_cuvid"):
        if name in blob:
            found.add(name)
    return found


def ffmpeg_cuda_available() -> bool:
    return bool(ffmpeg_cuvid_decoders())


def _even(n: int) -> int:
    n = max(16, int(n))
    return n if n % 2 == 0 else n - 1


def scaled_output_size(
    width: int,
    height: int,
    out_width: int,
) -> tuple[int, int]:
    """Letterbox-free width scale; both sides even for CUDA/NV12."""
    w = max(16, int(width))
    h = max(16, int(height))
    ow = int(out_width)
    if ow <= 0 or w <= ow:
        return _even(w), _even(h)
    nh = max(16, int(round(h * (ow / float(w)))))
    return _even(ow), _even(nh)


def build_ffmpeg_cuda_argv(
    device: str,
    *,
    width: int,
    height: int,
    fps: float,
    out_width: int,
    out_height: int,
    input_format: str = "mjpeg",
    decoder: str = "mjpeg_cuvid",
    full_pipe_fd: int | None = None,
    full_fps: float = 0.0,
) -> list[str]:
    """Build a low-latency CUVID pipeline that writes packed BGR to stdout.

    When ``full_pipe_fd`` is set, also writes camera-native BGR (for 4K gun crops)
    to that fd at ``full_fps`` (0 = same as preview fps).
    """
    w = _even(width)
    h = _even(height)
    ow = _even(out_width)
    oh = _even(out_height)
    fps_i = max(1, int(round(float(fps))))
    keep_full = full_pipe_fd is not None and (ow != w or oh != h)
    common = [
        _FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-c:v",
        decoder,
        "-f",
        "v4l2",
        "-thread_queue_size",
        "64",
        "-input_format",
        input_format,
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(fps_i),
        "-i",
        str(device),
        "-an",
        "-sn",
    ]
    preview_map = [
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    if keep_full:
        ffps = float(full_fps) if float(full_fps) > 0 else float(fps_i)
        ffps_i = max(1, int(round(min(ffps, float(fps_i)))))
        full_branch = "hwdownload,format=nv12"
        if ffps_i < fps_i:
            full_branch = f"hwdownload,format=nv12,fps={ffps_i}"
        graph = (
            f"[0:v]split=2[g1][g2];"
            f"[g1]scale_cuda={ow}:{oh},hwdownload,format=nv12[p1080];"
            f"[g2]{full_branch}[p4k]"
        )
        return [
            *common,
            "-filter_complex",
            graph,
            "-map",
            "[p1080]",
            *preview_map,
            "-map",
            "[p4k]",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            f"pipe:{int(full_pipe_fd)}",
        ]
    vf = "hwdownload,format=nv12"
    if ow != w or oh != h:
        vf = f"scale_cuda={ow}:{oh},hwdownload,format=nv12"
    return [*common, "-vf", vf, *preview_map]


class FFmpegCudaWebcamCapture:
    """Background reader: FFmpeg CUVID decode → latest BGR frame."""

    def __init__(
        self,
        device: str,
        *,
        width: int = 3840,
        height: int = 2160,
        fps: float = 60.0,
        out_width: int = 1920,
        input_format: str = "mjpeg",
        decoder: str = "mjpeg_cuvid",
        pipeline_argv: list[str] | None = None,
        keep_full_res: bool = False,
        full_fps: float = 30.0,
    ) -> None:
        if not ffmpeg_available():
            raise RuntimeError(f"{_FFMPEG} not found on PATH")
        self.device = str(device)
        self.cam_width = _even(width)
        self.cam_height = _even(height)
        self.fps = float(fps)
        self.width, self.height = scaled_output_size(self.cam_width, self.cam_height, int(out_width))
        self._frame_nbytes = self.width * self.height * 3
        self._decoder = str(decoder)
        self._input_format = str(input_format)
        self.keep_full_res = bool(keep_full_res) and (self.width != self.cam_width or self.height != self.cam_height)
        self.full_width = self.cam_width if self.keep_full_res else self.width
        self.full_height = self.cam_height if self.keep_full_res else self.height
        self.full_fps = float(full_fps) if float(full_fps) > 0 else float(fps)
        self._full_nbytes = self.full_width * self.full_height * 3
        self._full_r: int | None = None
        self._full_w: int | None = None
        self._pipeline_argv = list(pipeline_argv) if pipeline_argv is not None else []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._full_lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._full_frame: np.ndarray | None = None
        self._seq = 0
        self._full_seq = 0
        self._thread: threading.Thread | None = None
        self._full_thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._last_error: str | None = None
        if not self._pipeline_argv:
            self._rebuild_argv()

    def _rebuild_argv(self, full_pipe_fd: int | None = None) -> None:
        self._pipeline_argv = build_ffmpeg_cuda_argv(
            self.device,
            width=self.cam_width,
            height=self.cam_height,
            fps=self.fps,
            out_width=self.width,
            out_height=self.height,
            input_format=self._input_format,
            decoder=self._decoder,
            full_pipe_fd=full_pipe_fd if self.keep_full_res else None,
            full_fps=self.full_fps,
        )

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
        if self.keep_full_res:
            self._full_r, self._full_w = os.pipe()
            os.set_inheritable(self._full_w, True)
            if fcntl is not None:
                try:
                    fcntl.fcntl(self._full_r, fcntl.F_SETPIPE_SZ, 1024 * 1024)
                except OSError:
                    pass
            self._rebuild_argv(full_pipe_fd=int(self._full_w))
        self._thread = threading.Thread(target=self._run, name="ffmpeg-cuda-webcam", daemon=True)
        self._thread.start()
        if self.keep_full_res:
            self._full_thread = threading.Thread(target=self._run_full, name="ffmpeg-cuda-full", daemon=True)
            self._full_thread.start()

    def _spawn(self) -> subprocess.Popen[bytes]:
        env = os.environ.copy()
        pass_fds: tuple[int, ...] = ()
        if self._full_w is not None:
            pass_fds = (int(self._full_w),)
        proc = subprocess.Popen(
            self._pipeline_argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=self._frame_nbytes * 2,
            pass_fds=pass_fds,
            close_fds=True,
        )
        if self._full_w is not None:
            try:
                os.close(self._full_w)
            except OSError:
                pass
            self._full_w = None
        return proc

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
            self._last_error = f"ffmpeg spawn failed: {exc}"
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
                    f"ffmpeg pipeline produced no frames (rc={rc}). {err.strip() or self.pipeline_description}"
                )
            self._terminate_proc()

    def _run_full(self) -> None:
        fd = self._full_r
        if fd is None:
            return
        scratch = np.empty((self.full_height, self.full_width, 3), dtype=np.uint8)
        try:
            while not self._stop.is_set():
                raw = self._read_exact_fd(fd, self._full_nbytes)
                if raw is None:
                    break
                np.copyto(scratch, np.frombuffer(raw, dtype=np.uint8).reshape(self.full_height, self.full_width, 3))
                with self._full_lock:
                    if self._full_frame is None or self._full_frame.shape != scratch.shape:
                        self._full_frame = np.empty_like(scratch)
                    np.copyto(self._full_frame, scratch)
                    self._full_seq += 1
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            self._full_r = None

    def _read_exact_fd(self, fd: int, nbytes: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < nbytes and not self._stop.is_set():
            try:
                chunk = os.read(fd, nbytes - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf) if len(buf) == nbytes else None

    def wait_first_frame(self, timeout_s: float = 8.0) -> bool:
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

    def snapshot_full(self) -> np.ndarray | None:
        """Latest camera-native frame for gun crops, or None if not ready."""
        if not self.keep_full_res:
            return None
        with self._full_lock:
            if self._full_frame is None:
                return None
            return self._full_frame.copy()

    @property
    def has_full_res(self) -> bool:
        return bool(self.keep_full_res)

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
        ft = self._full_thread
        if ft is not None and ft.is_alive():
            ft.join(timeout=float(join_timeout_s))
        if self._full_r is not None:
            try:
                os.close(self._full_r)
            except OSError:
                pass
            self._full_r = None
        if self._full_w is not None:
            try:
                os.close(self._full_w)
            except OSError:
                pass
            self._full_w = None

    def __enter__(self) -> FFmpegCudaWebcamCapture:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


def open_ffmpeg_cuda_webcam(
    device: str,
    *,
    width: int,
    height: int,
    fps: float,
    out_width: int,
    keep_full_res: bool = False,
    full_fps: float = 30.0,
) -> FFmpegCudaWebcamCapture:
    """Try MJPEG CUVID, then H.264 CUVID. Raises RuntimeError if neither produces a frame."""
    available = ffmpeg_cuvid_decoders()
    attempts: list[tuple[str, str]] = []
    if "mjpeg_cuvid" in available:
        attempts.append(("mjpeg", "mjpeg_cuvid"))
    if "h264_cuvid" in available:
        attempts.append(("h264", "h264_cuvid"))
    if not attempts:
        raise RuntimeError("FFmpeg has no mjpeg_cuvid/h264_cuvid decoder")
    errors: list[str] = []
    modes = [True, False] if keep_full_res else [False]
    for use_full in modes:
        for fmt, dec in attempts:
            cap = FFmpegCudaWebcamCapture(
                device,
                width=width,
                height=height,
                fps=fps,
                out_width=out_width,
                input_format=fmt,
                decoder=dec,
                keep_full_res=use_full,
                full_fps=full_fps,
            )
            cap.start()
            if cap.wait_first_frame(timeout_s=8.0):
                return cap
            errors.append(f"{fmt}/{dec} full={use_full}: {cap.last_error or 'no frames'}")
            cap.stop()
    raise RuntimeError(" ; ".join(errors))
