"""Background OpenCV reader: MJPEG decode off the infer/overlay hot path."""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np


class LiveWebcamCapture:
    """Single consumer thread reads an already-configured OpenCV capture."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._capture_ns = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="live-webcam-capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        misses = 0
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except cv2.error:
                # HDMI UVC MJPEG often yields empty JPEG buffers; OpenCV 5 asserts in imdecode.
                misses += 1
                if misses >= 30:
                    self._stop.wait(0.02)
                continue
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                misses += 1
                if misses >= 30:
                    self._stop.wait(0.02)
                continue
            misses = 0
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            with self._lock:
                if self._frame is None or self._frame.shape != frame.shape:
                    self._frame = np.empty_like(frame)
                np.copyto(self._frame, frame)
                self._seq += 1
                self._capture_ns = time.monotonic_ns()

    def copy_latest_into(self, dst: np.ndarray) -> int:
        with self._lock:
            if self._frame is None:
                return -1
            if dst.shape != self._frame.shape:
                raise ValueError(f"dst shape {dst.shape} != capture {self._frame.shape}")
            np.copyto(dst, self._frame)
            return int(self._seq)

    @property
    def last_capture_ns(self) -> int:
        with self._lock:
            return int(self._capture_ns)

    def snapshot(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def shape(self) -> tuple[int, ...] | None:
        with self._lock:
            return None if self._frame is None else tuple(self._frame.shape)

    def stop(self, join_timeout_s: float = 3.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=float(join_timeout_s))
        try:
            self._cap.release()
        except Exception:
            pass

    def __enter__(self) -> LiveWebcamCapture:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()
