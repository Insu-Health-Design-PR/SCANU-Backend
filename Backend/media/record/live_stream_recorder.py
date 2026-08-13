"""Record live preview frames to MP4 with optional faster-than-real playback FPS."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FrameSupplier = Callable[[], tuple[np.ndarray, int] | None]


@dataclass(frozen=True)
class RecordingSnapshot:
    sensor: str
    path: str
    recording: bool
    frames_written: int
    source_fps: float
    playback_speed: float
    writer_fps: float
    started_at: float | None
    last_error: str | None = None


class LiveStreamRecorder:
    """
    Poll a frame supplier in a background thread and write MP4 via OpenCV.

    ``playback_speed`` sets container FPS to ``source_fps * playback_speed`` so a
    ~15 fps live stream plays back at ~1.5× (e.g. 22.5 fps) in normal players.
    """

    def __init__(self, *, sensor: str, frame_supplier: FrameSupplier) -> None:
        self.sensor = str(sensor)
        self._frame_supplier = frame_supplier
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._frames_written = 0
        self._source_fps = 15.0
        self._playback_speed = 1.5
        self._writer_fps = 22.5
        self._started_at: float | None = None
        self._last_error: str | None = None
        self._last_seq = -1
        self._last_frame: np.ndarray | None = None

    @property
    def recording(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def snapshot(self) -> RecordingSnapshot:
        with self._lock:
            return RecordingSnapshot(
                sensor=self.sensor,
                path=str(self._path) if self._path is not None else "",
                recording=self.recording,
                frames_written=int(self._frames_written),
                source_fps=float(self._source_fps),
                playback_speed=float(self._playback_speed),
                writer_fps=float(self._writer_fps),
                started_at=self._started_at,
                last_error=self._last_error,
            )

    def start(
        self,
        output_path: Path,
        *,
        source_fps: float = 15.0,
        playback_speed: float = 1.5,
    ) -> RecordingSnapshot:
        if self.recording:
            raise RuntimeError(f"{self.sensor} recording already active")
        self._stop.clear()
        self._path = Path(output_path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._frames_written = 0
        self._source_fps = max(1.0, float(source_fps))
        self._playback_speed = max(1.0, float(playback_speed))
        self._writer_fps = self._source_fps * self._playback_speed
        self._started_at = time.time()
        self._last_error = None
        self._last_seq = -1
        self._last_frame = None
        self._writer = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"live-record-{self.sensor}",
            daemon=True,
        )
        self._thread.start()
        return self.snapshot()

    def stop(self) -> RecordingSnapshot:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=5.0)
        self._thread = None
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
        return self.snapshot()

    def _should_skip_frame(self, frame: np.ndarray, seq: int) -> bool:
        if seq >= 0:
            if seq == self._last_seq:
                return True
            self._last_seq = seq
            return False
        if self._last_frame is not None and self._last_frame.shape == frame.shape:
            if np.array_equal(self._last_frame, frame):
                return True
        self._last_frame = frame.copy()
        return False

    def _ensure_writer(self, frame: np.ndarray) -> bool:
        if self._writer is not None:
            return True
        if self._path is None:
            return False
        h, w = frame.shape[:2]
        if h < 2 or w < 2:
            return False
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self._path), fourcc, float(self._writer_fps), (w, h))
        if not writer.isOpened():
            self._last_error = f"Cannot open VideoWriter for {self._path}"
            return False
        self._writer = writer
        return True

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                got = self._frame_supplier()
                if got is None:
                    time.sleep(0.01)
                    continue
                frame, seq = got
                if frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)
                if self._should_skip_frame(frame, int(seq)):
                    time.sleep(0.005)
                    continue
                with self._lock:
                    if not self._ensure_writer(frame):
                        time.sleep(0.05)
                        continue
                    assert self._writer is not None
                    self._writer.write(frame)
                    self._frames_written += 1
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
        finally:
            with self._lock:
                if self._writer is not None:
                    self._writer.release()
                    self._writer = None
