"""Shared-memory latest-frame writers."""

from __future__ import annotations

import mmap
import threading
from pathlib import Path

import numpy as np

from media.ipc.frame_reader import (
    BGR_HEADER_SIZE,
    BGR_HEADER_STRUCT,
    DEFAULT_BGR_CAPACITY,
    DEFAULT_JPEG_CAPACITY,
    JPEG_HEADER_SIZE,
    JPEG_HEADER_STRUCT,
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class LiveFrameWriter:
    """Write latest JPEG bytes into a fixed-size mmap region."""

    def __init__(self, path: Path, capacity: int = DEFAULT_JPEG_CAPACITY) -> None:
        self.path = path.expanduser().resolve()
        self.capacity = int(max(64 * 1024, capacity))
        _ensure_parent(self.path)
        total = JPEG_HEADER_SIZE + self.capacity
        with open(self.path, "wb") as f:
            f.truncate(total)
        self._fp = open(self.path, "r+b", buffering=0)
        self._mm = mmap.mmap(self._fp.fileno(), total, access=mmap.ACCESS_WRITE)
        self._seq = 0
        self._lock = threading.Lock()

    def write(self, payload: bytes) -> bool:
        data = bytes(payload or b"")
        if not data:
            return False
        size = len(data)
        if size > self.capacity:
            return False
        with self._lock:
            self._seq += 1
            self._mm.seek(0)
            self._mm.write(JPEG_HEADER_STRUCT.pack(self._seq, 0))
            self._mm.seek(JPEG_HEADER_SIZE)
            self._mm.write(data)
            self._seq += 1
            self._mm.seek(0)
            self._mm.write(JPEG_HEADER_STRUCT.pack(self._seq, size))
            self._mm.flush(0, JPEG_HEADER_SIZE + size)
        return True

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fp.close()


class LiveBgrFrameWriter:
    """Write latest raw BGR frame bytes into a fixed-size mmap region."""

    def __init__(self, path: Path, capacity: int = DEFAULT_BGR_CAPACITY) -> None:
        self.path = path.expanduser().resolve()
        self.capacity = int(max(256 * 1024, capacity))
        _ensure_parent(self.path)
        total = BGR_HEADER_SIZE + self.capacity
        with open(self.path, "wb") as f:
            f.truncate(total)
        self._fp = open(self.path, "r+b", buffering=0)
        self._mm = mmap.mmap(self._fp.fileno(), total, access=mmap.ACCESS_WRITE)
        self._seq = 0
        self._lock = threading.Lock()

    def _grow_to_fit(self, needed: int) -> None:
        new_cap = max(self.capacity * 2, int(needed) + 65536, DEFAULT_BGR_CAPACITY)
        if new_cap <= self.capacity:
            return
        path = self.path
        try:
            self._mm.close()
        except OSError:
            pass
        try:
            self._fp.close()
        except OSError:
            pass
        self.capacity = int(new_cap)
        total = BGR_HEADER_SIZE + self.capacity
        with open(path, "wb") as f:
            f.truncate(total)
        self._fp = open(path, "r+b", buffering=0)
        self._mm = mmap.mmap(self._fp.fileno(), total, access=mmap.ACCESS_WRITE)
        self._seq = 0

    def write_bgr_ndarray(self, frame) -> bool:
        """Write a contiguous `uint8` HxWxC array without an extra bytes copy."""
        if frame is None:
            return False
        arr = np.asarray(frame)
        if arr.size == 0 or arr.dtype != np.uint8:
            return False
        if arr.ndim == 2:
            height, width = int(arr.shape[0]), int(arr.shape[1])
            channels = 1
        elif arr.ndim == 3:
            height, width = int(arr.shape[0]), int(arr.shape[1])
            channels = int(arr.shape[2])
        else:
            return False
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        size = int(arr.nbytes)
        with self._lock:
            if size > self.capacity:
                self._grow_to_fit(size)
            if size > self.capacity:
                return False
            self._seq += 1
            self._mm.seek(0)
            self._mm.write(BGR_HEADER_STRUCT.pack(self._seq, 0, 0, 0, 0))
            self._mm.seek(BGR_HEADER_SIZE)
            self._mm.write(memoryview(arr).cast("B"))
            self._seq += 1
            self._mm.seek(0)
            self._mm.write(BGR_HEADER_STRUCT.pack(self._seq, size, height, width, channels))
            self._mm.flush(0, BGR_HEADER_SIZE + size)
        return True

    def write(self, payload: bytes, *, height: int, width: int, channels: int) -> bool:
        data = bytes(payload or b"")
        if not data or height <= 0 or width <= 0 or channels <= 0:
            return False
        size = len(data)
        with self._lock:
            if size > self.capacity:
                self._grow_to_fit(size)
            if size > self.capacity:
                return False
            self._seq += 1
            self._mm.seek(0)
            self._mm.write(BGR_HEADER_STRUCT.pack(self._seq, 0, 0, 0, 0))
            self._mm.seek(BGR_HEADER_SIZE)
            self._mm.write(data)
            self._seq += 1
            self._mm.seek(0)
            self._mm.write(BGR_HEADER_STRUCT.pack(self._seq, size, int(height), int(width), int(channels)))
            self._mm.flush(0, BGR_HEADER_SIZE + size)
        return True

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fp.close()


class FrameWriter:
    """Small facade for code that wants one object for both JPEG and BGR."""

    def __init__(self, jpeg_path: Path | None = None, bgr_path: Path | None = None) -> None:
        self._jpeg = LiveFrameWriter(jpeg_path) if jpeg_path is not None else None
        self._bgr = LiveBgrFrameWriter(bgr_path) if bgr_path is not None else None

    def write_jpeg(self, data: bytes) -> bool:
        if self._jpeg is None:
            return False
        return self._jpeg.write(data)

    def write_bgr(self, frame) -> bool:
        if self._bgr is None:
            return False
        return self._bgr.write_bgr_ndarray(frame)
