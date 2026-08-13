"""Shared-memory latest-frame readers.

This is the clean home for the old Layer 4 `live_frame_ipc` reader logic.
The format is a simple seqlock: odd sequence means writer in progress, even
sequence with matching before/after headers means stable payload.
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path

from media.encode.jpeg import is_valid_jpeg

JPEG_HEADER_STRUCT = struct.Struct("<II")  # seq, size
JPEG_HEADER_SIZE = JPEG_HEADER_STRUCT.size
BGR_HEADER_STRUCT = struct.Struct("<IIIII")  # seq, size, height, width, channels
BGR_HEADER_SIZE = BGR_HEADER_STRUCT.size
DEFAULT_JPEG_CAPACITY = 2 * 1024 * 1024
DEFAULT_BGR_CAPACITY = 32 * 1024 * 1024


class LiveFrameReader:
    """Read latest stable JPEG bytes from an mmap region."""

    def __init__(self, path: Path, capacity: int = DEFAULT_JPEG_CAPACITY) -> None:
        self.path = path.expanduser().resolve()
        self.capacity = int(max(64 * 1024, capacity))
        self._fp = None
        self._mm = None
        self._total = JPEG_HEADER_SIZE + self.capacity

    def _open_if_needed(self) -> bool:
        if self._mm is not None:
            return True
        if not self.path.is_file():
            return False
        try:
            self._fp = open(self.path, "r+b", buffering=0)
            self._mm = mmap.mmap(self._fp.fileno(), self._total, access=mmap.ACCESS_READ)
            return True
        except OSError:
            self.close()
            return False

    def read_latest_with_seq(self) -> tuple[bytes, int] | None:
        if not self._open_if_needed():
            return None
        assert self._mm is not None
        try:
            for _ in range(12):
                self._mm.seek(0)
                seq1, size = JPEG_HEADER_STRUCT.unpack(self._mm.read(JPEG_HEADER_SIZE))
                if (seq1 & 1) != 0:
                    continue
                if size <= 0 or size > self.capacity:
                    return None
                self._mm.seek(JPEG_HEADER_SIZE)
                payload = self._mm.read(size)
                self._mm.seek(0)
                seq2, _size2 = JPEG_HEADER_STRUCT.unpack(self._mm.read(JPEG_HEADER_SIZE))
                if seq1 == seq2 and (seq2 & 1) == 0:
                    return (payload, int(seq2)) if is_valid_jpeg(payload) else None
            return None
        except (ValueError, OSError):
            self.close()
            return None

    def read_latest(self) -> bytes | None:
        payload = self.read_latest_with_seq()
        return payload[0] if payload is not None else None

    def close(self) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except OSError:
                pass
            self._mm = None
        if self._fp is not None:
            try:
                self._fp.close()
            except OSError:
                pass
            self._fp = None


class LiveBgrFrameReader:
    """Read latest stable raw BGR frame from an mmap region."""

    def __init__(self, path: Path, capacity: int = DEFAULT_BGR_CAPACITY) -> None:
        self.path = path.expanduser().resolve()
        self.capacity = int(max(256 * 1024, capacity))
        self._fp = None
        self._mm = None
        self._total = BGR_HEADER_SIZE + self.capacity

    def _open_if_needed(self) -> bool:
        if not self.path.is_file():
            return False
        try:
            file_size = int(self.path.stat().st_size)
            if self._mm is not None and file_size == self._total:
                return True
            if self._mm is not None:
                self.close()
            if file_size < BGR_HEADER_SIZE + 256 * 1024:
                return False
            self._total = file_size
            self.capacity = file_size - BGR_HEADER_SIZE
            self._fp = open(self.path, "r+b", buffering=0)
            self._mm = mmap.mmap(self._fp.fileno(), self._total, access=mmap.ACCESS_READ)
            return True
        except OSError:
            self.close()
            return False

    def read_latest_with_seq(self) -> tuple[tuple[bytes, int, int, int], int] | None:
        if not self._open_if_needed():
            return None
        assert self._mm is not None
        try:
            for _ in range(12):
                self._mm.seek(0)
                seq1, size, height, width, channels = BGR_HEADER_STRUCT.unpack(
                    self._mm.read(BGR_HEADER_SIZE)
                )
                if (seq1 & 1) != 0:
                    continue
                if size <= 0 or size > self.capacity:
                    return None
                if height <= 0 or width <= 0 or channels <= 0:
                    return None
                self._mm.seek(BGR_HEADER_SIZE)
                payload = self._mm.read(size)
                self._mm.seek(0)
                seq2, size2, height2, width2, channels2 = BGR_HEADER_STRUCT.unpack(
                    self._mm.read(BGR_HEADER_SIZE)
                )
                if (
                    seq1 == seq2
                    and (seq2 & 1) == 0
                    and size == size2
                    and height == height2
                    and width == width2
                    and channels == channels2
                ):
                    return (payload, int(height), int(width), int(channels)), int(seq2)
            return None
        except (ValueError, OSError):
            self.close()
            return None

    def read_latest(self) -> tuple[bytes, int, int, int] | None:
        payload = self.read_latest_with_seq()
        return payload[0] if payload is not None else None

    def close(self) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except OSError:
                pass
            self._mm = None
        if self._fp is not None:
            try:
                self._fp.close()
            except OSError:
                pass
            self._fp = None


class FrameReader:
    """Small facade for code that wants one object for both JPEG and BGR."""

    def __init__(self, jpeg_path: Path, bgr_path: Path | None = None) -> None:
        self._jpeg = LiveFrameReader(jpeg_path)
        self._bgr = LiveBgrFrameReader(bgr_path) if bgr_path is not None else None

    def read_jpeg(self) -> bytes | None:
        return self._jpeg.read_latest()

    def read_bgr(self) -> tuple[bytes, int, int, int] | None:
        if self._bgr is None:
            return None
        return self._bgr.read_latest()
