"""Linux V4L2 capture helpers."""

from __future__ import annotations

import ctypes
import mmap
import os
import select
import struct
from typing import Any

import numpy as np

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_NONE = 1

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_READ = 2
_IOC_WRITE = 1
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS


def _ioc(dir_: int, typ: str, nr: int, size: int) -> int:
    return (
        (dir_ << _IOC_DIRSHIFT)
        | (ord(typ) << _IOC_TYPESHIFT)
        | (nr << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


def _iow(typ: str, nr: int, size: int) -> int:
    return _ioc(_IOC_WRITE, typ, nr, size)


def _iowr(typ: str, nr: int, size: int) -> int:
    return _ioc(_IOC_READ | _IOC_WRITE, typ, nr, size)


def v4l2_fourcc(a: str, b: str, c: str, d: str) -> int:
    return ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24)


V4L2_PIX_FMT_GREY = v4l2_fourcc("G", "R", "E", "Y")
V4L2_PIX_FMT_Y16 = v4l2_fourcc("Y", "1", "6", " ")

# Linux uapi: sizeof(struct v4l2_format) on x86_64.
_V4L2_FORMAT_SIZE = 208


class _v4l2_pix_format(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class _v4l2_format_union(ctypes.Union):
    _fields_ = [
        ("pix", _v4l2_pix_format),
        ("raw", ctypes.c_uint8 * 200),
    ]


class v4l2_format(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("fmt", _v4l2_format_union),
        ("_pad", ctypes.c_uint8 * 4),
    ]


class v4l2_requestbuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 1),
    ]


class _v4l2_timeval(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_long),
        ("tv_usec", ctypes.c_long),
    ]


class _v4l2_timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class _v4l2_buffer_m(ctypes.Union):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_ulong),
        ("fd", ctypes.c_int32),
    ]


class _v4l2_buffer_tail(ctypes.Union):
    _fields_ = [
        ("request_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class v4l2_buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", _v4l2_timeval),
        ("timecode", _v4l2_timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", _v4l2_buffer_m),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("u", _v4l2_buffer_tail),
    ]


VIDIOC_S_FMT = _iowr("V", 5, ctypes.sizeof(v4l2_format))
VIDIOC_REQBUFS = _iowr("V", 8, ctypes.sizeof(v4l2_requestbuffers))
VIDIOC_QUERYBUF = _iowr("V", 9, ctypes.sizeof(v4l2_buffer))
VIDIOC_QBUF = _iowr("V", 15, ctypes.sizeof(v4l2_buffer))
VIDIOC_DQBUF = _iowr("V", 17, ctypes.sizeof(v4l2_buffer))
VIDIOC_STREAMON = _iow("V", 18, ctypes.sizeof(ctypes.c_int))
VIDIOC_STREAMOFF = _iow("V", 19, ctypes.sizeof(ctypes.c_int))


def _fourcc_name(code: int) -> str:
    return "".join(chr((int(code) >> (8 * i)) & 0xFF) for i in range(4)).strip()


class V4L2ThermalCapture:
    """Mmap V4L2 reader for PureThermal GREY/Y16 frames."""

    def __init__(
        self,
        device: str | int,
        *,
        width: int = 80,
        height: int = 60,
        fps: float = 9.0,
        read_timeout_s: float = 2.0,
    ) -> None:
        self.device = str(device) if str(device).startswith("/dev/video") else f"/dev/video{int(device)}"
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.read_timeout_s = float(read_timeout_s)
        self._fd: int | None = None
        self._bufs: list[memoryview] = []
        self._streaming = False
        self.pixelformat = 0
        self.bytesperline = 0
        self.sizeimage = 0
        self.negotiated: tuple[int, int, str] = (0, 0, "")
        self._open_and_configure()

    def _ioctl(self, req: int, arg: Any) -> None:
        assert self._fd is not None
        fcntl = __import__("fcntl")
        fcntl.ioctl(self._fd, req, arg)

    def _set_format(self, pixfmt: int) -> tuple[int, int, str, int, int]:
        assert self._fd is not None
        buf = bytearray(_V4L2_FORMAT_SIZE)
        struct.pack_into("<I", buf, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        # Kernel v4l2_pix_format layout on this platform: width@+4, height@+12, fourcc@+16.
        struct.pack_into("<I", buf, 4, int(self.width))
        struct.pack_into("<I", buf, 12, int(self.height))
        struct.pack_into("<I", buf, 16, int(pixfmt))
        struct.pack_into("<I", buf, 20, V4L2_FIELD_NONE)
        self._ioctl(VIDIOC_S_FMT, buf)
        aw = int(struct.unpack_from("<I", buf, 4)[0])
        ah = int(struct.unpack_from("<I", buf, 12)[0])
        pf = int(struct.unpack_from("<I", buf, 16)[0])
        bpl = int(struct.unpack_from("<I", buf, 24)[0])
        sizeimage = int(struct.unpack_from("<I", buf, 28)[0])
        return aw, ah, _fourcc_name(pf), bpl, sizeimage

    def _open_and_configure(self) -> None:
        self._fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK, 0)
        last_err: Exception | None = None
        negotiated: tuple[int, int, str, int, int] | None = None
        # PureThermal: prefer 16-bit Y16 (radiometric) over 8-bit GREY.
        for pixfmt in (V4L2_PIX_FMT_Y16, V4L2_PIX_FMT_GREY):
            try:
                negotiated = self._set_format(pixfmt)
                self.pixelformat = pixfmt
                break
            except OSError as exc:
                last_err = exc
        if negotiated is None:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            raise RuntimeError(f"Cannot set V4L2 format on {self.device}: {last_err}")

        aw, ah, fcc, bpl, sizeimage = negotiated
        self.width = aw
        self.height = ah
        self.bytesperline = bpl if bpl > 0 else (aw * (2 if self.pixelformat == V4L2_PIX_FMT_Y16 else 1))
        self.sizeimage = sizeimage
        self.negotiated = (aw, ah, fcc)

        req = v4l2_requestbuffers()
        req.count = 2
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = V4L2_MEMORY_MMAP
        self._ioctl(VIDIOC_REQBUFS, req)
        if int(req.count) < 1:
            raise RuntimeError(f"VIDIOC_REQBUFS returned count={req.count} for {self.device}")

        self._bufs = []
        for index in range(int(req.count)):
            buf = v4l2_buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            buf.index = index
            self._ioctl(VIDIOC_QUERYBUF, buf)
            mm = mmap.mmap(
                self._fd,
                int(buf.length),
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=buf.m.offset,
            )
            self._bufs.append(mm)
            self._ioctl(VIDIOC_QBUF, buf)

        buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
        self._ioctl(VIDIOC_STREAMON, buf_type)
        self._streaming = True

    def _raw_to_frame(self, raw: bytes) -> np.ndarray | None:
        h, w = int(self.height), int(self.width)
        bpl = int(self.bytesperline)
        if h <= 0 or w <= 0:
            return None
        if self.pixelformat == V4L2_PIX_FMT_Y16:
            row_px = max(w, bpl // 2)
            need = h * row_px
            arr = np.frombuffer(raw, dtype=np.uint16)
            if arr.size < need:
                return None
            plane = arr[:need].reshape(h, row_px)
            return plane[:, :w].copy()
        row_px = max(w, bpl)
        need = h * row_px
        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size < need:
            return None
        plane = arr[:need].reshape(h, row_px)
        return plane[:, :w].copy()

    @property
    def is_open(self) -> bool:
        return self._fd is not None and self._streaming

    def read(self) -> np.ndarray | None:
        if self._fd is None:
            return None
        rlist, _, _ = select.select([self._fd], [], [], self.read_timeout_s)
        if not rlist:
            return None
        buf = v4l2_buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        try:
            self._ioctl(VIDIOC_DQBUF, buf)
        except OSError:
            return None
        index = int(buf.index)
        used = int(buf.bytesused)
        if index < 0 or index >= len(self._bufs) or used <= 0:
            self._ioctl(VIDIOC_QBUF, buf)
            return None
        raw = bytes(self._bufs[index][:used])
        self._ioctl(VIDIOC_QBUF, buf)
        return self._raw_to_frame(raw)

    def close(self) -> None:
        if self._fd is None:
            return
        if self._streaming:
            try:
                buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
                self._ioctl(VIDIOC_STREAMOFF, buf_type)
            except OSError:
                pass
            self._streaming = False
        for mm in self._bufs:
            try:
                mm.close()
            except OSError:
                pass
        self._bufs = []
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None

    def __enter__(self) -> V4L2ThermalCapture:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def list_devices() -> list[dict]:
    return []
