"""Frame source selection for live previews.

The current runtime owns camera capture and `/dev/shm` IPC. These helpers are
the new API-facing seam for selecting BGR IPC, JPEG IPC, disk frames, or a
placeholder without moving the capture code in this slice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from runtime import sensor_runner, thermal_runner  # noqa: E402
from layer8_ui.preview_media import mjpeg_placeholder_jpeg  # noqa: E402
from layer8_ui.thermal_ipc_fanout import get_thermal_ipc_fanout  # noqa: E402
from layer8_ui.webcam_ipc_fanout import get_webcam_ipc_fanout  # noqa: E402
from layer8_ui.multi_camera_ipc_fanout import get_multi_camera_ipc_fanout  # noqa: E402
from media.encode.jpeg import encode_preview_jpeg, is_valid_jpeg
from media.ipc import LiveBgrFrameReader, LiveFrameReader

_WEBCAM_LIVE_JPEG_Q = 62
_MJPEG_MAX_WIDTH = 960


class IpcFrameSources:
    def __init__(self) -> None:
        self.webcam_jpeg = LiveFrameReader(Path("/dev/shm/scanu_webcam_live_frame.bin"))
        self.webcam_bgr = LiveBgrFrameReader(Path("/dev/shm/scanu_webcam_live_bgr_frame.bin"))
        self.multi_camera_jpeg = LiveFrameReader(Path("/dev/shm/scanu_multi_camera_live_frame.bin"))
        self.multi_camera_bgr = LiveBgrFrameReader(Path("/dev/shm/scanu_multi_camera_live_bgr_frame.bin"))
        self.thermal_jpeg = LiveFrameReader(Path("/dev/shm/scanu_thermal_live_frame.bin"))
        self.thermal_bgr = LiveBgrFrameReader(Path("/dev/shm/scanu_thermal_live_bgr_frame.bin"))
        # Avoid re-encoding the same IPC seq for every MJPEG client / poll.
        self._jpeg_by_seq: dict[str, tuple[int, bytes]] = {}

    def latest_valid_jpeg(self, sensor: str) -> bytes | None:
        reader = self.thermal_jpeg if sensor == "thermal" else self.webcam_jpeg
        jpg = reader.read_latest()
        return jpg if is_valid_jpeg(jpg) else None

    def _encode_bgr_jpeg(self, bgr: np.ndarray) -> bytes | None:
        return encode_preview_jpeg(
            bgr,
            quality=int(_WEBCAM_LIVE_JPEG_Q),
            max_width=int(_MJPEG_MAX_WIDTH),
        )

    def _cached_encode(self, key: str, seq: int | None, bgr: np.ndarray | None) -> bytes | None:
        if bgr is None:
            return None
        if seq is not None and seq >= 0:
            hit = self._jpeg_by_seq.get(key)
            if hit is not None and hit[0] == int(seq):
                return hit[1]
        encoded = self._encode_bgr_jpeg(bgr)
        if encoded and seq is not None and seq >= 0:
            self._jpeg_by_seq[key] = (int(seq), encoded)
        return encoded

    def _bgr_payload_to_jpeg(
        self,
        key: str,
        bgr_payload: tuple[tuple[bytes, int, int, int], int] | tuple[bytes, int, int, int] | None,
    ) -> bytes | None:
        if bgr_payload is None:
            return None
        seq: int | None = None
        if isinstance(bgr_payload, tuple) and len(bgr_payload) == 2:
            inner, seq = bgr_payload
            raw, height, width, channels = inner
        else:
            raw, height, width, channels = bgr_payload  # type: ignore[misc]
        expected = int(height) * int(width) * int(channels)
        if channels != 3 or len(raw) != expected:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = arr.reshape((int(height), int(width), 3))
        return self._cached_encode(key, seq, bgr)

    def _bgr_payload_to_ndarray(
        self, bgr_payload: tuple[bytes, int, int, int] | None
    ) -> np.ndarray | None:
        if bgr_payload is None:
            return None
        raw, height, width, channels = bgr_payload
        expected = int(height) * int(width) * int(channels)
        if channels != 3 or len(raw) != expected:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        return arr.reshape((int(height), int(width), 3))

    def runner_frame_bgr_webcam_with_seq(self) -> tuple[np.ndarray, int] | None:
        """Latest annotated webcam BGR frame for recording (IPC / fanout)."""
        fanout = get_webcam_ipc_fanout()
        bgr = fanout.refresh_from_ipc_sync()
        if bgr is not None:
            return bgr, -1
        payload = self.webcam_bgr.read_latest_with_seq()
        if payload is None:
            return None
        (raw, height, width, channels), seq = payload
        frame = self._bgr_payload_to_ndarray((raw, height, width, channels))
        if frame is None:
            return None
        return frame, int(seq)

    def runner_frame_bgr_thermal_with_seq(self) -> tuple[np.ndarray, int] | None:
        """Latest annotated thermal BGR frame for recording (IPC / fanout)."""
        fanout = get_thermal_ipc_fanout()
        bgr = fanout.refresh_from_ipc_sync()
        if bgr is not None:
            return bgr, -1
        payload = self.thermal_bgr.read_latest_with_seq()
        if payload is None:
            return None
        (raw, height, width, channels), seq = payload
        frame = self._bgr_payload_to_ndarray((raw, height, width, channels))
        if frame is None:
            return None
        return frame, int(seq)

    def runner_frame_bgr_multi_camera_with_seq(self) -> tuple[np.ndarray, int] | None:
        """Latest annotated multi_camera BGR frame for recording / screenshots."""
        fanout = get_multi_camera_ipc_fanout()
        bgr = fanout.refresh_from_ipc_sync()
        if bgr is not None:
            return bgr, -1
        payload = self.multi_camera_bgr.read_latest_with_seq()
        if payload is None:
            return None
        (raw, height, width, channels), seq = payload
        frame = self._bgr_payload_to_ndarray((raw, height, width, channels))
        if frame is None:
            return None
        return frame, int(seq)

    def runner_frame_jpeg_webcam(
        self,
        rpath: Path | None,
        *,
        allow_disk_fallback: bool,
    ) -> bytes | None:
        """
        JPEG IPC (if present) → BGR IPC (downscaled preview JPEG, cached by seq) → fanout.
        """
        jpg = self.webcam_jpeg.read_latest()
        if jpg and is_valid_jpeg(jpg):
            return jpg
        encoded = self._bgr_payload_to_jpeg("webcam", self.webcam_bgr.read_latest_with_seq())
        if encoded:
            return encoded
        fanout = get_webcam_ipc_fanout()
        bgr = fanout.refresh_from_ipc_sync()
        if bgr is not None:
            encoded = self._cached_encode("webcam_fanout", -1, bgr)
            if encoded:
                return encoded
        if allow_disk_fallback and rpath is not None and rpath.is_file():
            try:
                disk = rpath.read_bytes()
                if disk and is_valid_jpeg(disk):
                    return disk
            except OSError:
                pass
        return None

    def runner_frame_jpeg_multi_camera(
        self,
        rpath: Path | None,
        *,
        allow_disk_fallback: bool,
    ) -> bytes | None:
        jpg = self.multi_camera_jpeg.read_latest()
        if jpg and is_valid_jpeg(jpg):
            return jpg
        encoded = self._bgr_payload_to_jpeg(
            "multi_camera", self.multi_camera_bgr.read_latest_with_seq()
        )
        if encoded:
            return encoded
        fanout = get_multi_camera_ipc_fanout()
        bgr = fanout.refresh_from_ipc_sync()
        if bgr is not None:
            encoded = self._cached_encode("multi_camera_fanout", -1, bgr)
            if encoded:
                return encoded
        if allow_disk_fallback and rpath is not None and rpath.is_file():
            try:
                disk = rpath.read_bytes()
                if disk and is_valid_jpeg(disk):
                    return disk
            except OSError:
                pass
        return None

    def runner_frame_jpeg_thermal(
        self,
        rpath: Path | None,
        *,
        allow_disk_fallback: bool,
    ) -> bytes | None:
        encoded = self._bgr_payload_to_jpeg("thermal", self.thermal_bgr.read_latest_with_seq())
        if encoded:
            return encoded
        fanout = get_thermal_ipc_fanout()
        bgr = fanout.refresh_from_ipc_sync()
        if bgr is not None:
            encoded = self._cached_encode("thermal_fanout", -1, bgr)
            if encoded:
                return encoded
        jpg = self.thermal_jpeg.read_latest()
        if jpg and is_valid_jpeg(jpg):
            return jpg
        if allow_disk_fallback and rpath is not None and rpath.is_file():
            try:
                disk = rpath.read_bytes()
                if disk and is_valid_jpeg(disk):
                    return disk
            except OSError:
                pass
        return None

    @staticmethod
    def thermal_uses_v4l2_preview(settings: dict[str, Any]) -> bool:
        return thermal_runner.thermal_uses_inprocess_v4l2_preview(settings)

    @staticmethod
    def thermal_disk_jpeg(settings: dict[str, Any], layer8_dir: Path) -> bytes | None:
        from layer8_ui.artifact_paths import resolved_artifact_path

        rel = (settings.get("thermal") or {}).get("live_frame") or ""
        rpath = resolved_artifact_path(settings, relative_to_software=str(rel), layer8_dir=layer8_dir)
        if rpath is None or not rpath.is_file():
            return None
        try:
            raw = rpath.read_bytes()
            return raw if is_valid_jpeg(raw) else None
        except OSError:
            return None

    @staticmethod
    def thermal_runner_ipc_placeholder(layer8_dir: Path) -> bytes:
        """Placeholder when infer subprocess is on but shared-memory IPC has no frame yet."""
        st = sensor_runner.status("thermal", layer8_dir)
        tail = (st.get("log_tail") or "").strip()
        detail = "Waiting for first BGR frame on /dev/shm (same path as AI overlay)."
        if "Cannot open" in tail:
            detail = "Camera failed to open — see layer8_ui/logs/thermal.log (needs /dev/videoN)."
        elif "select() timeout" in tail or "capture miss" in tail:
            detail = "Camera stuck (select timeout) — Stop thermal, unplug/replug PureThermal USB."
        return mjpeg_placeholder_jpeg(
            "Thermal AI is running; preview has no live frame yet.",
            detail,
            "layer8_ui/logs/thermal.log",
        )
