"""Frame source selection for live previews.

The current runtime owns camera capture and `/dev/shm` IPC. These helpers are
the new API-facing seam for selecting BGR IPC, JPEG IPC, disk frames, or a
placeholder without moving the capture code in this slice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from runtime import sensor_runner, thermal_runner  # noqa: E402
from layer8_ui.preview_media import mjpeg_placeholder_jpeg  # noqa: E402
from layer8_ui.thermal_ipc_fanout import get_thermal_ipc_fanout  # noqa: E402
from layer8_ui.webcam_ipc_fanout import get_webcam_ipc_fanout  # noqa: E402
from layer8_ui.multi_camera_ipc_fanout import get_multi_camera_ipc_fanout  # noqa: E402
from media.encode.jpeg import is_valid_jpeg
from media.ipc import LiveBgrFrameReader, LiveFrameReader

_WEBCAM_LIVE_JPEG_Q = 85


class IpcFrameSources:
    def __init__(self) -> None:
        self.webcam_jpeg = LiveFrameReader(Path("/dev/shm/scanu_webcam_live_frame.bin"))
        self.webcam_bgr = LiveBgrFrameReader(Path("/dev/shm/scanu_webcam_live_bgr_frame.bin"))
        self.multi_camera_jpeg = LiveFrameReader(Path("/dev/shm/scanu_multi_camera_live_frame.bin"))
        self.multi_camera_bgr = LiveBgrFrameReader(Path("/dev/shm/scanu_multi_camera_live_bgr_frame.bin"))
        self.thermal_jpeg = LiveFrameReader(Path("/dev/shm/scanu_thermal_live_frame.bin"))
        self.thermal_bgr = LiveBgrFrameReader(Path("/dev/shm/scanu_thermal_live_bgr_frame.bin"))

    def latest_valid_jpeg(self, sensor: str) -> bytes | None:
        reader = self.thermal_jpeg if sensor == "thermal" else self.webcam_jpeg
        jpg = reader.read_latest()
        return jpg if is_valid_jpeg(jpg) else None

    def _encode_bgr_jpeg(self, bgr: np.ndarray) -> bytes | None:
        ok, buf = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(_WEBCAM_LIVE_JPEG_Q)],
        )
        return buf.tobytes() if ok else None

    def _bgr_payload_to_jpeg(self, bgr_payload: tuple[bytes, int, int, int] | None) -> bytes | None:
        if bgr_payload is None:
            return None
        raw, height, width, channels = bgr_payload
        expected = int(height) * int(width) * int(channels)
        if channels != 3 or len(raw) != expected:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = arr.reshape((int(height), int(width), 3))
        return self._encode_bgr_jpeg(bgr)

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
        Same sources as ``WebcamJpegTrack.recv`` when the runner is on: JPEG IPC → BGR IPC → fanout.

        Prefer the JPEG IPC slot (written atomically with each infer frame) to avoid re-encoding
        large BGR buffers for MJPEG preview.
        """
        jpg = self.webcam_jpeg.read_latest()
        if jpg and is_valid_jpeg(jpg):
            return jpg
        fanout = get_webcam_ipc_fanout()
        bgr = fanout.refresh_from_ipc_sync()
        if bgr is not None:
            encoded = self._encode_bgr_jpeg(bgr)
            if encoded:
                return encoded
        encoded = self._bgr_payload_to_jpeg(self.webcam_bgr.read_latest())
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
        fanout = get_multi_camera_ipc_fanout()
        bgr = fanout.refresh_from_ipc_sync()
        if bgr is not None:
            encoded = self._encode_bgr_jpeg(bgr)
            if encoded:
                return encoded
        encoded = self._bgr_payload_to_jpeg(self.multi_camera_bgr.read_latest())
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
        fanout = get_thermal_ipc_fanout()
        bgr = fanout.refresh_from_ipc_sync()
        if bgr is not None:
            encoded = self._encode_bgr_jpeg(bgr)
            if encoded:
                return encoded
        encoded = self._bgr_payload_to_jpeg(self.thermal_bgr.read_latest())
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
