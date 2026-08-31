"""Save a single annotated preview frame as JPEG."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2

from api.streaming.frame_sources import IpcFrameSources

ScreenshotSensor = Literal["thermal", "webcam", "multi_camera"]

# Operator-requested save location (Desktop/Screenshots under the login home).
DEFAULT_SCREENSHOTS_DIR = Path.home() / "Desktop" / "Screenshots"


class ScreenshotService:
    def __init__(
        self,
        frame_sources: IpcFrameSources,
        screenshots_dir: Path | None = None,
    ) -> None:
        self.frame_sources = frame_sources
        self.screenshots_dir = Path(screenshots_dir or DEFAULT_SCREENSHOTS_DIR).expanduser()

    def _grab_bgr(self, sensor: ScreenshotSensor):
        if sensor == "webcam":
            out = self.frame_sources.runner_frame_bgr_webcam_with_seq()
        elif sensor == "multi_camera":
            out = self.frame_sources.runner_frame_bgr_multi_camera_with_seq()
        else:
            out = self.frame_sources.runner_frame_bgr_thermal_with_seq()
        if out is None:
            return None
        return out[0]

    def capture(self, sensor: ScreenshotSensor) -> dict[str, Any]:
        frame = self._grab_bgr(sensor)
        if frame is None:
            return {
                "ok": False,
                "error": "no_frame",
                "message": f"No live {sensor} frame available. Start the sensor runner and wait for preview.",
            }
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{sensor}_{stamp}.jpg"
        path = (self.screenshots_dir / filename).resolve()
        ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return {"ok": False, "error": "write_failed", "path": str(path)}
        return {
            "ok": True,
            "sensor": sensor,
            "path": str(path),
            "filename": filename,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }

    def capture_jpeg_bytes(self, sensor: ScreenshotSensor, *, quality: int = 85) -> bytes | None:
        """Return one annotated JPEG from IPC without opening an MJPEG stream."""
        frame = self._grab_bgr(sensor)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return buf.tobytes()
