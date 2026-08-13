"""Start/stop live preview recording (API-side, not infer subprocess)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from api.streaming.frame_sources import IpcFrameSources
from media.record import LiveStreamRecorder, RecordingSnapshot

SensorName = Literal["thermal", "webcam"]


class RecordingService:
    def __init__(self, layer8_dir: Path, frame_sources: IpcFrameSources) -> None:
        self.layer8_dir = Path(layer8_dir).resolve()
        self.frame_sources = frame_sources
        self._recorders: dict[str, LiveStreamRecorder] = {}

    def _default_output_path(self, sensor: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rel = Path("layer8_ui/artifacts/recordings") / f"{sensor}_live_{stamp}.mp4"
        return (self.layer8_dir.parent / rel).resolve()

    def _frame_supplier(self, sensor: SensorName):
        if sensor == "webcam":
            return self.frame_sources.runner_frame_bgr_webcam_with_seq
        return self.frame_sources.runner_frame_bgr_thermal_with_seq

    def _resolve_output_path(self, output_path: str | None, sensor: str) -> Path:
        if not output_path or not str(output_path).strip():
            return self._default_output_path(sensor)
        p = Path(str(output_path).strip()).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (self.layer8_dir.parent / p).resolve()

    def _default_source_fps(self, settings: dict[str, Any], sensor: SensorName) -> float:
        block = settings.get(sensor) or {}
        if sensor == "thermal":
            base_fps = float(block.get("thermal_fps") or block.get("fps") or 0.0)
        else:
            base_fps = float(block.get("fps") or 0.0)
        stride = max(1, int(block.get("weapon_live_infer_stride") or 1))
        if base_fps >= 1.0:
            return max(1.0, base_fps / stride)
        return 15.0

    def start(
        self,
        sensor: SensorName,
        settings: dict[str, Any],
        *,
        playback_speed: float = 1.5,
        source_fps: float | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        existing = self._recorders.get(sensor)
        if existing is not None and existing.recording:
            snap = existing.snapshot()
            return {
                "ok": False,
                "error": "recording_already_active",
                "status": self._snapshot_to_dict(snap),
            }
        src_fps = float(source_fps) if source_fps is not None else self._default_source_fps(settings, sensor)
        out = self._resolve_output_path(output_path, sensor)
        out.parent.mkdir(parents=True, exist_ok=True)
        recorder = LiveStreamRecorder(sensor=sensor, frame_supplier=self._frame_supplier(sensor))
        snap = recorder.start(out, source_fps=src_fps, playback_speed=float(playback_speed))
        self._recorders[sensor] = recorder
        return {"ok": True, "status": self._snapshot_to_dict(snap)}

    def stop(self, sensor: SensorName) -> dict[str, Any]:
        recorder = self._recorders.get(sensor)
        if recorder is None or not recorder.recording:
            snap = recorder.snapshot() if recorder is not None else None
            return {
                "ok": False,
                "error": "not_recording",
                "status": self._snapshot_to_dict(snap) if snap is not None else None,
            }
        snap = recorder.stop()
        return {"ok": True, "status": self._snapshot_to_dict(snap)}

    def status(self, sensor: SensorName | None = None) -> dict[str, Any]:
        if sensor is not None:
            recorder = self._recorders.get(sensor)
            if recorder is None:
                return {"sensor": sensor, "recording": False, "path": ""}
            return self._snapshot_to_dict(recorder.snapshot())
        return {
            sensor: self._snapshot_to_dict(rec.snapshot())
            if (rec := self._recorders.get(sensor)) is not None
            else {"sensor": sensor, "recording": False, "path": ""}
            for sensor in ("thermal", "webcam")
        }

    def latest_recording_path(self, sensor: SensorName) -> Path | None:
        recorder = self._recorders.get(sensor)
        if recorder is None:
            return None
        snap = recorder.snapshot()
        if not snap.path:
            return None
        p = Path(snap.path)
        return p if p.is_file() else None

    @staticmethod
    def _snapshot_to_dict(snap: RecordingSnapshot) -> dict[str, Any]:
        elapsed = (time.time() - snap.started_at) if snap.started_at is not None else 0.0
        return {
            "sensor": snap.sensor,
            "recording": snap.recording,
            "path": snap.path,
            "frames_written": snap.frames_written,
            "source_fps": round(snap.source_fps, 3),
            "playback_speed": round(snap.playback_speed, 3),
            "writer_fps": round(snap.writer_fps, 3),
            "elapsed_s": round(elapsed, 2),
            "last_error": snap.last_error,
        }
