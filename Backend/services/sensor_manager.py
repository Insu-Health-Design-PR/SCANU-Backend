"""Spawn, kill, and status for sensor runners."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from fastapi.responses import JSONResponse

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from runtime import sensor_runner  # noqa: E402

SensorName = Literal["thermal", "webcam", "mmwave", "multi_camera"]

logger = logging.getLogger("scanu.sensors")


class SensorManager:
    def __init__(
        self,
        layer8_dir: Path,
        *,
        thermal_stream: Any,
        webcam_stream: Any,
        multi_camera_stream: Any,
    ) -> None:
        self.layer8_dir = Path(layer8_dir)
        self._thermal_stream = thermal_stream
        self._webcam_stream = webcam_stream
        self._multi_camera_stream = multi_camera_stream

    def status(self, sensor_type: str) -> dict[str, Any]:
        return sensor_runner.status(sensor_type, self.layer8_dir)  # type: ignore[arg-type]

    def all_status(self) -> dict[str, Any]:
        return {
            name: sensor_runner.status(name, self.layer8_dir)
            for name in ("thermal", "webcam", "multi_camera", "mmwave")
        }

    def build_command(self, sensor: SensorName, settings: dict[str, Any]) -> dict[str, Any]:
        cmd = sensor_runner.build_command(sensor, settings, self.layer8_dir)
        run_cwd = str(sensor_runner.command_cwd(sensor, settings))
        return {
            "command": cmd,
            "cwd": run_cwd,
            "software_root": str(sensor_runner.resolved_software_root(settings)),
        }

    def stop(self, sensor_type: str) -> dict[str, Any]:
        out = sensor_runner.stop(sensor_type, self.layer8_dir)  # type: ignore[arg-type]
        if sensor_type == "thermal":
            self._thermal_stream.resume_after_thermal_subprocess_attempt()
        return out

    def _pause_preview_stream(self, sensor_type: str) -> None:
        if sensor_type == "webcam":
            logger.info("Pausing Front Cam V4L2 preview reader before infer subprocess")
            self._webcam_stream.force_release_camera()
        elif sensor_type == "multi_camera":
            logger.info("Pausing Back Cam V4L2 preview reader before infer subprocess")
            self._multi_camera_stream.force_release_camera()

    def _resume_preview_stream(self, sensor_type: str, *, start_ok: bool = False) -> None:
        if start_ok:
            # Infer subprocess owns the device; MJPEG/WebRTC use IPC until Stop.
            return
        if sensor_type == "webcam":
            self._webcam_stream.resume_after_webcam_subprocess_attempt()
        elif sensor_type == "multi_camera":
            self._multi_camera_stream.resume_after_multi_camera_subprocess_attempt()

    def start(self, sensor_type: str, settings: dict[str, Any] | None = None) -> dict[str, Any] | JSONResponse:
        from layer8_ui.settings_store import load

        s = load(self.layer8_dir) if settings is None else settings
        if sensor_type == "thermal":
            self._thermal_stream.force_release_camera()
            result: dict[str, Any] = {"ok": False}
            try:
                result = sensor_runner.start("thermal", s, self.layer8_dir)
            finally:
                if not result.get("ok"):
                    self._thermal_stream.resume_after_thermal_subprocess_attempt()
            if not result.get("ok"):
                return JSONResponse(result, status_code=409)
            return result
        if sensor_type in ("webcam", "multi_camera"):
            self._pause_preview_stream(sensor_type)
            result: dict[str, Any] = {"ok": False}
            try:
                result = sensor_runner.start(sensor_type, s, self.layer8_dir)  # type: ignore[arg-type]
            finally:
                self._resume_preview_stream(sensor_type, start_ok=bool(result.get("ok")))
            if not result.get("ok"):
                return JSONResponse(result, status_code=409)
            return result
        result = sensor_runner.start(sensor_type, s, self.layer8_dir)  # type: ignore[arg-type]
        if not result.get("ok"):
            return JSONResponse(result, status_code=409)
        return result

    def restart(self, sensor_type: str, settings: dict[str, Any] | None = None) -> dict[str, Any] | JSONResponse:
        from layer8_ui.settings_store import load

        s = load(self.layer8_dir) if settings is None else settings
        if sensor_type == "thermal":
            self._thermal_stream.force_release_camera()
            result: dict[str, Any] = {"ok": False}
            try:
                result = sensor_runner.restart("thermal", s, self.layer8_dir)
            finally:
                if not result.get("ok"):
                    self._thermal_stream.resume_after_thermal_subprocess_attempt()
            if not result.get("ok"):
                return JSONResponse(result, status_code=409)
            return result
        if sensor_type in ("webcam", "multi_camera"):
            self._pause_preview_stream(sensor_type)
            result: dict[str, Any] = {"ok": False}
            try:
                result = sensor_runner.restart(sensor_type, s, self.layer8_dir)  # type: ignore[arg-type]
            finally:
                self._resume_preview_stream(sensor_type, start_ok=bool(result.get("ok")))
            if not result.get("ok"):
                return JSONResponse(result, status_code=409)
            return result
        result = sensor_runner.restart(sensor_type, s, self.layer8_dir)  # type: ignore[arg-type]
        if not result.get("ok"):
            return JSONResponse(result, status_code=409)
        return result

    def run_sensor(self, sensor: SensorName) -> dict[str, Any] | JSONResponse:
        from layer8_ui.settings_store import load

        s = load(self.layer8_dir)
        if sensor == "thermal":
            self._thermal_stream.pause_for_thermal_subprocess()
        elif sensor in ("webcam", "multi_camera"):
            self._pause_preview_stream(sensor)
        result: dict[str, Any] = {"ok": False}
        try:
            result = sensor_runner.start(sensor, s, self.layer8_dir)
        finally:
            if sensor == "thermal":
                self._thermal_stream.resume_after_thermal_subprocess_attempt()
            elif sensor in ("webcam", "multi_camera"):
                self._resume_preview_stream(sensor, start_ok=bool(result.get("ok")))
        if not result.get("ok"):
            return JSONResponse(result, status_code=409)
        return result

    def restart_sensor(self, sensor: SensorName) -> dict[str, Any] | JSONResponse:
        from layer8_ui.settings_store import load

        s = load(self.layer8_dir)
        if sensor == "thermal":
            self._thermal_stream.pause_for_thermal_subprocess()
        elif sensor in ("webcam", "multi_camera"):
            self._pause_preview_stream(sensor)
        result: dict[str, Any] = {"ok": False}
        try:
            result = sensor_runner.restart(sensor, s, self.layer8_dir)
        finally:
            if sensor == "thermal":
                self._thermal_stream.resume_after_thermal_subprocess_attempt()
            elif sensor in ("webcam", "multi_camera"):
                self._resume_preview_stream(sensor, start_ok=bool(result.get("ok")))
        if not result.get("ok"):
            return JSONResponse(result, status_code=409)
        return result

    def run_all(self) -> dict[str, Any] | JSONResponse:
        from layer8_ui.settings_store import load

        s = load(self.layer8_dir)
        results: dict[str, Any] = {}
        started: list[SensorName] = []
        self._thermal_stream.pause_for_thermal_subprocess()
        try:
            for sensor in ("thermal", "webcam", "multi_camera", "mmwave"):
                if sensor == "webcam":
                    self._webcam_stream.force_release_camera()
                elif sensor == "multi_camera":
                    self._multi_camera_stream.force_release_camera()
                res: dict[str, Any] = {"ok": False}
                try:
                    res = sensor_runner.start(sensor, s, self.layer8_dir)  # type: ignore[arg-type]
                finally:
                    if sensor == "webcam" and not res.get("ok"):
                        self._webcam_stream.resume_after_webcam_subprocess_attempt()
                    elif sensor == "multi_camera" and not res.get("ok"):
                        self._multi_camera_stream.resume_after_multi_camera_subprocess_attempt()
                results[sensor] = res
                if res.get("ok"):
                    started.append(sensor)  # type: ignore[arg-type]
                    continue
                for started_sensor in started:
                    sensor_runner.stop(started_sensor, self.layer8_dir)
                return JSONResponse(
                    {
                        "ok": False,
                        "error": f"Failed to start {sensor}",
                        "results": results,
                    },
                    status_code=409,
                )
            return {"ok": True, "results": results}
        finally:
            self._thermal_stream.resume_after_thermal_subprocess_attempt()

    def stop_all(self) -> dict[str, Any]:
        return {
            "ok": True,
            "results": {
                sensor: sensor_runner.stop(sensor, self.layer8_dir)
                for sensor in ("thermal", "webcam", "multi_camera", "mmwave")
            },
        }

    def restart_all(self) -> dict[str, Any] | JSONResponse:
        sensor_runner.stop("thermal", self.layer8_dir)
        sensor_runner.stop("webcam", self.layer8_dir)
        sensor_runner.stop("multi_camera", self.layer8_dir)
        sensor_runner.stop("mmwave", self.layer8_dir)
        return self.run_all()
