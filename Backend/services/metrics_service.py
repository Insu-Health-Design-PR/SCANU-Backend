"""System and model metrics collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from layer8_ui import system_metrics  # noqa: E402
from runtime import sensor_runner  # noqa: E402
from layer8_ui.artifact_paths import resolved_artifact_path  # noqa: E402
from layer8_ui.settings_store import load  # noqa: E402


class MetricsService:
    def __init__(self, layer8_dir: Path) -> None:
        self.layer8_dir = Path(layer8_dir)

    def system_snapshot(self) -> dict[str, Any]:
        return system_metrics.snapshot()

    def dashboard_metrics(self) -> dict[str, Any]:
        settings = load(self.layer8_dir)
        thermal_running = bool(sensor_runner.status("thermal", self.layer8_dir).get("running"))
        webcam_running = bool(sensor_runner.status("webcam", self.layer8_dir).get("running"))
        multi_camera_running = bool(sensor_runner.status("multi_camera", self.layer8_dir).get("running"))
        if thermal_running:
            rel = (
                (settings.get("thermal") or {}).get("metrics_json")
                or "layer8_ui/artifacts/live_thermal_threat_metrics.json"
            )
            note = "Thermal infer active; metrics from thermal.metrics_json."
        elif multi_camera_running:
            rel = (
                (settings.get("multi_camera") or {}).get("metrics_json")
                or "layer8_ui/artifacts/live_multi_camera_threat_metrics.json"
            )
            note = "Multi_Camera infer active; metrics from multi_camera.metrics_json."
        elif webcam_running:
            rel = (
                (settings.get("webcam") or {}).get("metrics_json")
                or "layer8_ui/artifacts/live_threat_metrics.json"
            )
            note = "AI Camera infer active; metrics from webcam.metrics_json."
        else:
            rel = (
                (settings.get("webcam") or {}).get("metrics_json")
                or "layer8_ui/artifacts/live_threat_metrics.json"
            )
            note = "Start Thermal, AI Camera, or Multi_Camera runner for live threat metrics."

        path = resolved_artifact_path(settings, relative_to_software=str(rel), layer8_dir=self.layer8_dir)
        base: dict[str, Any] = {
            "unsafe_pct": None,
            "unsafe_score": None,
            "gun_detected": None,
            "object_gun_peak": None,
            "weapon_gun_peak": None,
            "persons_with_gun": None,
            "persons_total": None,
            "prediction": None,
            "mmwave_torso_score": None,
            "frame": None,
            "ts": None,
            "infer_fps": None,
            "note": note,
        }
        if path is None or not path.is_file():
            return base
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {**base, "note": "Metrics file exists but is not valid JSON yet."}
        if not isinstance(data, dict):
            return {**base, "note": "Metrics JSON must be an object."}
        out = {**base}
        for key in out.keys() - {"note"}:
            if key in data:
                out[key] = data[key]
        if out.get("mmwave_torso_score") is None:
            try:
                from layer8_ui.artifact_paths import software_root_from_settings
                from services.mmwave_metrics_service import live_metrics_snapshot

                settings = load(self.layer8_dir)
                mm = live_metrics_snapshot(
                    settings,
                    layer8_dir=self.layer8_dir,
                    software_root=software_root_from_settings(settings),
                )
                if isinstance(mm, dict) and mm.get("mmwave_torso_score") is not None:
                    out["mmwave_torso_score"] = mm["mmwave_torso_score"]
            except Exception:
                pass
        out["note"] = ""
        return out

    def threat_metrics_for(self, sensor: str) -> dict[str, Any]:
        """Read threat JSON for one camera runner (webcam = front, multi_camera = back)."""
        settings = load(self.layer8_dir)
        running = bool(sensor_runner.status(sensor, self.layer8_dir).get("running"))
        if sensor == "webcam":
            rel = (
                (settings.get("webcam") or {}).get("metrics_json")
                or "layer8_ui/artifacts/live_threat_metrics.json"
            )
            label = "front_camera"
        elif sensor == "multi_camera":
            rel = (
                (settings.get("multi_camera") or {}).get("metrics_json")
                or "layer8_ui/artifacts/live_multi_camera_threat_metrics.json"
            )
            label = "back_camera"
        else:
            return {"running": False, "sensor": sensor, "gun_detected": False, "persons_total": 0}

        path = resolved_artifact_path(settings, relative_to_software=str(rel), layer8_dir=self.layer8_dir)
        base: dict[str, Any] = {
            "sensor": label,
            "runner": sensor,
            "running": running,
            "unsafe_pct": None,
            "unsafe_score": None,
            "gun_detected": False,
            "object_gun_peak": None,
            "weapon_gun_peak": None,
            "persons_with_gun": None,
            "persons_total": 0,
            "prediction": None,
            "frame": None,
            "ts": None,
            "byte_tracks": [],
            "firearm_tracks": [],
        }
        if not running:
            base["note"] = f"{label} runner stopped"
            return base
        if path is None or not path.is_file():
            base["note"] = "Metrics file not ready yet"
            return base
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {**base, "note": "Invalid metrics JSON"}
        if not isinstance(data, dict):
            return base
        out = {**base}
        for key in out.keys():
            if key in data:
                out[key] = data[key]
        out["note"] = ""
        return out
