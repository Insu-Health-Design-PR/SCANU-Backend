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
        out["note"] = ""
        return out
