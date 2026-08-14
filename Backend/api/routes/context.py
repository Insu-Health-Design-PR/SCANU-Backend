"""Shared dependencies for modular API route builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from runtime import multi_camera_runner, thermal_runner, webcam_runner  # noqa: E402

from api.streaming.frame_sources import IpcFrameSources
from services.global_id_service import init_global_id_service
from services.metrics_service import MetricsService
from services.recording_service import RecordingService
from services.screenshot_service import ScreenshotService
from services.sensor_manager import SensorManager
from services.settings_service import SettingsService


@dataclass
class RouterContext:
    layer8_dir: Path
    thermal_stream: Any
    webcam_stream: Any
    multi_camera_stream: Any
    frame_sources: IpcFrameSources
    settings: SettingsService
    metrics: MetricsService
    sensors: SensorManager
    recording: RecordingService
    screenshots: ScreenshotService
    webrtc_peers: set[Any] = field(default_factory=set)
    global_id: Any = None


def create_router_context(layer8_dir: Path) -> RouterContext:
    layer8_dir = Path(layer8_dir).resolve()
    thermal_stream = thermal_runner.get_thermal_shared_stream(layer8_dir)
    webcam_stream = webcam_runner.get_webcam_shared_stream(layer8_dir)
    multi_camera_stream = multi_camera_runner.get_multi_camera_shared_stream(layer8_dir)
    frame_sources = IpcFrameSources()
    settings = SettingsService(layer8_dir)
    metrics = MetricsService(layer8_dir)
    sensors = SensorManager(
        layer8_dir,
        thermal_stream=thermal_stream,
        webcam_stream=webcam_stream,
        multi_camera_stream=multi_camera_stream,
    )
    recording = RecordingService(layer8_dir, frame_sources)
    screenshots = ScreenshotService(frame_sources)

    settings_dict = settings.get()
    global_id = init_global_id_service(layer8_dir, settings_dict, start=True)
    # Prefer IPC fanout BGR for Re-ID crops when available.
    try:
        from layer8_ui.webcam_ipc_fanout import get_webcam_ipc_fanout
        from layer8_ui.multi_camera_ipc_fanout import get_multi_camera_ipc_fanout

        cfg = global_id.config
        global_id.set_frame_provider(cfg.camera_front, get_webcam_ipc_fanout())
        global_id.set_frame_provider(cfg.camera_back, get_multi_camera_ipc_fanout())
    except Exception:
        pass

    return RouterContext(
        layer8_dir=layer8_dir,
        thermal_stream=thermal_stream,
        webcam_stream=webcam_stream,
        multi_camera_stream=multi_camera_stream,
        frame_sources=frame_sources,
        settings=settings,
        metrics=metrics,
        sensors=sensors,
        recording=recording,
        screenshots=screenshots,
        global_id=global_id,
    )
