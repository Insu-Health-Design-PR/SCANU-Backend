"""
Thermal live preview (shared V4L2 reader) and weapon-inference subprocess command.

``thermal_pipeline`` in settings:
  shared       — overlay polls ``thermal.live_frame`` (camera owned elsewhere; default)
  direct       — infer opens V4L2 directly
  capture_only — plain colormap capture via layer1 ``thermal_only_capture.py``
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

from layer8_ui.artifact_paths import abs_software_path, software_root_from_settings
from layer8_ui.thermal_device import (
    _blocked_for_thermal,
    detect_working_thermal_device,
    resolve_thermal_device_or_raise,
    v4l2_device_path,
)
from layer8_ui.weapon_cli_args import resolve_gun_model_path
from runtime.webcam_runner import _webcam_structured_weapon_args

_THERMAL_CAM_CFG_LEN = 9
_DEFAULT_WEAPON_CHECKPOINT = "trained_models/gun_detection/gun_prob_best.pt"
_DEFAULT_THERMAL_LIVE_BGR_IPC = Path("/dev/shm/scanu_thermal_live_bgr_frame.bin")


def _configure_thermal_capture(
    cap: cv2.VideoCapture,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Configure PureThermal UVC for greyscale thermal capture (Y16 preferred)."""
    w = int(width) if int(width) > 0 else 80
    h = int(height) if int(height) > 0 else 60
    # BGR3 only supports 80x60; Y16 supports 80x60 and 80x63. Prefer Y16.
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    except Exception:
        pass
    configured = False
    for fourcc, tw, th in (
        ("Y16 ", w, h),
        ("Y16 ", 80, 63 if h >= 63 else 60),
        ("Y16 ", 80, 60),
        ("GREY", 80, 60),
        ("BGR3", 80, 60),
    ):
        try:
            if fourcc == "Y16 ":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "1", "6", " "))
            elif fourcc == "GREY":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"GREY"))
            else:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"BGR3"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(tw))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(th))
            configured = True
            break
        except Exception:
            continue
    if not configured:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 80)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 60)
    try:
        cap.set(cv2.CAP_PROP_FPS, max(1, int(fps)))
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def layer1_examples_dir(software_root: Path) -> Path:
    return software_root / "layer1_radar" / "examples"


def thermal_capture_script(software_root: Path) -> Path:
    return layer1_examples_dir(software_root) / "thermal_only_capture.py"


def _thermal_pipeline(t: dict[str, Any]) -> str:
    raw = str(t.get("thermal_pipeline") or "direct").strip().lower()
    if raw in ("shared", "direct", "capture_only"):
        return raw
    return "direct"


def _resolve_thermal_device(t: dict[str, Any]) -> int:
    return resolve_thermal_device_or_raise(t)


def _build_thermal_capture_only_command(settings: dict[str, Any]) -> list[str]:
    import os as _os
    import sys

    sw = software_root_from_settings(settings)
    py = _os.environ.get("THERMAL_PYTHON") or _os.environ.get("PYTHON")
    if not py:
        py = (
            shutil.which("python3.12")
            or shutil.which("python3.11")
            or shutil.which("python3.10")
            or sys.executable
        )
    t = settings.get("thermal") or {}
    script = thermal_capture_script(sw)
    video = (t.get("video") or "").strip()
    live = abs_software_path(settings, str(t.get("live_frame") or ""))
    out = (t.get("output") or "").strip()
    thermal_device = _resolve_thermal_device(t)

    cmd = [
        py,
        str(script),
        "--frames",
        str(int(t.get("frames", 0))),
        "--fps",
        str(float(t.get("fps", 10))),
        "--thermal-device",
        v4l2_device_path(thermal_device),
        "--thermal-width",
        str(int(t.get("thermal_width", 80))),
        "--thermal-height",
        str(int(t.get("thermal_height", 60))),
        "--thermal-fps",
        str(int(t.get("thermal_fps", 30))),
        "--panel-w",
        str(int(t.get("panel_w", 640))),
        "--panel-h",
        str(int(t.get("panel_h", 480))),
    ]
    if video:
        cmd.extend(["--video", video])
    if out:
        cmd.extend(["--output", out])
    if live:
        cmd.extend(["--live-frame", live])
    return cmd


def _build_thermal_infer_command(settings: dict[str, Any], *, shared: bool) -> list[str]:
    import os as _os
    import sys

    t = settings.get("thermal") or {}
    sw = software_root_from_settings(settings)
    py = _os.environ.get("PYTHON", sys.executable)
    metrics_json = abs_software_path(
        settings,
        str(t.get("metrics_json") or "layer8_ui/artifacts/live_thermal_threat_metrics.json"),
    )
    video = abs_software_path(settings, str(t.get("video") or ""))
    live_poll = abs_software_path(settings, str(t.get("live_frame") or ""))
    ck_abs = resolve_gun_model_path(sw, _DEFAULT_WEAPON_CHECKPOINT)
    cmd = [
        py,
        "-m",
        "runtime.thermal_layer8_runner",
        "--thermal-fps",
        str(float(t.get("thermal_fps", 30))),
        "--panel-w",
        str(int(t.get("panel_w", 640))),
        "--panel-h",
        str(int(t.get("panel_h", 480))),
        "--checkpoint",
        ck_abs,
        "--live-ipc-bgr-frame",
        str(_DEFAULT_THERMAL_LIVE_BGR_IPC),
    ]
    tw = int(t.get("thermal_width", 80))
    th = int(t.get("thermal_height", 60))
    if shared:
        if not live_poll:
            raise ValueError("thermal.live_frame must be set for shared overlay pipeline")
        cmd.extend(["--live-frame-poll", live_poll])
        if not int(t.get("thermal_external_capture", 0)):
            dev = _resolve_thermal_device(t)
            cmd.extend(
                [
                    "--thermal-capture-feed",
                    v4l2_device_path(dev),
                    "--thermal-width",
                    str(tw),
                    "--thermal-height",
                    str(th),
                ]
            )
    else:
        dev = _resolve_thermal_device(t)
        cmd.extend(
            [
                "--thermal-device",
                v4l2_device_path(dev),
                "--thermal-width",
                str(tw),
                "--thermal-height",
                str(th),
            ]
        )
    if metrics_json:
        cmd.extend(["--metrics-json", metrics_json])
    if live_poll:
        cmd.extend(["--live-frame", live_poll])
    if video:
        cmd.extend(["--video", video])
    frames = int(t.get("frames", 0))
    if frames > 0:
        cmd.extend(["--frames", str(frames)])
    extra = _webcam_structured_weapon_args(
        t, sw, sentinel=settings.get("sentinel") if isinstance(settings, dict) else None
    ).strip()
    if extra:
        cmd.extend(["--weapon-extra-args", extra])
    return cmd


def thermal_preview_only(settings: dict[str, Any]) -> bool:
    """Live colormap preview in the API process; no weapon-inference subprocess."""
    return _thermal_pipeline(settings.get("thermal") or {}) == "capture_only"


def build_thermal_command(settings: dict[str, Any], _layer8_dir: Path) -> list[str]:
    pipeline = _thermal_pipeline(settings.get("thermal") or {})
    if pipeline == "capture_only":
        return _build_thermal_capture_only_command(settings)
    return _build_thermal_infer_command(settings, shared=(pipeline == "shared"))


def thermal_command_cwd(settings: dict[str, Any]) -> Path:
    pipeline = _thermal_pipeline(settings.get("thermal") or {})
    if pipeline == "capture_only":
        return software_root_from_settings(settings)
    return software_root_from_settings(settings)


def thermal_uses_inprocess_v4l2_preview(settings: dict[str, Any]) -> bool:
    """Only ``capture_only`` should open V4L2 inside the uvicorn process."""
    return _thermal_pipeline(settings.get("thermal") or {}) == "capture_only"


class ThermalSharedStream:
    """Single thermal camera reader shared by MJPEG / WebSocket clients."""

    def __init__(self, layer8_dir: Path) -> None:
        self._layer8_dir = Path(layer8_dir).resolve()
        self._lock = threading.Lock()
        self._cfg: tuple[Any, ...] | None = None
        self._cfg_dirty = False
        self._latest_jpg: bytes | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._clients = 0
        self._resolved_device: int | None = None
        self._next_detect_retry_ts = 0.0

    @property
    def layer8_dir(self) -> Path:
        return self._layer8_dir

    @staticmethod
    def _cfg_from_settings(settings: dict[str, Any]) -> tuple[Any, ...]:
        t = settings.get("thermal") or {}
        requested_device = int(t.get("thermal_device", 0))
        thermal_auto_detect = int(t.get("thermal_auto_detect", 1))
        thermal_detect_max_index = int(t.get("thermal_detect_max_index", 6))
        thermal_detect_retry_s = float(t.get("thermal_detect_retry_s", 12.0))
        thermal_width = int(t.get("thermal_width", 80))
        thermal_height = int(t.get("thermal_height", 60))
        thermal_fps = int(t.get("thermal_fps", 30))
        return (
            requested_device,
            thermal_width,
            thermal_height,
            thermal_fps,
            int(t.get("panel_w", 640)),
            int(t.get("panel_h", 480)),
            thermal_auto_detect,
            thermal_detect_max_index,
            thermal_detect_retry_s,
        )

    def sync_settings(self, settings: dict[str, Any]) -> None:
        new_cfg = self._cfg_from_settings(settings)
        with self._lock:
            if new_cfg == self._cfg:
                return
            self._cfg = new_cfg
            self._cfg_dirty = True
            self._resolved_device = None
            self._next_detect_retry_ts = 0.0

    def add_client(self, settings: dict[str, Any]) -> None:
        new_cfg = self._cfg_from_settings(settings)
        with self._lock:
            self._clients += 1
            if self._cfg != new_cfg:
                self._cfg = new_cfg
                self._cfg_dirty = True
                self._resolved_device = None
                self._next_detect_retry_ts = 0.0
            if self._thread is None or not self._thread.is_alive():
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._run, name="thermal-shared-stream", daemon=True)
                self._thread.start()

    def remove_client(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)
            if self._clients == 0:
                self._stop_event.set()

    def pause_for_thermal_subprocess(self, join_timeout_s: float = 6.0) -> None:
        self.force_release_camera(join_timeout_s=join_timeout_s)

    def force_release_camera(self, join_timeout_s: float = 6.0) -> None:
        """Stop the shared reader and release V4L2 (required before infer subprocess opens camera)."""
        with self._lock:
            self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=float(join_timeout_s))

    def resume_after_thermal_subprocess_attempt(self) -> None:
        if self._subprocess_running():
            return
        with self._lock:
            self._stop_event.clear()
            if self._clients > 0 and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, name="thermal-shared-stream", daemon=True)
                self._thread.start()

    def latest_jpg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpg

    def _subprocess_running(self) -> bool:
        from runtime import sensor_runner

        return bool(sensor_runner.status("thermal", self._layer8_dir).get("running"))

    def _run(self) -> None:
        cap: cv2.VideoCapture | None = None
        active_camera_cfg: tuple[Any, ...] | None = None
        next_open_retry_ts = 0.0
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    clients = self._clients
                    cfg = self._cfg
                    cfg_dirty = self._cfg_dirty
                    if self._cfg_dirty:
                        self._cfg_dirty = False

                if clients <= 0 or cfg is None:
                    time.sleep(0.05)
                    continue

                settings_cam_cfg = cfg[:_THERMAL_CAM_CFG_LEN]
                if cap is None or cfg_dirty or settings_cam_cfg != active_camera_cfg:
                    if self._subprocess_running():
                        if cap is not None:
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = None
                            active_camera_cfg = None
                        time.sleep(0.18)
                        continue
                    now_ts = time.time()
                    if now_ts < next_open_retry_ts:
                        time.sleep(0.05)
                        continue
                    if cap is not None:
                        cap.release()
                        cap = None
                    preferred_device, width, height, fps, _, _, thermal_auto_detect, detect_max_index, detect_retry_s = (
                        settings_cam_cfg
                    )
                    device = self._resolved_device if self._resolved_device is not None else int(preferred_device)
                    # Only auto-detect when preferred is a webcam node or unknown — don't probe
                    # every reopen (v4l2-ctl/OpenCV fights exclusive PureThermal access).
                    if self._resolved_device is None and (
                        _blocked_for_thermal(device) or (thermal_auto_detect and not Path(v4l2_device_path(device)).exists())
                    ):
                        detected = detect_working_thermal_device(
                            preferred=int(preferred_device),
                            width=int(width),
                            height=int(height),
                            fps=int(fps),
                            search_max_index=int(detect_max_index),
                        )
                        if detected is not None:
                            device = int(detected)
                            self._resolved_device = device
                    cap = cv2.VideoCapture(v4l2_device_path(device), cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        if thermal_auto_detect:
                            fallback_device = None
                            if now_ts >= self._next_detect_retry_ts:
                                fallback_device = detect_working_thermal_device(
                                    preferred=int(preferred_device),
                                    width=int(width),
                                    height=int(height),
                                    fps=int(fps),
                                    search_max_index=int(detect_max_index),
                                )
                                self._next_detect_retry_ts = now_ts + max(0.5, float(detect_retry_s))
                            if fallback_device is not None:
                                try_cap = cv2.VideoCapture(v4l2_device_path(fallback_device), cv2.CAP_V4L2)
                                if try_cap.isOpened():
                                    cap = try_cap
                                    self._resolved_device = int(fallback_device)
                                else:
                                    try_cap.release()
                        if cap is not None and cap.isOpened():
                            active_camera_cfg = settings_cam_cfg
                            next_open_retry_ts = 0.0
                            _configure_thermal_capture(cap, int(width), int(height), int(fps))
                            continue
                        next_open_retry_ts = now_ts + 1.5
                        time.sleep(0.1)
                        continue
                    _configure_thermal_capture(cap, int(width), int(height), int(fps))
                    self._resolved_device = int(device)
                    active_camera_cfg = settings_cam_cfg
                    next_open_retry_ts = 0.0

                if self._subprocess_running():
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                        active_camera_cfg = None
                    time.sleep(0.18)
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.03)
                    continue

                _, _, _, _, panel_w, panel_h, _a, _b, _c = settings_cam_cfg
                if frame.dtype == np.uint16:
                    f32 = frame.astype("float32")
                    mn = float(f32.min())
                    mx = float(f32.max())
                    if mx - mn > 1e-6:
                        gray = ((f32 - mn) / (mx - mn) * 255.0).astype("uint8")
                    else:
                        gray = cv2.convertScaleAbs(frame)
                elif len(frame.shape) == 2:
                    gray = frame
                elif len(frame.shape) == 3:
                    ch = int(frame.shape[2])
                    if ch == 1:
                        gray = frame[:, :, 0]
                    elif ch == 2:
                        gray = np.mean(frame, axis=2).astype(np.uint8)
                    elif ch == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    elif ch == 4:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                    else:
                        gray = np.mean(frame, axis=2).astype(np.uint8)
                else:
                    gray = frame

                heat = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
                heat = cv2.resize(heat, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)

                ok_enc, jpg = cv2.imencode(".jpg", heat, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_enc:
                    with self._lock:
                        self._latest_jpg = jpg.tobytes()
                time.sleep(0.01)
        finally:
            if cap is not None:
                cap.release()


_thermal_stream: ThermalSharedStream | None = None


def get_thermal_shared_stream(layer8_dir: Path) -> ThermalSharedStream:
    global _thermal_stream
    if _thermal_stream is None:
        _thermal_stream = ThermalSharedStream(Path(layer8_dir))
    return _thermal_stream
