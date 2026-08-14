"""
Webcam / Front Camera live preview and weapon-inference subprocess command.

Front Camera runs ``runtime.webcam_layer8_runner`` → ``weapon_ai.infer_objects``
(multi-class Multi-Camera path; formerly single-class infer_thermal_objects).
"""

from __future__ import annotations

import os
import shlex
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

from layer8_ui.artifact_paths import abs_software_path, software_root_from_settings
from layer8_ui.webcam_device import detect_working_webcam_device
from layer8_ui.weapon_cli_args import build_structured_weapon_args, resolve_gun_model_path

_WEBCAM_CAM_CFG_LEN = 7


def _capture_frame_sleep_s(capture_fps: float) -> float:
    """Rough pacing after each grab+encode; clamp to safe range."""
    f = max(1.0, min(60.0, float(capture_fps)))
    return min(0.2, max(0.005, 1.0 / f))


_DEFAULT_WEAPON_CHECKPOINT = "trained_models/gun_detection/gun_sohas_7class_phone_black_mix_v1.pt"
_DEFAULT_WEBCAM_LIVE_IPC = Path("/dev/shm/scanu_webcam_live_frame.bin")
_DEFAULT_WEBCAM_LIVE_BGR_IPC = Path("/dev/shm/scanu_webcam_live_bgr_frame.bin")


def _frame_to_bgr_for_jpeg(frame: Any) -> np.ndarray | None:
    """V4L2 can yield 2-channel or odd layouts; JPEG needs 1/3/4 ch — normalize to BGR."""
    if frame is None or not hasattr(frame, "shape"):
        return None
    if frame.size == 0:
        return None
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3:
        return None
    ch = int(frame.shape[2])
    if ch == 3:
        return frame
    if ch == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if ch == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if ch == 2:
        packed = np.ascontiguousarray(frame)
        # Most USB webcams expose YUYV/YUY2 (YUV 4:2:2 packed) as 2-channel in OpenCV.
        try:
            return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_YUY2)
        except cv2.error:
            pass
        # Fallback for UYVY cameras.
        try:
            return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_UYVY)
        except cv2.error:
            pass
        # Last resort (keeps pipeline alive, colors may be wrong).
        a = np.ascontiguousarray(frame[:, :, 0])
        b = np.ascontiguousarray(frame[:, :, 1])
        return cv2.merge([a, b, a])
    return None


def _webcam_structured_weapon_args(
    w: dict[str, Any], sw: Path, *, sentinel: dict[str, Any] | None = None,
    global_id: dict[str, Any] | None = None,
) -> str:
    ww = dict(w)
    if not str(ww.get("camera_id") or "").strip():
        ww["camera_id"] = "camera_1"
    g = global_id if isinstance(global_id, dict) else {}
    if int(g.get("enable", 0) or 0):
        ww["weapon_global_id_overlay"] = 1
        if g.get("state_json"):
            ww["global_id_state_json"] = str(g["state_json"])
        if g.get("camera_front"):
            ww["camera_id"] = str(g["camera_front"])
    return build_structured_weapon_args(
        ww, sw, include_overlay_classes=True, sentinel=sentinel
    )


def _webcam_pipeline(settings: dict[str, Any]) -> str:
    raw = str((settings.get("webcam") or {}).get("webcam_pipeline") or "infer").strip().lower()
    if raw in ("preview_only", "preview", "blank", "capture_only"):
        return "preview_only"
    return "infer"


def webcam_preview_only(settings: dict[str, Any]) -> bool:
    """Live V4L2 preview in the API process; no weapon-inference subprocess."""
    return _webcam_pipeline(settings) == "preview_only"


def begin_webcam_preview_only(settings: dict[str, Any], layer8_dir: Path) -> None:
    """Hold the shared webcam reader open while preview-only Run is active."""
    get_webcam_shared_stream(layer8_dir).add_client(settings)


def end_webcam_preview_only(layer8_dir: Path) -> None:
    get_webcam_shared_stream(layer8_dir).remove_client()


def build_webcam_command(settings: dict[str, Any], layer8_dir: Path) -> list[str]:
    """CLI for ``runtime.webcam_layer8_runner`` (Front Camera · Multi-Camera / infer_objects)."""
    if webcam_preview_only(settings):
        raise ValueError("webcam_pipeline is preview_only — no infer subprocess (camera-only preview).")
    import os as _os
    import sys

    del layer8_dir  # cwd/log use paths from settings
    w = settings.get("webcam") or {}
    sw = software_root_from_settings(settings)
    py = _os.environ.get("PYTHON", sys.executable)
    live = abs_software_path(settings, str(w.get("live_frame") or ""))
    # Recording is opt-in: only use webcam.video when explicitly set.
    video = abs_software_path(settings, str(w.get("video") or ""))
    metrics_json = abs_software_path(
        settings,
        str(w.get("metrics_json") or "layer8_ui/artifacts/live_threat_metrics.json"),
    )
    webcam_device = int(w.get("webcam_device", 0))
    webcam_width = int(w.get("webcam_width", 3840))
    webcam_height = int(w.get("webcam_height", 2160))
    webcam_fps = float(w.get("fps", 30))
    ck_raw = str(
        w.get("weapon_checkpoint")
        or w.get("weapon_gun_yolo_model")
        or _DEFAULT_WEAPON_CHECKPOINT
    )
    ck_abs = resolve_gun_model_path(sw, ck_raw)
    cmd = [
        py,
        "-m",
        "runtime.webcam_layer8_runner",
        "--webcam-device",
        f"/dev/video{int(webcam_device)}",
        "--capture-width",
        str(int(webcam_width)),
        "--capture-height",
        str(int(webcam_height)),
        "--capture-fps",
        str(float(webcam_fps)),
        "--checkpoint",
        ck_abs,
        "--live-ipc-bgr-frame",
        str(_DEFAULT_WEBCAM_LIVE_BGR_IPC),
    ]
    # For 1080p smooth WebRTC, avoid per-frame JPEG encode in infer subprocess.
    # Keep raw BGR mmap as primary live transport; UI still falls back to MJPEG paths
    # when WebRTC is unavailable or runner is off.
    _ = live
    if metrics_json:
        cmd.extend(["--metrics-json", metrics_json])
    if video:
        cmd.extend(["--video", video])
    frames = int(w.get("frames", 0))
    if frames > 0:
        cmd.extend(["--frames", str(frames)])
    extra = _webcam_structured_weapon_args(
        w,
        sw,
        sentinel=settings.get("sentinel") if isinstance(settings, dict) else None,
        global_id=settings.get("global_id") if isinstance(settings, dict) else None,
    ).strip()
    if extra:
        cmd.extend(["--weapon-extra-args", extra])
    return cmd


def webcam_command_cwd(settings: dict[str, Any]) -> Path:
    return software_root_from_settings(settings)


class WebcamSharedStream:
    """Single webcam reader shared by MJPEG clients (BGR → JPEG)."""

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
        w = settings.get("webcam") or {}
        requested_device = int(w.get("webcam_device", 0))
        webcam_auto_detect = int(w.get("webcam_auto_detect", 1))
        webcam_detect_max_index = int(w.get("webcam_detect_max_index", 8))
        webcam_detect_retry_s = float(w.get("webcam_detect_retry_s", 8.0))
        capture_fps = float(w.get("fps", 30) or 30)
        return (
            requested_device,
            int(w.get("webcam_width", 3840)),
            int(w.get("webcam_height", 2160)),
            webcam_auto_detect,
            webcam_detect_max_index,
            webcam_detect_retry_s,
            capture_fps,
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
                self._thread = threading.Thread(target=self._run, name="webcam-shared-stream", daemon=True)
                self._thread.start()

    def remove_client(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)
            if self._clients == 0:
                self._stop_event.set()

    def pause_for_webcam_subprocess(self, join_timeout_s: float = 6.0) -> None:
        with self._lock:
            self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=float(join_timeout_s))

    def resume_after_webcam_subprocess_attempt(self) -> None:
        with self._lock:
            self._stop_event.clear()
            if self._clients > 0 and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, name="webcam-shared-stream", daemon=True)
                self._thread.start()

    def latest_jpg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpg

    def _subprocess_running(self) -> bool:
        from runtime import sensor_runner

        return bool(sensor_runner.status("webcam", self._layer8_dir).get("running"))

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

                settings_cam_cfg = cfg[:_WEBCAM_CAM_CFG_LEN]
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
                    (
                        preferred_device,
                        width,
                        height,
                        webcam_auto_detect,
                        detect_max_index,
                        detect_retry_s,
                        capture_fps,
                    ) = settings_cam_cfg
                    device = self._resolved_device if self._resolved_device is not None else int(preferred_device)
                    cap = cv2.VideoCapture(int(device), cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        if webcam_auto_detect:
                            fallback_device = None
                            if now_ts >= self._next_detect_retry_ts:
                                fallback_device = detect_working_webcam_device(
                                    preferred=int(preferred_device),
                                    width=int(width),
                                    height=int(height),
                                    search_max_index=int(detect_max_index),
                                    fps=float(capture_fps),
                                )
                                self._next_detect_retry_ts = now_ts + max(0.5, float(detect_retry_s))
                            # If detect returns the same index as the failed open, we must open again:
                            # probe+read in detect can succeed while a bare VideoCapture just failed
                            # (V4L2 quirk, timing, or another client briefly holding the node).
                            if fallback_device is not None:
                                try_cap = cv2.VideoCapture(int(fallback_device), cv2.CAP_V4L2)
                                if try_cap.isOpened():
                                    cap = try_cap
                                    self._resolved_device = int(fallback_device)
                                else:
                                    try_cap.release()
                        if cap is not None and cap.isOpened():
                            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                            cap.set(cv2.CAP_PROP_FPS, float(capture_fps))
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            active_camera_cfg = settings_cam_cfg
                            next_open_retry_ts = 0.0
                            continue
                        next_open_retry_ts = now_ts + 1.5
                        time.sleep(0.1)
                        continue
                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                    cap.set(cv2.CAP_PROP_FPS, float(capture_fps))
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self._resolved_device = int(device)
                    active_camera_cfg = settings_cam_cfg
                    next_open_retry_ts = 0.0
                    continue

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
                    time.sleep(min(0.05, _capture_frame_sleep_s(settings_cam_cfg[6])))
                    continue

                _device, width, height, _a, _b, _c, capture_fps = settings_cam_cfg
                bgr = _frame_to_bgr_for_jpeg(frame)
                if bgr is None:
                    time.sleep(min(0.05, _capture_frame_sleep_s(capture_fps)))
                    continue
                preview = cv2.resize(bgr, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
                ok_enc, jpg = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok_enc:
                    with self._lock:
                        self._latest_jpg = jpg.tobytes()
                time.sleep(_capture_frame_sleep_s(capture_fps))
        finally:
            if cap is not None:
                cap.release()


_webcam_stream: WebcamSharedStream | None = None


def get_webcam_shared_stream(layer8_dir: Path) -> WebcamSharedStream:
    global _webcam_stream
    if _webcam_stream is None:
        _webcam_stream = WebcamSharedStream(Path(layer8_dir))
    return _webcam_stream
