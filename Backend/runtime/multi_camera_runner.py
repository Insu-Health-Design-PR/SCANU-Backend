"""
Back Camera live preview (shared V4L2 reader) and multi-class weapon-inference subprocess.

Weapon pipeline runs ``runtime.multi_camera_layer8_runner`` → ``weapon_ai.infer_objects``.
Source may be local ``/dev/videoN`` or a Jetson IP stream (RTSP/HTTP URL).
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
from layer8_ui.webcam_device import (
    camera_missing_hint,
    detect_working_webcam_device,
    wait_for_webcam_device,
)
from layer8_ui.weapon_cli_args import build_structured_weapon_args, resolve_gun_model_path

_MULTI_CAMERA_CAM_CFG_LEN = 7


def _capture_frame_sleep_s(capture_fps: float) -> float:
    """Rough pacing after each grab+encode; clamp to safe range."""
    f = max(1.0, min(60.0, float(capture_fps)))
    return min(0.2, max(0.005, 1.0 / f))


_DEFAULT_WEAPON_CHECKPOINT = "trained_models/gun_detection/gun_sohas_6class.pt"
_DEFAULT_MULTI_CAMERA_LIVE_IPC = Path("/dev/shm/scanu_multi_camera_live_frame.bin")
_DEFAULT_MULTI_CAMERA_LIVE_BGR_IPC = Path("/dev/shm/scanu_multi_camera_live_bgr_frame.bin")


def resolve_multi_camera_source(w: dict[str, Any]) -> tuple[str | None, bool]:
    """
    Resolve Back Camera capture source.

    Returns ``(source, is_network)``. When ``is_network`` is True, ``source`` is an
    OpenCV-openable URL (RTSP/HTTP). When False, ``source`` is None and the caller
    should resolve a local ``/dev/videoN`` index.
    """
    mode = str(w.get("source_mode") or "local").strip().lower()
    explicit = str(w.get("jetson_stream_url") or "").strip()
    use_jetson = mode in ("jetson", "ip", "network", "rtsp") or bool(explicit)
    if not use_jetson:
        return None, False
    if explicit:
        return explicit, True
    ip = str(w.get("jetson_ip") or "").strip()
    if not ip:
        raise ValueError(
            "Back Camera Jetson mode: set jetson_ip (e.g. 192.168.1.50) or jetson_stream_url "
            "(full rtsp://… / http://… URL)."
        )
    port = str(w.get("jetson_stream_port") or "8554").strip() or "8554"
    path = str(w.get("jetson_stream_path") or "/stream").strip() or "/stream"
    if not path.startswith("/"):
        path = "/" + path
    scheme = str(w.get("jetson_stream_scheme") or "rtsp").strip() or "rtsp"
    return f"{scheme}://{ip}:{port}{path}", True


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


def _multi_camera_structured_weapon_args(
    w: dict[str, Any], sw: Path, *, sentinel: dict[str, Any] | None = None
) -> str:
    return build_structured_weapon_args(
        w, sw, include_overlay_classes=True, sentinel=sentinel
    )


def build_multi_camera_command(settings: dict[str, Any], layer8_dir: Path) -> list[str]:
    """CLI for ``runtime.multi_camera_layer8_runner`` (local USB or Jetson IP stream)."""
    import os as _os
    import sys

    del layer8_dir  # cwd/log use paths from settings
    w = settings.get("multi_camera") or {}
    sw = software_root_from_settings(settings)
    py = _os.environ.get("PYTHON", sys.executable)
    live = abs_software_path(settings, str(w.get("live_frame") or ""))
    # Recording is opt-in: only use webcam.video when explicitly set.
    video = abs_software_path(settings, str(w.get("video") or ""))
    metrics_json = abs_software_path(
        settings,
        str(w.get("metrics_json") or "layer8_ui/artifacts/live_threat_metrics.json"),
    )
    webcam_width = int(w.get("webcam_width", 3840))
    webcam_height = int(w.get("webcam_height", 2160))
    webcam_fps = float(w.get("fps", 30))
    network_source, is_network = resolve_multi_camera_source(w)
    if is_network:
        webcam_source = str(network_source)
    else:
        webcam_device = int(w.get("webcam_device", 0))
        auto_detect = int(w.get("webcam_auto_detect", 1))
        detect_max = int(w.get("webcam_detect_max_index", 8))
        # Wait for USB cams that enumerate late / briefly drop during reconnect.
        wait_s = float(w.get("webcam_start_wait_s", 20) or 20)
        if auto_detect:
            resolved = wait_for_webcam_device(
                preferred=webcam_device,
                width=min(1280, webcam_width),
                height=min(720, webcam_height),
                search_max_index=detect_max,
                fps=min(30.0, webcam_fps),
                timeout_s=wait_s,
            )
        else:
            resolved = wait_for_webcam_device(
                preferred=webcam_device,
                width=min(1280, webcam_width),
                height=min(720, webcam_height),
                search_max_index=webcam_device,
                fps=min(30.0, webcam_fps),
                timeout_s=wait_s,
            )
            if resolved is not None and int(resolved) != int(webcam_device):
                resolved = None
        if resolved is None:
            # Last chance: node exists but 4K probe failed — accept any existing index.
            if auto_detect:
                resolved = detect_working_webcam_device(
                    preferred=webcam_device,
                    width=640,
                    height=480,
                    search_max_index=detect_max,
                    fps=15.0,
                )
        if resolved is None:
            raise ValueError(
                f"Multi-Cam: webcam not ready (wanted /dev/video{webcam_device}). {camera_missing_hint()}"
            )
        webcam_device = int(resolved)
        w["webcam_device"] = webcam_device
        webcam_source = f"/dev/video{int(webcam_device)}"
    ck_raw = str(
        w.get("weapon_checkpoint")
        or w.get("weapon_gun_yolo_model")
        or _DEFAULT_WEAPON_CHECKPOINT
    )
    ck_abs = resolve_gun_model_path(sw, ck_raw)
    cmd = [
        py,
        "-m",
        "runtime.multi_camera_layer8_runner",
        "--webcam-device",
        webcam_source,
        "--capture-width",
        str(int(webcam_width)),
        "--capture-height",
        str(int(webcam_height)),
        "--capture-fps",
        str(float(webcam_fps)),
        "--checkpoint",
        ck_abs,
        "--live-ipc-bgr-frame",
        str(_DEFAULT_MULTI_CAMERA_LIVE_BGR_IPC),
    ]
    # GStreamer path expects v4l2src; Jetson RTSP/HTTP must use OpenCV VideoCapture.
    if is_network:
        cmd.append("--no-gstreamer-capture")
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
    extra = _multi_camera_structured_weapon_args(
        w, sw, sentinel=settings.get("sentinel") if isinstance(settings, dict) else None
    ).strip()
    if extra:
        cmd.extend(["--weapon-extra-args", extra])
    return cmd


def multi_camera_command_cwd(settings: dict[str, Any]) -> Path:
    return software_root_from_settings(settings)


class MultiCameraSharedStream:
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
        w = settings.get("multi_camera") or {}
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
                self._thread = threading.Thread(target=self._run, name="multi-camera-shared-stream", daemon=True)
                self._thread.start()

    def remove_client(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)
            if self._clients == 0:
                self._stop_event.set()

    def pause_for_multi_camera_subprocess(self, join_timeout_s: float = 6.0) -> None:
        with self._lock:
            self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=float(join_timeout_s))

    def resume_after_multi_camera_subprocess_attempt(self) -> None:
        with self._lock:
            self._stop_event.clear()
            if self._clients > 0 and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, name="multi-camera-shared-stream", daemon=True)
                self._thread.start()

    def latest_jpg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpg

    def _subprocess_running(self) -> bool:
        from runtime import sensor_runner

        return bool(sensor_runner.status("multi_camera", self._layer8_dir).get("running"))

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

                settings_cam_cfg = cfg[:_MULTI_CAMERA_CAM_CFG_LEN]
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


_multi_camera_stream: MultiCameraSharedStream | None = None


def get_multi_camera_shared_stream(layer8_dir: Path) -> MultiCameraSharedStream:
    global _multi_camera_stream
    if _multi_camera_stream is None:
        _multi_camera_stream = MultiCameraSharedStream(Path(layer8_dir))
    return _multi_camera_stream
