"""JSON settings for Layer 8 sensor dashboard."""

from __future__ import annotations

import json
import re
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS: dict[str, Any] = {
    "software_root": "",
    "thermal": {
        "frames": 0,
        "fps": 30.0,
        "video": "",
        "live_frame": "layer8_ui/artifacts/live_thermal.jpg",
        "output": "",
        "thermal_device": 2,
        "thermal_auto_detect": 1,
        "thermal_detect_max_index": 12,
        "thermal_detect_retry_s": 12.0,
        "thermal_width": 160,
        "thermal_height": 120,
        "thermal_fps": 9,
        "panel_w": 640,
        "panel_h": 480,
        "thermal_pipeline": "capture_only",
        "thermal_external_capture": 0,
        "metrics_json": "layer8_ui/artifacts/live_thermal_threat_metrics.json",
        "person_detection_model": "yolov8n.pt",
        "weapon_unsafe_threshold": 0.5,
        "weapon_gun_threshold": 0.0,
        "weapon_yolo_model": "yolov8n.pt",
        "weapon_conf": 0.22,
        "weapon_person_conf": 0.40,
        "weapon_image_size": 224,
        "weapon_gun_conf": 0.1,
        "weapon_gun_imgsz": 160,
        "weapon_min_box_px": 24,
        "weapon_gun_min_box_px": 4,
        "weapon_gun_max_vs_person_height": 0.45,
        "weapon_gun_max_vs_person_area": 0.12,
        "weapon_gun_max_vs_person_width": 0.70,
        "weapon_gun_thermal": 1,
        "weapon_gun_only": 1,
        "weapon_gun_full_frame": 1,
        "weapon_byte_track": 0,
        "weapon_byte_track_firearms": 0,
        "weapon_byte_track_buffer": 45,
        "weapon_byte_track_fps": 9,
        "weapon_byte_track_firearm_ghost_frames": 18,
        "weapon_byte_track_firearm_draw_ghost": 1,
        "weapon_live_infer_stride": 1,
        "weapon_live_infer_max_width": 0,
        "weapon_live_ipc_max_width": 640,
        "weapon_live_gpu_preprocess": 1,
        "weapon_live_pipeline_publish": 1,
        "weapon_live_publish_workers": 1,
        "weapon_live_metrics_every_n": 30,
        "webrtc_smooth_display": 0,
        "webrtc_fps": 24,
        "webrtc_max_width": 1280,
        "webrtc_ipc_poll_fps": 30,
        "weapon_no_gun_yolo": 0,
        "weapon_show_yolo_name": 0,
        "weapon_gun_yolo_model": "gun_enhanced_cctv_v2.pt",
        "weapon_yolo_classes": "0",
        "weapon_yolo_device": "cuda",
        "weapon_classifier_device": "cuda",
        "weapon_gun_take_best": 1,
        "weapon_gun_take_best_infer_conf": 0.05,
        "weapon_gun_roi_pad_frac": 0.12,
        "weapon_gun_roi_pad_px": 48,
        "weapon_extra_args": "",
        "verbose": False,
        "weapon_gun_emit_min_conf": 0.02,
        "weapon_gun_class_emit_min": "",
        "weapon_gun_label_object_min": 0.15,
        "weapon_gun_label_weapon_min": 0.55,
        "weapon_assoc_motion": 1,
        "weapon_assoc_velocity_weight": 0.35,
        "weapon_assoc_min_speed": 1.5,
    },
    "webcam": {
        "frames": 0,
        "fps": 30.0,
        "video": "",
        "live_frame": "layer8_ui/artifacts/live_webcam.jpg",
        "output": "",
        "webcam_device": 0,
        "webcam_auto_detect": 1,
        "webcam_pipeline": "infer",
        "webcam_detect_max_index": 8,
        "webcam_detect_retry_s": 8.0,
        "webcam_width": 3840,
        "webcam_height": 2160,
        "metrics_json": "layer8_ui/artifacts/live_threat_metrics.json",
        "active_model_profile_id": "",
        "weapon_checkpoint": "trained_models/gun_detection/gun_sohas_7class_phone_black_mix_v1.pt",
        "person_detection_model": "yolov8n.pt",
        "weapon_unsafe_threshold": 0.92,
        "weapon_gun_threshold": 0.0,
        "weapon_yolo_model": "yolov8n.pt",
        "weapon_conf": 0.25,
        "weapon_person_conf": 0.40,
        "weapon_image_size": 224,
        "weapon_gun_conf": 0.25,
        "weapon_gun_imgsz": 640,
        "weapon_gun_batch": 1,
        "weapon_yolo_imgsz": 640,
        "weapon_min_box_px": 24,
        "weapon_gun_min_box_px": 8,
        "weapon_gun_max_vs_person_height": 0.45,
        "weapon_gun_max_vs_person_area": 0.12,
        "weapon_gun_max_vs_person_width": 0.70,
        "weapon_gun_roi_upper_frac": 0.80,
        "weapon_gun_thermal": 0,
        "weapon_byte_track": 1,
        "weapon_byte_track_firearms": 0,
        "weapon_byte_track_buffer": 45,
        "weapon_byte_track_fps": 30,
        "weapon_byte_track_firearm_ghost_frames": 18,
        "weapon_byte_track_firearm_draw_ghost": 1,
        "weapon_live_infer_stride": 1,
        "weapon_live_infer_max_width": 1920,
        "weapon_live_ipc_max_width": 1920,
        "weapon_live_gpu_preprocess": 1,
        "weapon_live_pipeline_publish": 1,
        "weapon_live_publish_workers": 1,
        "weapon_live_metrics_every_n": 30,
        "webrtc_smooth_display": 0,
        "webrtc_fps": 24,
        "webrtc_max_width": 1280,
        "webrtc_ipc_poll_fps": 30,
        "weapon_no_gun_yolo": 0,
        "weapon_show_yolo_name": 1,
        "weapon_overlay_classes": "all",
        "weapon_phone_overlay_label": "object",
        "weapon_gun_yolo_model": "gun_sohas_7class_phone_black_mix_v1.pt",
        "weapon_yolo_classes": "0",
        "weapon_yolo_device": "cuda",
        "weapon_classifier_device": "cuda",
        "weapon_gun_take_best": 0,
        "weapon_gun_take_best_infer_conf": 0.10,
        "weapon_gun_roi_pad_frac": 0.12,
        "weapon_gun_roi_pad_px": 48,
        "weapon_gun_only": 0,
        "weapon_gun_full_frame": 0,
        "weapon_gun_class_emit_min": "",
        "weapon_gun_emit_min_conf": 0.02,
        "weapon_gun_label_object_min": 0.15,
        "weapon_gun_label_weapon_min": 0.55,
        "weapon_assoc_motion": 1,
        "weapon_assoc_velocity_weight": 0.35,
        "weapon_assoc_min_speed": 1.5,
        "weapon_extra_args": "",
        "verbose": False,
    },
    "multi_camera": {
        "frames": 0,
        "fps": 30.0,
        "video": "",
        "live_frame": "layer8_ui/artifacts/live_multi_camera.jpg",
        "output": "",
        "webcam_device": 0,
        "webcam_auto_detect": 1,
        "webcam_detect_max_index": 8,
        "webcam_detect_retry_s": 8.0,
        "webcam_width": 3840,
        "webcam_height": 2160,
        "source_mode": "local",
        "jetson_ip": "",
        "jetson_stream_port": "8554",
        "jetson_stream_path": "/stream",
        "jetson_stream_scheme": "rtsp",
        "jetson_stream_url": "",
        "metrics_json": "layer8_ui/artifacts/live_multi_camera_threat_metrics.json",
        "active_model_profile_id": "",
        "weapon_checkpoint": "trained_models/gun_detection/gun_sohas_6class.pt",
        "person_detection_model": "yolov8n.pt",
        "weapon_unsafe_threshold": 0.92,
        "weapon_gun_threshold": 0.0,
        "weapon_yolo_model": "yolov8n.pt",
        "weapon_conf": 0.25,
        "weapon_person_conf": 0.40,
        "weapon_image_size": 224,
        "weapon_gun_conf": 0.25,
        "weapon_gun_imgsz": 640,
        "weapon_gun_batch": 1,
        "weapon_yolo_imgsz": 640,
        "weapon_min_box_px": 24,
        "weapon_gun_min_box_px": 8,
        "weapon_gun_max_vs_person_height": 0.45,
        "weapon_gun_max_vs_person_area": 0.12,
        "weapon_gun_max_vs_person_width": 0.70,
        "weapon_gun_roi_upper_frac": 0.80,
        "weapon_gun_thermal": 0,
        "weapon_byte_track": 1,
        "weapon_byte_track_firearms": 0,
        "weapon_byte_track_buffer": 45,
        "weapon_byte_track_fps": 30,
        "weapon_byte_track_firearm_ghost_frames": 18,
        "weapon_byte_track_firearm_draw_ghost": 1,
        "weapon_live_infer_stride": 1,
        "weapon_live_infer_max_width": 1920,
        "weapon_live_ipc_max_width": 1920,
        "weapon_live_gpu_preprocess": 1,
        "weapon_live_pipeline_publish": 1,
        "weapon_live_publish_workers": 1,
        "weapon_live_metrics_every_n": 30,
        "webrtc_smooth_display": 0,
        "webrtc_fps": 24,
        "webrtc_max_width": 1280,
        "webrtc_ipc_poll_fps": 30,
        "weapon_no_gun_yolo": 0,
        "weapon_show_yolo_name": 1,
        "weapon_overlay_classes": "all",
        "weapon_phone_overlay_label": "object",
        "weapon_pose_enable": 1,
        "weapon_pose_draw": 1,
        "weapon_pose_hand_boost": 1,
        "weapon_pose_hand_boost_max": 0.20,
        "weapon_pose_hand_radius_px": 100,
        "weapon_pose_min_kpt_conf": 0.35,
        "weapon_pose_model": "yolov8s-pose.pt",
        "weapon_hand_fingers": 1,
        "weapon_hand_draw": 1,
        "weapon_hand_finger_boost": 1,
        "weapon_hand_boost_grip": 0.12,
        "weapon_hand_boost_trigger": 0.18,
        "weapon_hand_boost_shoot": 0.25,
        "weapon_hand_model": "",
        "weapon_gun_yolo_model": "gun_sohas_6class.pt",
        "weapon_yolo_classes": "0",
        "weapon_yolo_device": "cuda",
        "weapon_classifier_device": "cuda",
        "weapon_gun_take_best": 1,
        "weapon_gun_take_best_infer_conf": 0.10,
        "weapon_gun_roi_pad_frac": 0.12,
        "weapon_gun_roi_pad_px": 36,
        "weapon_gun_only": 0,
        "weapon_gun_full_frame": 0,
        "weapon_gun_class_emit_min": "",
        "weapon_assoc_motion": 1,
        "weapon_assoc_velocity_weight": 0.35,
        "weapon_assoc_min_speed": 1.5,
        "weapon_extra_args": "",
        "verbose": False,
    },
    "sentinel": {
        "enable": 1,
        "baseline_m": 4.0,
        "person_height_m": 1.70,
        "vertical_fov_deg": 60.0,
        "note": "Beta: monocular PersonN (Xm) on overlay when enable=1.",
    },
    "global_id": {
        "enable": 0,
        "embed_interval_s": 0.5,
        "embed_min_box_px": 40,
        "embed_device": "auto",
        "embed_backend": "auto",
        "embed_model": "osnet_x0_25",
        "embed_weights": "trained_models/reid/osnet_x0_25_msmt17.pth",
        "embed_onnx": "trained_models/reid/osnet_x0_25_msmt17.onnx",
        "embed_engine": "trained_models/reid/osnet_x0_25_msmt17.engine",
        "embed_input_h": 256,
        "embed_input_w": 128,
        "similarity_threshold": 0.72,
        "soft_similarity_threshold": 0.58,
        "max_embedding_history": 12,
        "track_timeout_s": 8.0,
        "weight_reid": 0.70,
        "weight_depth": 0.20,
        "weight_temporal": 0.10,
        "baseline_m": 5.0,
        "depth_tolerance_frac": 0.30,
        "weapon_hold_s": 2.5,
        "weapon_conf_decay": 0.15,
        "camera_front": "camera_1",
        "camera_back": "camera_2",
        "state_json": "layer8_ui/configs/global_person_ids.json",
        "note": (
            "Cross-camera Person Re-ID + Global ID. Keep Front/Back infer independent; "
            "association runs in the API Global ID service. Set enable=1 to activate."
        ),
    },
    "mmwave": {
        "frames": 0,
        "mmwave_only": 1,
        "pipeline": "lab_replay",
        "session": "",
        "front_session": "",
        "back_session": "",
        "perception": "",
        "frames_jsonl": "",
        "plot_fps": 2.0,
        "config": "lab/mmwave77_usb/configs/awr1843boost_sdk_3_4_profile_3d.cfg",
        "cli_port": "",
        "data_port": "",
        "front_cli_port": "",
        "front_data_port": "",
        "back_cli_port": "",
        "back_data_port": "",
        "mmwave_show_plot": 1,
        "video": "",
        "live_frame": "layer8_ui/artifacts/live_mmwave.jpg",
        "live_frame_back": "layer8_ui/artifacts/live_mmwave_back.jpg",
        "output": "layer8_ui/artifacts/mmwave_frames.json",
        "no_frame_timeout_s": 30.0,
        "verbose": False,
        "extra_args": "",
    },
}


def _migrate_weapon_extra_args(block: dict[str, Any]) -> dict[str, Any]:
    """Move known flags from ``weapon_extra_args`` into structured JSON keys."""
    out = dict(block)
    extra = str(out.get("weapon_extra_args") or "").strip()
    if not extra:
        return out
    known_val = {
        "--yolo_classes": "weapon_yolo_classes",
        "--gun_take_best_infer_conf": "weapon_gun_take_best_infer_conf",
        "--gun_roi_pad_frac": "weapon_gun_roi_pad_frac",
        "--gun_roi_pad_px": "weapon_gun_roi_pad_px",
        "--yolo_device": "weapon_yolo_device",
        "--classifier_device": "weapon_classifier_device",
        "--gun_roi_upper_frac": "weapon_gun_roi_upper_frac",
    }
    known_flags = {
        "--gun_take_best": "weapon_gun_take_best",
        "--gun_full_frame": "weapon_gun_full_frame",
        "--gun_only": "weapon_gun_only",
        "--gun_batch": "weapon_gun_batch",
    }
    try:
        tokens = shlex.split(extra)
    except ValueError:
        tokens = re.findall(r"[^\s\"']+|\"[^\"]*\"|'[^']*'", extra)
    remaining: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in known_val:
            key = known_val[t]
            if i + 1 < len(tokens) and not out.get(key):
                out[key] = tokens[i + 1].strip("\"'")
                i += 2
                continue
        if t in known_flags and not int(out.get(known_flags[t], 0) or 0):
            out[known_flags[t]] = 1
            i += 1
            continue
        remaining.append(t)
        i += 1
    out["weapon_extra_args"] = " ".join(remaining).strip()
    return out


def settings_path(layer8_dir: Path) -> Path:
    return layer8_dir / "ui_settings.json"


def _sanitize_thermal_block(thermal: dict[str, Any], webcam: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fix common misconfig: thermal pointed at AI webcam with HD resolution."""
    t = dict(thermal)
    w = int(t.get("thermal_width", 160) or 160)
    h = int(t.get("thermal_height", 120) or 120)
    if w > 512 or h > 512:
        t["thermal_width"] = 160
        t["thermal_height"] = 120
        t["thermal_fps"] = 9
    td = int(t.get("thermal_device", 2))
    wd = int((webcam or {}).get("webcam_device", -1))
    if wd >= 0 and td == wd:
        t["thermal_device"] = 2
    return t


def _sanitize_sensor_ipc_settings(block: dict[str, Any]) -> dict[str, Any]:
    """Live IPC mmap writers are not safe with multiple concurrent publish threads."""
    out = dict(block)
    try:
        workers = int(out.get("weapon_live_publish_workers") or 1)
    except (TypeError, ValueError):
        workers = 1
    if workers > 1:
        out["weapon_live_publish_workers"] = 1
    return out


def load(layer8_dir: Path) -> dict[str, Any]:
    path = settings_path(layer8_dir)
    if not path.is_file():
        data = deepcopy(DEFAULT_SETTINGS)
        save(layer8_dir, data)
        return data
    with open(path) as f:
        merged = deepcopy(DEFAULT_SETTINGS)
        user = json.load(f)
        if isinstance(user, dict):
            merged.update({k: v for k, v in user.items() if k in ("software_root",)})
            if (
                "webcam" not in user
                and "infineon" in user
                and isinstance(user.get("infineon"), dict)
            ):
                merged["webcam"] = {**merged["webcam"], **user["infineon"]}
            for key in ("thermal", "webcam", "multi_camera", "mmwave", "sentinel", "global_id"):
                if key in user and isinstance(user[key], dict):
                    merged[key] = {**merged[key], **user[key]}
            merged["thermal"] = _sanitize_thermal_block(
                merged.get("thermal") or {},
                merged.get("webcam") or {},
            )
            for key in ("webcam", "multi_camera", "thermal"):
                if key in merged and isinstance(merged[key], dict):
                    merged[key] = _sanitize_sensor_ipc_settings(
                        _migrate_weapon_extra_args(merged[key])
                    )
        return merged


def reset_multi_camera_weapon_defaults(layer8_dir: Path) -> dict[str, Any]:
    """Reset only ``weapon_*`` and ``verbose`` under ``multi_camera``; camera paths unchanged."""
    data = load(layer8_dir)
    defs = deepcopy(DEFAULT_SETTINGS["multi_camera"])
    w = {**(data.get("multi_camera") or {})}
    for k, v in defs.items():
        if k.startswith("weapon_") or k in ("verbose", "person_detection_model"):
            w[k] = v
    data["multi_camera"] = w
    save(layer8_dir, data)
    return load(layer8_dir)


def reset_webcam_weapon_defaults(layer8_dir: Path) -> dict[str, Any]:
    """Reset only ``weapon_*`` and ``verbose`` under ``webcam`` (Model tab); paths/frames unchanged."""
    data = load(layer8_dir)
    defs = deepcopy(DEFAULT_SETTINGS["webcam"])
    w = {**(data.get("webcam") or {})}
    for k, v in defs.items():
        if k.startswith("weapon_") or k in ("verbose", "person_detection_model"):
            w[k] = v
    data["webcam"] = w
    save(layer8_dir, data)
    return load(layer8_dir)


def reset_thermal_weapon_defaults(layer8_dir: Path) -> dict[str, Any]:
    """Reset only ``weapon_*`` and ``verbose`` under ``thermal``; device/resolution unchanged."""
    data = load(layer8_dir)
    defs = deepcopy(DEFAULT_SETTINGS["thermal"])
    t = {**(data.get("thermal") or {})}
    for k, v in defs.items():
        if k.startswith("weapon_") or k in ("verbose", "person_detection_model"):
            t[k] = v
    data["thermal"] = t
    save(layer8_dir, data)
    return load(layer8_dir)


def save(layer8_dir: Path, data: dict[str, Any]) -> None:
    path = settings_path(layer8_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)
