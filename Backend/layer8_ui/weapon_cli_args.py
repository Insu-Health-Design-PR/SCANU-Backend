"""Build structured weapon-inference CLI flags from dashboard JSON settings."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any


def resolve_gun_model_path(sw: Path, raw: str) -> str:
    val = str(raw or "").strip()
    if not val:
        return ""
    p = Path(val).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    if "/" not in val and "\\" not in val:
        return str((sw / "trained_models" / "gun_detection" / val).resolve())
    return str((sw / val).resolve())


def resolve_person_model_path(sw: Path, raw: str) -> str:
    """Resolve person YOLO ``.pt`` / ``.engine`` under trained_models/person_detection when bare name."""
    val = str(raw or "").strip()
    if not val:
        return ""
    p = Path(val).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    if "/" not in val and "\\" not in val:
        local = (sw / "trained_models" / "person_detection" / val).resolve()
        if local.is_file():
            return str(local)
        # Ultralytics hub / CWD weights (e.g. yolov8n.pt)
        return val
    return str((sw / val).resolve())


def resolve_pose_model_path(sw: Path, raw: str) -> str:
    """Resolve pose YOLO weights under trained_models/pose when bare name."""
    val = str(raw or "").strip()
    if not val:
        return ""
    p = Path(val).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    if "/" not in val and "\\" not in val:
        local = (sw / "trained_models" / "pose" / val).resolve()
        if local.is_file():
            return str(local)
        # Ultralytics hub / CWD weights (e.g. yolov8n-pose.pt)
        return val
    return str((sw / val).resolve())


def build_structured_weapon_args(
    w: dict[str, Any],
    sw: Path,
    *,
    include_overlay_classes: bool = False,
    sentinel: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []
    no_gun = int(w.get("weapon_no_gun_yolo", 0))

    def _f(flag: str, key: str, caster: type) -> None:
        raw = w.get(key)
        if raw is None or str(raw).strip() == "":
            return
        try:
            parts.extend([flag, str(caster(raw))])
        except (TypeError, ValueError):
            return

    def _device(flag: str, key: str) -> None:
        dev = str(w.get(key) or "").strip().lower()
        if dev and dev not in ("auto", ""):
            parts.extend([flag, dev])

    _f("--unsafe_threshold", "weapon_unsafe_threshold", float)
    if not no_gun:
        gt_raw = w.get("weapon_gun_threshold")
        if gt_raw is not None and str(gt_raw).strip() != "":
            try:
                if float(gt_raw) > 0:
                    parts.extend(["--gun_threshold", str(float(gt_raw))])
            except (TypeError, ValueError):
                pass
    ym = str(w.get("person_detection_model") or w.get("weapon_yolo_model") or "").strip()
    if ym:
        parts.extend(["--yolo_model", resolve_person_model_path(sw, ym)])
    yc = str(w.get("weapon_yolo_classes") or "").strip()
    if yc:
        parts.extend(["--yolo_classes", yc])
    _device("--yolo_device", "weapon_yolo_device")
    _device("--classifier_device", "weapon_classifier_device")
    _f("--conf", "weapon_conf", float)
    _f("--person_conf", "weapon_person_conf", float)
    _f("--image_size", "weapon_image_size", int)
    if not no_gun:
        _f("--gun_conf", "weapon_gun_conf", float)
        _f("--gun_imgsz", "weapon_gun_imgsz", int)
        _f("--yolo_imgsz", "weapon_yolo_imgsz", int)
        if w.get("weapon_gun_batch") is not None and str(w.get("weapon_gun_batch")).strip() != "":
            if int(w.get("weapon_gun_batch", 1)):
                parts.append("--gun_batch")
            else:
                parts.append("--no-gun_batch")
        _f("--gun_label_object_min", "weapon_gun_label_object_min", float)
        _f("--gun_label_weapon_min", "weapon_gun_label_weapon_min", float)
        _f("--gun_emit_min_conf", "weapon_gun_emit_min_conf", float)
        cem = str(w.get("weapon_gun_class_emit_min") or "").strip()
        if cem:
            parts.extend(["--gun_class_emit_min", cem])
        if int(w.get("weapon_gun_take_best", 0)):
            parts.append("--gun_take_best")
            _f("--gun_take_best_infer_conf", "weapon_gun_take_best_infer_conf", float)
        _f("--gun_roi_pad_frac", "weapon_gun_roi_pad_frac", float)
        _f("--gun_roi_pad_px", "weapon_gun_roi_pad_px", int)
    _f("--min_box_px", "weapon_min_box_px", int)
    if not no_gun:
        _f("--gun_min_box_px", "weapon_gun_min_box_px", int)
        _f("--gun_max_vs_person_height", "weapon_gun_max_vs_person_height", float)
        _f("--gun_max_vs_person_area", "weapon_gun_max_vs_person_area", float)
        _f("--gun_max_vs_person_width", "weapon_gun_max_vs_person_width", float)
        if int(w.get("weapon_gun_thermal", 0)):
            parts.append("--gun_thermal")
        if int(w.get("weapon_gun_only", 0)):
            parts.append("--gun_only")
        if int(w.get("weapon_gun_full_frame", 0)):
            parts.append("--gun_full_frame")
        _f("--gun_roi_upper_frac", "weapon_gun_roi_upper_frac", float)
        if w.get("weapon_gun_phone_conflict_suppress") is not None and str(
            w.get("weapon_gun_phone_conflict_suppress")
        ).strip() != "":
            if int(w.get("weapon_gun_phone_conflict_suppress", 1)):
                parts.append("--gun_phone_conflict_suppress")
            else:
                parts.append("--no-gun_phone_conflict_suppress")
        _f("--gun_phone_conflict_iou", "weapon_gun_phone_conflict_iou", float)
        _f("--gun_phone_conflict_margin", "weapon_gun_phone_conflict_margin", float)
        am = w.get("weapon_assoc_motion")
        if am is not None and str(am).strip() != "" and not int(am):
            parts.append("--no-assoc_motion")
        _f("--assoc_velocity_weight", "weapon_assoc_velocity_weight", float)
        _f("--assoc_min_speed", "weapon_assoc_min_speed", float)
    bt_raw = w.get("weapon_byte_track")
    if bt_raw is not None and str(bt_raw).strip() != "" and not int(bt_raw):
        parts.append("--no-byte_track")
    if int(w.get("weapon_byte_track_firearms", 0)):
        parts.append("--byte_track_firearms")
    _f("--byte_track_buffer", "weapon_byte_track_buffer", int)
    _f("--byte_track_fps", "weapon_byte_track_fps", float)
    _f("--live_infer_stride", "weapon_live_infer_stride", int)
    _f("--live_infer_max_width", "weapon_live_infer_max_width", int)
    _f("--live_ipc_max_width", "weapon_live_ipc_max_width", int)
    _f("--live_metrics_every_n", "weapon_live_metrics_every_n", int)
    _f("--live_publish_workers", "weapon_live_publish_workers", int)
    if int(w.get("weapon_live_gpu_preprocess", 1)):
        parts.append("--live_gpu_preprocess")
    else:
        parts.append("--no-live_gpu_preprocess")
    if int(w.get("weapon_live_pipeline_publish", 1)):
        parts.append("--live_pipeline_publish")
    else:
        parts.append("--no-live_pipeline_publish")
    _f("--byte_track_firearm_ghost_frames", "weapon_byte_track_firearm_ghost_frames", int)
    if w.get("weapon_byte_track_firearm_draw_ghost") is not None and str(
        w.get("weapon_byte_track_firearm_draw_ghost")
    ).strip() != "":
        if not int(w.get("weapon_byte_track_firearm_draw_ghost", 1)):
            parts.append("--no-byte_track_firearm_draw_ghost")
    parts.append("--no-analysis_report")
    if no_gun:
        parts.append("--no_gun_yolo")
    if int(w.get("weapon_show_yolo_name", 0)):
        parts.append("--show_yolo_name")
    if include_overlay_classes:
        oc = str(w.get("weapon_overlay_classes") or "").strip()
        if oc and oc.lower() not in ("all", "*", "any"):
            parts.extend(["--gun_overlay_classes", oc])
        phone_lab = str(w.get("weapon_phone_overlay_label") or "object").strip().lower()
        if phone_lab not in ("object", "smartphone"):
            phone_lab = "object"
        parts.extend(["--phone_overlay_label", phone_lab])
        if w.get("weapon_pose_enable") is not None and str(w.get("weapon_pose_enable")).strip() != "":
            if not int(w.get("weapon_pose_enable", 1)):
                parts.append("--no-pose")
        if int(w.get("weapon_pose_draw", 1)):
            parts.append("--pose_draw")
        else:
            parts.append("--no-pose_draw")
        if int(w.get("weapon_pose_hand_boost", 1)):
            parts.append("--pose_hand_boost")
        else:
            parts.append("--no-pose_hand_boost")
        _f("--pose_hand_boost_max", "weapon_pose_hand_boost_max", float)
        _f("--pose_hand_radius_px", "weapon_pose_hand_radius_px", float)
        _f("--pose_min_kpt_conf", "weapon_pose_min_kpt_conf", float)
        _f("--pose_imgsz", "weapon_pose_imgsz", int)
        _f("--pose_max_persons", "weapon_pose_max_persons", int)
        _f("--hand_max_persons", "weapon_hand_max_persons", int)
        pm = str(w.get("weapon_pose_model") or "").strip()
        if pm:
            parts.extend(["--pose_model", resolve_pose_model_path(sw, pm)])
        if w.get("weapon_hand_fingers") is not None and str(w.get("weapon_hand_fingers")).strip() != "":
            if int(w.get("weapon_hand_fingers", 1)):
                parts.append("--hand_fingers")
            else:
                parts.append("--no-hand_fingers")
        else:
            parts.append("--hand_fingers")
        if int(w.get("weapon_hand_draw", 1)):
            parts.append("--hand_draw")
        else:
            parts.append("--no-hand_draw")
        if int(w.get("weapon_hand_finger_boost", 1)):
            parts.append("--hand_finger_boost")
        else:
            parts.append("--no-hand_finger_boost")
        _f("--hand_boost_grip", "weapon_hand_boost_grip", float)
        _f("--hand_boost_trigger", "weapon_hand_boost_trigger", float)
        _f("--hand_boost_shoot", "weapon_hand_boost_shoot", float)
        hm = str(w.get("weapon_hand_model") or "").strip()
        if hm:
            parts.extend(["--hand_model", hm])
        else:
            from weapon_ai.pose.hand_fingers import default_hand_model_path

            parts.extend(["--hand_model", str(default_hand_model_path())])
    # Beta monocular person distance (Sentinel settings → --person_distance).
    sent = sentinel if isinstance(sentinel, dict) else {}
    dist_enable = sent.get("enable", w.get("weapon_person_distance", 0))
    try:
        dist_on = bool(int(dist_enable)) if str(dist_enable).strip() != "" else False
    except (TypeError, ValueError):
        dist_on = str(dist_enable).strip().lower() in {"1", "true", "yes", "on"}
    if dist_on:
        parts.append("--person_distance")
        ph = sent.get("person_height_m", w.get("weapon_person_height_m"))
        if ph is not None and str(ph).strip() != "":
            try:
                parts.extend(["--person_height_m", str(float(ph))])
            except (TypeError, ValueError):
                pass
        vf = sent.get("vertical_fov_deg", w.get("weapon_person_vertical_fov_deg"))
        if vf is not None and str(vf).strip() != "":
            try:
                parts.extend(["--person_vertical_fov_deg", str(float(vf))])
            except (TypeError, ValueError):
                pass

    # Cross-camera Global ID / Re-ID (per-sensor camera_id + optional state JSON overlay).
    cam_id = str(w.get("camera_id") or w.get("weapon_camera_id") or "").strip()
    if cam_id:
        parts.extend(["--camera_id", cam_id])
    state_json = str(w.get("global_id_state_json") or "").strip()
    overlay_flag = w.get("weapon_global_id_overlay", sent.get("global_id_overlay", 0))
    try:
        overlay_on = bool(int(overlay_flag)) if str(overlay_flag).strip() != "" else False
    except (TypeError, ValueError):
        overlay_on = str(overlay_flag).strip().lower() in {"1", "true", "yes", "on"}
    if state_json:
        sp = Path(state_json).expanduser()
        abs_state = str(sp.resolve()) if sp.is_absolute() else str((sw / state_json).resolve())
        parts.extend(["--global_id_state_json", abs_state])
    elif overlay_on:
        default_state = sw / "layer8_ui" / "configs" / "global_person_ids.json"
        parts.extend(["--global_id_state_json", str(default_state.resolve())])

    if not no_gun:
        gpath = str(w.get("weapon_gun_yolo_model") or "").strip()
        if gpath:
            abs_g = resolve_gun_model_path(sw, gpath)
            parts.extend(["--gun_yolo_model", abs_g])

    built = shlex.join(parts) if parts else ""
    manual = (w.get("weapon_extra_args") or "").strip()
    if manual:
        return f"{built} {manual}".strip() if built else manual
    return built
