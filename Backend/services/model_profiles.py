"""Model profile persistence and apply helpers.

This is extracted from the legacy Layer 8 router. It intentionally keeps the
same on-disk profile format during the first migration slice.
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

MODEL_PROFILE_WEBCAM_KEYS: tuple[str, ...] = (
    "webcam_device",
    "webcam_auto_detect",
    "webcam_pipeline",
    "webcam_detect_max_index",
    "webcam_detect_retry_s",
    "webcam_width",
    "webcam_height",
    "fps",
    "webrtc_smooth_display",
    "webrtc_fps",
    "webrtc_max_width",
    "webrtc_ipc_poll_fps",
    "metrics_json",
    "person_detection_model",
    "weapon_yolo_model",
    "weapon_conf",
    "weapon_min_box_px",
    "weapon_show_yolo_name",
    "weapon_unsafe_threshold",
    "weapon_gun_threshold",
    "weapon_image_size",
    "weapon_gun_conf",
    "weapon_gun_imgsz",
    "weapon_gun_min_box_px",
    "weapon_gun_max_vs_person_height",
    "weapon_gun_max_vs_person_area",
    "weapon_gun_max_vs_person_width",
    "weapon_gun_emit_min_conf",
    "weapon_gun_class_emit_min",
    "weapon_gun_label_object_min",
    "weapon_gun_label_weapon_min",
    "weapon_gun_thermal",
    "output",
    "weapon_byte_track",
    "weapon_byte_track_firearms",
    "weapon_byte_track_buffer",
    "weapon_byte_track_fps",
    "weapon_byte_track_firearm_ghost_frames",
    "weapon_byte_track_firearm_draw_ghost",
    "weapon_live_infer_stride",
    "weapon_live_infer_max_width",
    "weapon_live_ipc_max_width",
    "weapon_live_gpu_preprocess",
    "weapon_live_pipeline_publish",
    "weapon_live_publish_workers",
    "weapon_live_metrics_every_n",
    "weapon_no_gun_yolo",
    "weapon_gun_yolo_model",
    "weapon_yolo_classes",
    "weapon_yolo_device",
    "weapon_classifier_device",
    "weapon_gun_take_best",
    "weapon_gun_take_best_infer_conf",
    "weapon_gun_roi_pad_frac",
    "weapon_gun_roi_pad_px",
    "weapon_gun_only",
    "weapon_gun_full_frame",
    "weapon_gun_roi_upper_frac",
    "weapon_gun_phone_conflict_suppress",
    "weapon_gun_phone_conflict_iou",
    "weapon_gun_phone_conflict_margin",
    "weapon_overlay_classes",
    "weapon_phone_overlay_label",
    "weapon_pose_enable",
    "weapon_pose_draw",
    "weapon_pose_hand_boost",
    "weapon_pose_hand_boost_max",
    "weapon_pose_hand_radius_px",
    "weapon_pose_min_kpt_conf",
    "weapon_pose_imgsz",
    "weapon_pose_max_persons",
    "weapon_pose_model",
    "weapon_hand_fingers",
    "weapon_hand_draw",
    "weapon_hand_finger_boost",
    "weapon_hand_max_persons",
    "weapon_hand_boost_grip",
    "weapon_hand_boost_trigger",
    "weapon_hand_boost_shoot",
    "weapon_hand_model",
    "weapon_assoc_motion",
    "weapon_assoc_velocity_weight",
    "weapon_assoc_min_speed",
    "weapon_extra_args",
)

PROFILE_FILE_META_KEYS = frozenset({"version", "schema", "__schema__", "_meta"})


def model_profiles_path(layer8_dir: Path) -> Path:
    return Path(layer8_dir) / "profiles" / "model_profiles.json"


def load_model_profiles_raw(layer8_dir: Path) -> dict[str, Any]:
    path = model_profiles_path(layer8_dir)
    if not path.is_file():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def coerce_profile_entry(profile_id: str, value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    pid = str(profile_id).strip()
    if not pid or pid in PROFILE_FILE_META_KEYS:
        return None
    if "values" in value and isinstance(value.get("values"), dict):
        return {
            "label": str(value.get("label") or pid),
            "description": str(value.get("description") or ""),
            "values": dict(value["values"]),
        }
    return {"label": pid, "description": "", "values": dict(value)}


def normalize_profiles_document(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return `profile_id -> {label, description, values}` for all supported formats."""
    if isinstance(raw.get("profiles"), list):
        out: dict[str, dict[str, Any]] = {}
        for row in raw["profiles"]:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("id") or "").strip()
            if not pid:
                continue
            values = row.get("values") if isinstance(row.get("values"), dict) else row.get("webcam")
            out[pid] = {
                "label": str(row.get("label") or row.get("name") or pid),
                "description": str(row.get("description") or ""),
                "values": dict(values) if isinstance(values, dict) else {},
            }
        return out

    if isinstance(raw.get("profiles"), dict):
        rows = raw["profiles"].items()
    else:
        rows = ((k, v) for k, v in raw.items() if k not in PROFILE_FILE_META_KEYS)

    out = {}
    for pid, value in rows:
        entry = coerce_profile_entry(str(pid), value)
        if entry:
            out[str(pid)] = entry
    return out


def serialize_profiles_to_disk(norm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        pid: {
            "label": entry.get("label", pid),
            "description": entry.get("description", ""),
            "values": dict(entry.get("values") or {}),
        }
        for pid, entry in norm.items()
    }


def save_model_profiles(layer8_dir: Path, data: dict[str, Any]) -> None:
    path = model_profiles_path(layer8_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            shutil.copy2(path, path.with_suffix(".json.bak"))
        except OSError:
            pass
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def get_model_profiles_normalized(layer8_dir: Path) -> dict[str, dict[str, Any]]:
    return normalize_profiles_document(load_model_profiles_raw(layer8_dir))


def ai_camera_profiles_public_list(layer8_dir: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "id": pid,
            "name": str(entry.get("label") or pid),
            "description": str(entry.get("description") or ""),
            "values": dict(entry.get("values") or {}),
        }
        for pid, entry in get_model_profiles_normalized(layer8_dir).items()
    ]
    rows.sort(key=lambda r: (r["name"].lower(), r["id"]))
    return rows


def extract_profile_values(webcam: dict[str, Any]) -> dict[str, Any]:
    return {key: webcam[key] for key in MODEL_PROFILE_WEBCAM_KEYS if key in webcam}


def apply_values_to_webcam(webcam: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    merged = {**webcam, **values}
    person_model = merged.get("person_detection_model")
    if person_model is not None and str(person_model).strip():
        merged["weapon_yolo_model"] = str(person_model).strip()
    gun_model = str(merged.get("weapon_gun_yolo_model") or "").strip()
    if gun_model:
        merged["weapon_checkpoint"] = (
            gun_model if "/" in gun_model else f"trained_models/gun_detection/{gun_model}"
        )
    return merged


def apply_values_to_multi_camera(multi_camera: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Same keys as webcam profiles (device, resolution, weapon_*)."""
    return apply_values_to_webcam(multi_camera, values)


def apply_weapon_profile_to_thermal(thermal: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    merged = {**thermal}
    for key, value in values.items():
        if key.startswith("weapon_") or key in ("person_detection_model", "output"):
            merged[key] = value
    person_model = merged.get("person_detection_model")
    if person_model is not None and str(person_model).strip():
        merged["weapon_yolo_model"] = str(person_model).strip()
    return merged


def profile_ids_matching_name(norm: dict[str, dict[str, Any]], name: str) -> list[str]:
    want = name.strip().casefold()
    if not want:
        return []
    matches = [
        pid
        for pid, entry in norm.items()
        if str(entry.get("label") or pid).strip().casefold() == want
    ]
    matches.sort()
    return matches


def profile_exists_by_name(layer8_dir: Path, name: str) -> tuple[bool, str | None]:
    matches = [
        pid
        for pid, entry in get_model_profiles_normalized(layer8_dir).items()
        if str(entry.get("label") or pid) == name.strip()
    ]
    return (bool(matches), matches[0] if matches else None)


def unique_profile_key_from_name(name: str, existing: set[str]) -> str:
    slug = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", slug.lower().strip())
    slug = re.sub(r"_+", "_", slug).strip("_") or "profile"
    base = slug[:80]
    if base not in existing:
        return base
    suffix = 2
    while f"{base}_{suffix}" in existing:
        suffix += 1
    return f"{base}_{suffix}"

