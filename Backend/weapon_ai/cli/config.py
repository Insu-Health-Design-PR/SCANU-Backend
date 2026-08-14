"""CLI config-file helpers for weapon inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keys accepted inside --config (flat dict); unknown keys are ignored with a warning.
INFER_CONFIG_KEYS = frozenset(
    {
        "checkpoint",
        "source",
        "image_size",
        "yolo_model",
        "conf",
        "person_conf",
        "min_box_px",
        "yolo_classes",
        "unsafe_threshold",
        "safe_max",
        "unsafe_border_thick",
        "unsafe_border_color",
        "composite_mode",
        "thermal_panel",
        "show_yolo_name",
        "show_fps",
        "output",
        "no_imshow",
        "max_frames",
        "gun_yolo_model",
        "no_gun_yolo",
        "gun_conf",
        "gun_max_area_frac",
        "gun_max_side_frac",
        "gun_full_frame",
        "assoc_motion",
        "assoc_velocity_weight",
        "assoc_min_speed",
        "gun_roi_pad_frac",
        "gun_roi_pad_px",
        "gun_roi_upper_frac",
        "gun_hand_zone",
        "gun_hand_zone_min_overlap",
        "pose",
        "pose_model",
        "pose_draw",
        "pose_hand_boost",
        "pose_hand_boost_max",
        "pose_hand_radius_px",
        "pose_min_kpt_conf",
        "hand_fingers",
        "hand_model",
        "hand_draw",
        "hand_finger_boost",
        "hand_boost_grip",
        "hand_boost_trigger",
        "hand_boost_shoot",
        "gun_imgsz",
        "gun_batch",
        "yolo_imgsz",
        "gun_thermal_debug",
        "gun_thermal",
        "gun_min_box_px",
        "gun_take_best",
        "gun_take_best_per_person",
        "gun_take_best_infer_conf",
        "gun_emit_min_conf",
        "gun_class_emit_min",
        "gun_label_weapon_min",
        "gun_label_object_min",
        "gun_threshold",
        "fuse_gun_to_prob",
        "gun_prob_floor",
        "gun_conf_scale",
        "live_jpg",
        "live_ipc_frame",
        "live_ipc_bgr_frame",
        "live_metrics_json",
        "capture_width",
        "capture_height",
        "capture_fps",
        "thermal_v4l2",
        "thin_overlay",
        "panel_w",
        "panel_h",
        "live_frame_poll",
        "classifier_device",
        "yolo_device",
        "cuda_empty_cache",
        "cuda_empty_cache_every",
        "batch_warmup_passes",
        "batch_async_infer",
        "live_infer_stride",
        "live_infer_max_width",
        "live_ipc_max_width",
        "live_gpu_preprocess",
        "live_pipeline_publish",
        "live_publish_workers",
        "live_metrics_every_n",
        "live_threaded_capture",
        "gstreamer_capture",
        "output_fps",
        "byte_track",
        "byte_track_firearms",
        "byte_track_buffer",
    "byte_track_fps",
    "byte_track_firearm_ghost_frames",
    "byte_track_firearm_draw_ghost",
        "person_distance",
        "person_height_m",
        "person_vertical_fov_deg",
        "camera_id",
        "global_id_state_json",
        "yolo_non_person_inside_person",
        "yolo_nonperson_min_conf",
        "yolo_nonperson_min_short_side_px",
        "yolo_nonperson_min_area_px",
        "yolo_nonperson_max_aspect",
        "analysis_report",
        "analysis_report_path",
        "analysis_report_flush_every",
    }
)


def load_infer_config(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(raw)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SystemExit("YAML config requires PyYAML. Install with: pip install PyYAML") from exc
        data = yaml.safe_load(raw)
    else:
        raise SystemExit(f"Unsupported config format (use .json, .yaml, .yml): {path}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config root must be a JSON/YAML object, got {type(data).__name__}")
    return data


def coerce_infer_config_values(cfg: dict) -> dict:
    """Normalize types from JSON/YAML for argparse/set_defaults."""
    out = dict(cfg)
    for key in (
        "checkpoint",
        "output",
        "gun_yolo_model",
        "live_jpg",
        "live_metrics_json",
        "analysis_report_path",
    ):
        if key in out and out[key] is not None and out[key] != "":
            out[key] = Path(out[key])
    if "source" in out and out["source"] is not None and not isinstance(out["source"], str):
        out["source"] = str(out["source"])
    return out


def filter_infer_config(cfg: dict, path: Path) -> dict:
    extra = set(cfg) - INFER_CONFIG_KEYS
    if extra:
        print(f"Warning: ignoring unknown config keys in {path}: {sorted(extra)}", file=sys.stderr)
    return {key: value for key, value in cfg.items() if key in INFER_CONFIG_KEYS}


def pop_infer_config_file(argv: list[str]) -> tuple[Path | None, list[str]]:
    """Strip exact `--config` / `--config=` tokens from argv."""
    cfg: Path | None = None
    out: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            if i + 1 >= len(argv):
                raise SystemExit("--config requires a path to a .json or .yaml file")
            cfg = Path(argv[i + 1])
            i += 2
            continue
        if arg.startswith("--config="):
            cfg = Path(arg.split("=", 1)[1])
            i += 1
            continue
        out.append(arg)
        i += 1
    return cfg, out


def add_bool_optional_arg(
    parser: argparse.ArgumentParser,
    flag_name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    """Add --flag / --no-flag on both modern and legacy argparse versions."""
    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is not None:
        parser.add_argument(flag_name, action=bool_action, default=default, help=help_text)
        return

    dest = flag_name.lstrip("-").replace("-", "_")
    parser.add_argument(flag_name, dest=dest, action="store_true", default=default, help=help_text)
    parser.add_argument(
        f"--no-{dest.replace('_', '-')}",
        dest=dest,
        action="store_false",
        help=argparse.SUPPRESS,
    )

