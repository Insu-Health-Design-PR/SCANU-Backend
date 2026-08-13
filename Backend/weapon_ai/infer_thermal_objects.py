"""
Thermal stream: YOLO boxes (person + optional objects) -> per-crop safe/unsafe score -> border on unsafe.

By default only COCO class **0 (person)** is drawn; firearm boxes come from the second-stage gun YOLO.
Use ``--yolo_classes`` to add more (e.g. ``"0,39"`` for bottles, ``all`` for every COCO class).

Optionally draws firearm boxes from a second YOLO detector (default: Subh775/Firearm_Detection_Yolov8n
on Hugging Face, AGPL-3.0). By default firearm YOLO runs only **inside each person box** (crop),
then boxes are mapped back to the full frame—reduces chair/background false positives. Use
--gun_full_frame to scan the whole image like before. Trained on visible-light imagery; on thermal it
often scores below normal conf thresholds and may predict huge, soft boxes—see --gun_thermal_debug
or lower --gun_conf and relax --gun_max_*_frac. Use --gun_thermal for a balanced thermal preset
(--gun_take_best + ``--gun_min_box_px``). Fine-tuning on thermal gun crops is the durable fix.

Use a dedicated thermal extract, e.g. clip_thermal.mp4 from split_panels, or pass a composite
recording with --composite_mode and --thermal_panel.

For fresh-start training (`video_mode=gun_prob`), this script reads a BCE checkpoint and uses
sigmoid(logit) as p(gun). Person armed / unsafe use ``--unsafe_threshold``; firearm overlay
``object`` when score > 0.25, ``weapon`` when score > 0.70 (75-frame track hold).

Examples:
  python -m weapon_ai.infer_thermal_objects --config weapon_ai/infer_thermal.example.yaml
  # CLI flags override values from the JSON/YAML file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import warnings
from collections.abc import Mapping
from typing import Any
import urllib.request
from pathlib import Path
from time import sleep, time

import cv2
import numpy as np
import torch

from weapon_ai.analysis_report import ClipAnalysisReporter, default_analysis_base
from weapon_ai.detection.firearms import (
    clamp_box as _clamp_box,
    box_center as _box_center,
    dedupe_gun_candidates as _dedupe_gun_candidates,
    dedupe_person_rows as _dedupe_person_rows,
    expand_person_roi_for_gun as _expand_person_roi_for_gun,
    firearm_kind_for_detection as _firearm_kind_for_detection,
    gun_detection_valid as _gun_detection_valid,
    gun_yolo_binary_class_sets as _gun_yolo_binary_class_sets,
    nearest_person_ridx_for_gun as _nearest_person_ridx_for_gun,
    xyxy_center_inside_any_person as _xyxy_center_inside_any_person,
    yolo_keep_nonperson_detection as _yolo_keep_nonperson_detection,
)
from weapon_ai.cli.config import (
    add_bool_optional_arg,
    coerce_infer_config_values,
    filter_infer_config,
    load_infer_config,
    pop_infer_config_file,
)
from weapon_ai.tracking.bytetrack import (
    ByteTrackConfig,
    DisplayTrackIds,
    GunStableIdTracker,
    IndexedBoxByteTracker,
    PersonArmedLatch,
    ThermalByteTracker,
    WeaponPersonAssociator,
)
from weapon_ai.live_frame_ipc import LiveBgrFrameWriter, LiveFrameWriter, is_valid_jpeg
from weapon_ai.threat.metrics import write_live_metrics_json
from weapon_ai.threat.buckets import SAFE_MAX as _SAFE_MAX
from weapon_ai.threat.buckets import overlay_score as _overlay_score
from weapon_ai.threat.buckets import threat_bucket as _threat_bucket
from weapon_ai.overlay import (
    COLOR_GUN_GHOST_BGR as _COLOR_GUN_GHOST_BGR,
    COLOR_GUN_OBJECT_BGR as _COLOR_GUN_OBJECT_BGR,
    COLOR_GUN_WEAPON_BGR as _COLOR_GUN_WEAPON_BGR,
    COLOR_PERSON_ARMED_BGR as _COLOR_PERSON_ARMED_BGR,
    COLOR_PERSON_ARMED_CONCEALED_BGR as _COLOR_PERSON_ARMED_CONCEALED_BGR,
    FIREARM_GHOST_CONF as _FIREARM_GHOST_CONF,
    OVERLAY_SCALE_STATUS as _OVERLAY_SCALE_STATUS,
    OVERLAY_THICK_STATUS as _OVERLAY_THICK_STATUS,
    draw_label_above_box as _draw_label_above_box,
    gun_box_person_ridx as _gun_box_person_ridx,
    overlay_draw_style as _overlay_draw_style,
    person_key_for_row as _person_key_for_row,
    person_overlay_label as _person_overlay_label,
    person_public_name as _person_public_name,
    person_weapon_bracket as _person_weapon_bracket,
)
from weapon_ai.live_preprocess import (
    downscale_to_max_width,
    resize_bgr_max_width,
    resize_bgr_to,
    scale_gun_boxes,
    scale_person_rows,
)
from media.capture.gstreamer_webcam import (
    GStreamerWebcamCapture,
    gstreamer_available,
    nvidia_gst_jpeg_available,
)
from media.capture.live_webcam_capture import LiveWebcamCapture

_REPO_ROOT = Path(os.environ.get("SCANU_MODEL_ROOT", Path(__file__).resolve().parent.parent)).resolve()
_DEFAULT_FIREARM_YOLO = _REPO_ROOT / "trained_models" / "gun_detection" / "firearm_yolov8n_best.pt"
# When present at repo root, prefer this checkpoint over the HF default (Layer 4 / SCANU workflow).
_PREFERRED_FIREARM_YOLO = _REPO_ROOT / "trained_models" / "gun_detection" / "gun_enhanced_cctv_v2.pt"
_FIREARM_HF_URL = (
    "https://huggingface.co/Subh775/Firearm_Detection_Yolov8n/resolve/main/weights/best.pt"
)


def _default_firearm_yolo_path() -> Path:
    """Prefer ``gun_enhanced_cctv_v2.pt`` when installed; else HF-download default."""
    p = _PREFERRED_FIREARM_YOLO
    if p.is_file() and p.stat().st_size > 1_000_000:
        return p
    return _DEFAULT_FIREARM_YOLO


def _extract_thermal_column(bgr: np.ndarray, panel: str) -> np.ndarray:
    h, w = bgr.shape[:2]
    if panel == "left":
        return bgr[:, : w // 3]
    if panel == "center":
        return bgr[:, w // 3 : 2 * w // 3]
    if panel == "right":
        return bgr[:, 2 * w // 3 :]
    raise ValueError(panel)


def _print_run_summary(
    source: str,
    all_probs: list[float],
    frame_count: int,
    frame_max_probs: list[float],
    unsafe_min: float,
    score_name: str,
    *,
    safe_max: float = _SAFE_MAX,
) -> None:
    print()
    print("=" * 60)
    print(f"Run finished: {source}")
    print(f"Frames read: {frame_count}")
    if not all_probs:
        print(f"No person crops scored — cannot summarize {score_name}.")
        print("=" * 60)
        return
    arr = np.array(all_probs, dtype=np.float64)
    print(f"Person crops scored (total boxes): {len(all_probs)}")
    print(f"{score_name} over all crops — min: {arr.min():.4f}  mean: {arr.mean():.4f}  max: {arr.max():.4f}")
    if frame_max_probs:
        fm = np.array(frame_max_probs, dtype=np.float64)
        print(
            f"Per-frame max {score_name} — mean: {fm.mean():.4f}  peak (worst frame): {fm.max():.4f}"
        )
    peak = float(arr.max())
    verdict = _threat_bucket(peak, unsafe_min, safe_max).upper()
    print(
        f"FINAL (clip): {verdict}  |  peak {score_name}={peak:.4f}  "
        f"(safe: score<{unsafe_min:.2f}; unsafe: score≥{unsafe_min:.2f}; peak={_overlay_score(peak)})"
    )
    print("=" * 60)


def _parse_yolo_classes(s: str | None) -> list[int] | None:
    if not s or not s.strip():
        return None
    t = s.strip().lower()
    if t == "all":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _frame_to_bgr_for_infer(frame: np.ndarray | None) -> np.ndarray | None:
    """Normalize odd V4L2/OpenCV frames (1/2/4-channel) into 3-channel BGR."""
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
        # Typical webcam packed YUV 4:2:2 stream (YUYV/YUY2).
        try:
            return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_YUY2)
        except cv2.error:
            pass
        # Alternate packed layout used by some devices.
        try:
            return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_UYVY)
        except cv2.error:
            pass
        # Last resort keeps inference alive even if color decode is imperfect.
        a = np.ascontiguousarray(frame[:, :, 0])
        b = np.ascontiguousarray(frame[:, :, 1])
        return cv2.merge([a, b, a])
    return None


def _frame_valid_for_infer(frame: np.ndarray | None) -> bool:
    """Reject empty / 1D / degenerate buffers that break YOLO letterbox resize."""
    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return False
    if frame.ndim != 3 or int(frame.shape[2]) < 3:
        return False
    h, w = int(frame.shape[0]), int(frame.shape[1])
    return h >= 2 and w >= 2


def _is_batch_video_file_source(src: object) -> bool:
    """True when ``--source`` is a pre-recorded file (not a webcam index or ``/dev/video*``)."""
    if isinstance(src, int):
        return False
    raw = str(src).strip()
    if raw.startswith("/dev/video") or raw.isdigit():
        return False
    try:
        return Path(raw).expanduser().resolve().is_file()
    except OSError:
        return False


def _thermal_uint16_to_gray(frame: np.ndarray) -> np.ndarray | None:
    if frame.ndim != 2:
        return None
    if frame.dtype == np.uint16:
        f32 = frame.astype(np.float32)
        mn = float(f32.min())
        mx = float(f32.max())
        if mx - mn > 1e-6:
            return ((f32 - mn) / (mx - mn) * 255.0).astype(np.uint8)
        # Flat frame (lens cap / uniform scene): avoid sudden black preview.
        mid = float(np.median(f32))
        return np.clip(f32 * (128.0 / max(mid, 1.0)), 0, 255).astype(np.uint8)
    return frame if frame.dtype == np.uint8 else cv2.convertScaleAbs(frame)


def _thermal_capture_to_infer_and_vis(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Y16 / grayscale thermal frame → BGR for YOLO and inferno colormap for operator preview."""
    if raw is None or not hasattr(raw, "shape") or raw.size == 0:
        return None
    gray = _thermal_uint16_to_gray(raw)
    if gray is None or gray.ndim != 2 or gray.shape[0] < 2 or gray.shape[1] < 2:
        return None
    infer_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    if not _frame_valid_for_infer(infer_bgr):
        return None
    return infer_bgr, vis_bgr


def main() -> None:
    infer_config_path, argv_rest = pop_infer_config_file(sys.argv[1:])
    file_defaults: dict = {}
    if infer_config_path is not None:
        loaded = load_infer_config(infer_config_path)
        file_defaults = coerce_infer_config_values(filter_infer_config(loaded, infer_config_path))

    p = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog=(
            "Use --config path.yaml or path.json (before other flags) to load options; "
            "CLI arguments override the file. See weapon_ai/infer_thermal.example.yaml."
        ),
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--source", type=str, default=None, help="thermal .mp4 path or webcam index")
    p.add_argument(
        "--capture_width",
        type=int,
        default=3840,
        help="Requested capture width when source is a webcam index.",
    )
    p.add_argument(
        "--capture_height",
        type=int,
        default=2160,
        help="Requested capture height when source is a webcam index.",
    )
    p.add_argument(
        "--capture_fps",
        type=float,
        default=30.0,
        help="Requested capture FPS when source is a webcam index.",
    )
    p.add_argument(
        "--output_fps",
        type=float,
        default=0.0,
        metavar="FPS",
        help="Output video FPS for --output. For pre-recorded files, subsamples decoded frames "
        "(stride ≈ round(source_fps / FPS)) so duration matches the source while running fewer infers. "
        "0 = use source FPS (default). Webcam: sets writer FPS only; no subsampling.",
    )
    p.add_argument(
        "--live_infer_stride",
        type=int,
        default=1,
        metavar="N",
        help="Live webcam/thermal: run YOLO+gun infer every N captured frames (1=every frame). "
        "Overlay/IPC still updates every frame; with ByteTrack on, gap frames use MOT prediction "
        "(use N=2 at 30fps capture for ~15 infer/s and smoother 30fps MJPEG).",
    )
    p.add_argument(
        "--live_infer_max_width",
        type=int,
        default=0,
        metavar="PX",
        help="Live only: downscale frames to this width before YOLO (0=full capture res). "
        "Boxes are mapped back to full resolution for overlay. Use 1920 at 4K capture on server GPUs.",
    )
    p.add_argument(
        "--live_ipc_max_width",
        type=int,
        default=0,
        metavar="PX",
        help="Live only: downscale annotated BGR before mmap IPC/WebRTC (0=overlay resolution). "
        "Overlay drawing stays full-res; stream uses this width.",
    )
    add_bool_optional_arg(
        p,
        "--live_gpu_preprocess",
        default=True,
        help_text=(
            "Use CUDA torch bilinear resize for live_infer_max_width / live_ipc_max_width when available. "
            "MJPEG decode stays on CPU (OpenCV); this offloads downscale from the overlay hot path."
        ),
    )
    add_bool_optional_arg(
        p,
        "--live_pipeline_publish",
        default=True,
        help_text=(
            "Publish BGR/JPEG IPC on a background thread pool so OpenCV draw + mmap write overlap the next frame."
        ),
    )
    p.add_argument(
        "--live_publish_workers",
        type=int,
        default=2,
        metavar="N",
        help="Thread pool size for --live_pipeline_publish (default 2).",
    )
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--min_box_px", type=int, default=24)
    p.add_argument(
        "--yolo_classes",
        type=str,
        default="0",
        help=(
            'COCO class ids to keep, comma-separated. Default "0" = person only (firearm = gun YOLO). '
            'Add "39" for bottle, "67" for cell phone, or "all" for every class.'
        ),
    )
    add_bool_optional_arg(
        p,
        "--yolo_non_person_inside_person",
        default=True,
        help_text=(
            "With multi-class YOLO, keep bottle and other non-person boxes only when their center lies inside "
            "some person box (drops background clutter). Use --no-yolo_non_person_inside_person for full-frame objects."
        ),
    )
    p.add_argument(
        "--yolo_nonperson_min_conf",
        type=float,
        default=0.50,
        help="YOLO bottle/other non-person rows: minimum detection confidence after NMS (default 0.50; person still uses --conf).",
    )
    p.add_argument(
        "--yolo_nonperson_min_short_side_px",
        type=int,
        default=44,
        help="YOLO bottle/other non-person: discard boxes with min(w,h) below this (reduces shirt text).",
    )
    p.add_argument(
        "--yolo_nonperson_min_area_px",
        type=int,
        default=1100,
        metavar="AREA",
        help="YOLO bottle/other non-person: discard boxes smaller than width*height in pixels².",
    )
    p.add_argument(
        "--yolo_nonperson_max_aspect",
        type=float,
        default=7.0,
        help="YOLO bottle/other non-person: discard if max(w/h,h/w) exceeds this (0 = disable).",
    )
    p.add_argument(
        "--unsafe_threshold",
        type=float,
        default=0.75,
        help=(
            "Person UNSAFE (red border) when threat score ≥ this (0–1, same scale as firearm YOLO conf). "
            "Shown as decimals on labels (e.g. 0.75); default 0.75."
        ),
    )
    p.add_argument(
        "--safe_max",
        type=float,
        default=_SAFE_MAX,
        help="Deprecated (SUSPICIOUS tier removed). Kept for config compatibility only.",
    )
    p.add_argument(
        "--gun_threshold",
        type=float,
        default=None,
        help="Alias for --unsafe_threshold when using a gun-probability checkpoint.",
    )
    add_bool_optional_arg(
        p,
        "--fuse_gun_to_prob",
        default=True,
        help_text="If a firearm box is detected in the frame, boost person p(gun) using gun confidence.",
    )
    p.add_argument(
        "--gun_prob_floor",
        type=float,
        default=0.60,
        help="Minimum fused p(gun) when firearm YOLO detects a valid box in the frame.",
    )
    p.add_argument(
        "--gun_conf_scale",
        type=float,
        default=2.0,
        help="Scale factor for firearm confidence when fusing into p(gun).",
    )
    p.add_argument(
        "--unsafe_border_thick",
        type=int,
        default=6,
        help="Border thickness in pixels for UNSAFE boxes.",
    )
    p.add_argument(
        "--unsafe_border_color",
        type=str,
        default="red",
        choices=["red", "black", "white", "yellow"],
        help="Border color for UNSAFE boxes (BGR via named preset).",
    )
    p.add_argument(
        "--composite_mode",
        action="store_true",
        help="Source is 3-panel composite; thermal is taken from --thermal_panel strip only.",
    )
    p.add_argument(
        "--thermal_panel",
        choices=["left", "center", "right"],
        default="left",
        help="Which third of composite is thermal (when --composite_mode).",
    )
    p.add_argument(
        "--thermal_v4l2",
        action="store_true",
        help="Live PureThermal / Y16 V4L2 capture (not MJPG webcam). Applies inferno colormap to preview.",
    )
    p.add_argument(
        "--thin_overlay",
        action="store_true",
        help="Use 1px overlay strokes for native low-res thermal (before panel upscale).",
    )
    p.add_argument(
        "--panel_w",
        type=int,
        default=0,
        help="Upscale annotated preview width before live_jpg/IPC (0 = native capture size).",
    )
    p.add_argument(
        "--panel_h",
        type=int,
        default=0,
        help="Upscale annotated preview height before live_jpg/IPC (0 = native capture size).",
    )
    p.add_argument(
        "--live_frame_poll",
        type=Path,
        default=None,
        help="Poll this JPEG path for thermal frames (overlay-only; do not open V4L2). "
        "Use when another process owns the camera and writes live_frame.",
    )
    p.add_argument(
        "--show_yolo_name",
        action="store_true",
        help="Prefix label with YOLO class name (e.g. person).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write annotated preview to this video file (e.g. preview.mp4). Same overlays as the live window.",
    )
    p.add_argument(
        "--playground_jpg",
        type=Path,
        default=None,
        help="Single-frame playground mode: write annotated preview JPEG here and exit after one frame.",
    )
    p.add_argument(
        "--playground_json",
        type=Path,
        default=None,
        help="Single-frame playground mode: write threat summary JSON here (same schema as live_metrics_json).",
    )
    p.add_argument(
        "--live_jpg",
        type=Path,
        default=None,
        help="Each processed frame, atomically write this JPEG path (for Layer 8 MJPEG preview).",
    )
    p.add_argument(
        "--live_ipc_frame",
        type=Path,
        default=None,
        help="Optional mmap frame channel path (single-producer latest JPEG) for low-latency UI preview.",
    )
    p.add_argument(
        "--live_ipc_bgr_frame",
        type=Path,
        default=None,
        help="Optional mmap frame channel path (single-producer latest raw BGR) for low-CPU WebRTC preview.",
    )
    p.add_argument(
        "--live_metrics_json",
        type=Path,
        default=None,
        help="Per-frame threat summary JSON for Layer 8 dashboard.",
    )
    p.add_argument(
        "--live_metrics_every_n",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Write live_metrics_json every N frames (0 = auto: 15 live / 1 batch). "
            "Overlay and BGR IPC still update every frame."
        ),
    )
    add_bool_optional_arg(
        p,
        "--live_threaded_capture",
        default=True,
        help_text=(
            "Live UVC webcam: decode MJPEG on a background thread so overlay/IPC stay on the hot path. "
            "Disable with --no-live_threaded_capture."
        ),
    )
    add_bool_optional_arg(
        p,
        "--gstreamer_capture",
        default=False,
        help_text=(
            "Live UVC webcam: capture via GStreamer (v4l2src → jpegdec/nvjpegdec → BGR pipe) instead of "
            "OpenCV CAP_V4L2. Falls back to OpenCV if the pipeline fails. "
            "Disable with --no-gstreamer_capture."
        ),
    )
    add_bool_optional_arg(
        p,
        "--analysis_report",
        default=True,
        help_text=(
            "Write live .log + updating .json + final .txt analysis (person T<id>, firearm objectN/weaponN, "
            "frame/time spans, peaks). Default on; base path from --analysis_report_path or next to --output."
        ),
    )
    p.add_argument(
        "--analysis_report_path",
        type=Path,
        default=None,
        help="Base path for analysis artifacts (extensions .json, .txt, .log added). Default: {output_stem}_analysis.",
    )
    p.add_argument(
        "--analysis_report_flush_every",
        type=int,
        default=60,
        metavar="N",
        help="Rewrite analysis JSON and append a summary log line every N frames (default 60).",
    )
    p.add_argument(
        "--no_imshow",
        action="store_true",
        help="Do not open cv2.imshow (batch / headless). Use with --output.",
    )
    add_bool_optional_arg(
        p,
        "--show_fps",
        default=True,
        help_text="Draw smoothed infer FPS on the annotated overlay (top-right).",
    )
    p.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Stop after this many frames (0 = entire clip).",
    )
    p.add_argument(
        "--gun_yolo_model",
        type=Path,
        default=None,
        help=(
            "Firearm-detection YOLO .pt (Ultralytics). Default: repo-root ``gun_enhanced_cctv_apr17.pt`` if present, "
            f"else {_DEFAULT_FIREARM_YOLO} (auto-download if missing)."
        ),
    )
    p.add_argument(
        "--no_gun_yolo",
        action="store_true",
        help="Disable second-stage firearm boxes.",
    )
    p.add_argument(
        "--gun_only",
        action="store_true",
        help="Skip person YOLO; run firearm YOLO on the full frame only (use with thermal low-res).",
    )
    p.add_argument(
        "--gun_conf",
        type=float,
        default=0.25,
        help="Firearm YOLO NMS confidence. With --gun-take-best (default): infer uses min(this, --gun_take_best_infer_conf).",
    )
    p.add_argument(
        "--gun_take_best_infer_conf",
        type=float,
        default=0.15,
        help=(
            "With --gun-take-best: cap Ultralytics infer conf (default 0.15). "
            "Older builds used 0.01 and caused 0.01–0.05 score flicker on video."
        ),
    )
    p.add_argument(
        "--gun_emit_min_conf",
        type=float,
        default=0.25,
        help=(
            "Do not draw a firearm box unless the detection score is > this "
            "(default 0.25; scores above become object, > --gun_label_weapon_min become weapon)."
        ),
    )
    p.add_argument(
        "--gun_min_box_px",
        type=int,
        default=14,
        help="Min width and height (px) for an orange firearm box to count; filters tiny flicker when the subject is far. "
        "If distant guns vanish, lower (e.g. 10). Default 14.",
    )
    add_bool_optional_arg(
        p,
        "--gun_take_best",
        default=True,
        help_text=(
            "Pick one best firearm box per person ROI from candidates above --gun_emit_min_conf "
            "(default: on)."
        ),
    )
    p.add_argument(
        "--gun_thermal",
        action="store_true",
        help="Preset for RGB firearm weights on thermal: take-best, lower gun_conf, relaxed size limits, imgsz≥800, ROI pad.",
    )
    p.add_argument(
        "--gun_max_area_frac",
        type=float,
        default=0.22,
        help="Skip firearm boxes if area exceeds this fraction of the frame (thermal false positives are often huge). Set to 1.0 to disable.",
    )
    p.add_argument(
        "--gun_max_side_frac",
        type=float,
        default=0.65,
        help="Skip firearm boxes if width or height exceeds this fraction of frame W/H. Set to 1.0 to disable.",
    )
    p.add_argument(
        "--gun_label_object_min",
        type=float,
        default=0.25,
        help="Firearm overlay: score > this and ≤ --gun_label_weapon_min → objectN (default 0.25).",
    )
    p.add_argument(
        "--gun_label_weapon_min",
        type=float,
        default=0.70,
        help="Firearm overlay: score > this → weaponN (default 0.70). Track must confirm over ~75 frames.",
    )
    p.add_argument(
        "--gun_full_frame",
        action="store_true",
        help="Run firearm YOLO on the full thermal frame (legacy). Default: only inside each person box crop.",
    )
    add_bool_optional_arg(
        p,
        "--assoc_motion",
        default=True,
        help_text=(
            "Associate each firearm/object with the person it moves with (position + "
            "velocity) instead of nearest centroid. Robust when a bystander walks in "
            "front of the armed person. Use --no-assoc_motion for legacy behavior."
        ),
    )
    p.add_argument(
        "--assoc_velocity_weight",
        type=float,
        default=0.35,
        help="Weight of the velocity-agreement term (cosine of gun vs person velocity) in motion association. 0 disables velocity.",
    )
    p.add_argument(
        "--assoc_min_speed",
        type=float,
        default=1.5,
        help="Min centroid speed (px/frame) for both gun and person before the velocity term is used; below this, association is spatial only.",
    )
    p.add_argument(
        "--gun_roi_pad_frac",
        type=float,
        default=0.40,
        help="Expand each person box by this fraction of its width/height (each side) before firearm YOLO (clamped to frame). Default 0.40 for stretched arms / weapon at reach.",
    )
    p.add_argument(
        "--gun_roi_pad_px",
        type=int,
        default=72,
        help="After fractional pad, expand the person ROI by this many pixels on each side (clamped to frame). Default 72.",
    )
    p.add_argument(
        "--gun_roi_upper_frac",
        type=float,
        default=0.60,
        metavar="FRAC",
        help="Keep firearm search/detections in the upper FRAC of each person box (from head down). "
        "0.78 ≈ drop bottom 22%% (shoes/feet); 1.0 = full height; 0 = disable.",
    )
    add_bool_optional_arg(
        p,
        "--gun_hand_zone",
        default=False,
        help_text="Require firearm boxes to overlap estimated hand/carry bands on the person.",
    )
    p.add_argument(
        "--gun_hand_zone_min_overlap",
        type=float,
        default=0.10,
        metavar="FRAC",
        help="Min overlap with a hand/carry band when --gun_hand_zone is on (default 0.10).",
    )
    p.add_argument(
        "--gun_imgsz",
        type=int,
        default=640,
        help="Inference image size for firearm YOLO (larger can help tiny regions; slower).",
    )
    p.add_argument(
        "--gun_thermal_debug",
        action="store_true",
        help="RGB firearm model on thermal: set conf=0.02 and disable size filters (1.0). Expect large, imprecise boxes; use for demos only.",
    )
    p.add_argument(
        "--yolo_device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Ultralytics YOLO device. On Jetson OOM try cpu (slow).",
    )
    p.add_argument(
        "--classifier_device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="MobileNet gun/safe head. Use cpu on Jetson to avoid CUBLAS OOM with dual YOLO + high-res.",
    )
    add_bool_optional_arg(
        p,
        "--cuda_empty_cache",
        default=False,
        help_text="Call torch.cuda.empty_cache() periodically (helps fragmented Jetson VRAM). Use with --cuda-empty-cache-every.",
    )
    p.add_argument(
        "--cuda_empty_cache_every",
        type=int,
        default=30,
        metavar="N",
        help="With --cuda-empty-cache, run empty_cache every N frames (default: 30). Use 1 for legacy per-frame behavior.",
    )
    p.add_argument(
        "--batch_warmup_passes",
        type=int,
        default=2,
        metavar="N",
        help="Video files only: run N full infer passes on the first good frame, rewind, then process. "
        "Warms CUDA/kernels before the written output. 0 disables (default 2).",
    )
    p.add_argument(
        "--batch_async_infer",
        action="store_true",
        help="Video files only: legacy threaded infer (overlap decode with next infer; first frames may lack overlays). "
        "Default for files is synchronous infer per frame (no startup overlay lag).",
    )
    add_bool_optional_arg(
        p,
        "--byte_track",
        default=True,
        help_text="Track person boxes with ByteTrack (default on); labels prefix with T<id>. Use --no-byte_track to disable.",
    )
    add_bool_optional_arg(
        p,
        "--byte_track_firearms",
        default=False,
        help_text=(
            "Also run ByteTrack on firearm boxes (MOT ghost hold). Default off; firearm overlay ids still use "
            "stable object<N>/weapon<N> without this. Use --byte_track_firearms when you want MOT on gun boxes too."
        ),
    )
    p.add_argument(
        "--byte_track_buffer",
        type=int,
        default=30,
        metavar="N",
        help="ByteTrack frames to keep lost tracks (Ultralytics default 30 at 30 fps reference).",
    )
    p.add_argument(
        "--byte_track_fps",
        type=float,
        default=0.0,
        metavar="FPS",
        help="Override FPS for ByteTrack timing (lost-track length). 0 = use capture or reported video FPS.",
    )
    p.add_argument(
        "--byte_track_firearm_ghost_frames",
        type=int,
        default=18,
        metavar="N",
        help="When firearm ByteTrack is on: repeat last real gun box up to N frames as a low-conf ghost "
        "detection so IDs stay stable across brief YOLO dropouts (0 = disable).",
    )
    add_bool_optional_arg(
        p,
        "--byte_track_firearm_draw_ghost",
        default=True,
        help_text="When ghost hold is active, draw faint dashed-style gun boxes for ghost-only frames.",
    )
    add_bool_optional_arg(
        p,
        "--quiet_third_party_warnings",
        default=True,
        help_text="Suppress noisy dependency warnings (e.g. torchvision compatibility) and keep app logs focused.",
    )
    if file_defaults:
        p.set_defaults(**file_defaults)
    args = p.parse_args(argv_rest)

    if infer_config_path is not None:
        print(f"Loaded infer config: {infer_config_path.resolve()}")

    if args.source is None:
        raise SystemExit(
            "Missing --source. Add it to your config file or pass it on the command line."
        )
    if args.gun_threshold is not None:
        args.unsafe_threshold = float(args.gun_threshold)
    args.unsafe_threshold = float(args.unsafe_threshold)
    args.safe_max = float(args.safe_max)
    if args.unsafe_threshold <= args.safe_max:
        args.unsafe_threshold = args.safe_max + 0.05
        print(
            f"Warning: --unsafe_threshold must be > --safe_max; using unsafe_threshold={args.unsafe_threshold:.3f}",
            flush=True,
        )
    args.cuda_empty_cache_every = max(1, int(args.cuda_empty_cache_every))
    args.live_infer_stride = max(1, int(getattr(args, "live_infer_stride", 1) or 1))
    args.live_infer_max_width = max(0, int(getattr(args, "live_infer_max_width", 0) or 0))
    args.live_ipc_max_width = max(0, int(getattr(args, "live_ipc_max_width", 0) or 0))
    args.live_publish_workers = max(1, int(getattr(args, "live_publish_workers", 2) or 2))
    if args.quiet_third_party_warnings:
        warnings.filterwarnings(
            "ignore",
            message=r".*torchvision==.*incompatible with torch==.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*For a full compatibility table.*",
        )
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        try:
            from ultralytics.utils import LOGGER as _UL_LOGGER  # local import to avoid hard dependency here

            _UL_LOGGER.setLevel(logging.ERROR)
        except Exception:
            pass
    # Import after warning filters so startup compatibility warnings can be muted.
    from ultralytics import YOLO

    if args.gun_thermal_debug:
        args.gun_conf = 0.02
        args.gun_max_area_frac = 1.0
        args.gun_max_side_frac = 1.0
    elif args.gun_thermal:
        args.gun_take_best = True
        args.gun_conf = min(float(args.gun_conf), 0.15)
        args.gun_take_best_infer_conf = min(float(args.gun_take_best_infer_conf), 0.15)
        args.gun_max_area_frac = 0.5
        args.gun_max_side_frac = 0.88
        if not args.gun_only:
            args.gun_imgsz = max(args.gun_imgsz, 800)
        args.gun_roi_pad_frac = max(args.gun_roi_pad_frac, 0.40)
        args.gun_roi_pad_px = max(int(args.gun_roi_pad_px), 72)

    if args.gun_only:
        if args.no_gun_yolo:
            raise SystemExit("--gun_only requires firearm YOLO (do not use --no_gun_yolo).")
        args.gun_full_frame = True

    _unsafe_bgr = {
        "red": (0, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "yellow": (0, 255, 255),
    }[args.unsafe_border_color]
    _unsafe_text_bgr = (0, 0, 255) if args.unsafe_border_color == "black" else _unsafe_bgr

    def _resolve_torch_device(choice: str) -> torch.device:
        if choice == "cpu":
            return torch.device("cpu")
        if choice == "cuda":
            if not torch.cuda.is_available():
                raise SystemExit(
                    "CUDA requested (--yolo_device / --classifier_device) but torch.cuda.is_available() is False"
                )
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_torch = _resolve_torch_device(args.yolo_device)
    det_device: int | str = 0 if yolo_torch.type == "cuda" else "cpu"
    score_name = "gun_conf"

    yolo_classes = _parse_yolo_classes(args.yolo_classes)
    person_only = yolo_classes is not None and yolo_classes == [0]
    detector: YOLO | None = None
    if not args.gun_only:
        detector = YOLO(args.yolo_model)

    gun_detector: YOLO | None = None
    if not args.no_gun_yolo:
        gpath = Path(args.gun_yolo_model).resolve() if args.gun_yolo_model is not None else _default_firearm_yolo_path()
        if gpath.resolve() == _DEFAULT_FIREARM_YOLO.resolve():
            _ensure_firearm_yolo_weights(gpath)
        elif not (gpath.is_file() and gpath.stat().st_size > 1_000_000):
            raise SystemExit(f"Firearm YOLO weights not found or too small: {gpath}")
        gun_detector = YOLO(str(gpath))

    gun_binary_class_sets: tuple[set[int], set[int]] | None = None
    gun_model_names: dict = {}
    if gun_detector is not None:
        gun_model_names = dict(getattr(gun_detector, "names", {}) or {})
        gun_binary_class_sets = _gun_yolo_binary_class_sets(gun_model_names)
        if gun_binary_class_sets is not None:
            print(
                "Firearm YOLO binary classes: "
                f"no_weapon={sorted(gun_binary_class_sets[0])} weapon={sorted(gun_binary_class_sets[1])}"
            )

    src_raw = str(args.source or "").strip()
    is_linux_video_node = src_raw.startswith("/dev/video") or src_raw.isdigit()
    if sys.platform.startswith("linux") and is_linux_video_node:
        src = src_raw if src_raw.startswith("/dev/video") else f"/dev/video{int(src_raw)}"
    else:
        src = int(src_raw) if src_raw.isdigit() else src_raw
    live_frame_poll_path: Path | None = None
    v4l2_cap: Any = None
    gst_live_capture: GStreamerWebcamCapture | None = None
    if args.live_frame_poll is not None:
        live_frame_poll_path = args.live_frame_poll.expanduser().resolve()
        live_frame_poll_path.parent.mkdir(parents=True, exist_ok=True)
        cap = None
        print(
            f"Thermal overlay: polling {live_frame_poll_path} (camera not opened; "
            "another process should write this JPEG).",
            flush=True,
        )
    elif is_linux_video_node:
        cap = None
        v4l2_path = str(src)
        if not os.path.exists(v4l2_path):
            hint = ""
            try:
                p = subprocess.run(
                    ["v4l2-ctl", "--list-devices"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                text = p.stdout or ""
                if "PureThermal" in text or "PureTh" in text:
                    hint = " Check v4l2-ctl --list-devices for the current PureThermal /dev/videoN node."
            except Exception:
                pass
            raise SystemExit(
                f"Cannot open {v4l2_path}: device missing.{hint} "
                "No /dev/videoN node — plug in the camera, check `v4l2-ctl --list-devices`, "
                "then set thermal_device in Layer 8 settings."
            )
        if bool(getattr(args, "thermal_v4l2", False)) and sys.platform.startswith("linux"):
            try:
                from media.capture import V4L2ThermalCapture

                v4l2_cap = V4L2ThermalCapture(
                    v4l2_path,
                    width=int(args.capture_width),
                    height=int(args.capture_height),
                    fps=float(args.capture_fps),
                )
                aw_mmap, ah_mmap, fcc_mmap = v4l2_cap.negotiated
                print(
                    f"Thermal V4L2 mmap {v4l2_path}: {aw_mmap}x{ah_mmap} @ {float(args.capture_fps):.2f} fps; "
                    f"FOURCC '{fcc_mmap}' bytesperline={int(v4l2_cap.bytesperline)}",
                    flush=True,
                )
                if aw_mmap != int(args.capture_width) or ah_mmap != int(args.capture_height):
                    print(
                        f"Note: driver negotiated {aw_mmap}x{ah_mmap} "
                        f"(requested {int(args.capture_width)}x{int(args.capture_height)}).",
                        flush=True,
                    )
                pw_mmap = int(getattr(args, "panel_w", 0) or 0)
                ph_mmap = int(getattr(args, "panel_h", 0) or 0)
                if pw_mmap > 0 and ph_mmap > 0:
                    print(f"Thermal preview upscale: {pw_mmap}x{ph_mmap}", flush=True)
            except Exception as exc:
                v4l2_cap = None
                print(f"Warning: V4L2 mmap capture failed ({exc}); trying OpenCV", flush=True)
        want_gst = (
            v4l2_cap is None
            and not bool(getattr(args, "thermal_v4l2", False))
            and bool(getattr(args, "gstreamer_capture", False))
        )
        if want_gst:
            if not gstreamer_available():
                print(
                    "Warning: --gstreamer_capture set but gst-launch-1.0 missing; using OpenCV.",
                    flush=True,
                )
            else:
                try:
                    gst_try = GStreamerWebcamCapture(
                        v4l2_path,
                        width=int(args.capture_width),
                        height=int(args.capture_height),
                        fps=float(args.capture_fps),
                    )
                    gst_try.start()
                    if gst_try.wait_first_frame(timeout_s=6.0):
                        gst_live_capture = gst_try
                        hw = "nvjpegdec (GPU)" if nvidia_gst_jpeg_available() else "jpegdec (CPU)"
                        print(
                            f"GStreamer capture {v4l2_path}: "
                            f"{int(args.capture_width)}x{int(args.capture_height)} @ "
                            f"{float(args.capture_fps):.2f} fps; decoder={hw}",
                            flush=True,
                        )
                        if not nvidia_gst_jpeg_available():
                            print(
                                "Note: NVIDIA GStreamer JPEG/NVDEC plugins not installed "
                                "(DeepStream nvjpegdec/nvv4l2decoder). MJPEG decode stays on CPU; "
                                "RTX still used for YOLO/CUDA preprocess.",
                                flush=True,
                            )
                    else:
                        err = gst_try.last_error or "no frames"
                        gst_try.stop()
                        print(
                            f"Warning: GStreamer capture failed ({err}); falling back to OpenCV.",
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        f"Warning: GStreamer capture failed ({exc}); falling back to OpenCV.",
                        flush=True,
                    )
        if v4l2_cap is None and gst_live_capture is None:
            # Some UVC devices fail briefly after reconnect; retry before hard fail.
            for _attempt in range(20):
                c = cv2.VideoCapture(v4l2_path, cv2.CAP_V4L2)
                if c.isOpened():
                    cap = c
                    break
                c.release()
                sleep(0.15)
            if cap is None:
                raise SystemExit(f"Cannot open {v4l2_path}")
    else:
        cap = cv2.VideoCapture(src)
    if cap is not None and not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")
    if cap is not None and is_linux_video_node:
        def _get_fourcc_str(cobj: Any) -> str:
            fcc_i = int(cobj.get(cv2.CAP_PROP_FOURCC) or 0)
            return "".join(chr((fcc_i >> (8 * i)) & 0xFF) for i in range(4)).strip()

        def _configure_thermal_capture(cobj: Any) -> tuple[int, int, float, str]:
            # PureThermal: prefer GREY @ 80x60; Y16 is a fallback if GREY read fails.
            try:
                cobj.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"GREY"))
            except Exception:
                pass
            cobj.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.capture_width))
            cobj.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.capture_height))
            cobj.set(cv2.CAP_PROP_FPS, float(args.capture_fps))
            cobj.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            warmed = False
            for _ in range(3):
                ok_w, _ = cobj.read()
                if ok_w:
                    warmed = True
                    break
                sleep(0.08)
            if not warmed:
                try:
                    cobj.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
                except Exception:
                    pass
                for _ in range(2):
                    ok_w, _ = cobj.read()
                    if ok_w:
                        break
                    sleep(0.08)
            aw_l = int(round(cobj.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
            ah_l = int(round(cobj.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
            afps_l = float(cobj.get(cv2.CAP_PROP_FPS) or 0.0)
            fcc_l = _get_fourcc_str(cobj)
            return aw_l, ah_l, afps_l, fcc_l

        def _configure_capture(cobj: Any) -> tuple[int, int, float, str]:
            # Many UVC webcams default to low-fps YUYV at 1080p; request MJPG + target FPS.
            try:
                cobj.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            cobj.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.capture_width))
            cobj.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.capture_height))
            req_fps = float(args.capture_fps)
            # UVC MJPEG is typically 30 fps max at 1080p/2K; asking 60 can pick a slower fallback mode.
            if (
                int(args.capture_width) >= 1920
                and int(args.capture_height) >= 1080
                and req_fps > 30.0
            ):
                req_fps = 30.0
            cobj.set(cv2.CAP_PROP_FPS, req_fps)
            # Webcam path should be normal RGB/BGR for YOLO; avoid raw 2-channel V4L2 formats.
            cobj.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            cobj.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Read once so driver applies/locks negotiated mode.
            _ = cobj.read()
            aw_l = int(round(cobj.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
            ah_l = int(round(cobj.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
            afps_l = float(cobj.get(cv2.CAP_PROP_FPS) or 0.0)
            fcc_l = _get_fourcc_str(cobj)
            return aw_l, ah_l, afps_l, fcc_l

        if bool(getattr(args, "thermal_v4l2", False)) and v4l2_cap is None:
            aw, ah, afps, fcc = _configure_thermal_capture(cap)
            ew, eh = int(args.capture_width), int(args.capture_height)
            print(
                f"Thermal Y16 negotiated {aw}x{ah} @ {afps:.2f} fps (requested {ew}x{eh} @ {float(args.capture_fps):.2f}); "
                f"FOURCC '{fcc}'",
                flush=True,
            )
            pw = int(getattr(args, "panel_w", 0) or 0)
            ph = int(getattr(args, "panel_h", 0) or 0)
            if pw > 0 and ph > 0:
                print(f"Thermal preview upscale: {pw}x{ph}", flush=True)
        else:
            aw, ah, afps, fcc = _configure_capture(cap)
            if "MJPG" not in fcc.upper():
                # Hard attempt to escape YUYV low-fps mode: reopen and re-negotiate MJPG.
                for _retry in range(4):
                    try:
                        cap.release()
                    except Exception:
                        pass
                    if sys.platform.startswith("linux"):
                        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
                    else:
                        cap = cv2.VideoCapture(src)
                    if not cap.isOpened():
                        sleep(0.1)
                        continue
                    aw, ah, afps, fcc = _configure_capture(cap)
                    if "MJPG" in fcc.upper():
                        break

            ew, eh = int(args.capture_width), int(args.capture_height)
            if aw > 0 and ah > 0 and (abs(aw - ew) > 8 or abs(ah - eh) > 8):
                print(
                    f"Webcam negotiated {aw}x{ah} (requested {ew}x{eh}). "
                    "The device may still be capturing or scaling in a higher-latency mode.",
                    flush=True,
                )
            if afps > 0:
                print(f"Webcam negotiated FPS {afps:.2f} (requested {float(args.capture_fps):.2f})", flush=True)
            if fcc:
                print(f"Webcam negotiated FOURCC '{fcc}'", flush=True)
            if "MJPG" not in fcc.upper():
                print(
                    "Warning: webcam did not lock MJPG; low-fps YUYV mode may still be active.",
                    flush=True,
                )

    det_mode = (
        "gun only (full frame)"
        if args.gun_only
        else ("person only" if person_only else ("all YOLO classes" if yolo_classes is None else str(yolo_classes)))
    )
    print(
        f"q=quit | YOLO ({det_mode}) device={det_device} | "
        f"label = armed / score (0–1 on frame) | "
        f"UNSAFE = red border when score ≥ {float(args.unsafe_threshold):.2f} ({_overlay_score(float(args.unsafe_threshold))})"
    )
    if bool(args.yolo_non_person_inside_person) and not person_only:
        print(
            "YOLO: non-person boxes (e.g. bottle) only when centered inside a person box.",
            flush=True,
        )
    if not person_only and float(getattr(args, "yolo_nonperson_min_conf", 0.5)) > 0:
        print(
            f"YOLO: non-person extra filter conf≥{float(args.yolo_nonperson_min_conf):.2f}, "
            f"min short side {int(args.yolo_nonperson_min_short_side_px)} px, "
            f"area≥{int(args.yolo_nonperson_min_area_px)} px², "
            f"max aspect {float(args.yolo_nonperson_max_aspect):.1f}.",
            flush=True,
        )
    if args.cuda_empty_cache and torch.cuda.is_available():
        print(
            f"CUDA: empty_cache every {args.cuda_empty_cache_every} frame(s) "
            "(Jetson VRAM workaround; use --cuda-empty-cache-every 1 for every frame).",
            flush=True,
        )
    if gun_detector is not None:
        roi = "full frame" if args.gun_full_frame else "inside each person box only"
        print(
            f"Firearm YOLO: orange boxes (AGPL-3.0) — search: {roi}. "
            f"Size filter vs ROI/frame: area≤{args.gun_max_area_frac:.0%}, side≤{args.gun_max_side_frac:.0%}."
        )
        print(f"Firearm YOLO imgsz={args.gun_imgsz}.")
        if args.gun_thermal_debug:
            print(
                "WARNING: --gun_thermal_debug — low conf, size filters off; boxes are often wrong on thermal."
            )
        if args.gun_roi_pad_frac > 0:
            print(f"Person ROI pad for firearm pass: {args.gun_roi_pad_frac:.0%} of box size.")
        if int(args.gun_roi_pad_px) > 0:
            print(f"Person ROI extra pad for firearm pass: {int(args.gun_roi_pad_px)} px per side.")
        upper_frac = float(getattr(args, "gun_roi_upper_frac", 0.0) or 0.0)
        if 0.0 < upper_frac < 1.0:
            print(
                f"Firearm pass limited to upper {upper_frac:.0%} of each person box "
                "(torso/arms; legs and shoes excluded).",
                flush=True,
            )
        if args.gun_thermal:
            print("Preset --gun_thermal: take-best firearm box, relaxed size limits, larger imgsz.")
        if args.gun_take_best:
            print(
                f"Firearm YOLO take-best: infer conf=min(--gun_conf, {float(args.gun_take_best_infer_conf):.2f}); "
                f"emit/draw only if score ≥ {float(args.gun_emit_min_conf):.2f}."
            )
        print(
            f"Firearm track labels: objectN (> {float(args.gun_label_object_min):.2f}) / "
            f"weaponN (> {float(args.gun_label_weapon_min):.2f}) by vote majority; "
            f"gun box → {'nearest person (full frame)' if args.gun_full_frame else 'person whose ROI crop produced the detection'} "
            f"(dedupe on ROI overlap). Armed latch after weapon track confirmed (75-frame hold).",
            flush=True,
        )
    if args.output is not None:
        args.output = args.output.resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.live_jpg is not None:
        args.live_jpg = args.live_jpg.expanduser().resolve()
        args.live_jpg.parent.mkdir(parents=True, exist_ok=True)
    live_ipc_writer: LiveFrameWriter | None = None
    if args.live_ipc_frame is not None:
        args.live_ipc_frame = args.live_ipc_frame.expanduser().resolve()
        args.live_ipc_frame.parent.mkdir(parents=True, exist_ok=True)
        live_ipc_writer = LiveFrameWriter(args.live_ipc_frame)
    live_ipc_bgr_writer: LiveBgrFrameWriter | None = None
    if args.live_ipc_bgr_frame is not None:
        args.live_ipc_bgr_frame = args.live_ipc_bgr_frame.expanduser().resolve()
        args.live_ipc_bgr_frame.parent.mkdir(parents=True, exist_ok=True)
        live_ipc_bgr_writer = LiveBgrFrameWriter(args.live_ipc_bgr_frame)
        boot_pw = int(getattr(args, "panel_w", 0) or 0) or int(getattr(args, "capture_width", 640) or 640)
        boot_ph = int(getattr(args, "panel_h", 0) or 0) or int(getattr(args, "capture_height", 480) or 480)
        boot = np.zeros((boot_ph, boot_pw, 3), dtype=np.uint8)
        cv2.putText(
            boot,
            "Thermal AI starting...",
            (16, max(28, boot_ph // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )
        live_ipc_bgr_writer.write_bgr_ndarray(boot)
    if args.live_metrics_json is not None:
        args.live_metrics_json = args.live_metrics_json.expanduser().resolve()
        args.live_metrics_json.parent.mkdir(parents=True, exist_ok=True)

    def _infer_annotations(
        thermal_frame: np.ndarray,
    ) -> tuple[
        list[tuple[int, int, int, int, float, int | None, str, float]],
        list[tuple[int, int, int, int, str, str, float]],
        int,
        list[float],
    ]:
        h, w = thermal_frame.shape[:2]
        rows: list[tuple[int, int, int, int, float, int | None, str, float]] = []
        if not args.gun_only and detector is not None:
            pred_kw: dict = dict(
                source=thermal_frame,
                conf=args.conf,
                verbose=False,
                device=det_device,
            )
            if yolo_classes is not None:
                pred_kw["classes"] = yolo_classes
            results = detector.predict(**pred_kw)
            boxes = results[0].boxes if results else None
            id_to_name = results[0].names if results and hasattr(results[0], "names") else {}
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
                conf_np = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                for i, row in enumerate(xyxy):
                    x1, y1, x2, y2 = _clamp_box(row, w, h)
                    if (x2 - x1) < args.min_box_px or (y2 - y1) < args.min_box_px:
                        continue
                    cid = int(cls_ids[i]) if cls_ids is not None and i < len(cls_ids) else None
                    yolo_tag = id_to_name.get(cid, str(cid)) if cid is not None else "obj"
                    det_c = float(conf_np[i]) if conf_np is not None and i < len(conf_np) else float(args.conf)
                    rows.append((x1, y1, x2, y2, 0.0, cid, yolo_tag, det_c))

        if bool(args.yolo_non_person_inside_person) and rows:
            person_boxes = [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rows if r[5] == 0]
            if person_boxes:
                rows = [
                    r
                    for r in rows
                    if r[5] == 0
                    or r[5] is None
                    or _xyxy_center_inside_any_person(int(r[0]), int(r[1]), int(r[2]), int(r[3]), person_boxes)
                ]
            else:
                rows = [r for r in rows if r[5] == 0 or r[5] is None]

        rows = [
            r
            for r in rows
            if _yolo_keep_nonperson_detection(
                r[5] if r[5] is None else int(r[5]),
                int(r[0]),
                int(r[1]),
                int(r[2]),
                int(r[3]),
                float(r[7]),
                args,
            )
        ]
        rows = _dedupe_person_rows(rows)

        gun_count = 0
        person_gun_best_conf: dict[int, float] = {}
        gun_boxes: list[tuple[int, int, int, int, str, str, float, int, str]] = []
        all_person_rows: list[tuple[int, tuple[int, int, int, int]]] = [
            (ridx, (int(r[0]), int(r[1]), int(r[2]), int(r[3])))
            for ridx, r in enumerate(rows)
            if r[5] in (None, 0)
        ]
        if gun_detector is not None:
            if args.gun_thermal_debug:
                infer_gun_conf = float(args.gun_conf)
            elif args.gun_take_best:
                infer_gun_conf = min(
                    float(args.gun_conf),
                    float(args.gun_take_best_infer_conf),
                )
            else:
                infer_gun_conf = float(args.gun_conf)
            gun_emit_min = float(args.gun_emit_min_conf)
            gnames: dict = {}

            def _person_torso_ok(gx1: int, gy1: int, gx2: int, gy2: int, person_ridx: int) -> bool:
                del gx1, gy1, gx2, gy2, person_ridx
                return True

            def _emit_firearm_overlay(
                gx1: int,
                gy1: int,
                gx2: int,
                gy2: int,
                gname: str,
                gc: float,
                *,
                person_ridx: int = -1,
                roi_w: int | None = None,
                roi_h: int | None = None,
                yolo_cls: int | None = None,
            ) -> None:
                nonlocal gun_count
                _ = gname
                if float(gc) <= gun_emit_min:
                    return
                if not _person_torso_ok(gx1, gy1, gx2, gy2, person_ridx):
                    return
                gx1, gy1, gx2, gy2 = max(0, gx1), max(0, gy1), min(w, gx2), min(h, gy2)
                ref_w, ref_h = roi_w, roi_h
                if person_ridx >= 0 and (ref_w is None or ref_h is None):
                    pr0, pr1, pr2, pr3 = (
                        int(rows[person_ridx][0]),
                        int(rows[person_ridx][1]),
                        int(rows[person_ridx][2]),
                        int(rows[person_ridx][3]),
                    )
                    ref_w, ref_h = pr2 - pr0, pr3 - pr1
                gun_count += 1
                kind = _firearm_kind_for_detection(
                    float(gc),
                    gx1,
                    gy1,
                    gx2,
                    gy2,
                    yolo_cls=yolo_cls,
                    gnames=gun_model_names or gnames,
                    binary_sets=gun_binary_class_sets,
                    object_min=float(args.gun_label_object_min),
                    weapon_min=float(args.gun_label_weapon_min),
                    ref_w=ref_w,
                    ref_h=ref_h,
                )
                glabel = f"{kind} {_overlay_score(gc)}"
                gun_boxes.append(
                    (gx1, gy1, gx2, gy2, glabel, kind, float(gc), int(person_ridx), str(gname))
                )

            def _draw_gun_from_candidates(
                candidates: list[tuple[float, int, int, int, int, str]],
                pr: int,
                pb: int,
                *,
                person_ridx: int = -1,
            ) -> None:
                if not candidates:
                    return
                normed: list[tuple[float, int, int, int, int, str]] = []
                for gc, gx1, gy1, gx2, gy2, gnm in candidates:
                    gx1, gy1, gx2, gy2 = max(0, gx1), max(0, gy1), min(w, gx2), min(h, gy2)
                    normed.append((gc, gx1, gy1, gx2, gy2, gnm))
                if args.gun_take_best:
                    for gc, gx1, gy1, gx2, gy2, gnm in sorted(normed, key=lambda t: -t[0]):
                        if float(gc) <= gun_emit_min:
                            continue
                        if _gun_detection_valid(
                            gx1, gy1, gx2, gy2, w, h,
                            args.gun_max_area_frac, args.gun_max_side_frac, args.gun_min_box_px,
                            ref_w=pr, ref_h=pb,
                        ):
                            _emit_firearm_overlay(
                                gx1, gy1, gx2, gy2, gnm, gc,
                                person_ridx=person_ridx, roi_w=pr, roi_h=pb,
                            )
                            return
                    return
                for gc, gx1, gy1, gx2, gy2, gnm in normed:
                    if float(gc) <= gun_emit_min:
                        continue
                    if not _gun_detection_valid(
                        gx1, gy1, gx2, gy2, w, h,
                        args.gun_max_area_frac, args.gun_max_side_frac, args.gun_min_box_px,
                        ref_w=pr, ref_h=pb,
                    ):
                        continue
                    _emit_firearm_overlay(
                        gx1, gy1, gx2, gy2, gnm, gc,
                        person_ridx=person_ridx, roi_w=pr, roi_h=pb,
                    )

            def _best_valid_conf(
                candidates: list[tuple[float, int, int, int, int, str]],
                pr: int,
                pb: int,
            ) -> float:
                best = 0.0
                for gc, gx1, gy1, gx2, gy2, _gnm in candidates:
                    if float(gc) <= gun_emit_min:
                        continue
                    gx1, gy1, gx2, gy2 = max(0, gx1), max(0, gy1), min(w, gx2), min(h, gy2)
                    if not _gun_detection_valid(
                        gx1, gy1, gx2, gy2, w, h,
                        args.gun_max_area_frac, args.gun_max_side_frac, args.gun_min_box_px,
                        ref_w=pr, ref_h=pb,
                    ):
                        continue
                    best = max(best, float(gc))
                return best

            if args.gun_full_frame:
                gres = gun_detector.predict(
                    source=thermal_frame, conf=infer_gun_conf, imgsz=args.gun_imgsz, verbose=False, device=det_device
                )
                gboxes = gres[0].boxes if gres else None
                gnames = dict(gres[0].names) if gres and hasattr(gres[0], "names") else {}
                candidates_ff: list[tuple[float, int, int, int, int, str]] = []
                if gboxes is not None and len(gboxes) > 0:
                    g_xyxy = gboxes.xyxy.cpu().numpy()
                    g_cls = gboxes.cls.cpu().numpy().astype(int) if gboxes.cls is not None else None
                    g_conf = gboxes.conf.cpu().numpy() if gboxes.conf is not None else None
                    for j, grow in enumerate(g_xyxy):
                        gx1, gy1, gx2, gy2 = _clamp_box(grow, w, h)
                        cid = int(g_cls[j]) if g_cls is not None and j < len(g_cls) else 0
                        gnm = gnames.get(cid, "gun")
                        gc = float(g_conf[j]) if g_conf is not None and j < len(g_conf) else 0.0
                        candidates_ff.append((gc, gx1, gy1, gx2, gy2, gnm, cid))
                person_rows_ff: list[tuple[int, tuple[int, int, int, int]]] = list(all_person_rows)
                raw_ff: list[tuple[float, int, int, int, int, str, int, int]] = []
                for gc, gx1, gy1, gx2, gy2, gnm, cid in candidates_ff:
                    if not _gun_detection_valid(
                        gx1, gy1, gx2, gy2, w, h,
                        args.gun_max_area_frac, args.gun_max_side_frac, args.gun_min_box_px,
                        ref_w=w, ref_h=h,
                    ):
                        continue
                    owner = _nearest_person_ridx_for_gun(gx1, gy1, gx2, gy2, person_rows_ff)
                    if not _person_torso_ok(gx1, gy1, gx2, gy2, owner):
                        continue
                    raw_ff.append((gc, gx1, gy1, gx2, gy2, gnm, owner, cid))
                if args.gun_take_best and raw_ff:
                    by_owner_ff: dict[int, list] = {}
                    for item in raw_ff:
                        by_owner_ff.setdefault(item[6], []).append(item)
                    raw_ff = [max(g, key=lambda t: float(t[0])) for g in by_owner_ff.values()]
                for gc, gx1, gy1, gx2, gy2, gnm, owner, cid in _dedupe_gun_candidates(raw_ff):
                    if owner >= 0:
                        person_gun_best_conf[owner] = max(
                            person_gun_best_conf.get(owner, 0.0), float(gc)
                        )
                    _emit_firearm_overlay(
                        gx1, gy1, gx2, gy2, gnm, gc,
                        person_ridx=owner, roi_w=w, roi_h=h, yolo_cls=cid,
                    )
            elif rows:
                person_rows: list[tuple[int, tuple[int, int, int, int]]] = list(all_person_rows)

                raw_gun: list[tuple[float, int, int, int, int, str, int, int]] = []
                for ridx, (px1, py1, px2, py2) in person_rows:
                    qx1, qy1, qx2, qy2 = _expand_person_roi_for_gun(
                        px1, py1, px2, py2, w, h, args.gun_roi_pad_frac, int(args.gun_roi_pad_px)
                    )
                    pr, pb = qx2 - qx1, qy2 - qy1
                    if pr < args.min_box_px or pb < args.min_box_px:
                        continue
                    pcrop = thermal_frame[qy1:qy2, qx1:qx2]
                    if pcrop.size == 0:
                        continue
                    gres = gun_detector.predict(
                        source=pcrop, conf=infer_gun_conf, imgsz=args.gun_imgsz, verbose=False, device=det_device
                    )
                    if gres and hasattr(gres[0], "names"):
                        gnames = dict(gres[0].names)
                    gboxes = gres[0].boxes if gres else None
                    if gboxes is not None and len(gboxes) > 0:
                        g_xyxy = gboxes.xyxy.cpu().numpy()
                        g_cls = gboxes.cls.cpu().numpy().astype(int) if gboxes.cls is not None else None
                        g_conf = gboxes.conf.cpu().numpy() if gboxes.conf is not None else None
                        cw, ch = pcrop.shape[1], pcrop.shape[0]
                        for j, grow in enumerate(g_xyxy):
                            lx1, ly1, lx2, ly2 = _clamp_box(grow, cw, ch)
                            gx1, gy1 = qx1 + lx1, qy1 + ly1
                            gx2, gy2 = qx1 + lx2, qy1 + ly2
                            cid = int(g_cls[j]) if g_cls is not None and j < len(g_cls) else 0
                            gnm = gnames.get(cid, "gun")
                            gc = float(g_conf[j]) if g_conf is not None and j < len(g_conf) else 0.0
                            if not _gun_detection_valid(
                                gx1, gy1, gx2, gy2, w, h,
                                args.gun_max_area_frac, args.gun_max_side_frac, args.gun_min_box_px,
                                ref_w=pr, ref_h=pb,
                            ):
                                continue
                            owner = int(ridx)
                            gcx, gcy = _box_center(gx1, gy1, gx2, gy2)
                            if not (px1 <= gcx <= px2 and py1 <= gcy <= py2):
                                continue
                            if not _person_torso_ok(gx1, gy1, gx2, gy2, owner):
                                continue
                            raw_gun.append((gc, gx1, gy1, gx2, gy2, gnm, owner, cid))

                if args.gun_take_best:
                    by_owner: dict[int, list[tuple[float, int, int, int, int, str, int, int]]] = {}
                    for item in raw_gun:
                        by_owner.setdefault(item[6], []).append(item)
                    raw_gun = []
                    for _owner, group in by_owner.items():
                        raw_gun.append(max(group, key=lambda t: float(t[0])))
                raw_gun = _dedupe_gun_candidates(raw_gun)

                for gc, gx1, gy1, gx2, gy2, gnm, owner, cid in raw_gun:
                    if float(gc) <= gun_emit_min:
                        continue
                    person_gun_best_conf[owner] = max(person_gun_best_conf.get(owner, 0.0), float(gc))
                    orw, orh = 0, 0
                    if owner >= 0:
                        for ridx_o, (ox1, oy1, ox2, oy2) in person_rows:
                            if ridx_o == owner:
                                orw, orh = ox2 - ox1, oy2 - oy1
                                break
                    _emit_firearm_overlay(
                        gx1, gy1, gx2, gy2, gnm, gc,
                        person_ridx=owner,
                        roi_w=orw if orw > 0 else None,
                        roi_h=orh if orh > 0 else None,
                        yolo_cls=cid,
                    )

        if rows:
            rows = [
                (x1, y1, x2, y2, float(person_gun_best_conf.get(ridx, 0.0)), cid, ytag, det_c)
                for ridx, (x1, y1, x2, y2, _prob, cid, ytag, det_c) in enumerate(rows)
            ]
        probs = [r[4] for r in rows] if rows else []
        return rows, gun_boxes, int(gun_count), probs

    is_batch_file = live_frame_poll_path is None and _is_batch_video_file_source(src)
    is_live_capture = live_frame_poll_path is None and not is_batch_file
    use_gpu_preprocess = bool(getattr(args, "live_gpu_preprocess", True)) and torch.cuda.is_available()

    def _infer_annotations_for_frame(
        frame: np.ndarray,
    ) -> tuple[
        list[tuple[int, int, int, int, float, int | None, str, float]],
        list[tuple[int, int, int, int, str, str, float, int]],
        int,
        list[float],
    ]:
        if not _frame_valid_for_infer(frame):
            return [], [], 0, []
        infer_mw = int(getattr(args, "live_infer_max_width", 0) or 0)
        if is_live_capture and infer_mw > 0:
            small, sx, sy = downscale_to_max_width(frame, infer_mw, use_gpu=use_gpu_preprocess)
            rows, guns, gc, probs = _infer_annotations(small)
            if sx != 1.0 or sy != 1.0:
                rows = scale_person_rows(rows, sx, sy)
                guns = scale_gun_boxes(guns, sx, sy)
            return rows, guns, gc, probs
        return _infer_annotations(frame)

    if is_batch_file and int(args.batch_warmup_passes) > 0:
        wp = max(0, int(args.batch_warmup_passes))
        t_warm = time()
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            pass
        warm_np: np.ndarray | None = None
        for _ in range(400):
            ok_w, bgr_w = cap.read()
            if not ok_w:
                break
            if args.composite_mode:
                tr_w = _extract_thermal_column(bgr_w, args.thermal_panel)
            else:
                tr_w = bgr_w
            warm_np = _frame_to_bgr_for_infer(tr_w)
            if warm_np is not None:
                break
        if warm_np is not None:
            for _ in range(wp):
                _ = _infer_annotations(warm_np.copy())
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(
                f"Batch warmup: {wp} full infer pass(es) on first good frame in {time() - t_warm:.2f}s",
                flush=True,
            )
        else:
            print("Batch warmup: skipped (no decodable BGR frame at start).", flush=True)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            pass

    live_capture: LiveWebcamCapture | GStreamerWebcamCapture | None = None
    live_frame_buf: np.ndarray | None = None
    vis_buf: np.ndarray | None = None
    panel_out_buf: np.ndarray | None = None
    last_capture_seq = -1
    if gst_live_capture is not None:
        live_capture = gst_live_capture
        print(
            "Live webcam: GStreamer MJPEG decode on background process; BGR mmap overlay without JPEG re-encode.",
            flush=True,
        )
    use_live_threaded = (
        live_capture is None
        and cap is not None
        and live_frame_poll_path is None
        and v4l2_cap is None
        and not bool(getattr(args, "thermal_v4l2", False))
        and not is_batch_file
        and bool(getattr(args, "live_threaded_capture", True))
    )
    if use_live_threaded:
        live_capture = LiveWebcamCapture(cap)
        cap = None
        live_capture.start()
        print(
            "Live webcam: MJPEG decode on background thread; BGR mmap overlay without JPEG re-encode.",
            flush=True,
        )
    if is_live_capture and use_gpu_preprocess:
        parts = ["Live server preprocess: CUDA resize enabled"]
        if int(getattr(args, "live_infer_max_width", 0) or 0) > 0:
            parts.append(f"infer≤{int(args.live_infer_max_width)}px")
        if int(getattr(args, "live_ipc_max_width", 0) or 0) > 0:
            parts.append(f"IPC≤{int(args.live_ipc_max_width)}px")
        print("; ".join(parts) + ".", flush=True)

    use_infer_pool = not (is_batch_file and not args.batch_async_infer)
    infer_pool: concurrent.futures.ThreadPoolExecutor | None = None
    if use_infer_pool:
        infer_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    use_pipeline_publish = is_live_capture and bool(getattr(args, "live_pipeline_publish", True))
    publish_pool: concurrent.futures.ThreadPoolExecutor | None = None
    if use_pipeline_publish and (
        args.live_jpg is not None or live_ipc_writer is not None or live_ipc_bgr_writer is not None
    ):
        publish_workers = 1  # IPC mmap writers must not be updated concurrently
        if int(getattr(args, "live_publish_workers", 1) or 1) > 1:
            print(
                "Note: live_publish_workers capped to 1 for shared-memory IPC preview.",
                flush=True,
            )
        publish_pool = concurrent.futures.ThreadPoolExecutor(max_workers=publish_workers)
        print(
            f"Live IPC publish: background thread pool ({publish_workers} workers).",
            flush=True,
        )

    def _publish_live_outputs(vis_bgr: np.ndarray) -> None:
        stream = vis_bgr
        ipc_mw = int(getattr(args, "live_ipc_max_width", 0) or 0)
        if is_live_capture and ipc_mw > 0 and stream.shape[1] > ipc_mw:
            stream = resize_bgr_max_width(stream, ipc_mw, use_gpu=use_gpu_preprocess)
        if args.live_jpg is not None or live_ipc_writer is not None:
            ok_lj, buf = cv2.imencode(".jpg", stream, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok_lj:
                jb = buf.tobytes()
                if args.live_jpg is not None:
                    tmp = args.live_jpg.with_suffix(args.live_jpg.suffix + ".tmp")
                    tmp.write_bytes(jb)
                    tmp.replace(args.live_jpg)
                if live_ipc_writer is not None:
                    live_ipc_writer.write(jb)
        if live_ipc_bgr_writer is not None:
            live_ipc_bgr_writer.write_bgr_ndarray(stream)

    def _publish_live_outputs_owned(vis_bgr: np.ndarray) -> None:
        _publish_live_outputs(vis_bgr.copy())

    print("Summary prints when the video ends or you press q.")
    all_probs: list[float] = []
    frame_max_probs: list[float] = []
    frame_count = 0
    bad_frame_count = 0
    v4l2_reopen_every = 90

    def _try_reopen_v4l2() -> bool:
        nonlocal v4l2_cap, bad_frame_count
        if not bool(getattr(args, "thermal_v4l2", False)):
            return False
        try:
            from media.capture import V4L2ThermalCapture
        except Exception as exc:
            print(f"Warning: thermal V4L2 reopen unavailable: {exc}", flush=True)
            return False
        try:
            if v4l2_cap is not None:
                v4l2_cap.close()
        except Exception:
            pass
        try:
            v4l2_cap = V4L2ThermalCapture(
                v4l2_path,
                width=int(args.capture_width),
                height=int(args.capture_height),
                fps=float(args.capture_fps),
            )
            aw_mmap, ah_mmap, fcc_mmap = v4l2_cap.negotiated
            print(
                f"Thermal V4L2 reopened {v4l2_path}: {aw_mmap}x{ah_mmap} @ {float(args.capture_fps):.2f} fps; "
                f"FOURCC '{fcc_mmap}'",
                flush=True,
            )
            bad_frame_count = 0
            return True
        except Exception as exc:
            print(f"Warning: thermal V4L2 reopen failed: {exc}", flush=True)
            v4l2_cap = None
            return False
    writer: cv2.VideoWriter | None = None
    infer_every_n = max(1, int(getattr(args, "live_infer_stride", 1) or 1))
    last_infer_frame = 0
    if infer_every_n > 1 and live_frame_poll_path is None and not is_batch_file:
        print(
            f"Live infer stride: run detection every {infer_every_n} frame(s).",
            flush=True,
        )
    cached_rows: list[tuple[int, int, int, int, float, int | None, str, float]] = []
    cached_gun_boxes: list[tuple[int, int, int, int, str, str, float, int, str]] = []
    cached_gun_count = 0
    last_frame_ts: float | None = None
    fps_ema: float | None = None
    infer_future: concurrent.futures.Future | None = None
    live_metrics_every_n = max(1, int(getattr(args, "live_metrics_every_n", 0) or 0))
    if int(getattr(args, "live_metrics_every_n", 0) or 0) <= 0:
        live_metrics_every_n = 15 if (live_frame_poll_path is None and not is_batch_file) else 1
    if args.live_metrics_json is not None and live_metrics_every_n > 1:
        print(f"Live metrics JSON: write every {live_metrics_every_n} frame(s).", flush=True)
    out_fps_tgt = float(getattr(args, "output_fps", 0) or 0)
    writer_fps_override: float | None = None
    decode_stride = 1
    if out_fps_tgt > 0:
        writer_fps_override = out_fps_tgt
        if is_batch_file:
            src_fps_b = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if src_fps_b >= 1.0:
                decode_stride = max(1, int(round(src_fps_b / out_fps_tgt)))
            print(
                f"Output target {out_fps_tgt:g} fps: writer {writer_fps_override:g} fps, "
                f"decode stride {decode_stride} (source reports {src_fps_b:.2f} fps).",
                flush=True,
            )
        else:
            print(
                f"Output writer fps set to {out_fps_tgt:g} (live capture; no decode subsampling).",
                flush=True,
            )
    _bt_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap is not None else 0.0
    if _bt_fps < 1.0 and v4l2_cap is not None:
        _bt_fps = float(args.capture_fps)
    if _bt_fps < 1.0:
        if live_frame_poll_path is not None or isinstance(src, int) or is_linux_video_node:
            _bt_fps = float(args.capture_fps)
        else:
            _bt_fps = 30.0
    if float(getattr(args, "byte_track_fps", 0.0) or 0.0) >= 1.0:
        _bt_fps = float(args.byte_track_fps)
    byte_tracker: ThermalByteTracker | None = None
    gun_byte_tracker: IndexedBoxByteTracker | None = None
    if bool(args.byte_track):
        byte_tracker = ThermalByteTracker(
            ByteTrackConfig(frame_rate=_bt_fps, track_buffer=int(args.byte_track_buffer))
        )
        msg = (
            f"ByteTrack on (Person<id>, buffer={int(args.byte_track_buffer)}, timing_fps≈{_bt_fps:.2f})"
        )
        if bool(args.byte_track_firearms):
            gun_byte_tracker = IndexedBoxByteTracker(
                ByteTrackConfig.for_firearms(
                    frame_rate=_bt_fps, track_buffer=int(args.byte_track_buffer)
                )
            )
            msg += "; firearm ByteTrack (arms only after first weapon box; object warm-up has no gun<N>)"
            gf = max(0, int(getattr(args, "byte_track_firearm_ghost_frames", 0) or 0))
            if gf > 0:
                msg += f", ghost hold up to {gf} frame(s) on dropout"
        else:
            msg += "; firearm ByteTrack off (--no-byte_track_firearms)"
        print(msg + ".", flush=True)
    _ghost_cfg_fm = max(0, int(getattr(args, "byte_track_firearm_ghost_frames", 0) or 0))
    _stable_miss = max(75, _ghost_cfg_fm * 6) if _ghost_cfg_fm > 0 else 75
    gun_stable = GunStableIdTracker(iou_threshold=0.15, max_missed_frames=int(_stable_miss))
    person_tid_display = DisplayTrackIds()
    person_armed = PersonArmedLatch(
        confirm_weapon_seconds=0.0,
        confirm_weapon_frames=1,
        confirm_break_grace_seconds=0.15,
        unlatch_object_frames=18,
    )
    weapon_person_assoc: WeaponPersonAssociator | None = (
        WeaponPersonAssociator(
            velocity_weight=float(getattr(args, "assoc_velocity_weight", 0.35)),
            min_speed_px=float(getattr(args, "assoc_min_speed", 1.5)),
        )
        if bool(getattr(args, "assoc_motion", True))
        else None
    )
    gun_carry_state: dict[int, tuple[tuple[int, int, int, int], float, int]] = {}
    firearm_mot_armed = False
    analysis_reporter: ClipAnalysisReporter | None = None
    if bool(args.analysis_report):
        _ar_base = args.analysis_report_path
        if _ar_base is None:
            _ar_base = default_analysis_base(args.output, str(src))
        if _ar_base is not None:
            analysis_reporter = ClipAnalysisReporter(
                Path(_ar_base),
                source=str(src),
                fps=_bt_fps,
                unsafe_threshold=float(args.unsafe_threshold),
                flush_every=int(args.analysis_report_flush_every),
            )
            print(
                f"Analysis report: {analysis_reporter._json_path.with_suffix('')}"
                f" (.json / .txt / .log)",
                flush=True,
            )
    _poll_last_bytes: bytes | None = None
    _poll_last_mtime_ns: int | None = None

    def _read_live_frame_poll() -> tuple[bool, np.ndarray | None]:
        nonlocal _poll_last_bytes, _poll_last_mtime_ns
        if live_frame_poll_path is None:
            return False, None
        if not live_frame_poll_path.is_file():
            sleep(0.05)
            return False, None
        try:
            st = live_frame_poll_path.stat()
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
            if _poll_last_mtime_ns is not None and mtime_ns == _poll_last_mtime_ns:
                sleep(0.02)
                return False, None
            raw = live_frame_poll_path.read_bytes()
        except OSError:
            sleep(0.05)
            return False, None
        if not raw or (raw == _poll_last_bytes and _poll_last_mtime_ns is not None):
            sleep(0.02)
            return False, None
        if not is_valid_jpeg(raw):
            sleep(0.02)
            return False, None
        _poll_last_bytes = raw
        _poll_last_mtime_ns = mtime_ns
        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size == 0:
            return False, None
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return False, None
        return True, bgr

    try:
        with torch.no_grad():
            while True:
                if live_frame_poll_path is not None:
                    ok, bgr_full = _read_live_frame_poll()
                    if not ok or bgr_full is None:
                        continue
                elif live_capture is not None:
                    if live_frame_buf is None:
                        cap_shape = live_capture.shape
                        if cap_shape is None:
                            sleep(0.005)
                            continue
                        live_frame_buf = np.empty(cap_shape, dtype=np.uint8)
                    cap_seq = live_capture.copy_latest_into(live_frame_buf)
                    if cap_seq < 0:
                        sleep(0.002)
                        continue
                    if cap_seq == last_capture_seq:
                        sleep(0.001)
                        continue
                    last_capture_seq = cap_seq
                    bgr_full = live_frame_buf
                    ok = True
                elif v4l2_cap is not None:
                    bgr_full = v4l2_cap.read()
                    ok = bgr_full is not None
                    if not ok:
                        bad_frame_count += 1
                        if bad_frame_count <= 3 or bad_frame_count % 60 == 0:
                            print(
                                f"Warning: thermal V4L2 mmap miss (count={bad_frame_count}); retrying…",
                                flush=True,
                            )
                        if bad_frame_count > 0 and bad_frame_count % v4l2_reopen_every == 0:
                            _try_reopen_v4l2()
                        sleep(0.08)
                        continue
                else:
                    ok, bgr_full = cap.read()
                    if not ok:
                        if bool(getattr(args, "thermal_v4l2", False)):
                            bad_frame_count += 1
                            if bad_frame_count <= 3 or bad_frame_count % 60 == 0:
                                print(
                                    f"Warning: thermal capture miss (count={bad_frame_count}); retrying…",
                                    flush=True,
                                )
                            sleep(0.08)
                            continue
                        break
                if decode_stride > 1 and cap is not None and live_capture is None:
                    for _ in range(decode_stride - 1):
                        if not cap.grab():
                            break
                frame_count += 1
                if args.max_frames and frame_count > args.max_frames:
                    break

                if args.composite_mode:
                    thermal_raw = _extract_thermal_column(bgr_full, args.thermal_panel)
                else:
                    thermal_raw = bgr_full

                if bool(getattr(args, "thermal_v4l2", False)):
                    pair = _thermal_capture_to_infer_and_vis(thermal_raw)
                    if pair is None:
                        bad_frame_count += 1
                        if bad_frame_count <= 3 or bad_frame_count % 30 == 0:
                            shp = getattr(thermal_raw, "shape", None)
                            print(
                                f"Warning: skipped frame {frame_count} due to unsupported thermal shape={shp}.",
                                flush=True,
                            )
                        continue
                    thermal, vis = pair
                    bad_frame_count = 0
                else:
                    thermal = _frame_to_bgr_for_infer(thermal_raw)
                    if thermal is None:
                        bad_frame_count += 1
                        if bad_frame_count <= 3 or bad_frame_count % 30 == 0:
                            shp = getattr(thermal_raw, "shape", None)
                            print(
                                f"Warning: skipped frame {frame_count} due to unsupported frame shape={shp}.",
                                flush=True,
                            )
                        continue
                    if vis_buf is None or vis_buf.shape != thermal.shape:
                        vis_buf = np.empty_like(thermal)
                    np.copyto(vis_buf, thermal)
                    vis = vis_buf
                h, w = vis.shape[:2]
                if infer_pool is not None:
                    if infer_future is not None and infer_future.done():
                        try:
                            rows_new, gun_boxes_new, gun_count_new, probs = infer_future.result()
                            cached_rows = list(rows_new)
                            cached_gun_boxes = list(gun_boxes_new)
                            cached_gun_count = int(gun_count_new)
                            if probs:
                                all_probs.extend(probs)
                                frame_max_probs.append(max(probs))
                            last_infer_frame = frame_count
                        except Exception as exc:
                            print(f"Warning: async inference failed on frame {frame_count}: {exc}", flush=True)
                        infer_future = None

                    if infer_future is None and (frame_count - last_infer_frame) >= infer_every_n:
                        infer_frame = live_capture.snapshot() if live_capture is not None else thermal.copy()
                        if _frame_valid_for_infer(infer_frame):
                            infer_future = infer_pool.submit(_infer_annotations_for_frame, infer_frame)
                else:
                    if (frame_count - last_infer_frame) >= infer_every_n:
                        try:
                            rows_new, gun_boxes_new, gun_count_new, probs = _infer_annotations_for_frame(
                                thermal.copy()
                            )
                            cached_rows = list(rows_new)
                            cached_gun_boxes = list(gun_boxes_new)
                            cached_gun_count = int(gun_count_new)
                            if probs:
                                all_probs.extend(probs)
                                frame_max_probs.append(max(probs))
                            last_infer_frame = frame_count
                        except Exception as exc:
                            print(f"Warning: sync inference failed on frame {frame_count}: {exc}", flush=True)

                rows = list(cached_rows)
                gun_count = int(cached_gun_count)
                row_track: dict[int, int] = {}
                if byte_tracker is not None:
                    row_track = byte_tracker.update(rows, (h, w), thermal)
                gun_track: dict[int, int] = {}
                ghost_draw_specs: list[tuple[int, int, int, int, int]] = []
                ghost_max = max(0, int(getattr(args, "byte_track_firearm_ghost_frames", 0) or 0))
                draw_ghost = bool(getattr(args, "byte_track_firearm_draw_ghost", True))

                if weapon_person_assoc is not None and cached_gun_boxes:
                    _persons_assoc: list[tuple[str, tuple[int, int, int, int]]] = []
                    _key_to_ridx: dict[str, int] = {}
                    for _ridx, _r in enumerate(rows):
                        if _r[5] not in (None, 0):
                            continue
                        _k = _person_key_for_row(_ridx, row_track, person_tid_display)
                        if not _k:
                            continue
                        _persons_assoc.append(
                            (_k, (int(_r[0]), int(_r[1]), int(_r[2]), int(_r[3])))
                        )
                        _key_to_ridx[_k] = _ridx
                    _guns_assoc = [
                        (gi, (int(g[0]), int(g[1]), int(g[2]), int(g[3])))
                        for gi, g in enumerate(cached_gun_boxes)
                    ]
                    _owner_keys = weapon_person_assoc.associate(_persons_assoc, _guns_assoc)
                    _reowned = list(cached_gun_boxes)
                    for gi, g in enumerate(cached_gun_boxes):
                        _rid = _key_to_ridx.get(_owner_keys.get(gi, ""), -1)
                        if _rid >= 0 and _rid != _gun_box_person_ridx(g):
                            _reowned[gi] = tuple(g[:7]) + (int(_rid),) + tuple(g[8:])
                    cached_gun_boxes = _reowned

                any_weapon_firearm = any(str(g[5]) == "weapon" for g in cached_gun_boxes)
                firearm_dets_stable = [
                    (
                        int(g[0]),
                        int(g[1]),
                        int(g[2]),
                        int(g[3]),
                        str(g[5]),
                        _person_key_for_row(_gun_box_person_ridx(g), row_track, person_tid_display),
                    )
                    for g in cached_gun_boxes
                ]
                stable_gun_ids = gun_stable.update(firearm_dets_stable)

                person_object_peak: dict[str, float] = {}
                visible_weapon_peak: dict[str, float] = {}
                for gidx, g in enumerate(cached_gun_boxes):
                    sid = stable_gun_ids.get(gidx)
                    stable_pk = gun_stable.person_key_for_sid(sid) if sid is not None else ""
                    _pk = stable_pk or _person_key_for_row(
                        _gun_box_person_ridx(g), row_track, person_tid_display
                    )
                    if not _pk:
                        continue
                    kind = str(g[5])
                    conf = float(g[6])
                    if kind == "weapon":
                        visible_weapon_peak[_pk] = max(visible_weapon_peak.get(_pk, 0.0), conf)
                    else:
                        person_object_peak[_pk] = max(person_object_peak.get(_pk, 0.0), conf)

                if gun_byte_tracker is None:
                    gun_carry_state.clear()
                    if any_weapon_firearm and not firearm_mot_armed:
                        firearm_mot_armed = True
                else:
                    if not firearm_mot_armed:
                        if not any_weapon_firearm:
                            _ = gun_byte_tracker.update(
                                [], (h, w), thermal, yolo_cls=0.0, min_iou_match=0.025
                            )
                            gun_carry_state.clear()
                        else:
                            firearm_mot_armed = True
                            gun_byte_tracker.reset()
                            gun_carry_state.clear()
                    if firearm_mot_armed:
                        if ghost_max <= 0:
                            gun_carry_state.clear()
                            gun_entries = [
                                (int(g[0]), int(g[1]), int(g[2]), int(g[3]), float(g[6]))
                                for g in cached_gun_boxes
                            ]
                            gun_track_ext = gun_byte_tracker.update(
                                gun_entries, (h, w), thermal, yolo_cls=0.0, min_iou_match=0.025
                            )
                            gun_track = {
                                i: gun_track_ext[i]
                                for i in gun_track_ext
                                if i < len(gun_entries)
                            }
                        else:
                            raw_entries = [
                                (int(g[0]), int(g[1]), int(g[2]), int(g[3]), float(g[6]))
                                for g in cached_gun_boxes
                            ]
                            extended = list(raw_entries)
                            for tid in sorted(gun_carry_state.keys()):
                                xyxy, _lc, miss = gun_carry_state[tid]
                                if 0 < miss <= ghost_max:
                                    extended.append((*xyxy, float(_FIREARM_GHOST_CONF)))
                            gun_track_ext = gun_byte_tracker.update(
                                extended, (h, w), thermal, yolo_cls=0.0, min_iou_match=0.025
                            )
                            n_raw = len(raw_entries)
                            gun_track = {
                                i: gun_track_ext[i] for i in gun_track_ext if i < n_raw
                            }
                            if draw_ghost:
                                j = 0
                                for tid in sorted(gun_carry_state.keys()):
                                    xyxy, _lc, miss = gun_carry_state[tid]
                                    if not (0 < miss <= ghost_max):
                                        continue
                                    ext_i = n_raw + j
                                    j += 1
                                    if ext_i in gun_track_ext:
                                        rtid = int(gun_track_ext[ext_i])
                                        x1, y1, x2, y2 = xyxy
                                        ghost_draw_specs.append((x1, y1, x2, y2, rtid))
                            matched_real: set[int] = set()
                            for gidx in range(n_raw):
                                if gidx in gun_track_ext:
                                    tid = int(gun_track_ext[gidx])
                                    matched_real.add(tid)
                                    x1, y1, x2, y2, cf = raw_entries[gidx]
                                    gun_carry_state[tid] = ((x1, y1, x2, y2), float(cf), 0)
                            for tid in list(gun_carry_state.keys()):
                                if tid in matched_real:
                                    continue
                                xyxy, lc, miss = gun_carry_state[tid]
                                miss += 1
                                if miss > ghost_max:
                                    del gun_carry_state[tid]
                                else:
                                    gun_carry_state[tid] = (xyxy, lc, miss)

                if ghost_draw_specs and draw_ghost:
                    _ov_person, _ov_gun_obj, _ov_gun_wpn, _ov_label_thick, _ov_scale, _ov_unsafe_border = (
                        _overlay_draw_style(args)
                    )
                    for gx1, gy1, gx2, gy2, _rtid in ghost_draw_specs:
                        cv2.rectangle(
                            vis,
                            (gx1, gy1),
                            (gx2, gy2),
                            _COLOR_GUN_GHOST_BGR,
                            max(1, _ov_gun_obj - 1),
                            lineType=cv2.LINE_AA,
                        )
                else:
                    _ov_person, _ov_gun_obj, _ov_gun_wpn, _ov_label_thick, _ov_scale, _ov_unsafe_border = (
                        _overlay_draw_style(args)
                    )

                for ridx, (_x1, _y1, _x2, _y2, prob, cid, _ytag, _det_c) in enumerate(rows):
                    if cid not in (None, 0) or ridx not in row_track:
                        continue
                    _pk = _person_key_for_row(ridx, row_track, person_tid_display)
                    gun_stable.reconcile_tentative_weapon(_pk)
                    person_armed.update(
                        _pk,
                        visible_weapon_conf=visible_weapon_peak.get(_pk, 0.0),
                        track_confirmed_weapon=gun_stable.person_weapon_confirmed(_pk),
                        track_object_majority=gun_stable.person_track_object_majority(_pk),
                        ever_had_weapon=gun_stable.person_ever_had_weapon(_pk),
                        instant_arm_threshold=float(args.unsafe_threshold),
                        min_weapon_conf=float(args.gun_label_weapon_min),
                    )

                if analysis_reporter is not None:
                    _ar_unsafe = float(args.unsafe_threshold)
                    _ar_safe = float(args.safe_max)
                    _ar_persons: list[dict] = []
                    for ridx, (_x1, _y1, _x2, _y2, prob, cid, _ytag, _det_c) in enumerate(rows):
                        if cid not in (None, 0) or ridx not in row_track:
                            continue
                        _pk = _person_key_for_row(ridx, row_track, person_tid_display)
                        _pnum = person_tid_display.display_num(row_track[ridx])
                        _eff = person_armed.effective_gun_conf(_pk, float(prob))
                        _armed = person_armed.is_armed(_pk)
                        _vis_w = visible_weapon_peak.get(_pk, 0.0) > 0.0
                        _bucket = _threat_bucket(_eff, _ar_unsafe, _ar_safe)
                        _ar_persons.append(
                            {
                                "display": _person_public_name(_pnum),
                                "person_key": _pk,
                                "gun_conf": float(_eff),
                                "object_gun_conf": float(person_object_peak.get(_pk, 0.0)),
                                "weapon_gun_conf": float(visible_weapon_peak.get(_pk, 0.0)),
                                "bucket": _bucket,
                                "armed": _armed,
                                "armed_concealed": _armed and not _vis_w,
                                "visible_weapon": _vis_w,
                            }
                        )
                    _ar_firearms: list[dict] = []
                    for gidx, g in enumerate(cached_gun_boxes):
                        _sid = stable_gun_ids.get(gidx)
                        _tag = gun_stable.display_tag(_sid) if _sid is not None else None
                        _pridx = _gun_box_person_ridx(g)
                        _ar_firearms.append(
                            {
                                "display_tag": _tag,
                                "person": _person_key_for_row(_pridx, row_track, person_tid_display),
                                "kind": str(g[5]),
                                "majority_kind": (
                                    gun_stable.majority_kind(_sid)
                                    if _sid is not None
                                    else str(g[5])
                                ),
                                "conf": float(g[6]),
                            }
                        )
                    _ar_fmax = 0.0
                    for ridx, (_x1, _y1, _x2, _y2, prob, cid, _ytag, _det_c) in enumerate(rows):
                        if cid not in (None, 0) or ridx not in row_track:
                            continue
                        _pk = _person_key_for_row(ridx, row_track, person_tid_display)
                        _ar_fmax = max(
                            _ar_fmax,
                            person_armed.effective_gun_conf(_pk, float(prob)),
                        )
                    analysis_reporter.on_frame(
                        frame_count,
                        persons=_ar_persons,
                        firearms=_ar_firearms,
                        frame_max_gun_conf=_ar_fmax,
                    )

                for gidx, g in enumerate(cached_gun_boxes):
                    gx1, gy1, gx2, gy2 = int(g[0]), int(g[1]), int(g[2]), int(g[3])
                    glabel, gkind, _gconf = str(g[4]), str(g[5]), float(g[6])
                    sid = stable_gun_ids.get(gidx)
                    track_kind = gkind
                    if sid is not None:
                        mk = gun_stable.majority_kind(sid)
                        if mk is not None:
                            track_kind = mk
                    gcolor = _COLOR_GUN_WEAPON_BGR if track_kind == "weapon" else _COLOR_GUN_OBJECT_BGR
                    gthick = _ov_gun_wpn if track_kind == "weapon" else _ov_gun_obj
                    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), gcolor, gthick)
                    _draw_label_above_box(
                        vis,
                        gx1,
                        gy1,
                        f"{track_kind} {_overlay_score(_gconf)}",
                        gcolor,
                        scale=_ov_scale,
                        thickness=_ov_label_thick,
                    )

                unsafe_thr = float(args.unsafe_threshold)
                unsafe_first: list[tuple[int, int, int, int, float, str]] = []
                armed_concealed_list: list[tuple[int, int, int, int, float, str]] = []
                safe_list: list[tuple[int, int, int, int, float, str]] = []
                for ridx, (x1, y1, x2, y2, prob, cid, ytag, _det_c) in enumerate(rows):
                    if cid not in (None, 0) or ridx not in row_track:
                        continue
                    _pk = _person_key_for_row(ridx, row_track, person_tid_display)
                    _pnum = person_tid_display.display_num(row_track[ridx])
                    eff_prob = person_armed.effective_gun_conf(_pk, float(prob))
                    armed = person_armed.is_armed(_pk)
                    vis_w = visible_weapon_peak.get(_pk, 0.0) > 0.0
                    bucket = _threat_bucket(eff_prob, unsafe_thr)
                    weapon_bracket = _person_weapon_bracket(_pk, cached_gun_boxes, row_track, person_tid_display) if armed else ""
                    prefix = f"{ytag} " if args.show_yolo_name else ""
                    label_txt = prefix + _person_overlay_label(
                        _pnum,
                    )
                    if armed and vis_w:
                        unsafe_first.append((x1, y1, x2, y2, eff_prob, label_txt))
                    elif armed:
                        armed_concealed_list.append((x1, y1, x2, y2, eff_prob, label_txt))
                    elif bucket == "unsafe":
                        unsafe_first.append((x1, y1, x2, y2, eff_prob, label_txt))
                    else:
                        safe_list.append((x1, y1, x2, y2, eff_prob, label_txt))

                for x1, y1, x2, y2, _prob, label_txt in safe_list:
                    c = (0, 220, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), c, _ov_person)
                    _draw_label_above_box(
                        vis,
                        x1,
                        y1,
                        label_txt,
                        c,
                        scale=_ov_scale,
                        thickness=_ov_label_thick,
                    )

                for x1, y1, x2, y2, _prob, label_txt in armed_concealed_list:
                    c = _COLOR_PERSON_ARMED_CONCEALED_BGR
                    thick = max(_ov_person, _ov_unsafe_border)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), c, thick)
                    _draw_label_above_box(
                        vis,
                        x1,
                        y1,
                        label_txt,
                        c,
                        scale=_ov_scale,
                        thickness=_ov_label_thick,
                    )

                for x1, y1, x2, y2, _prob, label_txt in unsafe_first:
                    armed_visible = "(Armed)" in label_txt
                    if armed_visible:
                        c = _COLOR_PERSON_ARMED_BGR
                        text_c = _COLOR_PERSON_ARMED_BGR
                    else:
                        c = _unsafe_bgr
                        text_c = _unsafe_text_bgr
                    cv2.rectangle(
                        vis,
                        (x1, y1),
                        (x2, y2),
                        c,
                        thickness=max(1, _ov_unsafe_border),
                    )
                    _draw_label_above_box(
                        vis,
                        x1,
                        y1,
                        label_txt,
                        text_c,
                        scale=_ov_scale,
                        thickness=_ov_label_thick,
                    )

                now_ts = time()
                if last_frame_ts is not None:
                    dt = max(1e-6, now_ts - last_frame_ts)
                    inst_fps = 1.0 / dt
                    if fps_ema is None:
                        fps_ema = inst_fps
                    else:
                        fps_ema = (0.85 * fps_ema) + (0.15 * inst_fps)
                last_frame_ts = now_ts
                if bool(getattr(args, "show_fps", True)) and fps_ema is not None:
                    fps_txt = f"FPS {fps_ema:.1f}"
                    (tw, th), _ = cv2.getTextSize(
                        fps_txt, cv2.FONT_HERSHEY_SIMPLEX, _OVERLAY_SCALE_STATUS, _OVERLAY_THICK_STATUS
                    )
                    tx = max(8, w - tw - 12)
                    ty = max(th + 8, 24)
                    cv2.putText(
                        vis,
                        fps_txt,
                        (tx + 1, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        _OVERLAY_SCALE_STATUS,
                        (0, 0, 0),
                        _OVERLAY_THICK_STATUS + 1,
                    )
                    cv2.putText(
                        vis,
                        fps_txt,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        _OVERLAY_SCALE_STATUS,
                        (255, 255, 255),
                        _OVERLAY_THICK_STATUS,
                    )

                if args.output is not None:
                    if writer is None:
                        if writer_fps_override is not None:
                            fps = float(writer_fps_override)
                        else:
                            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap is not None else float(args.capture_fps)
                            if fps < 1.0:
                                fps = 30.0
                        hh, ww = vis.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (ww, hh))
                        if not writer.isOpened():
                            raise SystemExit(f"Cannot open VideoWriter for {args.output}")
                    writer.write(vis)

                panel_w = int(getattr(args, "panel_w", 0) or 0)
                panel_h = int(getattr(args, "panel_h", 0) or 0)
                vis_out = vis
                if panel_w > 0 and panel_h > 0:
                    vh, vw = vis.shape[:2]
                    if vw != panel_w or vh != panel_h:
                        if use_gpu_preprocess and is_live_capture:
                            vis_out = resize_bgr_to(vis, panel_w, panel_h, use_gpu=True)
                        else:
                            if panel_out_buf is None or panel_out_buf.shape[:2] != (panel_h, panel_w):
                                panel_out_buf = np.empty((panel_h, panel_w, 3), dtype=np.uint8)
                            cv2.resize(vis, (panel_w, panel_h), dst=panel_out_buf, interpolation=cv2.INTER_LINEAR)
                            vis_out = panel_out_buf

                if (
                    args.live_jpg is not None
                    or live_ipc_writer is not None
                    or live_ipc_bgr_writer is not None
                ):
                    if publish_pool is not None:
                        publish_pool.submit(_publish_live_outputs_owned, vis_out)
                    else:
                        _publish_live_outputs(vis_out)

                if args.live_metrics_json is not None and (
                    frame_count % live_metrics_every_n == 0 or frame_count == 1
                ):
                    person_rows = [r for r in rows if r[5] in (None, 0)]
                    persons_total = int(len(person_rows))
                    persons_with_gun = int(sum(1 for r in person_rows if float(r[4]) >= unsafe_thr))
                    gun_detected = bool(persons_with_gun > 0 or gun_count > 0)
                    person_peak = max((float(r[4]) for r in person_rows), default=0.0)
                    risk_score = person_peak
                    unsafe_pct = (
                        round(100.0 * float(persons_with_gun) / float(persons_total), 1)
                        if persons_total > 0
                        else 0.0
                    )
                    if persons_total <= 0:
                        if args.gun_only:
                            prediction = "armed" if gun_count > 0 else "clear"
                        else:
                            prediction = "no_person"
                    elif persons_with_gun > 0 or gun_count > 0:
                        prediction = "armed"
                    else:
                        prediction = _threat_bucket(risk_score, unsafe_thr)
                    object_gun_peak = max(person_object_peak.values(), default=0.0)
                    weapon_gun_peak = max(visible_weapon_peak.values(), default=0.0)
                    payload = {
                        "ts": time(),
                        "frame": int(frame_count),
                        "infer_fps": round(float(fps_ema), 2) if fps_ema is not None else None,
                        "unsafe_score": round(float(risk_score), 4),
                        "unsafe_pct": unsafe_pct,
                        "unsafe_threshold": unsafe_thr,
                        "gun_detected": gun_detected,
                        "object_gun_peak": round(float(object_gun_peak), 4),
                        "weapon_gun_peak": round(float(weapon_gun_peak), 4),
                        "persons_with_gun": persons_with_gun,
                        "persons_total": persons_total,
                        "prediction": prediction,
                        "mmwave_torso_score": None,
                        "byte_tracks": [
                            {
                                "track_id": int(row_track[ridx]),
                                "display_id": int(person_tid_display.display_num(row_track[ridx])),
                                "row": int(ridx),
                                "threat": round(float(rows[ridx][4]), 4),
                                "object_gun_conf": round(
                                    float(person_object_peak.get(
                                        _person_key_for_row(ridx, row_track, person_tid_display), 0.0
                                    )),
                                    4,
                                ),
                                "weapon_gun_conf": round(
                                    float(visible_weapon_peak.get(
                                        _person_key_for_row(ridx, row_track, person_tid_display), 0.0
                                    )),
                                    4,
                                ),
                                "bucket": _threat_bucket(float(rows[ridx][4]), unsafe_thr),
                            }
                            for ridx in sorted(row_track)
                        ],
                        "firearm_tracks": [
                            {
                                "track_id": int(gun_track[gidx]) if gidx in gun_track else -1,
                                "stable_sid": int(stable_gun_ids.get(gidx, -1)),
                                "display_tag": (
                                    gun_stable.display_tag(int(stable_gun_ids[gidx]))
                                    if gidx in stable_gun_ids
                                    else None
                                ),
                                "majority_kind": (
                                    gun_stable.majority_kind(int(stable_gun_ids[gidx]))
                                    if gidx in stable_gun_ids
                                    else str(cached_gun_boxes[gidx][5])
                                ),
                                "idx": int(gidx),
                                "conf": round(float(cached_gun_boxes[gidx][6]), 4),
                                "kind": str(cached_gun_boxes[gidx][5]),
                            }
                            for gidx in range(len(cached_gun_boxes))
                        ],
                    }
                    write_live_metrics_json(args.live_metrics_json, payload)

                pg_jpg = getattr(args, "playground_jpg", None)
                pg_json = getattr(args, "playground_json", None)
                if pg_jpg is not None or pg_json is not None:
                    if pg_jpg is not None:
                        pg_jpg = pg_jpg.expanduser().resolve()
                        pg_jpg.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(pg_jpg), vis_out)
                    if pg_json is not None:
                        person_rows = [r for r in rows if r[5] in (None, 0)]
                        persons_total = int(len(person_rows))
                        persons_with_gun = int(sum(1 for r in person_rows if float(r[4]) >= unsafe_thr))
                        gun_detected = bool(persons_with_gun > 0 or gun_count > 0)
                        person_peak = max((float(r[4]) for r in person_rows), default=0.0)
                        risk_score = person_peak
                        unsafe_pct = (
                            round(100.0 * float(persons_with_gun) / float(persons_total), 1)
                            if persons_total > 0
                            else 0.0
                        )
                        if persons_total <= 0:
                            if args.gun_only:
                                prediction = "armed" if gun_count > 0 else "clear"
                            else:
                                prediction = "no_person"
                        elif persons_with_gun > 0 or gun_count > 0:
                            prediction = "armed"
                        else:
                            prediction = _threat_bucket(risk_score, unsafe_thr)
                        object_gun_peak = max(person_object_peak.values(), default=0.0)
                        weapon_gun_peak = max(visible_weapon_peak.values(), default=0.0)
                        pg_payload = {
                            "ts": time(),
                            "frame": int(frame_count),
                            "unsafe_score": round(float(risk_score), 4),
                            "unsafe_pct": unsafe_pct,
                            "unsafe_threshold": unsafe_thr,
                            "gun_detected": gun_detected,
                            "object_gun_peak": round(float(object_gun_peak), 4),
                            "weapon_gun_peak": round(float(weapon_gun_peak), 4),
                            "persons_with_gun": persons_with_gun,
                            "persons_total": persons_total,
                            "prediction": prediction,
                            "persons": [
                                {
                                    "x1": int(r[0]),
                                    "y1": int(r[1]),
                                    "x2": int(r[2]),
                                    "y2": int(r[3]),
                                    "threat": round(float(r[4]), 4),
                                    "yolo_class": int(r[5]) if r[5] is not None else None,
                                    "det_conf": round(float(r[7]), 4),
                                }
                                for r in person_rows
                            ],
                            "firearms": [
                                {
                                    "x1": int(g[0]),
                                    "y1": int(g[1]),
                                    "x2": int(g[2]),
                                    "y2": int(g[3]),
                                    "label": str(g[4]),
                                    "kind": str(g[5]),
                                    "conf": round(float(g[6]), 4),
                                }
                                for g in cached_gun_boxes
                            ],
                        }
                        pg_json = pg_json.expanduser().resolve()
                        pg_json.parent.mkdir(parents=True, exist_ok=True)
                        pg_json.write_text(json.dumps(pg_payload, indent=2), encoding="utf-8")
                    break

                if args.cuda_empty_cache and torch.cuda.is_available():
                    if frame_count % int(args.cuda_empty_cache_every) == 0:
                        torch.cuda.empty_cache()

                if not args.no_imshow:
                    title = "thermal | objects + threat (border = unsafe)"
                    cv2.imshow(title, vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if infer_future is not None and infer_pool is not None:
            try:
                infer_future.cancel()
            except Exception:
                pass
        if infer_pool is not None:
            infer_pool.shutdown(wait=False)
        if publish_pool is not None:
            publish_pool.shutdown(wait=True)
        if live_ipc_writer is not None:
            live_ipc_writer.close()
        if live_ipc_bgr_writer is not None:
            live_ipc_bgr_writer.close()
        if live_capture is not None:
            live_capture.stop()
        if writer is not None:
            writer.release()
        if v4l2_cap is not None:
            v4l2_cap.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if analysis_reporter is not None:
            _ar_txt = analysis_reporter.finalize()
            print(f"Analysis report written: {_ar_txt}", flush=True)
        _print_run_summary(
            str(src),
            all_probs,
            frame_count,
            frame_max_probs,
            float(args.unsafe_threshold),
            score_name,
            safe_max=float(args.safe_max),
        )


if __name__ == "__main__":
    main()
