#!/usr/bin/env python3
"""Offline Front/Back dual-video runner with Global ID + weapon sync.

Lockstep-processes two timestamp-synchronized MP4s (facing cameras ~baseline_m
apart), runs person + weapon YOLO, ByteTrack, OSNet Re-ID / GlobalIDManager in
process, and writes annotated outputs:

  front_camera_inferred.mp4
  back_camera_inferred.mp4

Example:

  python scripts/sync_runner.py \\
    --front /path/to/front_camera.mp4 \\
    --back  /path/to/back_camera.mp4 \\
    --baseline-m 5

Unlike the live API path (infer_objects ×2 + GlobalIDService JSON), this script
keeps both cameras in one process so frame N on Front matches frame N on Back.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer8_ui.sentinel_distance import depth_from_bbox_height
from weapon_ai.detection.firearms import (
    gun_overlay_class_allowed,
    normalize_firearm_class_display,
    parse_gun_overlay_classes,
    person_armed_latch_class_allowed,
)
from weapon_ai.overlay.draw import (
    COLOR_GUN_OBJECT_BGR,
    COLOR_GUN_WEAPON_BGR,
    COLOR_PERSON_ARMED_BGR,
    COLOR_PERSON_OBJECT_BGR,
    OVERLAY_RECT_GUN_OBJECT,
    OVERLAY_RECT_GUN_WEAPON,
    OVERLAY_RECT_PERSON,
    OVERLAY_SCALE_PERSON,
    OVERLAY_THICK,
    draw_label_above_box,
)
from weapon_ai.overlay.labels import person_overlay_label
from weapon_ai.reid.config import ReIDConfig
from weapon_ai.reid.embeddings import PersonReIDEmbedder, TrackEmbeddingCache
from weapon_ai.reid.global_manager import GlobalIDManager, LocalObservation
from weapon_ai.tracking import ByteTrackConfig, DisplayTrackIds, IndexedBoxByteTracker


DEFAULT_PERSON = ROOT / "trained_models" / "person_detection" / "yolov8n.engine"
if not DEFAULT_PERSON.is_file():
    DEFAULT_PERSON = ROOT / "yolov8n.pt"
DEFAULT_GUN = (
    ROOT / "trained_models" / "gun_detection" / "gun_sohas_7class_phone_black_mix_v1.pt"
)


@dataclass
class GunDet:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    kind: str  # weapon | object
    conf: float
    person_idx: int = -1


@dataclass
class CamState:
    camera_id: str
    tracker: IndexedBoxByteTracker
    display: DisplayTrackIds
    emb_cache: TrackEmbeddingCache
    # display_id -> last known armed from local gun
    local_armed: dict[int, float] = field(default_factory=dict)


def _resolve_model(path: Path | str, fallback: Path) -> Path:
    p = Path(path).expanduser() if path else fallback
    if not p.is_absolute():
        cand = (ROOT / p).resolve()
        if cand.is_file():
            return cand
    p = p.resolve()
    if not p.is_file():
        raise SystemExit(f"Model not found: {p}")
    return p


def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))
    return x1, y1, x2, y2


def _pad_roi(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int, *, frac: float, pad_px: int
) -> tuple[int, int, int, int]:
    bw, bh = x2 - x1, y2 - y1
    px = max(int(pad_px), int(round(max(bw, bh) * float(frac))))
    return _clamp_box(x1 - px, y1 - px, x2 + px, y2 + px, w, h)


def _center_in(box: tuple[int, int, int, int], person: tuple[int, int, int, int]) -> bool:
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    return person[0] <= cx <= person[2] and person[1] <= cy <= person[3]


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    return cap


def _writer_for(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(max(fps, 1.0)), size)
    if not writer.isOpened():
        raise SystemExit(f"Cannot open VideoWriter for {path}")
    return writer


def _load_yolo(path: Path, device: str):
    from ultralytics import YOLO

    model = YOLO(str(path), task="detect") if str(path).endswith(".engine") else YOLO(str(path))
    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        model.predict(source=dummy, conf=0.5, verbose=False, device=device)
    except Exception:
        pass
    return model


def detect_persons(
    model,
    frame: np.ndarray,
    *,
    conf: float,
    device: str,
    min_box_px: int,
) -> list[tuple[int, int, int, int, float]]:
    h, w = frame.shape[:2]
    res = model.predict(source=frame, conf=conf, classes=[0], verbose=False, device=device)
    out: list[tuple[int, int, int, int, float]] = []
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        return out
    xyxy = res[0].boxes.xyxy.cpu().numpy()
    confs = res[0].boxes.conf.cpu().numpy()
    for i, row in enumerate(xyxy):
        x1, y1, x2, y2 = _clamp_box(int(row[0]), int(row[1]), int(row[2]), int(row[3]), w, h)
        if (x2 - x1) < min_box_px or (y2 - y1) < min_box_px:
            continue
        out.append((x1, y1, x2, y2, float(confs[i])))
    # High-conf first for stable ByteTrack assignment
    out.sort(key=lambda t: -t[4])
    return out


def detect_guns_on_persons(
    gun_model,
    frame: np.ndarray,
    persons: list[tuple[int, int, int, int, float]],
    *,
    gun_conf: float,
    gun_imgsz: int,
    device: str,
    roi_pad_frac: float,
    roi_pad_px: int,
    gun_min_box_px: int,
    overlay_allowed: frozenset[str] | None,
    weapon_min: float = 0.55,
) -> list[GunDet]:
    if gun_model is None or not persons:
        return []
    h, w = frame.shape[:2]
    names = dict(getattr(gun_model, "names", {}) or {})
    guns: list[GunDet] = []
    for pi, (px1, py1, px2, py2, _pc) in enumerate(persons):
        rx1, ry1, rx2, ry2 = _pad_roi(
            px1, py1, px2, py2, w, h, frac=roi_pad_frac, pad_px=roi_pad_px
        )
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue
        gres = gun_model.predict(
            source=crop,
            conf=float(gun_conf),
            imgsz=int(gun_imgsz),
            verbose=False,
            device=device,
        )
        if not gres or gres[0].boxes is None or len(gres[0].boxes) == 0:
            continue
        gxy = gres[0].boxes.xyxy.cpu().numpy()
        gcf = gres[0].boxes.conf.cpu().numpy()
        gcl = gres[0].boxes.cls.cpu().numpy().astype(int)
        best: GunDet | None = None
        for i, row in enumerate(gxy):
            gx1 = int(row[0]) + rx1
            gy1 = int(row[1]) + ry1
            gx2 = int(row[2]) + rx1
            gy2 = int(row[3]) + ry1
            gx1, gy1, gx2, gy2 = _clamp_box(gx1, gy1, gx2, gy2, w, h)
            if (gx2 - gx1) < gun_min_box_px or (gy2 - gy1) < gun_min_box_px:
                continue
            raw_name = names.get(int(gcl[i]), str(int(gcl[i])))
            label = normalize_firearm_class_display(raw_name)
            if not gun_overlay_class_allowed(label, overlay_allowed):
                continue
            conf = float(gcf[i])
            kind = "weapon" if conf >= float(weapon_min) else "object"
            cand = GunDet(gx1, gy1, gx2, gy2, label, kind, conf, person_idx=pi)
            if best is None or conf > best.conf:
                best = cand
        if best is not None and _center_in(
            (best.x1, best.y1, best.x2, best.y2), (px1, py1, px2, py2)
        ):
            guns.append(best)
        elif best is not None:
            # Still keep if heavily overlapping person ROI (side draw)
            guns.append(best)
    return guns


def _downscale(frame: np.ndarray, max_w: int) -> tuple[np.ndarray, float]:
    if max_w <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame, 1.0
    scale = max_w / float(w)
    nh = max(1, int(round(h * scale)))
    small = cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_LINEAR)
    return small, scale


def process_camera_frame(
    state: CamState,
    frame: np.ndarray,
    *,
    person_model,
    gun_model,
    args: argparse.Namespace,
    overlay_allowed: frozenset[str] | None,
    now: float,
) -> tuple[
    list[tuple[int, int, int, int, int, float, float | None]],  # x1,y1,x2,y2,display_id,pconf,depth
    list[GunDet],
    list[LocalObservation],
]:
    """Detect/track one camera frame; return tracked people, guns, GlobalID observations."""
    infer_frame, scale = _downscale(frame, int(args.infer_max_width))
    persons = detect_persons(
        person_model,
        infer_frame,
        conf=float(args.person_conf),
        device=str(args.device),
        min_box_px=max(8, int(args.min_box_px * scale) if scale < 1 else args.min_box_px),
    )
    if scale != 1.0:
        inv = 1.0 / scale
        persons = [
            (
                int(round(x1 * inv)),
                int(round(y1 * inv)),
                int(round(x2 * inv)),
                int(round(y2 * inv)),
                c,
            )
            for x1, y1, x2, y2, c in persons
        ]

    h, w = frame.shape[:2]
    row_track = state.tracker.update(
        [(x1, y1, x2, y2, c) for x1, y1, x2, y2, c in persons],
        (h, w),
        frame,
    )

    guns = detect_guns_on_persons(
        gun_model,
        frame,
        persons,
        gun_conf=float(args.gun_conf),
        gun_imgsz=int(args.gun_imgsz),
        device=str(args.device),
        roi_pad_frac=float(args.gun_roi_pad_frac),
        roi_pad_px=int(args.gun_roi_pad_px),
        gun_min_box_px=int(args.gun_min_box_px),
        overlay_allowed=overlay_allowed,
        weapon_min=float(args.gun_weapon_min),
    )
    # Only keep guns whose owner is a confirmed track
    guns = [g for g in guns if g.person_idx in row_track]

    tracked: list[tuple[int, int, int, int, int, float, float | None]] = []
    observations: list[LocalObservation] = []
    armed_peak: dict[int, float] = {}

    for pi, g in enumerate(guns):
        if g.person_idx not in row_track:
            continue
        if not person_armed_latch_class_allowed(g.label, overlay_allowed):
            continue
        if g.kind != "weapon" and g.conf < float(args.gun_weapon_min):
            continue
        tid = row_track[g.person_idx]
        did = state.display.display_num(tid)
        armed_peak[did] = max(armed_peak.get(did, 0.0), float(g.conf))

    for pi, (x1, y1, x2, y2, pconf) in enumerate(persons):
        if pi not in row_track:
            continue
        tid = row_track[pi]
        did = state.display.display_num(tid)
        depth = None
        if args.person_distance:
            depth = depth_from_bbox_height(
                float(y2 - y1),
                float(h),
                person_height_m=float(args.person_height_m),
                vertical_fov_deg=float(args.person_vfov_deg),
            )
        tracked.append((x1, y1, x2, y2, did, pconf, depth))

        wconf = float(armed_peak.get(did, 0.0))
        weapon = wconf > 0.0
        if weapon:
            state.local_armed[did] = wconf
        emb = state.emb_cache.maybe_update(
            state.camera_id,
            did,
            frame,
            (x1, y1, x2, y2),
            now=now,
            force=True,
        )
        if emb is None:
            emb = state.emb_cache.representative(state.camera_id, did)
        observations.append(
            LocalObservation(
                camera_id=state.camera_id,
                local_track_id=int(did),
                embedding=emb,
                bbox=(x1, y1, x2, y2),
                depth_m=depth,
                weapon_detected=weapon,
                weapon_confidence=wconf if weapon else 0.0,
                timestamp=now,
            )
        )

    active = {t[4] for t in tracked}
    state.emb_cache.prune_missing(state.camera_id, active)
    for did in list(state.local_armed.keys()):
        if did not in active:
            state.local_armed.pop(did, None)

    return tracked, guns, observations


def draw_frame(
    frame: np.ndarray,
    tracked: list[tuple[int, int, int, int, int, float, float | None]],
    guns: list[GunDet],
    *,
    camera_id: str,
    manager: GlobalIDManager,
) -> np.ndarray:
    vis = frame.copy()
    # Guns first (under person labels)
    for g in guns:
        color = COLOR_GUN_WEAPON_BGR if g.kind == "weapon" else COLOR_GUN_OBJECT_BGR
        thick = OVERLAY_RECT_GUN_WEAPON if g.kind == "weapon" else OVERLAY_RECT_GUN_OBJECT
        cv2.rectangle(vis, (g.x1, g.y1), (g.x2, g.y2), color, thick)
        draw_label_above_box(
            vis, g.x1, g.y1, g.label, color, scale=OVERLAY_SCALE_PERSON, thickness=OVERLAY_THICK
        )

    for x1, y1, x2, y2, did, _pconf, depth in tracked:
        info = manager.overlay_for(camera_id, did)
        gid = int(info["global_id"]) if info else None
        global_weapon = bool(info and info.get("weapon_detected"))
        local_weapon = False
        for g in guns:
            if _center_in((g.x1, g.y1, g.x2, g.y2), (x1, y1, x2, y2)):
                if person_armed_latch_class_allowed(g.label, None) and g.kind == "weapon":
                    local_weapon = True
                    break

        armed = global_weapon or local_weapon
        if armed:
            # Synced or local weapon → red (matches live Global ID overlay policy)
            color = COLOR_PERSON_ARMED_BGR
        else:
            color = COLOR_PERSON_OBJECT_BGR

        label = person_overlay_label(did, distance_m=depth, global_id=gid, armed=armed)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, OVERLAY_RECT_PERSON)
        draw_label_above_box(
            vis, x1, y1, label, color, scale=OVERLAY_SCALE_PERSON, thickness=OVERLAY_THICK
        )
    return vis


def default_out_path(src: Path) -> Path:
    return src.with_name(f"{src.stem}_inferred{src.suffix or '.mp4'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--front", type=Path, required=True, help="Front camera MP4 (timestamp-synced)")
    p.add_argument("--back", type=Path, required=True, help="Back camera MP4 (timestamp-synced)")
    p.add_argument(
        "--front-out",
        type=Path,
        default=None,
        help="Default: <front_stem>_inferred.mp4 next to source",
    )
    p.add_argument(
        "--back-out",
        type=Path,
        default=None,
        help="Default: <back_stem>_inferred.mp4 next to source",
    )
    p.add_argument("--baseline-m", type=float, default=5.0, help="Corridor baseline meters")
    p.add_argument("--camera-front", default="camera_1")
    p.add_argument("--camera-back", default="camera_2")
    p.add_argument("--person-model", type=Path, default=DEFAULT_PERSON)
    p.add_argument("--gun-model", type=Path, default=DEFAULT_GUN)
    p.add_argument("--no-gun", action="store_true", help="Person + Re-ID only")
    p.add_argument("--device", default="cuda")
    p.add_argument("--person-conf", type=float, default=0.4)
    p.add_argument("--gun-conf", type=float, default=0.45)
    p.add_argument("--gun-imgsz", type=int, default=1280)
    p.add_argument("--gun-min-box-px", type=int, default=8)
    p.add_argument("--gun-roi-pad-frac", type=float, default=0.12)
    p.add_argument("--gun-roi-pad-px", type=int, default=36)
    p.add_argument("--gun-weapon-min", type=float, default=0.55)
    p.add_argument("--min-box-px", type=int, default=24)
    p.add_argument("--infer-max-width", type=int, default=1920)
    p.add_argument("--infer-stride", type=int, default=1, help="Run YOLO every N frames (1=all)")
    p.add_argument("--max-frames", type=int, default=0, help="0 = full shorter video")
    p.add_argument("--output-fps", type=float, default=0.0, help="0 = use Front source FPS")
    p.add_argument("--overlay-classes", default="gun_and_knife")
    p.add_argument("--person-distance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--person-height-m", type=float, default=1.7)
    p.add_argument("--person-vfov-deg", type=float, default=60.0)
    p.add_argument("--similarity-threshold", type=float, default=0.72)
    p.add_argument("--soft-similarity-threshold", type=float, default=0.58)
    p.add_argument("--weapon-hold-s", type=float, default=2.5)
    p.add_argument("--track-timeout-s", type=float, default=8.0)
    p.add_argument("--reid-backend", default="auto", help="auto|tensorrt|torchreid|...")
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for association summary JSON",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    front_path = args.front.expanduser().resolve()
    back_path = args.back.expanduser().resolve()
    if not front_path.is_file():
        raise SystemExit(f"--front not found: {front_path}")
    if not back_path.is_file():
        raise SystemExit(f"--back not found: {back_path}")

    front_out = (args.front_out or default_out_path(front_path)).expanduser().resolve()
    back_out = (args.back_out or default_out_path(back_path)).expanduser().resolve()

    person_model_path = _resolve_model(args.person_model, DEFAULT_PERSON)
    gun_model_path = None if args.no_gun else _resolve_model(args.gun_model, DEFAULT_GUN)
    overlay_allowed = parse_gun_overlay_classes(args.overlay_classes)

    print(f"Front : {front_path}")
    print(f"Back  : {back_path}")
    print(f"Out   : {front_out}")
    print(f"        {back_out}")
    print(f"Person model: {person_model_path}")
    print(f"Gun model   : {gun_model_path or '(disabled)'}")
    print(f"Baseline    : {args.baseline_m} m")

    person_model = _load_yolo(person_model_path, args.device)
    gun_model = _load_yolo(gun_model_path, args.device) if gun_model_path else None

    cfg = ReIDConfig(
        enable=True,
        baseline_m=float(args.baseline_m),
        similarity_threshold=float(args.similarity_threshold),
        soft_similarity_threshold=float(args.soft_similarity_threshold),
        weapon_hold_s=float(args.weapon_hold_s),
        track_timeout_s=float(args.track_timeout_s),
        camera_front=str(args.camera_front),
        camera_back=str(args.camera_back),
        embed_backend=str(args.reid_backend),
        embed_device=str(args.device),
    )
    manager = GlobalIDManager(cfg)
    embedder = PersonReIDEmbedder(
        device=str(args.device),
        backend=str(args.reid_backend),
    )
    print(f"Re-ID backend: {embedder.backend} dim={embedder.feature_dim}")

    fps_track = float(args.output_fps) if args.output_fps > 0 else 30.0
    front_state = CamState(
        camera_id=str(args.camera_front),
        tracker=IndexedBoxByteTracker(ByteTrackConfig(frame_rate=fps_track, track_buffer=45)),
        display=DisplayTrackIds(),
        emb_cache=TrackEmbeddingCache(embedder, interval_s=0.01, min_box_px=int(args.min_box_px)),
    )
    back_state = CamState(
        camera_id=str(args.camera_back),
        tracker=IndexedBoxByteTracker(ByteTrackConfig(frame_rate=fps_track, track_buffer=45)),
        display=DisplayTrackIds(),
        emb_cache=TrackEmbeddingCache(embedder, interval_s=0.01, min_box_px=int(args.min_box_px)),
    )

    cap_f = _open_video(front_path)
    cap_b = _open_video(back_path)
    src_fps_f = float(cap_f.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    src_fps_b = float(cap_b.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    n_f = int(cap_f.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_fps = float(args.output_fps) if args.output_fps > 0 else src_fps_f
    print(f"Source FPS front={src_fps_f:.2f} back={src_fps_b:.2f} → write @{out_fps:.2f}")
    print(f"Frames available front={n_f} back={n_b}")

    writer_f: cv2.VideoWriter | None = None
    writer_b: cv2.VideoWriter | None = None

    stride = max(1, int(args.infer_stride))
    max_frames = int(args.max_frames) if args.max_frames > 0 else 0
    frame_i = 0
    written = 0
    t0 = time.time()

    # Cache last annotations for stride gaps (still advance Global ID only on infer frames)
    last_f: tuple[Any, ...] | None = None
    last_b: tuple[Any, ...] | None = None

    try:
        while True:
            ok_f, frame_f = cap_f.read()
            ok_b, frame_b = cap_b.read()
            if not ok_f or not ok_b or frame_f is None or frame_b is None:
                break
            if max_frames and written >= max_frames:
                break

            do_infer = (frame_i % stride) == 0
            now = float(frame_i) / float(out_fps)

            if do_infer:
                tracked_f, guns_f, obs_f = process_camera_frame(
                    front_state,
                    frame_f,
                    person_model=person_model,
                    gun_model=gun_model,
                    args=args,
                    overlay_allowed=overlay_allowed,
                    now=now,
                )
                tracked_b, guns_b, obs_b = process_camera_frame(
                    back_state,
                    frame_b,
                    person_model=person_model,
                    gun_model=gun_model,
                    args=args,
                    overlay_allowed=overlay_allowed,
                    now=now,
                )
                # Lockstep Global ID update (both cameras same timestamp)
                manager.update_observations([*obs_f, *obs_b], now=now)
                last_f = (tracked_f, guns_f)
                last_b = (tracked_b, guns_b)
            else:
                tracked_f, guns_f = last_f if last_f else ([], [])
                tracked_b, guns_b = last_b if last_b else ([], [])

            vis_f = draw_frame(
                frame_f, tracked_f, guns_f, camera_id=front_state.camera_id, manager=manager
            )
            vis_b = draw_frame(
                frame_b, tracked_b, guns_b, camera_id=back_state.camera_id, manager=manager
            )

            if writer_f is None:
                hf, wf = vis_f.shape[:2]
                hb, wb = vis_b.shape[:2]
                writer_f = _writer_for(front_out, out_fps, (wf, hf))
                writer_b = _writer_for(back_out, out_fps, (wb, hb))

            assert writer_f is not None and writer_b is not None
            writer_f.write(vis_f)
            writer_b.write(vis_b)
            written += 1
            frame_i += 1
            if written % 50 == 0 or written == 1:
                elapsed = time.time() - t0
                rate = written / max(elapsed, 1e-6)
                print(
                    f"  frame {written}  persons_global={len(manager.persons)}  "
                    f"{rate:.1f} pair-fps",
                    flush=True,
                )
    finally:
        cap_f.release()
        cap_b.release()
        if writer_f is not None:
            writer_f.release()
        if writer_b is not None:
            writer_b.release()

    snap = manager.snapshot()
    summary = {
        "front": str(front_path),
        "back": str(back_path),
        "front_out": str(front_out),
        "back_out": str(back_out),
        "frames_written": written,
        "elapsed_s": round(time.time() - t0, 2),
        "baseline_m": float(args.baseline_m),
        "reid_backend": embedder.backend,
        "persons": snap.get("persons"),
        "association_log": snap.get("association_log"),
        "config": snap.get("config"),
    }
    summary_path = args.summary_json
    if summary_path is None:
        summary_path = front_out.with_name(front_out.stem + "_sync_summary.json")
    else:
        summary_path = summary_path.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Wrote {front_out}")
    print(f"Wrote {back_out}")
    print(f"Wrote {summary_path}")
    print(f"Frames={written}  global_persons={len(manager.persons)}  "
          f"elapsed={summary['elapsed_s']}s")
    for line in (snap.get("association_log") or [])[-10:]:
        print(" ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
