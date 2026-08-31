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
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from manual_annotations import (  # noqa: E402
    ManualAnnotationStore,
    draw_manual_frame,
    draw_manual_overlays,
    frame_for_draw,
)

from layer8_ui.sentinel_distance import depth_from_bbox_height
from weapon_ai.detection.firearms import (
    normalize_firearm_class_display,
    parse_gun_overlay_classes,
    person_armed_latch_class_allowed,
    suppress_gun_phone_conflicts,
)
from weapon_ai.overlay.draw import (
    COLOR_GUN_OBJECT_BGR,
    COLOR_GUN_WEAPON_BGR,
    COLOR_PERSON_ARMED_BGR,
    COLOR_PERSON_ARMED_CONCEALED_BGR,
    COLOR_PERSON_OBJECT_BGR,
    OVERLAY_RECT_GUN_OBJECT,
    OVERLAY_RECT_GUN_WEAPON,
    OVERLAY_RECT_PERSON,
    OVERLAY_SCALE_GUN,
    OVERLAY_THICK,
    draw_label_above_box,
)

# Larger labels for 4K dual-camera demo exports (sync_runner only).
SYNC_OVERLAY_SCALE_PERSON = 2.0
SYNC_OVERLAY_SCALE_GUN = max(OVERLAY_SCALE_GUN, 1.65)
from weapon_ai.overlay.labels import person_overlay_label
from weapon_ai.reid.config import ReIDConfig
from weapon_ai.reid.embeddings import PersonReIDEmbedder, TrackEmbeddingCache
from weapon_ai.reid.global_manager import GlobalIDManager, LocalObservation
from weapon_ai.tracking import (
    AppearanceDisplayIds,
    ByteTrackConfig,
    DisplayTrackIds,
    IndexedBoxBotSortTracker,
    IndexedBoxByteTracker,
    make_indexed_box_tracker,
)


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
    display_id: int = -1


@dataclass
class CamWeaponPolicy:
    """Per-camera weapon detection filter and confirm timing."""

    classes_allowed: frozenset[str] | None = None
    gun_conf: float = 0.65
    knife_conf: float = 0.40
    weapon_min: float = 0.65
    phone_conf: float = 0.25
    confirm_s: float = 2.5
    knife_confirm_s: float | None = None
    confirm_frames: int = 0
    confirm_break_s: float = 0.75
    draw_confirm_frames: int = 3
    draw_miss_break: int = 2
    latch_armed: bool = False
    hold_s: float = 2.5
    visible_hold_s: float = 2.5
    max_weapon_depth_m: float = 0.0


@dataclass
class CamState:
    camera_id: str
    tracker: IndexedBoxByteTracker | IndexedBoxBotSortTracker
    display: DisplayTrackIds
    emb_cache: TrackEmbeddingCache
    weapon_policy: CamWeaponPolicy
    appearance_display: AppearanceDisplayIds | None = None
    stabilize_display_ids: bool = False
    weapon_grace_until: dict[int, float] = field(default_factory=dict)
    # Time-based gun/knife confirm per local display id (phones never count).
    weapon_accum_s: dict[int, float] = field(default_factory=dict)
    knife_accum_s: dict[int, float] = field(default_factory=dict)
    weapon_hit_n: dict[int, int] = field(default_factory=dict)
    knife_last_hit: dict[int, float] = field(default_factory=dict)
    weapon_last_hit: dict[int, float] = field(default_factory=dict)
    weapon_confirmed: set[int] = field(default_factory=set)
    knife_confirmed: set[int] = field(default_factory=set)
    weapon_conf: dict[int, float] = field(default_factory=dict)
    phone_tagged: set[int] = field(default_factory=set)
    phone_veto_ids: set[int] = field(default_factory=set)
    # Demo override: these local display IDs are always unarmed (no gun boxes).
    force_safe_display_ids: set[int] = field(default_factory=set)
    # Demo override: match overlay label "Person N" when N is the global Re-ID.
    force_safe_global_ids: set[int] = field(default_factory=set)
    # Short draw latch for gun/knife boxes (phones are never drawn).
    draw_streak: dict[tuple[int, str], int] = field(default_factory=dict)
    draw_miss: dict[tuple[int, str], int] = field(default_factory=dict)
    draw_ok: set[tuple[int, str]] = field(default_factory=set)
    draw_box: dict[tuple[int, str], GunDet] = field(default_factory=dict)
    # Last time a qualifying weapon was seen (anchors the concealed/hold window).
    weapon_activity_ts: dict[int, float] = field(default_factory=dict)
    # Last raw YOLO weapon sighting — anchors the armed-red persistence window.
    weapon_visible_ts: dict[int, float] = field(default_factory=dict)
    # Local concealed: armed but red hold expired (single-camera, no dual sync required).
    concealed_latched_ids: set[int] = field(default_factory=set)


_GUN_SUPPRESS_LABELS = frozenset({"gun", "rifle", "shotgun", "long_gun"})
_FORCE_SAFE_STRIP_LABELS = frozenset({"gun", "rifle", "shotgun", "long_gun", "knife"})


def _parse_force_safe_ids(raw: str | None) -> set[int]:
    if not raw or not str(raw).strip():
        return set()
    out: set[int] = set()
    for part in str(raw).replace(" ", "").split(","):
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            raise SystemExit(f"Invalid force-safe person id (expected integers): {part!r}")
    return out


def _is_force_safe_person(
    state: CamState,
    manager: GlobalIDManager | None,
    camera_id: str,
    did: int,
) -> bool:
    if did in state.force_safe_display_ids:
        return True
    if manager is not None and state.force_safe_global_ids:
        gid = manager.get_global_id(str(camera_id), int(did))
        if gid is not None and int(gid) in state.force_safe_global_ids:
            return True
    return False


def _enforce_force_safe(state: CamState, did: int) -> None:
    """Clear armed/gun overlay state for demo-safe tracks."""
    state.weapon_confirmed.discard(did)
    state.knife_confirmed.discard(did)
    state.weapon_accum_s.pop(did, None)
    state.knife_accum_s.pop(did, None)
    state.weapon_hit_n.pop(did, None)
    state.weapon_last_hit.pop(did, None)
    state.knife_last_hit.pop(did, None)
    state.weapon_conf.pop(did, None)
    state.weapon_activity_ts.pop(did, None)
    state.weapon_visible_ts.pop(did, None)
    state.phone_veto_ids.add(did)
    state.weapon_grace_until.pop(did, None)
    for key in list(state.draw_ok) + list(state.draw_streak) + list(state.draw_box):
        if key[0] == did and key[1] in _FORCE_SAFE_STRIP_LABELS:
            state.draw_ok.discard(key)
            state.draw_streak.pop(key, None)
            state.draw_miss.pop(key, None)
            state.draw_box.pop(key, None)


def _embed_person_crop(
    state: CamState,
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if (x2 - x1) < state.emb_cache.min_box_px or (y2 - y1) < state.emb_cache.min_box_px:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return state.emb_cache.embedder.embed(crop)


def _resolve_track_display_ids(
    state: CamState,
    row_track: dict[int, int],
    persons: list[tuple[int, int, int, int, float]],
    frame: np.ndarray,
) -> dict[int, int]:
    """Map ByteTrack ``track_id`` -> stable on-screen Person N."""
    if not row_track:
        return {}
    if not state.stabilize_display_ids or state.appearance_display is None:
        return {int(tid): state.display.display_num(int(tid)) for tid in row_track.values()}

    track_items: list[tuple[int, np.ndarray | None]] = []
    seen: set[int] = set()
    for pi, tid in row_track.items():
        tid = int(tid)
        if tid in seen:
            continue
        seen.add(tid)
        x1, y1, x2, y2, _ = persons[pi]
        track_items.append((tid, _embed_person_crop(state, frame, (x1, y1, x2, y2))))
    return state.appearance_display.assign(track_items)


def _weapon_grace_s(policy: CamWeaponPolicy) -> float:
    return 3.0 if policy.latch_armed else 0.0


def _should_purge_weapon_state(state: CamState, did: int, *, now: float, policy: CamWeaponPolicy) -> bool:
    grace = _weapon_grace_s(policy)
    if grace <= 0:
        return True
    armedish = (
        did in state.weapon_confirmed
        or did in state.knife_confirmed
        or did in state.concealed_latched_ids
    )
    if armedish:
        state.weapon_grace_until[did] = max(state.weapon_grace_until.get(did, 0.0), now + grace)
    if now < state.weapon_grace_until.get(did, 0.0):
        return False
    state.weapon_grace_until.pop(did, None)
    return True


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
    weapon_min: float = 0.65,
    phone_conf: float = 0.25,
    knife_conf: float = 0.40,
    classes_allowed: frozenset[str] | None = None,
    phone_conflict_suppress: bool = True,
    phone_conflict_iou: float = 0.35,
    phone_conflict_margin: float = 0.08,
    phone_veto_gun_conf: float = 0.85,
) -> list[GunDet]:
    if gun_model is None or not persons:
        return []
    h, w = frame.shape[:2]
    names = dict(getattr(gun_model, "names", {}) or {})
    keep = frozenset({"gun", "knife", "smartphone", "rifle", "shotgun", "long_gun"})
    gun_labels = frozenset({"gun", "rifle", "shotgun", "long_gun"})
    conf_floor: list[float] = []
    if classes_allowed is None:
        conf_floor.extend([float(gun_conf), float(phone_conf), float(knife_conf)])
    else:
        if classes_allowed & gun_labels:
            conf_floor.append(float(gun_conf))
        if "knife" in classes_allowed:
            conf_floor.append(float(knife_conf))
        if "smartphone" in classes_allowed:
            conf_floor.append(float(phone_conf))
    # Always run phone head so we can veto phone→gun false positives even in gun_only/knife presets.
    if classes_allowed is not None and "smartphone" not in (classes_allowed or ()):
        conf_floor.append(float(phone_conf))
    predict_conf = min(conf_floor) if conf_floor else min(float(gun_conf), float(phone_conf), float(knife_conf))
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
            conf=float(predict_conf),
            imgsz=int(gun_imgsz),
            verbose=False,
            device=device,
        )
        if not gres or gres[0].boxes is None or len(gres[0].boxes) == 0:
            continue
        gxy = gres[0].boxes.xyxy.cpu().numpy()
        gcf = gres[0].boxes.conf.cpu().numpy()
        gcl = gres[0].boxes.cls.cpu().numpy().astype(int)
        best: dict[str, GunDet] = {}
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
            if label not in keep:
                continue
            if classes_allowed is not None and label not in classes_allowed:
                if label != "smartphone":
                    continue
            conf = float(gcf[i])
            is_phone = label == "smartphone"
            if is_phone:
                if conf < float(phone_conf):
                    continue
            elif label == "knife":
                if conf < float(knife_conf):
                    continue
            elif conf < float(gun_conf):
                continue
            if is_phone:
                kind = "object"
            elif label == "knife":
                kind = "weapon"
            else:
                kind = "weapon" if conf >= float(weapon_min) else "object"
            cand = GunDet(gx1, gy1, gx2, gy2, label, kind, conf, person_idx=pi)
            prev = best.get(label)
            if prev is None or conf > prev.conf:
                best[label] = cand
        phone_det = best.get("smartphone")
        # Always scan for smartphone (phones are often misclassified as guns).
        gres_phone = gun_model.predict(
            source=crop,
            conf=0.01,
            imgsz=int(gun_imgsz),
            verbose=False,
            device=device,
        )
        if gres_phone and gres_phone[0].boxes is not None and len(gres_phone[0].boxes) > 0:
            gcf2 = gres_phone[0].boxes.conf.cpu().numpy()
            gcl2 = gres_phone[0].boxes.cls.cpu().numpy().astype(int)
            gxy2 = gres_phone[0].boxes.xyxy.cpu().numpy()
            for i, cls_id in enumerate(gcl2):
                label2 = normalize_firearm_class_display(
                    names.get(int(cls_id), str(int(cls_id)))
                )
                if label2 != "smartphone":
                    continue
                conf2 = float(gcf2[i])
                if conf2 < float(phone_conf):
                    continue
                row = gxy2[i]
                gx1 = int(row[0]) + rx1
                gy1 = int(row[1]) + ry1
                gx2 = int(row[2]) + rx1
                gy2 = int(row[3]) + ry1
                gx1, gy1, gx2, gy2 = _clamp_box(gx1, gy1, gx2, gy2, w, h)
                cand2 = GunDet(gx1, gy1, gx2, gy2, "smartphone", "object", conf2, person_idx=pi)
                if phone_det is None or conf2 > float(phone_det.conf):
                    phone_det = cand2
                    best["smartphone"] = cand2
        phone_det = best.get("smartphone")
        if phone_det is not None and float(phone_det.conf) >= float(phone_veto_gun_conf):
            for gl in list(gun_labels):
                gun_box = best.get(gl)
                if gun_box is not None and float(gun_box.conf) < float(phone_det.conf):
                    best.pop(gl, None)
        elif phone_det is not None:
            for gl in list(gun_labels):
                gun_box = best.get(gl)
                if gun_box is not None and float(phone_det.conf) > float(gun_box.conf):
                    best.pop(gl, None)
        if best:
            items = [
                (g.conf, g.x1, g.y1, g.x2, g.y2, g.label, pi, 0) for g in best.values()
            ]
            if phone_conflict_suppress:
                items = suppress_gun_phone_conflicts(
                    items,
                    iou_thresh=float(phone_conflict_iou),
                    prefer_phone_margin=float(phone_conflict_margin),
                )
            keep_labels = {str(it[5]) for it in items}
            for label, cand in best.items():
                if label in keep_labels:
                    guns.append(cand)
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
    now: float,
    manager: GlobalIDManager | None = None,
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
    tid_to_did = _resolve_track_display_ids(state, row_track, persons, frame)

    policy = state.weapon_policy
    guns = detect_guns_on_persons(
        gun_model,
        frame,
        persons,
        gun_conf=float(policy.gun_conf),
        gun_imgsz=int(args.gun_imgsz),
        device=str(args.device),
        roi_pad_frac=float(args.gun_roi_pad_frac),
        roi_pad_px=int(args.gun_roi_pad_px),
        gun_min_box_px=int(args.gun_min_box_px),
        weapon_min=float(policy.weapon_min),
        phone_conf=float(policy.phone_conf),
        knife_conf=float(policy.knife_conf),
        classes_allowed=policy.classes_allowed,
        phone_conflict_suppress=bool(getattr(args, "gun_phone_conflict_suppress", True)),
        phone_conflict_iou=float(getattr(args, "gun_phone_conflict_iou", 0.35)),
        phone_conflict_margin=float(getattr(args, "gun_phone_conflict_margin", 0.08)),
        phone_veto_gun_conf=float(getattr(args, "phone_veto_gun_conf", 0.85)),
    )
    # Only keep guns whose owner is a confirmed track
    guns = [g for g in guns if g.person_idx in row_track]

    gun_labels = frozenset({"gun", "rifle", "shotgun", "long_gun"})
    phone_veto_conf = float(getattr(args, "phone_veto_gun_conf", 0.85))
    for g in guns:
        if g.person_idx not in row_track:
            continue
        did = tid_to_did[int(row_track[g.person_idx])]
        if g.label == "smartphone":
            state.phone_tagged.add(did)
            if float(g.conf) >= phone_veto_conf:
                state.phone_veto_ids.add(did)
    guns = [
        g
        for g in guns
        if g.person_idx not in row_track
        or g.label not in gun_labels
        or tid_to_did[int(row_track[g.person_idx])] not in state.phone_veto_ids
    ]
    if state.force_safe_display_ids or state.force_safe_global_ids:
        guns = [
            g
            for g in guns
            if g.person_idx not in row_track
            or g.label not in _FORCE_SAFE_STRIP_LABELS
            or not _is_force_safe_person(
                state, manager, state.camera_id, tid_to_did[int(row_track[g.person_idx])]
            )
        ]
    # Drop latched gun boxes for phone-confirmed tracks.
    for did in list(state.phone_veto_ids):
        for key in list(state.draw_ok):
            if key[0] == did and key[1] in gun_labels:
                state.draw_ok.discard(key)
                state.draw_streak.pop(key, None)
                state.draw_miss.pop(key, None)
                state.draw_box.pop(key, None)

    person_depth_by_pi: dict[int, float] = {}
    for pi, (x1, y1, x2, y2, _pconf) in enumerate(persons):
        if pi not in row_track:
            continue
        person_depth_by_pi[pi] = depth_from_bbox_height(
            float(y2 - y1),
            float(h),
            person_height_m=float(args.person_height_m),
            vertical_fov_deg=float(args.person_vfov_deg),
        )

    max_weapon_depth_m = float(getattr(policy, "max_weapon_depth_m", 0.0) or 0.0)
    if max_weapon_depth_m > 0:
        guns = [
            g
            for g in guns
            if g.kind != "weapon"
            or g.label == "smartphone"
            or person_depth_by_pi.get(g.person_idx, max_weapon_depth_m + 1.0) <= max_weapon_depth_m
        ]

    tracked: list[tuple[int, int, int, int, int, float, float | None]] = []
    observations: list[LocalObservation] = []
    confirm_s = max(0.0, float(policy.confirm_s))
    confirm_frames = max(0, int(policy.confirm_frames))
    break_s = max(0.05, float(policy.confirm_break_s))
    knife_confirm_s = max(
        0.0,
        float(policy.knife_confirm_s) if policy.knife_confirm_s is not None else confirm_s,
    )
    use_knife_confirm = policy.knife_confirm_s is not None

    raw_weapon_peak: dict[int, float] = {}
    for g in guns:
        if g.person_idx not in row_track:
            continue
        did = tid_to_did[int(row_track[g.person_idx])]
        if g.label == "smartphone":
            state.phone_tagged.add(did)
            continue
        if did in state.phone_veto_ids and g.label in ("gun", "rifle", "shotgun", "long_gun"):
            continue
        if not person_armed_latch_class_allowed(g.label, policy.classes_allowed):
            continue
        if g.kind != "weapon":
            continue
        raw_weapon_peak[did] = max(raw_weapon_peak.get(did, 0.0), float(g.conf))

    active_dids = _active_display_ids(row_track, tid_to_did)
    _update_weapon_draw_latch(
        state, guns, row_track, tid_to_did, active_dids, policy, manager=manager
    )
    drawn_dids = _weapon_drawn_display_ids(state)

    for pi, (x1, y1, x2, y2, pconf) in enumerate(persons):
        if pi not in row_track:
            continue
        tid = row_track[pi]
        did = tid_to_did[int(tid)]
        depth = person_depth_by_pi.get(
            pi,
            depth_from_bbox_height(
                float(y2 - y1),
                float(h),
                person_height_m=float(args.person_height_m),
                vertical_fov_deg=float(args.person_vfov_deg),
            ),
        )
        beyond_depth = (
            max_weapon_depth_m > 0
            and depth is not None
            and float(depth) > max_weapon_depth_m
        )
        tracked.append((x1, y1, x2, y2, did, pconf, depth))

        drawn_labels = _drawn_weapon_labels(state, did)
        has_knife_drawn = "knife" in drawn_labels
        has_other_weapon = bool(drawn_labels - {"knife"})

        # Only count weapon time when a gun/knife box is actually drawn on screen.
        if did in drawn_dids and not beyond_depth:
            state.weapon_visible_ts[did] = now
            if did in raw_weapon_peak:
                state.weapon_activity_ts[did] = now
                state.weapon_conf[did] = max(
                    state.weapon_conf.get(did, 0.0), raw_weapon_peak[did]
                )

            if use_knife_confirm and has_knife_drawn:
                last_k = state.knife_last_hit.get(did)
                if last_k is not None and (now - last_k) <= break_s:
                    state.knife_accum_s[did] = state.knife_accum_s.get(did, 0.0) + (now - last_k)
                else:
                    state.knife_accum_s[did] = 0.0
                state.knife_last_hit[did] = now
                if state.knife_accum_s.get(did, 0.0) >= knife_confirm_s:
                    state.knife_confirmed.add(did)

            if (not use_knife_confirm) or has_other_weapon:
                last = state.weapon_last_hit.get(did)
                if last is not None and (now - last) <= break_s:
                    state.weapon_accum_s[did] = state.weapon_accum_s.get(did, 0.0) + (now - last)
                    state.weapon_hit_n[did] = state.weapon_hit_n.get(did, 0) + 1
                else:
                    state.weapon_accum_s[did] = 0.0
                    state.weapon_hit_n[did] = 1
                state.weapon_last_hit[did] = now
                if confirm_frames > 0:
                    if state.weapon_hit_n.get(did, 0) >= confirm_frames:
                        state.weapon_confirmed.add(did)
                elif state.weapon_accum_s.get(did, 0.0) >= confirm_s:
                    state.weapon_confirmed.add(did)
        else:
            hold_s = max(0.0, float(policy.hold_s))
            in_hold = hold_s > 0 and (now - state.weapon_activity_ts.get(did, -1e9)) < hold_s
            last = state.weapon_last_hit.get(did)
            last_k = state.knife_last_hit.get(did)
            if last is not None and (now - last) >= break_s:
                if policy.latch_armed and did in state.weapon_confirmed:
                    pass
                elif did in state.concealed_latched_ids and did in state.weapon_confirmed:
                    pass
                elif in_hold and did in state.weapon_confirmed:
                    pass
                else:
                    state.weapon_accum_s.pop(did, None)
                    state.weapon_hit_n.pop(did, None)
                    state.weapon_last_hit.pop(did, None)
                    if did in state.weapon_confirmed:
                        state.weapon_confirmed.discard(did)
                    if not in_hold:
                        state.weapon_conf.pop(did, None)
            if last_k is not None and (now - last_k) >= break_s:
                if not (in_hold and did in state.knife_confirmed):
                    state.knife_accum_s.pop(did, None)
                    state.knife_last_hit.pop(did, None)
                    state.knife_confirmed.discard(did)

        confirmed = did in state.weapon_confirmed or did in state.knife_confirmed
        hold_s = max(0.0, float(policy.hold_s))
        in_hold = hold_s > 0 and (now - state.weapon_activity_ts.get(did, -1e9)) < hold_s
        report_armed = confirmed or in_hold or (policy.latch_armed and (did in state.weapon_confirmed or did in state.knife_confirmed))
        if did in state.concealed_latched_ids and (did in state.weapon_confirmed or did in state.knife_confirmed):
            report_armed = True
        if beyond_depth:
            report_armed = False
        if _is_force_safe_person(state, manager, state.camera_id, did):
            _enforce_force_safe(state, did)
            report_armed = False
        wconf = float(state.weapon_conf.get(did, 0.0)) if report_armed else 0.0
        if report_armed and wconf <= 0.0:
            wconf = 0.5
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
                lateral_norm=float(x1 + x2) / (2.0 * float(w)),
                depth_m=depth,
                weapon_detected=report_armed,
                weapon_confidence=wconf,
                timestamp=now,
            )
        )

    active = {t[4] for t in tracked}
    state.emb_cache.prune_missing(state.camera_id, active)
    for did in list(state.weapon_accum_s.keys()) + list(state.knife_accum_s.keys()) + list(
        state.weapon_confirmed
    ) + list(state.knife_confirmed) + list(state.phone_tagged):
        if did not in active:
            if not _should_purge_weapon_state(state, did, now=now, policy=policy):
                continue
            state.weapon_accum_s.pop(did, None)
            state.knife_accum_s.pop(did, None)
            state.weapon_hit_n.pop(did, None)
            state.weapon_last_hit.pop(did, None)
            state.knife_last_hit.pop(did, None)
            state.weapon_conf.pop(did, None)
            state.weapon_confirmed.discard(did)
            state.knife_confirmed.discard(did)
            state.phone_tagged.discard(did)
            state.phone_veto_ids.discard(did)
            state.weapon_activity_ts.pop(did, None)
            state.weapon_visible_ts.pop(did, None)
            state.concealed_latched_ids.discard(did)

    for x1, y1, x2, y2, did, _pconf, depth in tracked:
        if max_weapon_depth_m > 0 and depth is not None and float(depth) > max_weapon_depth_m:
            continue
        if did in state.weapon_confirmed or did in state.knife_confirmed:
            last_vis = state.weapon_visible_ts.get(did, -1e9)
            vis_hold = max(0.0, float(policy.visible_hold_s))
            if last_vis > 0 and vis_hold > 0 and (now - last_vis) >= vis_hold:
                state.concealed_latched_ids.add(did)

    for did in active:
        if _is_force_safe_person(state, manager, state.camera_id, did):
            _enforce_force_safe(state, did)

    drawn = [
        state.draw_box[k]
        for k in state.draw_ok
        if k in state.draw_box
        and not (
            _is_force_safe_person(state, manager, state.camera_id, int(k[0]))
            and str(k[1]) in _FORCE_SAFE_STRIP_LABELS
        )
    ]
    return tracked, drawn, observations


def _active_display_ids(row_track: dict[int, int], tid_to_did: dict[int, int]) -> set[int]:
    return {tid_to_did[int(tid)] for tid in row_track.values()}


def _weapon_drawn_display_ids(state: CamState) -> set[int]:
    """Display IDs with a gun/knife box actually on screen (draw latch passed)."""
    out: set[int] = set()
    for did, _label in state.draw_ok:
        out.add(int(did))
    return out


def _drawn_weapon_labels(state: CamState, did: int) -> set[str]:
    return {
        str(label)
        for d, label in state.draw_ok
        if int(d) == int(did) and str(label) != "smartphone"
    }


def _update_weapon_draw_latch(
    state: CamState,
    guns: list[GunDet],
    row_track: dict[int, int],
    tid_to_did: dict[int, int],
    active_dids: set[int],
    policy: CamWeaponPolicy,
    *,
    manager: GlobalIDManager | None = None,
) -> None:
    draw_n = max(1, int(policy.draw_confirm_frames))
    draw_miss_n = max(1, int(policy.draw_miss_break))
    hits: dict[tuple[int, str], GunDet] = {}
    for g in guns:
        if g.person_idx not in row_track:
            continue
        did = tid_to_did[int(row_track[g.person_idx])]
        if g.label == "smartphone":
            continue
        if _is_force_safe_person(state, manager, state.camera_id, did):
            continue
        g.display_id = did
        hits[(did, g.label)] = g

    for key, box in hits.items():
        state.draw_streak[key] = state.draw_streak.get(key, 0) + 1
        state.draw_miss[key] = 0
        state.draw_box[key] = box
        if state.draw_streak[key] >= draw_n:
            state.draw_ok.add(key)

    for key in set(state.draw_streak) | set(state.draw_ok) | set(state.draw_box):
        did, _label = key
        if did not in active_dids:
            state.draw_streak.pop(key, None)
            state.draw_miss.pop(key, None)
            state.draw_ok.discard(key)
            state.draw_box.pop(key, None)
            continue
        if key in hits:
            continue
        state.draw_streak[key] = 0
        state.draw_miss[key] = state.draw_miss.get(key, 0) + 1
        if state.draw_miss[key] >= draw_miss_n:
            state.draw_ok.discard(key)
            state.draw_streak.pop(key, None)
            state.draw_miss.pop(key, None)
            state.draw_box.pop(key, None)


def _person_is_force_safe(
    *,
    camera_id: str,
    did: int,
    manager: GlobalIDManager,
    force_safe_display_ids: set[int],
    force_safe_global_ids: set[int],
) -> bool:
    if int(did) in force_safe_display_ids:
        return True
    if force_safe_global_ids:
        gid = manager.get_global_id(str(camera_id), int(did))
        if gid is not None and int(gid) in force_safe_global_ids:
            return True
    return False


def _weapon_showing_red(
    *,
    did: int,
    now: float,
    visible_hold_s: float,
    visible_ts: dict[int, float],
    manager: GlobalIDManager,
    camera_id: str,
    force_safe_display_ids: set[int] | None = None,
    force_safe_global_ids: set[int] | None = None,
) -> bool:
    """Red while weapon recently visible locally or on a synced cross-camera identity."""
    if force_safe_display_ids and did in force_safe_display_ids:
        return False
    if force_safe_global_ids and manager is not None:
        gid = manager.get_global_id(str(camera_id), int(did))
        if gid is not None and int(gid) in force_safe_global_ids:
            return False
    hold = max(0.0, float(visible_hold_s))
    if hold > 0 and (now - visible_ts.get(did, -1e9)) < hold:
        return True
    if manager.synced_on_both_cameras(camera_id, did):
        return manager.weapon_recently_visible_at(camera_id, did, ts=now, hold_s=hold)
    return False


def _person_armed_for_draw(
    *,
    did: int,
    now: float,
    hold_s: float,
    confirmed_ids: set[int],
    activity_ts: dict[int, float],
    manager: GlobalIDManager,
    camera_id: str,
    force_safe_display_ids: set[int] | None = None,
    force_safe_global_ids: set[int] | None = None,
    concealed_latched_ids: set[int] | None = None,
) -> bool:
    if force_safe_display_ids and did in force_safe_display_ids:
        return False
    if force_safe_global_ids and manager is not None:
        gid = manager.get_global_id(str(camera_id), int(did))
        if gid is not None and int(gid) in force_safe_global_ids:
            return False
    if concealed_latched_ids and did in concealed_latched_ids:
        return True
    if did in confirmed_ids:
        return True
    if hold_s > 0 and (now - activity_ts.get(did, -1e9)) < hold_s:
        return True
    if manager.synced_on_both_cameras(camera_id, did):
        return manager.weapon_armed_at(camera_id, did, ts=now)
    return False


def draw_frame(
    frame: np.ndarray,
    tracked: list[tuple[int, int, int, int, int, float, float | None]],
    guns: list[GunDet],
    *,
    camera_id: str,
    manager: GlobalIDManager,
    confirmed_ids: set[int],
    now: float = 0.0,
    weapon_hold_s: float = 2.5,
    weapon_visible_hold_s: float = 2.5,
    weapon_activity_ts: dict[int, float] | None = None,
    weapon_visible_ts: dict[int, float] | None = None,
    force_safe_display_ids: set[int] | None = None,
    force_safe_global_ids: set[int] | None = None,
    concealed_latched_ids: set[int] | None = None,
) -> np.ndarray:
    vis = frame.copy()
    visible_ts = weapon_visible_ts or {}
    activity_ts = weapon_activity_ts or {}
    safe_display = force_safe_display_ids or set()
    safe_global = force_safe_global_ids or set()
    concealed_ids = concealed_latched_ids or set()

    for g in guns:
        if g.label == "smartphone":
            continue
        if g.display_id >= 0 and _person_is_force_safe(
            camera_id=camera_id,
            did=g.display_id,
            manager=manager,
            force_safe_display_ids=safe_display,
            force_safe_global_ids=safe_global,
        ) and g.label in _FORCE_SAFE_STRIP_LABELS:
            continue
        color = COLOR_GUN_WEAPON_BGR if g.kind == "weapon" else COLOR_GUN_OBJECT_BGR
        thick = OVERLAY_RECT_GUN_WEAPON if g.kind == "weapon" else OVERLAY_RECT_GUN_OBJECT
        cv2.rectangle(vis, (g.x1, g.y1), (g.x2, g.y2), color, thick)
        draw_label_above_box(
            vis,
            g.x1,
            g.y1,
            g.label,
            color,
            scale=SYNC_OVERLAY_SCALE_GUN,
            thickness=OVERLAY_THICK,
        )

    for x1, y1, x2, y2, did, _pconf, depth in tracked:
        info = manager.overlay_for(camera_id, did)
        gid = int(info["global_id"]) if info else None
        armed = _person_armed_for_draw(
            did=did,
            now=now,
            hold_s=weapon_hold_s,
            confirmed_ids=confirmed_ids,
            activity_ts=activity_ts,
            manager=manager,
            camera_id=camera_id,
            force_safe_display_ids=safe_display,
            force_safe_global_ids=safe_global,
            concealed_latched_ids=concealed_ids,
        )
        showing_red = _weapon_showing_red(
            did=did,
            now=now,
            visible_hold_s=weapon_visible_hold_s,
            visible_ts=visible_ts,
            manager=manager,
            camera_id=camera_id,
            force_safe_display_ids=safe_display,
            force_safe_global_ids=safe_global,
        )
        if armed and showing_red:
            color = COLOR_PERSON_ARMED_BGR
            concealed = False
        elif armed:
            color = COLOR_PERSON_ARMED_CONCEALED_BGR
            concealed = True
        else:
            color = COLOR_PERSON_OBJECT_BGR
            concealed = False

        label = person_overlay_label(
            did,
            distance_m=None,
            global_id=gid,
            armed=armed,
            concealed=concealed,
        )
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, OVERLAY_RECT_PERSON)
        draw_label_above_box(
            vis,
            x1,
            y1,
            label,
            color,
            scale=SYNC_OVERLAY_SCALE_PERSON,
            thickness=OVERLAY_THICK,
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
    p.add_argument(
        "--gun-conf",
        type=float,
        default=0.65,
        help="Minimum confidence for gun/knife boxes (YOLO filter + confirm).",
    )
    p.add_argument(
        "--phone-conf",
        type=float,
        default=0.25,
        help="Minimum confidence for smartphone boxes (does not arm the person).",
    )
    p.add_argument(
        "--gun-phone-conflict-suppress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop overlapping gun boxes when a competitive smartphone detection wins.",
    )
    p.add_argument("--gun-phone-conflict-iou", type=float, default=0.35)
    p.add_argument("--gun-phone-conflict-margin", type=float, default=0.08)
    p.add_argument(
        "--phone-veto-gun-conf",
        type=float,
        default=0.85,
        help="If smartphone conf >= this on a person, drop weaker gun detections on that person.",
    )
    p.add_argument(
        "--knife-conf",
        type=float,
        default=0.40,
        help="Minimum confidence for knife boxes; knife can arm the person after confirm.",
    )
    p.add_argument(
        "--front-gun-conf",
        type=float,
        default=None,
        help="Front/Camera A gun confidence floor (default: --gun-conf).",
    )
    p.add_argument(
        "--front-knife-conf",
        type=float,
        default=None,
        help="Front/Camera A knife confidence floor (default: --knife-conf).",
    )
    p.add_argument(
        "--front-gun-weapon-min",
        type=float,
        default=None,
        help="Front/Camera A weapon-min for gun kind (default: --gun-weapon-min).",
    )
    p.add_argument(
        "--back-gun-conf",
        type=float,
        default=None,
        help="Back/Camera B gun confidence floor (default: --gun-conf).",
    )
    p.add_argument(
        "--back-knife-conf",
        type=float,
        default=None,
        help="Back/Camera B knife confidence floor (default: --knife-conf).",
    )
    p.add_argument(
        "--back-gun-weapon-min",
        type=float,
        default=None,
        help="Back/Camera B weapon-min for gun kind (default: --gun-weapon-min).",
    )
    p.add_argument("--gun-imgsz", type=int, default=1280)
    p.add_argument("--gun-min-box-px", type=int, default=8)
    p.add_argument(
        "--gun-roi-pad-frac",
        type=float,
        default=0.0,
        help="Extra ROI around the person box for gun YOLO. 0 = person box only.",
    )
    p.add_argument("--gun-roi-pad-px", type=int, default=0)
    p.add_argument("--gun-weapon-min", type=float, default=0.65)
    p.add_argument(
        "--weapon-confirm-s",
        type=float,
        default=2.5,
        help="Seconds of sustained gun/knife before tagging the person as armed (ignored if --weapon-confirm-frames > 0).",
    )
    p.add_argument(
        "--weapon-confirm-frames",
        type=int,
        default=0,
        help="If >0, arm after this many consecutive gun/knife frames instead of --weapon-confirm-s.",
    )
    p.add_argument(
        "--weapon-confirm-break-s",
        type=float,
        default=0.75,
        help="Gap in seconds that resets an unconfirmed streak (or drops a confirmed latch).",
    )
    p.add_argument(
        "--draw-confirm-frames",
        type=int,
        default=3,
        help="Gun/knife box must appear this many consecutive frames before drawing.",
    )
    p.add_argument(
        "--draw-miss-break",
        type=int,
        default=2,
        help="Consecutive misses before dropping a drawn gun/knife box.",
    )
    p.add_argument(
        "--weapon-max-depth-m",
        type=float,
        default=0.0,
        help="Ignore gun/knife detections when estimated person depth exceeds this (0=off).",
    )
    p.add_argument(
        "--front-weapon-max-depth-m",
        type=float,
        default=None,
        help="Front/Camera A depth limit (default: --weapon-max-depth-m). 0=off.",
    )
    p.add_argument(
        "--back-weapon-max-depth-m",
        type=float,
        default=None,
        help="Back/Camera B depth limit (default: --weapon-max-depth-m). 0=off.",
    )
    p.add_argument(
        "--front-weapon-classes",
        default="all",
        help="Front/Camera A weapon filter: all|gun_only|knife|weapons|comma labels.",
    )
    p.add_argument(
        "--back-weapon-classes",
        default="all",
        help="Back/Camera B weapon filter: all|gun_only|knife|weapons|comma labels.",
    )
    p.add_argument(
        "--front-weapon-confirm-s",
        type=float,
        default=None,
        help="Front armed confirm seconds (default: --weapon-confirm-s). 0 = instant on gun.",
    )
    p.add_argument(
        "--front-knife-confirm-s",
        type=float,
        default=None,
        help="Front/Camera A: seconds of sustained knife before armed (default: --front-weapon-confirm-s).",
    )
    p.add_argument(
        "--back-weapon-confirm-s",
        type=float,
        default=None,
        help="Back armed confirm seconds (default: --weapon-confirm-s).",
    )
    p.add_argument(
        "--front-weapon-confirm-frames",
        type=int,
        default=None,
        help="Front armed confirm frames (overrides --front-weapon-confirm-s when >0).",
    )
    p.add_argument(
        "--back-weapon-confirm-frames",
        type=int,
        default=None,
        help="Back armed confirm frames (overrides --back-weapon-confirm-s when >0).",
    )
    p.add_argument("--front-draw-confirm-frames", type=int, default=None)
    p.add_argument("--back-draw-confirm-frames", type=int, default=None)
    p.add_argument("--min-box-px", type=int, default=24)
    p.add_argument("--infer-max-width", type=int, default=1920)
    p.add_argument("--infer-stride", type=int, default=1, help="Run YOLO every N frames (1=all)")
    p.add_argument("--max-frames", type=int, default=0, help="0 = full shorter video")
    p.add_argument("--output-fps", type=float, default=0.0, help="0 = use Front source FPS")
    p.add_argument("--person-distance", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--person-height-m", type=float, default=1.7)
    p.add_argument("--person-vfov-deg", type=float, default=60.0)
    p.add_argument("--similarity-threshold", type=float, default=0.72)
    p.add_argument("--soft-similarity-threshold", type=float, default=0.58)
    p.add_argument("--depth-boost-min-reid", type=float, default=0.48)
    p.add_argument("--depth-boost-min-depth", type=float, default=0.65)
    p.add_argument("--depth-boost-strong-min-reid", type=float, default=0.45)
    p.add_argument(
        "--tracker",
        choices=("bytetrack", "botsort"),
        default="botsort",
        help="Person MOT backend: botsort (Kalman+GMC, default) or bytetrack.",
    )
    p.add_argument("--track-buffer", type=int, default=45, help="Lost-track buffer frames.")
    p.add_argument("--weight-reid", type=float, default=0.35)
    p.add_argument("--weight-depth", type=float, default=0.55)
    p.add_argument("--depth-tolerance-frac", type=float, default=0.22)
    p.add_argument(
        "--lateral-tolerance-frac",
        type=float,
        default=0.20,
        help="Facing cameras: |cx_front + cx_back - 1| tolerance for cross-camera match.",
    )
    p.add_argument(
        "--weapon-visible-hold-s",
        type=float,
        default=2.5,
        help="Seconds to keep person RED (Armed) after last gun/knife detection; resets on re-detect.",
    )
    p.add_argument("--weapon-hold-s", type=float, default=3.0)
    p.add_argument(
        "--weapon-latch",
        action="store_true",
        help="Once armed, keep person armed/concealed until the track leaves the frame.",
    )
    p.add_argument("--track-timeout-s", type=float, default=8.0)
    p.add_argument("--reid-backend", default="auto", help="auto|tensorrt|torchreid|...")
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for association summary JSON",
    )
    p.add_argument(
        "--manual-annotations",
        type=Path,
        default=None,
        help="JSON keyframes for demo-perfect manual boxes (see scripts/manual_annotations.py).",
    )
    p.add_argument(
        "--manual-only",
        action="store_true",
        help="Skip YOLO/Re-ID; render only manual boxes from --manual-annotations.",
    )
    p.add_argument(
        "--force-safe-person-ids",
        default="",
        help="Comma-separated local Person N display IDs always shown unarmed (both cameras).",
    )
    p.add_argument(
        "--front-force-safe-person-ids",
        default=None,
        help="Front/Camera A force-safe display IDs (default: --force-safe-person-ids).",
    )
    p.add_argument(
        "--back-force-safe-person-ids",
        default=None,
        help="Back/Camera B force-safe display IDs (default: --force-safe-person-ids).",
    )
    p.add_argument(
        "--force-safe-global-ids",
        default="",
        help="Comma-separated global Re-ID numbers always unarmed (matches on-screen Person N when synced).",
    )
    p.add_argument(
        "--stabilize-display-ids",
        action="store_true",
        help="Use appearance Re-ID to keep Person N labels stable when ByteTrack swaps ids.",
    )
    p.add_argument(
        "--front-stabilize-display-ids",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Front/Camera A appearance-stable Person N (default: --stabilize-display-ids).",
    )
    p.add_argument(
        "--back-stabilize-display-ids",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Back/Camera B appearance-stable Person N (default: --stabilize-display-ids).",
    )
    return p.parse_args(argv)


def _build_weapon_policy(args: argparse.Namespace, *, side: str) -> CamWeaponPolicy:
    prefix = "front" if side == "front" else "back"
    classes_raw = str(getattr(args, f"{prefix}_weapon_classes", "all") or "all")
    confirm_s = getattr(args, f"{prefix}_weapon_confirm_s", None)
    if confirm_s is None:
        confirm_s = float(args.weapon_confirm_s)
    confirm_frames = getattr(args, f"{prefix}_weapon_confirm_frames", None)
    if confirm_frames is None:
        confirm_frames = int(args.weapon_confirm_frames)
    draw_n = getattr(args, f"{prefix}_draw_confirm_frames", None)
    if draw_n is None:
        draw_n = int(args.draw_confirm_frames)
    gun_conf = getattr(args, f"{prefix}_gun_conf", None)
    if gun_conf is None:
        gun_conf = float(args.gun_conf)
    knife_conf = getattr(args, f"{prefix}_knife_conf", None)
    if knife_conf is None:
        knife_conf = float(args.knife_conf)
    weapon_min = getattr(args, f"{prefix}_gun_weapon_min", None)
    if weapon_min is None:
        weapon_min = float(args.gun_weapon_min)
    knife_confirm = getattr(args, f"{prefix}_knife_confirm_s", None)
    depth_m = getattr(args, f"{prefix}_weapon_max_depth_m", None)
    if depth_m is None:
        depth_m = float(args.weapon_max_depth_m)
    return CamWeaponPolicy(
        classes_allowed=parse_gun_overlay_classes(classes_raw),
        gun_conf=float(gun_conf),
        knife_conf=float(knife_conf),
        weapon_min=float(weapon_min),
        phone_conf=float(args.phone_conf),
        confirm_s=float(confirm_s),
        knife_confirm_s=(float(knife_confirm) if knife_confirm is not None else None),
        confirm_frames=int(confirm_frames),
        confirm_break_s=float(args.weapon_confirm_break_s),
        draw_confirm_frames=int(draw_n),
        draw_miss_break=int(args.draw_miss_break),
        latch_armed=bool(getattr(args, "weapon_latch", False)),
        hold_s=float(args.weapon_hold_s),
        visible_hold_s=float(args.weapon_visible_hold_s),
        max_weapon_depth_m=float(depth_m),
    )


def _policy_summary(side: str, policy: CamWeaponPolicy) -> str:
    allowed = "all" if policy.classes_allowed is None else ",".join(sorted(policy.classes_allowed))
    knife_note = (
        f" knife_confirm={policy.knife_confirm_s:.1f}s" if policy.knife_confirm_s is not None else ""
    )
    return (
        f"{side}: classes={allowed} gun={policy.gun_conf:.2f} knife={policy.knife_conf:.2f} "
        f"weapon_min={policy.weapon_min:.2f} confirm={policy.confirm_s:.1f}s{knife_note} "
        f"frames={policy.confirm_frames} draw={policy.draw_confirm_frames}"
        f"{f' depth<={policy.max_weapon_depth_m:.1f}m' if policy.max_weapon_depth_m > 0 else ''}"
    )


def run_manual_only(args: argparse.Namespace) -> int:
    """Fast path: source MP4s + manual JSON → inferred outputs (no models)."""
    manual_path = args.manual_annotations
    if manual_path is None or not Path(manual_path).expanduser().is_file():
        raise SystemExit("--manual-only requires an existing --manual-annotations JSON file")

    front_path = args.front.expanduser().resolve()
    back_path = args.back.expanduser().resolve()
    front_out = (args.front_out or default_out_path(front_path)).expanduser().resolve()
    back_out = (args.back_out or default_out_path(back_path)).expanduser().resolve()

    store = ManualAnnotationStore.from_path(manual_path)
    if not store.has_any():
        raise SystemExit(f"No keyframes in {manual_path}")

    cf = str(args.camera_front)
    cb = str(args.camera_back)
    cap_f = _open_video(front_path)
    cap_b = _open_video(back_path)
    out_fps = float(args.output_fps) if args.output_fps > 0 else float(cap_f.get(cv2.CAP_PROP_FPS) or 30.0)

    writer_f = writer_b = None
    frame_i = 0
    written = 0
    t0 = time.time()
    max_frames = int(args.max_frames) if args.max_frames > 0 else 0

    print(f"Manual-only render: {manual_path}")
    print(f"Keyframes: {store.keyframe_count()}")

    try:
        while True:
            ok_f, frame_f = cap_f.read()
            ok_b, frame_b = cap_b.read()
            if not ok_f or not ok_b or frame_f is None or frame_b is None:
                break
            if max_frames and written >= max_frames:
                break

            hf, wf = frame_f.shape[:2]
            hb, wb = frame_b.shape[:2]
            mf = frame_for_draw(store, cf, frame_i, frame_w=wf, frame_h=hf)
            mb = frame_for_draw(store, cb, frame_i, frame_w=wb, frame_h=hb)
            vis_f = draw_manual_frame(frame_f, mf)
            vis_b = draw_manual_frame(frame_b, mb)

            if writer_f is None:
                writer_f = _writer_for(front_out, out_fps, (wf, hf))
                writer_b = _writer_for(back_out, out_fps, (wb, hb))
            writer_f.write(vis_f)
            writer_b.write(vis_b)
            written += 1
            frame_i += 1
            if written % 100 == 0 or written == 1:
                print(f"  frame {written}", flush=True)
    finally:
        cap_f.release()
        cap_b.release()
        if writer_f is not None:
            writer_f.release()
        if writer_b is not None:
            writer_b.release()

    print(f"Wrote {front_out}")
    print(f"Wrote {back_out}")
    print(f"Frames={written} elapsed={time.time() - t0:.1f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.manual_only:
        return run_manual_only(args)

    manual_store: ManualAnnotationStore | None = None
    if args.manual_annotations is not None:
        mp = Path(args.manual_annotations).expanduser().resolve()
        if mp.is_file():
            manual_store = ManualAnnotationStore.from_path(mp)
            print(f"Manual annotations: {mp} ({manual_store.keyframe_count()} keyframes)")
        else:
            print(f"Manual annotations: {mp} (not found, skipping)")

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

    print(f"Front : {front_path}")
    print(f"Back  : {back_path}")
    print(f"Out   : {front_out}")
    print(f"        {back_out}")
    print(f"Person model: {person_model_path}")
    print(f"Gun model   : {gun_model_path or '(disabled)'}")
    print(f"Baseline    : {args.baseline_m} m")
    print(
        f"Gun conf    : gun={args.gun_conf:.2f} knife={args.knife_conf:.2f} "
        f"weapon_min={args.gun_weapon_min:.2f} phone={args.phone_conf:.2f} "
        f"break={args.weapon_confirm_break_s:.1f}s "
        f"pad={args.gun_roi_pad_frac}/{args.gun_roi_pad_px}px "
        f"draw_miss={args.draw_miss_break} no_phone_draw "
        f"phone_veto={float(args.phone_veto_gun_conf):.2f}"
    )
    if float(args.weapon_max_depth_m) > 0:
        print(f"Weapon depth (default): ignore gun/knife beyond {float(args.weapon_max_depth_m):.1f}m")

    front_policy = _build_weapon_policy(args, side="front")
    back_policy = _build_weapon_policy(args, side="back")
    print(_policy_summary("Front", front_policy))
    print(_policy_summary("Back ", back_policy))
    if args.weapon_latch:
        print("Weapon latch: ON (armed stays red until track ends)")

    person_model = _load_yolo(person_model_path, args.device)
    gun_model = _load_yolo(gun_model_path, args.device) if gun_model_path else None

    cfg = ReIDConfig(
        enable=True,
        baseline_m=float(args.baseline_m),
        similarity_threshold=float(args.similarity_threshold),
        soft_similarity_threshold=float(args.soft_similarity_threshold),
        depth_boost_min_reid=float(args.depth_boost_min_reid),
        depth_boost_min_depth=float(args.depth_boost_min_depth),
        depth_boost_strong_min_reid=float(args.depth_boost_strong_min_reid),
        weapon_hold_s=float(args.weapon_hold_s),
        weapon_latch=bool(args.weapon_latch),
        track_timeout_s=float(args.track_timeout_s),
        camera_front=str(args.camera_front),
        camera_back=str(args.camera_back),
        embed_backend=str(args.reid_backend),
        embed_device=str(args.device),
        weight_reid=float(args.weight_reid),
        weight_depth=float(args.weight_depth),
        depth_tolerance_frac=float(args.depth_tolerance_frac),
        lateral_tolerance_frac=float(args.lateral_tolerance_frac),
    )
    manager = GlobalIDManager(cfg)
    embedder = PersonReIDEmbedder(
        device=str(args.device),
        backend=str(args.reid_backend),
    )
    print(f"Re-ID backend: {embedder.backend} dim={embedder.feature_dim}")
    print(f"Re-ID       : sim>={args.similarity_threshold:.2f} soft>={args.soft_similarity_threshold:.2f} "
          f"w_reid={args.weight_reid:.2f} w_depth={args.weight_depth:.2f} "
          f"depth_boost>={args.depth_boost_min_reid:.2f}@{args.depth_boost_min_depth:.2f} "
          f"depth_tol={args.depth_tolerance_frac:.2f} lateral_tol={args.lateral_tolerance_frac:.2f}")
    print(f"Tracker     : {args.tracker} buffer={args.track_buffer}")

    fps_track = float(args.output_fps) if args.output_fps > 0 else 30.0
    shared_safe = _parse_force_safe_ids(getattr(args, "force_safe_person_ids", "") or "")
    shared_global_safe = _parse_force_safe_ids(getattr(args, "force_safe_global_ids", "") or "")
    front_safe = _parse_force_safe_ids(getattr(args, "front_force_safe_person_ids", None))
    if not front_safe:
        front_safe = set(shared_safe)
    back_safe = _parse_force_safe_ids(getattr(args, "back_force_safe_person_ids", None))
    if not back_safe:
        back_safe = set(shared_safe)
    stabilize_both = bool(getattr(args, "stabilize_display_ids", False))
    front_stabilize = getattr(args, "front_stabilize_display_ids", None)
    if front_stabilize is None:
        front_stabilize = stabilize_both
    back_stabilize = getattr(args, "back_stabilize_display_ids", None)
    if back_stabilize is None:
        back_stabilize = stabilize_both
    front_state = CamState(
        camera_id=str(args.camera_front),
        tracker=make_indexed_box_tracker(
            tracker_type=str(args.tracker),
            frame_rate=fps_track,
            track_buffer=int(args.track_buffer),
        ),
        display=DisplayTrackIds(),
        emb_cache=TrackEmbeddingCache(embedder, interval_s=0.01, min_box_px=int(args.min_box_px)),
        weapon_policy=front_policy,
        appearance_display=AppearanceDisplayIds() if front_stabilize else None,
        stabilize_display_ids=bool(front_stabilize),
        force_safe_display_ids=front_safe,
        force_safe_global_ids=set(shared_global_safe),
    )
    back_state = CamState(
        camera_id=str(args.camera_back),
        tracker=make_indexed_box_tracker(
            tracker_type=str(args.tracker),
            frame_rate=fps_track,
            track_buffer=int(args.track_buffer),
        ),
        display=DisplayTrackIds(),
        emb_cache=TrackEmbeddingCache(embedder, interval_s=0.01, min_box_px=int(args.min_box_px)),
        weapon_policy=back_policy,
        appearance_display=AppearanceDisplayIds() if back_stabilize else None,
        stabilize_display_ids=bool(back_stabilize),
        force_safe_display_ids=back_safe,
        force_safe_global_ids=set(shared_global_safe),
    )
    if front_safe:
        print(f"Force-safe front person IDs: {sorted(front_safe)}")
    if back_safe:
        print(f"Force-safe back person IDs: {sorted(back_safe)}")
    if shared_global_safe:
        print(f"Force-safe global IDs: {sorted(shared_global_safe)}")
    if front_stabilize:
        print("Front: appearance-stable Person N labels ON")
    if back_stabilize:
        print("Back : appearance-stable Person N labels ON")

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
                    now=now,
                    manager=manager,
                )
                tracked_b, guns_b, obs_b = process_camera_frame(
                    back_state,
                    frame_b,
                    person_model=person_model,
                    gun_model=gun_model,
                    args=args,
                    now=now,
                    manager=manager,
                )
                # Lockstep Global ID update (both cameras same timestamp)
                manager.update_observations([*obs_f, *obs_b], now=now)
                if shared_global_safe:
                    manager.clear_weapon_state_for_globals(shared_global_safe)
                for cam_state in (front_state, back_state):
                    for did, vts in cam_state.weapon_visible_ts.items():
                        manager.note_weapon_visible(cam_state.camera_id, did, ts=vts)
                last_f = (tracked_f, guns_f)
                last_b = (tracked_b, guns_b)
            else:
                tracked_f, guns_f = last_f if last_f else ([], [])
                tracked_b, guns_b = last_b if last_b else ([], [])

            manager.update_concealed_latches(
                ts=now,
                visible_hold_s=float(args.weapon_visible_hold_s),
            )
            if shared_global_safe:
                manager.clear_weapon_state_for_globals(shared_global_safe)

            vis_f = draw_frame(
                frame_f,
                tracked_f,
                guns_f,
                camera_id=front_state.camera_id,
                manager=manager,
                confirmed_ids=front_state.weapon_confirmed | front_state.knife_confirmed,
                now=now,
                weapon_hold_s=float(args.weapon_hold_s),
                weapon_visible_hold_s=float(args.weapon_visible_hold_s),
                weapon_activity_ts=front_state.weapon_activity_ts,
                weapon_visible_ts=front_state.weapon_visible_ts,
                force_safe_display_ids=front_state.force_safe_display_ids,
                force_safe_global_ids=front_state.force_safe_global_ids,
                concealed_latched_ids=front_state.concealed_latched_ids,
            )
            vis_b = draw_frame(
                frame_b,
                tracked_b,
                guns_b,
                camera_id=back_state.camera_id,
                manager=manager,
                confirmed_ids=back_state.weapon_confirmed | back_state.knife_confirmed,
                now=now,
                weapon_hold_s=float(args.weapon_hold_s),
                weapon_visible_hold_s=float(args.weapon_visible_hold_s),
                weapon_activity_ts=back_state.weapon_activity_ts,
                weapon_visible_ts=back_state.weapon_visible_ts,
                force_safe_display_ids=back_state.force_safe_display_ids,
                force_safe_global_ids=back_state.force_safe_global_ids,
                concealed_latched_ids=back_state.concealed_latched_ids,
            )

            if manual_store is not None and manual_store.has_any():
                hf, wf = vis_f.shape[:2]
                hb, wb = vis_b.shape[:2]
                draw_manual_overlays(
                    vis_f,
                    frame_for_draw(
                        manual_store, front_state.camera_id, frame_i, frame_w=wf, frame_h=hf
                    ),
                )
                draw_manual_overlays(
                    vis_b,
                    frame_for_draw(
                        manual_store, back_state.camera_id, frame_i, frame_w=wb, frame_h=hb
                    ),
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
