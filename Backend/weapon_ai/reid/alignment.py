"""Cross-camera corridor alignment scoring for Front ↔ Back calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from layer8_ui.sentinel_distance import depth_from_bbox_height
from weapon_ai.reid.config import ReIDConfig
from weapon_ai.reid.embeddings import cosine_similarity
from weapon_ai.reid.global_manager import bbox_lateral_norm, lateral_mirror_score


@dataclass
class AlignmentTrack:
    camera_id: str
    local_id: int
    bbox: tuple[int, int, int, int] | None
    lateral_norm: float | None
    depth_m: float | None
    global_id: int | None = None
    embedding: Any = None


def _parse_bbox(row: dict[str, Any]) -> tuple[int, int, int, int] | None:
    bbox = row.get("bbox") or row.get("xyxy")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        except (TypeError, ValueError):
            return None
    keys = ("x1", "y1", "x2", "y2")
    if all(k in row for k in keys):
        try:
            return int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        except (TypeError, ValueError):
            return None
    return None


def _positive_int(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _float_or_none(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def depth_corridor_score(
    d_front: float | None,
    d_back: float | None,
    *,
    baseline_m: float,
    tolerance_frac: float,
) -> tuple[float, float | None]:
    """Return (score 0..1, residual_m)."""
    if d_front is None or d_back is None or baseline_m <= 0:
        return 0.0, None
    residual = abs(float(d_front) + float(d_back) - float(baseline_m))
    tol = max(1e-6, float(tolerance_frac) * float(baseline_m))
    if residual > tol:
        return 0.0, residual
    return 1.0 - residual / tol, residual


def tracks_from_metrics(
    data: dict[str, Any] | None,
    *,
    camera_id: str,
    person_height_m: float = 1.7,
    vertical_fov_deg: float = 60.0,
    compute_depth: bool = True,
) -> tuple[list[AlignmentTrack], int, int]:
    """Parse person tracks from live threat metrics JSON."""
    if not data or not isinstance(data, dict):
        return [], 0, 0
    frame_w = _positive_int(data.get("frame_w"))
    frame_h = _positive_int(data.get("frame_h"))
    rows = data.get("byte_tracks") or []
    if not isinstance(rows, list):
        rows = []
    out: list[AlignmentTrack] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            local_id = int(row.get("display_id") or row.get("track_id") or 0)
        except (TypeError, ValueError):
            continue
        if local_id <= 0:
            continue
        bbox = _parse_bbox(row)
        lateral = _float_or_none(row.get("lateral_norm"))
        if lateral is None and bbox is not None and frame_w > 0:
            lateral = bbox_lateral_norm(bbox, frame_w)
        depth_m = _float_or_none(row.get("depth_m"))
        if depth_m is None and compute_depth and bbox is not None and frame_h > 0:
            bh = max(1, int(bbox[3]) - int(bbox[1]))
            depth_m = depth_from_bbox_height(
                float(bh),
                float(frame_h),
                person_height_m=person_height_m,
                vertical_fov_deg=vertical_fov_deg,
            )
        emb = row.get("reid_embedding")
        out.append(
            AlignmentTrack(
                camera_id=str(camera_id),
                local_id=local_id,
                bbox=bbox,
                lateral_norm=lateral,
                depth_m=depth_m,
                embedding=emb,
            )
        )
    return out, frame_w, frame_h


def _global_id_lookup(global_snapshot: dict[str, Any] | None) -> dict[tuple[str, int], int]:
    """Map (camera_id, local_track_id) -> global_id."""
    out: dict[tuple[str, int], int] = {}
    if not global_snapshot or not isinstance(global_snapshot, dict):
        return out
    persons = global_snapshot.get("persons") or []
    if not isinstance(persons, list):
        return out
    for person in persons:
        if not isinstance(person, dict):
            continue
        try:
            gid = int(person.get("global_id") or 0)
        except (TypeError, ValueError):
            continue
        if gid <= 0:
            continue
        tracks = person.get("camera_tracks") or {}
        if not isinstance(tracks, dict):
            continue
        for cam, tid in tracks.items():
            try:
                out[(str(cam), int(tid))] = gid
            except (TypeError, ValueError):
                continue
    return out


def _reid_between(front: AlignmentTrack, back: AlignmentTrack) -> float | None:
    fe = front.embedding
    be = back.embedding
    if fe is None or be is None:
        return None
    try:
        import numpy as np

        a = np.asarray(fe, dtype=np.float32)
        b = np.asarray(be, dtype=np.float32)
        if a.size == 0 or b.size == 0:
            return None
        return float(cosine_similarity(a, b))
    except Exception:
        return None


def score_pair(
    front: AlignmentTrack,
    back: AlignmentTrack,
    cfg: ReIDConfig,
    *,
    reid_sim: float | None = None,
) -> dict[str, Any]:
    lat_tol = float(getattr(cfg, "lateral_tolerance_frac", 0.20))
    lat_score = 0.0
    lat_residual = None
    if front.lateral_norm is not None and back.lateral_norm is not None:
        lat_score = lateral_mirror_score(
            float(front.lateral_norm),
            float(back.lateral_norm),
            tolerance=lat_tol,
        )
        lat_residual = abs(float(front.lateral_norm) + float(back.lateral_norm) - 1.0)

    depth_score, depth_residual = depth_corridor_score(
        front.depth_m,
        back.depth_m,
        baseline_m=float(cfg.baseline_m),
        tolerance_frac=float(cfg.depth_tolerance_frac),
    )

    if reid_sim is None:
        reid_sim = _reid_between(front, back)

    lateral_ok = lat_score >= 0.45
    depth_ok = depth_score >= 0.50
    reid_ok = reid_sim is not None and float(reid_sim) >= float(cfg.soft_similarity_threshold)
    strong_depth = depth_score >= float(getattr(cfg, "depth_boost_strong_depth", 0.85))
    aligned = lateral_ok and (depth_ok or strong_depth or reid_ok)

    checks = {
        "lateral": {"ok": lateral_ok, "score": round(lat_score, 3), "residual": lat_residual},
        "depth": {
            "ok": depth_ok or strong_depth,
            "score": round(depth_score, 3),
            "residual_m": round(depth_residual, 3) if depth_residual is not None else None,
            "d_front_m": round(float(front.depth_m), 2) if front.depth_m else None,
            "d_back_m": round(float(back.depth_m), 2) if back.depth_m else None,
            "sum_m": (
                round(float(front.depth_m) + float(back.depth_m), 2)
                if front.depth_m and back.depth_m
                else None
            ),
        },
        "reid": {
            "ok": reid_ok,
            "score": round(float(reid_sim), 3) if reid_sim is not None else None,
        },
    }

    return {
        "front_track_id": int(front.local_id),
        "back_track_id": int(back.local_id),
        "global_id": front.global_id if front.global_id == back.global_id else None,
        "aligned": bool(aligned),
        "checks": checks,
        "front_lateral_norm": front.lateral_norm,
        "back_lateral_norm": back.lateral_norm,
        "front_bbox": list(front.bbox) if front.bbox else None,
        "back_bbox": list(back.bbox) if back.bbox else None,
    }


def _pair_score_rank(pair: dict[str, Any]) -> float:
    checks = pair.get("checks") or {}
    lat = float((checks.get("lateral") or {}).get("score") or 0.0)
    dep = float((checks.get("depth") or {}).get("score") or 0.0)
    reid_raw = (checks.get("reid") or {}).get("score")
    reid = float(reid_raw) if reid_raw is not None else 0.0
    bonus = 0.15 if pair.get("global_id") else 0.0
    return lat * 0.35 + dep * 0.45 + reid * 0.20 + bonus


def find_alignment_pairs(
    front_tracks: list[AlignmentTrack],
    back_tracks: list[AlignmentTrack],
    cfg: ReIDConfig,
    *,
    global_snapshot: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    gid_map = _global_id_lookup(global_snapshot)
    for t in front_tracks:
        t.global_id = gid_map.get((t.camera_id, t.local_id))
    for t in back_tracks:
        t.global_id = gid_map.get((t.camera_id, t.local_id))

    pairs: list[dict[str, Any]] = []
    used_back: set[int] = set()

    # Prefer global_id linked pairs first.
    for f in front_tracks:
        if f.global_id is None:
            continue
        for b in back_tracks:
            if b.local_id in used_back or b.global_id != f.global_id:
                continue
            scored = score_pair(f, b, cfg)
            pairs.append(scored)
            used_back.add(b.local_id)
            break

    for f in front_tracks:
        if any(p.get("front_track_id") == f.local_id for p in pairs):
            continue
        best: dict[str, Any] | None = None
        best_rank = -1.0
        for b in back_tracks:
            if b.local_id in used_back:
                continue
            scored = score_pair(f, b, cfg)
            rank = _pair_score_rank(scored)
            if rank > best_rank:
                best_rank = rank
                best = scored
        if best is not None:
            pairs.append(best)
            used_back.add(int(best["back_track_id"]))

    pairs.sort(key=_pair_score_rank, reverse=True)
    return pairs


def overlay_guides() -> dict[str, Any]:
    """Normalized overlay lines for corridor alignment UI."""
    return {
        "center_line_x": 0.5,
        "corridor_left_x": 0.15,
        "corridor_right_x": 0.85,
        "floor_band_y": 0.72,
        "hint": "Level cameras; corridor center at 50%. Person left on Front → right on Back.",
    }


def compute_alignment_status(
    *,
    front_metrics: dict[str, Any] | None,
    back_metrics: dict[str, Any] | None,
    settings: dict[str, Any] | None,
    global_snapshot: dict[str, Any] | None = None,
    front_running: bool = False,
    back_running: bool = False,
) -> dict[str, Any]:
    cfg = ReIDConfig.from_settings(settings)
    sent = (settings or {}).get("sentinel") if isinstance(settings, dict) else {}
    sent = sent if isinstance(sent, dict) else {}
    person_h = float(sent.get("person_height_m") or 1.7)
    vfov = float(sent.get("vertical_fov_deg") or 60.0)

    cam_front = str(cfg.camera_front)
    cam_back = str(cfg.camera_back)

    front_tracks, fw_w, fw_h = tracks_from_metrics(
        front_metrics,
        camera_id=cam_front,
        person_height_m=person_h,
        vertical_fov_deg=vfov,
    )
    back_tracks, bk_w, bk_h = tracks_from_metrics(
        back_metrics,
        camera_id=cam_back,
        person_height_m=person_h,
        vertical_fov_deg=vfov,
    )

    pairs = find_alignment_pairs(front_tracks, back_tracks, cfg, global_snapshot=global_snapshot)
    best = pairs[0] if pairs else None
    aligned = bool(best and best.get("aligned"))

    if not front_running or not back_running:
        phase = "runners_off"
        message = "Start Front and Back camera infer runners."
    elif not front_tracks and not back_tracks:
        phase = "no_person"
        message = "Stand in the corridor so both cameras see you."
    elif not front_tracks or not back_tracks:
        phase = "single_camera"
        message = "Person visible on one camera only — move into shared view."
    elif aligned:
        phase = "aligned"
        message = "Corridor geometry matches — cross-camera linking should work."
    else:
        phase = "misaligned"
        message = "Adjust camera angle, baseline, or VFOV until lateral + depth turn green."

    return {
        "phase": phase,
        "aligned": aligned,
        "message": message,
        "config": {
            "baseline_m": float(cfg.baseline_m),
            "depth_tolerance_frac": float(cfg.depth_tolerance_frac),
            "lateral_tolerance_frac": float(cfg.lateral_tolerance_frac),
            "person_height_m": person_h,
            "vertical_fov_deg": vfov,
            "camera_front": cam_front,
            "camera_back": cam_back,
        },
        "runners": {"front": bool(front_running), "back": bool(back_running)},
        "frames": {
            "front": {"w": fw_w, "h": fw_h, "persons": len(front_tracks)},
            "back": {"w": bk_w, "h": bk_h, "persons": len(back_tracks)},
        },
        "tracks": {
            "front": [
                {
                    "id": t.local_id,
                    "lateral_norm": t.lateral_norm,
                    "depth_m": round(t.depth_m, 2) if t.depth_m else None,
                    "bbox": list(t.bbox) if t.bbox else None,
                    "global_id": t.global_id,
                }
                for t in front_tracks
            ],
            "back": [
                {
                    "id": t.local_id,
                    "lateral_norm": t.lateral_norm,
                    "depth_m": round(t.depth_m, 2) if t.depth_m else None,
                    "bbox": list(t.bbox) if t.bbox else None,
                    "global_id": t.global_id,
                }
                for t in back_tracks
            ],
        },
        "pairs": pairs,
        "best_pair": best,
        "overlay": overlay_guides(),
    }
