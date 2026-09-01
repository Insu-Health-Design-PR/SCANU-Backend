"""Fuse mmWave point cloud evidence onto camera infer overlay frames."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from weapon_ai.overlay.mmwave_project import (
    MmwaveProjectConfig,
    depth_aligned_fallback_pixel,
    lateral_norm_to_m,
    native_capture_size,
    parse_point_dict,
    pixel_after_capture_rotate,
    project_radar_point_to_pixel,
)

COLOR_MMWAVE_BODY = (255, 180, 60)  # blue-ish BGR
COLOR_MMWAVE_ANOMALY = (0, 140, 255)  # orange
COLOR_MMWAVE_ANOMALY_PERSIST = (0, 69, 255)  # red-orange
COLOR_AGREEMENT = (80, 220, 80)

# Facing dual AWR1843: B→A is x'=-x, y'=D-y (and A→B is the same involution).
_DEFAULT_SENSOR_DISTANCE_M = 5.0


def mmwave_age_ms(
    metrics: dict[str, Any] | None,
    *,
    now_ns: int | None = None,
) -> float | None:
    """Age of a radar metrics blob.

    Lab publishers may store wall-clock ``timestamp_ns`` while Layer 8 uses
    ``ts_monotonic_ns``. Mixing those clocks used to yield age 0 (always
    fresh) and kept dots on camera video after mmWave was stopped.
    """
    if not isinstance(metrics, dict):
        return None
    if str(metrics.get("publisher") or "") == "stopped":
        return None
    raw = metrics.get("ts_monotonic_ns")
    if raw is None:
        raw = metrics.get("timestamp_ns")
    try:
        ts = int(raw)
    except (TypeError, ValueError):
        ts = None
    now_mono = int(now_ns if now_ns is not None else time.monotonic_ns())
    now_wall = time.time_ns()
    ages: list[float] = []
    if ts is not None:
        age_mono = (now_mono - ts) / 1_000_000.0
        age_wall = (now_wall - ts) / 1_000_000.0
        # Accept the clock that produces a plausible non-negative age (< 1 day).
        if 0.0 <= age_mono < 86_400_000.0:
            ages.append(age_mono)
        if 0.0 <= age_wall < 86_400_000.0:
            ages.append(age_wall)
    mtime = metrics.get("_source_mtime_ns")
    try:
        mt = int(mtime) if mtime is not None else None
    except (TypeError, ValueError):
        mt = None
    if mt is not None:
        ages.append(max(0.0, (now_wall - mt) / 1_000_000.0))
    if not ages:
        return None
    return min(ages)


def mmwave_fresh_for_threat(
    metrics: dict[str, Any] | None,
    *,
    max_age_ms: float = 300.0,
    now_ns: int | None = None,
) -> bool:
    age = mmwave_age_ms(metrics, now_ns=now_ns)
    if age is None:
        return False
    return float(age) <= float(max_age_ms)


def compute_mmwave_torso_score(
    metrics: dict[str, Any] | None,
    *,
    max_age_ms: float = 300.0,
    now_ns: int | None = None,
) -> float | None:
    """Engineering evidence score. Stale radar must not raise threat."""
    if not isinstance(metrics, dict):
        return None
    if not mmwave_fresh_for_threat(metrics, max_age_ms=max_age_ms, now_ns=now_ns):
        return None
    scores: list[float] = []
    for side_key in ("front", "back"):
        block = metrics.get(side_key)
        if not isinstance(block, dict):
            continue
        track = block.get("track")
        if isinstance(track, dict) and track.get("confidence") is not None:
            try:
                scores.append(float(track["confidence"]))
            except (TypeError, ValueError):
                pass
        for an in block.get("anomalies") or []:
            if isinstance(an, dict) and an.get("score") is not None:
                try:
                    scores.append(min(1.0, float(an["score"]) * 0.85))
                except (TypeError, ValueError):
                    pass
    if not scores:
        front = metrics.get("front")
        if isinstance(front, dict) and str(front.get("screening_state") or "") == "person":
            return 0.35
        return None
    return round(max(0.0, min(1.0, max(scores))), 4)


@dataclass(frozen=True)
class MmwaveFusionConfig:
    side: str = "front"
    depth_gate_m: float = 0.6
    lateral_gate_m: float = 0.5
    vertical_fov_deg: float = 60.0
    corridor_half_width_m: float = 1.5
    radar_mount_lateral_m: float = 0.0
    radar_mount_height_m: float = 1.2
    capture_rotate: int = 0
    draw_agreement_halo: bool = True
    draw_anomaly_links: bool = True
    sensor_distance_m: float = _DEFAULT_SENSOR_DISTANCE_M
    max_age_ms: float = 300.0


def transform_a_frame_to_b_local(
    x_m: float, y_m: float, z_m: float, *, distance_m: float
) -> tuple[float, float, float]:
    """Map a point from radar-A frame into radar-B local frame (facing pair, baseline D)."""
    d = float(distance_m)
    return (-float(x_m), d - float(y_m), float(z_m))


def _xyzsnr_points(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for p in raw:
        if isinstance(p, dict):
            parsed = parse_point_dict(p)
            if parsed is not None:
                x, y, z, snr = parsed
                out.append({"x": x, "y": y, "z": z, "snr": snr})
            continue
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            try:
                out.append(
                    {
                        "x": float(p[0]),
                        "y": float(p[1]),
                        "z": float(p[2]),
                        "snr": float(p[3]) if len(p) > 3 else 0.0,
                    }
                )
            except (TypeError, ValueError):
                continue
    return out


def _points_to_b_local(
    points: list[dict[str, Any]], *, distance_m: float
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in points:
        try:
            x, y, z = float(p["x"]), float(p["y"]), float(p["z"])
            snr = float(p.get("snr") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        xb, yb, zb = transform_a_frame_to_b_local(x, y, z, distance_m=distance_m)
        out.append({"x": xb, "y": yb, "z": zb, "snr": snr})
    return out


def _centroid_to_b_local(raw: Any, *, distance_m: float) -> list[float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    try:
        x, y, z = float(raw[0]), float(raw[1]), float(raw[2])
    except (TypeError, ValueError):
        return None
    xb, yb, zb = transform_a_frame_to_b_local(x, y, z, distance_m=distance_m)
    return [xb, yb, zb]


def _track_for_side(tracks: list[Any], *, side: str) -> dict[str, Any] | None:
    want = {"A", "FRONT", "RADAR_A", "SENSOR_A"} if side == "front" else {"B", "BACK", "RADAR_B", "SENSOR_B"}
    labeled: list[dict[str, Any]] = []
    unlabeled: list[dict[str, Any]] = []
    for t in tracks:
        if not isinstance(t, dict):
            continue
        views = {str(v).strip().upper() for v in (t.get("source_views") or [])}
        if not views:
            unlabeled.append(t)
        elif views & want:
            labeled.append(t)
    pick = labeled or unlabeled
    return pick[0] if pick else None


def _anomalies_for_side(cands: list[Any], *, side: str) -> list[dict[str, Any]]:
    want = {"A", "FRONT", "RADAR_A", "SENSOR_A"} if side == "front" else {"B", "BACK", "RADAR_B", "SENSOR_B"}
    out: list[dict[str, Any]] = []
    for an in cands:
        if not isinstance(an, dict):
            continue
        views = {str(v).strip().upper() for v in (an.get("source_views") or [])}
        if views and not (views & want):
            continue
        centroid = an.get("center_m") or an.get("centroid_m") or an.get("position_m")
        out.append(
            {
                "centroid_m": centroid,
                "score": an.get("score"),
                "persistent": an.get("persistent"),
            }
        )
    return out


def _resolve_sensor_distance_m(
    metrics: dict[str, Any], *, fallback_m: float | None = None
) -> float:
    for key in ("sensor_distance_m", "baseline_m", "radar_baseline_m"):
        raw = metrics.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            d = float(raw)
            if d > 0.1:
                return d
        except (TypeError, ValueError):
            continue
    calib = metrics.get("calibration") if isinstance(metrics.get("calibration"), dict) else {}
    for key in ("sensor_distance_m", "distance_m", "baseline_m"):
        raw = calib.get(key) if isinstance(calib, dict) else None
        if raw is None:
            continue
        try:
            d = float(raw)
            if d > 0.1:
                return d
        except (TypeError, ValueError):
            continue
    if fallback_m is not None and float(fallback_m) > 0.1:
        return float(fallback_m)
    return _DEFAULT_SENSOR_DISTANCE_M


def _points_b_published_in_a_frame(metrics: dict[str, Any]) -> bool:
    """Lab dual-live v2 publishes points_b already transformed into radar-A frame."""
    schema = str(metrics.get("schema_version") or "").lower()
    if "v2" in schema or "dual_live" in schema:
        return True
    frame = str(
        metrics.get("points_b_frame")
        or metrics.get("coord_frame")
        or (metrics.get("calibration") or {}).get("points_b_frame")
        or ""
    ).lower()
    if frame in ("a", "radar_a", "front", "sensor_a"):
        return True
    if metrics.get("transform_b_to_a") or (
        isinstance(metrics.get("calibration"), dict)
        and metrics["calibration"].get("transform_b_to_a")
    ):
        return True
    return False


def _localize_back_block(block: dict[str, Any], *, distance_m: float) -> dict[str, Any]:
    """Convert A-frame points/track/anomalies into B-local for Back camera overlay."""
    out = dict(block)
    out["points"] = _points_to_b_local(list(block.get("points") or []), distance_m=distance_m)
    track = block.get("track")
    if isinstance(track, dict):
        t2 = dict(track)
        for key in ("centroid_m", "position_m"):
            mapped = _centroid_to_b_local(t2.get(key), distance_m=distance_m)
            if mapped is not None:
                t2[key] = mapped
        out["track"] = t2
    anoms: list[dict[str, Any]] = []
    for an in block.get("anomalies") or []:
        if not isinstance(an, dict):
            continue
        a2 = dict(an)
        mapped = _centroid_to_b_local(
            a2.get("centroid_m") or a2.get("position_m"), distance_m=distance_m
        )
        if mapped is not None:
            a2["centroid_m"] = mapped
        anoms.append(a2)
    out["anomalies"] = anoms
    out["frame"] = "radar_b_local"
    return out


def normalize_mmwave_metrics_for_overlay(
    metrics: dict[str, Any] | None,
    *,
    sensor_distance_m: float | None = None,
) -> dict[str, Any] | None:
    """Accept lab v2 (points_a/points_b in A-frame) or overlay v1 (front/back local).

    Dual AWR v2 publishes ``points_b`` already mapped into radar-A coordinates using
    baseline D (``x'=-x, y'=D-y``). For Back camera dots we invert that transform so
    projection is relative to the Back radar (cameras ~D meters apart).
    """
    if not isinstance(metrics, dict):
        return None
    front = metrics.get("front")
    # Pure v1 already has per-side local clouds.
    if (
        isinstance(front, dict)
        and (front.get("points") or front.get("track"))
        and "points_a" not in metrics
        and "points_b" not in metrics
    ):
        return metrics
    if "points_a" not in metrics and "points_b" not in metrics:
        return metrics

    d = _resolve_sensor_distance_m(metrics, fallback_m=sensor_distance_m)
    tracks = metrics.get("tracks")
    if not isinstance(tracks, list):
        fused = metrics.get("fused") if isinstance(metrics.get("fused"), dict) else {}
        tracks = fused.get("active_tracks") or []
    cands = metrics.get("reflective_candidates")
    if not isinstance(cands, list):
        fused = metrics.get("fused") if isinstance(metrics.get("fused"), dict) else {}
        cands = fused.get("reflective_candidates") or []

    front_block = {
        "points": _xyzsnr_points(metrics.get("points_a")),
        "track": _track_for_side(tracks if isinstance(tracks, list) else [], side="front"),
        "anomalies": _anomalies_for_side(cands if isinstance(cands, list) else [], side="front"),
        "screening_state": str(metrics.get("state") or "live"),
        "frame": "radar_a_local",
    }
    back_raw = {
        "points": _xyzsnr_points(metrics.get("points_b")),
        "track": _track_for_side(tracks if isinstance(tracks, list) else [], side="back"),
        "anomalies": _anomalies_for_side(cands if isinstance(cands, list) else [], side="back"),
        "screening_state": str(metrics.get("state") or "live"),
    }
    if _points_b_published_in_a_frame(metrics):
        back_block = _localize_back_block(back_raw, distance_m=d)
    else:
        back_block = {**back_raw, "frame": "radar_b_local"}

    out = dict(metrics)
    out["schema_version"] = out.get("schema_version") or "scanu_mmwave_live_v1"
    out["ts_monotonic_ns"] = metrics.get("ts_monotonic_ns") or metrics.get("timestamp_ns")
    out["sensor_distance_m"] = float(d)
    out["front"] = front_block
    out["back"] = back_block
    return out


def _side_metrics(metrics: dict[str, Any], side: str) -> dict[str, Any]:
    block = metrics.get(side)
    return block if isinstance(block, dict) else {}


def _track_centroid(track: dict[str, Any] | None) -> tuple[float, float, float] | None:
    if not isinstance(track, dict):
        return None
    raw = track.get("centroid_m") or track.get("position_m")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    try:
        return float(raw[0]), float(raw[1]), float(raw[2])
    except (TypeError, ValueError):
        return None


def associate_person_to_radar(
    bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    side_metrics: dict[str, Any],
    cfg: MmwaveFusionConfig,
    *,
    person_height_m: float = 1.7,
) -> tuple[bool, tuple[float, float, float] | None]:
    x1, y1, x2, y2 = bbox
    from weapon_ai.overlay.distance import depth_from_bbox_height

    depth = depth_from_bbox_height(
        float(max(1, y2 - y1)),
        float(frame_h),
        person_height_m=person_height_m,
        vertical_fov_deg=cfg.vertical_fov_deg,
    )
    if depth is None:
        return False, None
    cx = 0.5 * (x1 + x2)
    lateral_norm = cx / max(1, frame_w)
    lateral_m = lateral_norm_to_m(lateral_norm, half_width_m=cfg.corridor_half_width_m)
    track = side_metrics.get("track")
    centroid = _track_centroid(track if isinstance(track, dict) else None)
    if centroid is None:
        return False, None
    rx, ry, rz = centroid
    if abs(float(depth) - float(ry)) > cfg.depth_gate_m:
        return False, centroid
    if abs(float(lateral_m) - float(rx)) > cfg.lateral_gate_m:
        return False, centroid
    return True, centroid


def draw_mmwave_fusion_overlay(
    vis: np.ndarray,
    metrics: dict[str, Any] | None,
    *,
    cfg: MmwaveFusionConfig,
    byte_tracks: list[dict[str, Any]] | None = None,
    person_height_m: float = 1.7,
) -> np.ndarray:
    metrics = normalize_mmwave_metrics_for_overlay(
        metrics, sensor_distance_m=cfg.sensor_distance_m
    )
    if not metrics:
        return vis
    draw_ttl_ms = max(float(cfg.max_age_ms), 1000.0)
    if not mmwave_fresh_for_threat(metrics, max_age_ms=draw_ttl_ms):
        return vis

    side = _side_metrics(metrics, cfg.side)
    if not side:
        return vis

    h, w = vis.shape[:2]
    orig_w, orig_h = native_capture_size(w, h, cfg.capture_rotate)
    proj_cfg = MmwaveProjectConfig(
        frame_w=orig_w,
        frame_h=orig_h,
        vertical_fov_deg=cfg.vertical_fov_deg,
        radar_mount_lateral_m=cfg.radar_mount_lateral_m,
        radar_mount_height_m=cfg.radar_mount_height_m,
        corridor_half_width_m=cfg.corridor_half_width_m,
    )

    def _to_display(pix: tuple[int, int] | None) -> tuple[int, int] | None:
        if pix is None:
            return None
        u, v = pixel_after_capture_rotate(pix[0], pix[1], orig_w, orig_h, cfg.capture_rotate)
        if u < 0 or v < 0 or u >= w or v >= h:
            return None
        return u, v

    for p in side.get("points") or []:
        if not isinstance(p, dict):
            continue
        parsed = parse_point_dict(p)
        if parsed is None:
            continue
        x_m, y_m, z_m, snr = parsed
        mapped = _to_display(project_radar_point_to_pixel(x_m, y_m, z_m, proj_cfg))
        if mapped is None:
            continue
        u, v = mapped
        radius = max(4, min(12, int(3 + snr / 10.0)))
        cv2.circle(vis, (u, v), radius + 1, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (u, v), radius, COLOR_MMWAVE_BODY, -1, lineType=cv2.LINE_AA)

    centroid = _track_centroid(side.get("track") if isinstance(side.get("track"), dict) else None)
    if centroid is not None:
        mapped = _to_display(
            project_radar_point_to_pixel(centroid[0], centroid[1], centroid[2], proj_cfg)
        )
        if mapped is not None:
            u, v = mapped
            cv2.circle(vis, (u, v), 10, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(vis, (u, v), 8, COLOR_MMWAVE_BODY, 2, lineType=cv2.LINE_AA)

    for an in side.get("anomalies") or []:
        if not isinstance(an, dict):
            continue
        raw = an.get("centroid_m") or an.get("position_m")
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            continue
        try:
            x_m, y_m, z_m = float(raw[0]), float(raw[1]), float(raw[2])
        except (TypeError, ValueError):
            continue
        mapped = _to_display(project_radar_point_to_pixel(x_m, y_m, z_m, proj_cfg))
        if mapped is None:
            continue
        u, v = mapped
        persistent = bool(an.get("persistent"))
        color = COLOR_MMWAVE_ANOMALY_PERSIST if persistent else COLOR_MMWAVE_ANOMALY
        sz = 6
        pts = np.array(
            [[u, v - sz], [u + sz, v], [u, v + sz], [u - sz, v]],
            dtype=np.int32,
        )
        cv2.fillPoly(vis, [pts], color, lineType=cv2.LINE_AA)

    for row in byte_tracks or []:
        bbox = row.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        ok, _ = associate_person_to_radar(
            (x1, y1, x2, y2),
            w,
            h,
            side,
            cfg,
            person_height_m=person_height_m,
        )
        if ok and cfg.draw_agreement_halo:
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_AGREEMENT, 2, lineType=cv2.LINE_AA)

    return vis


def load_mmwave_metrics(
    path: str | None,
    *,
    sensor_distance_m: float | None = None,
) -> dict[str, Any] | None:
    if not path:
        return None
    from pathlib import Path
    import json

    candidates = [Path(path)]
    shm = Path("/dev/shm/scanu_mmwave_live_metrics.json")
    if shm != candidates[0]:
        candidates.append(shm)
    for p in candidates:
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            try:
                data["_source_mtime_ns"] = int(p.stat().st_mtime_ns)
            except OSError:
                pass
            return normalize_mmwave_metrics_for_overlay(
                data, sensor_distance_m=sensor_distance_m
            )
    return None


def fusion_config_from_args(args: Any) -> MmwaveFusionConfig:
    return MmwaveFusionConfig(
        side=str(getattr(args, "mmwave_side", "front") or "front"),
        depth_gate_m=float(getattr(args, "mmwave_depth_gate_m", 0.6) or 0.6),
        lateral_gate_m=float(getattr(args, "mmwave_lateral_gate_m", 0.5) or 0.5),
        vertical_fov_deg=float(getattr(args, "person_vertical_fov_deg", 60.0) or 60.0),
        corridor_half_width_m=float(getattr(args, "mmwave_corridor_half_width_m", 1.5) or 1.5),
        radar_mount_lateral_m=float(getattr(args, "mmwave_mount_lateral_m", 0.0) or 0.0),
        radar_mount_height_m=float(getattr(args, "mmwave_mount_height_m", 1.2) or 1.2),
        capture_rotate=int(getattr(args, "capture_rotate", 0) or 0),
        sensor_distance_m=float(
            getattr(args, "mmwave_sensor_distance_m", _DEFAULT_SENSOR_DISTANCE_M)
            or _DEFAULT_SENSOR_DISTANCE_M
        ),
        max_age_ms=float(getattr(args, "mmwave_max_age_ms", 300.0) or 300.0),
    )
