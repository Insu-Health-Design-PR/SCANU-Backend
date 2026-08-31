"""Project mmWave corridor coordinates onto camera pixels (v1 pinhole + mount offset)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MmwaveProjectConfig:
    frame_w: int
    frame_h: int
    vertical_fov_deg: float = 60.0
    radar_mount_lateral_m: float = 0.0
    radar_mount_height_m: float = 1.2
    corridor_half_width_m: float = 1.5


def lateral_norm_to_m(lateral_norm: float, *, half_width_m: float) -> float:
    """Map bbox center norm [0,1] to lateral meters (0 = image center)."""
    return (float(lateral_norm) - 0.5) * 2.0 * float(half_width_m)


def native_capture_size(display_w: int, display_h: int, rotate: int) -> tuple[int, int]:
    """Unrotated capture size. 90/270 swap width and height."""
    rot = int(rotate) % 360
    if rot in (90, 270):
        return int(display_h), int(display_w)
    return int(display_w), int(display_h)


def pixel_after_capture_rotate(
    u: int,
    v: int,
    orig_w: int,
    orig_h: int,
    rotate: int,
) -> tuple[int, int]:
    """Map a pixel from unrotated capture into the display frame after capture_rotate."""
    rot = int(rotate) % 360
    if rot == 90:
        return int(orig_h) - 1 - int(v), int(u)
    if rot == 270:
        return int(v), int(orig_w) - 1 - int(u)
    if rot == 180:
        return int(orig_w) - 1 - int(u), int(orig_h) - 1 - int(v)
    return int(u), int(v)


def project_radar_point_to_pixel(
    x_m: float,
    y_m: float,
    z_m: float,
    cfg: MmwaveProjectConfig,
    *,
    clamp: bool = False,
) -> tuple[int, int] | None:
    """Project radar (x lateral, y depth, z height) to *unrotated* capture pixels."""
    if not (math.isfinite(x_m) and math.isfinite(y_m) and math.isfinite(z_m)):
        return None
    if y_m <= 0.05:
        return None
    vfov_deg = max(1.0, float(cfg.vertical_fov_deg))
    vfov = math.radians(vfov_deg)
    hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * (cfg.frame_w / max(cfg.frame_h, 1)))
    x_rel = float(x_m) + float(cfg.radar_mount_lateral_m)
    u = cfg.frame_w * (0.5 + math.tan(math.atan2(x_rel, y_m)) / (2.0 * math.tan(hfov / 2.0)))
    z_rel = float(z_m) - float(cfg.radar_mount_height_m)
    pitch = math.atan2(z_rel, y_m)
    v = cfg.frame_h * (0.5 - pitch / (vfov / 2.0))
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    ui, vi = int(round(u)), int(round(v))
    if clamp:
        ui = max(0, min(int(cfg.frame_w) - 1, ui))
        vi = max(0, min(int(cfg.frame_h) - 1, vi))
        return ui, vi
    if ui < 0 or vi < 0 or ui >= cfg.frame_w or vi >= cfg.frame_h:
        return None
    return ui, vi


def depth_aligned_fallback_pixel(
    depth_m: float,
    lateral_norm: float,
    cfg: MmwaveProjectConfig,
) -> tuple[int, int]:
    """Fallback dot at bbox floor row from monocular depth + lateral norm."""
    vfov = math.radians(max(1.0, cfg.vertical_fov_deg))
    # Place dot lower third based on depth (closer → lower in frame heuristic)
    depth_norm = min(1.0, max(0.0, float(depth_m) / 8.0))
    v = int(cfg.frame_h * (0.55 + 0.25 * depth_norm))
    u = int(float(lateral_norm) * cfg.frame_w)
    u = max(0, min(cfg.frame_w - 1, u))
    v = max(0, min(cfg.frame_h - 1, v))
    return u, v


def parse_point_dict(p: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        x = float(p.get("x"))
        y = float(p.get("y"))
        z = float(p.get("z"))
        snr = float(p.get("snr") or 0.0)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(snr)):
        return None
    return x, y, z, snr
