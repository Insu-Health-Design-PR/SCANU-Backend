"""Overlay drawing constants and helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
# Legibility on 4K / demo exports. Do not shrink without explicit product sign-off.
OVERLAY_SCALE_PERSON = 1.0
OVERLAY_SCALE_GUN = 1.1
OVERLAY_THICK = 3
OVERLAY_RECT_PERSON = 4
OVERLAY_RECT_GUN_OBJECT = 3
OVERLAY_RECT_GUN_WEAPON = 5
OVERLAY_SCALE_STATUS = 0.82
OVERLAY_THICK_STATUS = 2
COLOR_GUN_OBJECT_BGR = (0, 140, 255)
COLOR_GUN_WEAPON_BGR = (0, 0, 255)
COLOR_GUN_GHOST_BGR = (100, 200, 255)
COLOR_PERSON_ARMED_BGR = (0, 0, 255)  # red — armed + visible weapon
COLOR_PERSON_ARMED_CONCEALED_BGR = (0, 255, 255)  # yellow — armed latch, no visible weapon box
COLOR_PERSON_OBJECT_BGR = (0, 220, 0)  # green — safe / clear
COLOR_PERSON_AMBIGUOUS_BGR = (0, 255, 255)
FIREARM_GHOST_CONF = 0.12


def is_playground_export(args: Any) -> bool:
    return (
        getattr(args, "playground_jpg", None) is not None
        or getattr(args, "playground_json", None) is not None
    )


def is_thermal_preview(args: Any) -> bool:
    """Thin strokes for native low-res thermal only (not RGB webcam with --gun_thermal)."""
    if bool(getattr(args, "thin_overlay", False)):
        return True
    if bool(getattr(args, "thermal_v4l2", False)):
        return True
    w = int(getattr(args, "capture_width", 0) or 0)
    h = int(getattr(args, "capture_height", 0) or 0)
    return w > 0 and h > 0 and max(w, h) <= 320


def overlay_draw_style(args: Any) -> tuple[int, int, int, int, float, int]:
    """Return person/gun stroke, label thickness, label scale, unsafe border."""
    if is_playground_export(args):
        return (2, 2, 2, 2, 0.72, 2)
    if is_thermal_preview(args):
        return (1, 1, 1, 1, 0.45, 1)
    return (
        OVERLAY_RECT_PERSON,
        OVERLAY_RECT_GUN_OBJECT,
        OVERLAY_RECT_GUN_WEAPON,
        OVERLAY_THICK,
        OVERLAY_SCALE_PERSON,
        max(1, int(args.unsafe_border_thick)),
    )


def draw_label_above_box(
    vis: np.ndarray,
    x1: int,
    y1: int,
    text: str,
    color: tuple[int, int, int],
    *,
    scale: float,
    thickness: int,
) -> None:
    (_tw, th), baseline = cv2.getTextSize(text, OVERLAY_FONT, scale, thickness)
    ty = int(max(y1 - baseline - 6, th + 4))
    cv2.putText(vis, text, (x1, ty), OVERLAY_FONT, scale, color, thickness, lineType=cv2.LINE_AA)
