"""Person and weapon overlay labels."""

from __future__ import annotations

from typing import Any


THREAT_COLORS = {
    "low": (0, 255, 0),
    "medium": (0, 165, 255),
    "high": (0, 0, 255),
}


def label_for_class(class_name: str) -> str:
    return class_name


def person_key_for_row(
    person_ridx: int,
    row_track: dict[int, int],
    person_tid_display: Any,
) -> str:
    if person_ridx < 0 or person_ridx not in row_track:
        return ""
    return f"T{person_tid_display.display_num(row_track[person_ridx])}"


def gun_box_person_ridx(gun_box: tuple) -> int:
    return int(gun_box[7]) if len(gun_box) > 7 else -1


def person_public_name(display_num: int) -> str:
    return f"Person {int(display_num)}"


def format_distance_suffix(distance_m: float | None) -> str:
    """Beta monocular range tag, e.g. `` (5m)``."""
    if distance_m is None:
        return ""
    try:
        d = float(distance_m)
    except (TypeError, ValueError):
        return ""
    if not (d > 0.05) or d > 200.0:
        return ""
    meters = int(round(d))
    if meters < 1:
        meters = 1
    return f" ({meters}m)"


def gun_box_class_name(gun_box: tuple) -> str:
    if len(gun_box) > 8:
        return str(gun_box[8])
    return "weapon"


def armed_weapon_bracket(class_name: str) -> str:
    """Map YOLO class name to public tag: only ``[Gun]`` or ``[Knife]`` (never phone/object)."""
    name = str(class_name or "").strip().lower().replace("-", "_")
    if any(token in name for token in ("knife", "blade", "dagger", "machete")):
        return "[Knife]"
    if any(
        token in name
        for token in ("gun", "pistol", "handgun", "firearm", "rifle", "shotgun", "long_gun", "weapon")
    ):
        return "[Gun]"
    return ""


def person_weapon_bracket(
    person_key: str,
    cached_gun_boxes: list[tuple],
    row_track: dict[int, int],
    person_tid_display: Any,
    *,
    latch_class_ok: Any = None,
    fallback_bracket: str = "",
) -> str:
    """Best-effort ``[Gun]`` / ``[Knife]`` tag for an armed person (ignores smartphone/object)."""
    if not person_key:
        return str(fallback_bracket or "").strip() or "[Gun]"
    best_name = ""
    best_conf = -1.0
    for gun_box in cached_gun_boxes:
        if person_key_for_row(gun_box_person_ridx(gun_box), row_track, person_tid_display) != person_key:
            continue
        display_label = str(gun_box[4]) if len(gun_box) > 4 else ""
        if latch_class_ok is not None and not latch_class_ok(display_label):
            continue
        bracket = armed_weapon_bracket(gun_box_class_name(gun_box))
        if not bracket:
            # Also try the display label (normalized class) — still gun/knife only.
            bracket = armed_weapon_bracket(display_label)
        if not bracket:
            continue
        conf = float(gun_box[6])
        if conf > best_conf:
            best_conf = conf
            best_name = gun_box_class_name(gun_box) or display_label
    resolved = armed_weapon_bracket(best_name) if best_name else ""
    if resolved:
        return resolved
    fb = str(fallback_bracket or "").strip()
    if fb in ("[Gun]", "[Knife]"):
        return fb
    if fb and not fb.startswith("["):
        mapped = armed_weapon_bracket(fb)
        return mapped or "[Gun]"
    return "[Gun]"


def person_overlay_label(
    display_num: int,
    *,
    armed: bool = False,
    weapon_bracket: str = "",
    concealed: bool = False,
    visual_state: str = "clear",
    grip_state: str = "",
    distance_m: float | None = None,
    global_id: int | None = None,
) -> str:
    """On-frame person tag: ``Person N``, ``Person N (Armed)``, or ``Person N (Concealed)``.

    When ``global_id`` is set (cross-camera Re-ID), that ID is shown instead of the
    local display number so Front/Back share one identity.
    """
    del weapon_bracket, visual_state, grip_state
    num = int(global_id) if global_id is not None and int(global_id) > 0 else int(display_num)
    base = f"{person_public_name(num)}{format_distance_suffix(distance_m)}"
    if armed and concealed:
        return f"{base} (Concealed)"
    if armed:
        return f"{base} (Armed)"
    return base
