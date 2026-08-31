"""Firearm detection geometry and classification helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def firearm_display_kind(
    gc: float,
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    *,
    object_min: float,
    weapon_min: float,
    ref_w: int | None = None,
    ref_h: int | None = None,
) -> str:
    """Classify firearm overlay as ``object`` vs ``weapon`` from detection score only."""
    del gx1, gy1, gx2, gy2, ref_w, ref_h
    if float(gc) > float(weapon_min):
        return "weapon"
    if float(gc) > float(object_min):
        return "object"
    return "object"


def gun_yolo_binary_class_sets(gnames: dict) -> tuple[set[int], set[int]] | None:
    """If the firearm model is 2-class ``no_weapon`` / ``weapon``, return class id sets."""
    if not gnames or len(gnames) < 2:
        return None
    no_ids: set[int] = set()
    weapon_ids: set[int] = set()
    for i, name in gnames.items():
        n = str(name).strip().lower().replace("-", "_")
        if n in ("no_weapon", "no weapon", "object", "not_weapon", "not weapon"):
            no_ids.add(int(i))
        elif n in ("weapon", "gun", "firearm"):
            weapon_ids.add(int(i))
        elif any(
            t in n
            for t in (
                "phone",
                "smartphone",
                "umbrella",
                "bottle",
                "tablet",
                "cane",
                "chip",
                "tool",
                "monedero",
                "wallet",
                "purse",
                "billete",
                "bill",
                "banknote",
                "tarjeta",
                "card",
            )
        ):
            no_ids.add(int(i))
        elif any(t in n for t in ("pistol", "rifle", "shotgun", "handgun", "knife", "long_gun", "gun")):
            weapon_ids.add(int(i))
    if no_ids and weapon_ids:
        return no_ids, weapon_ids
    return None


ARMED_PERSON_DISPLAY_CLASSES = frozenset({"gun", "knife", "rifle", "shotgun", "long_gun"})
# Classes that may latch a person box red when detection is weapon-tier (high conf).
# Phones / wallets / orange object-tier boxes do not latch.
PERSON_ARMED_LATCH_CLASSES = frozenset({"gun", "knife", "rifle", "shotgun", "long_gun"})


def is_armed_person_display_class(label: str) -> bool:
    """True for weapon classes that may turn a person box red (gun/knife/rifle…)."""
    return normalize_firearm_class_display(label) in ARMED_PERSON_DISPLAY_CLASSES


def person_armed_latch_class_allowed(
    label: str,
    overlay_allowed: frozenset[str] | None,
) -> bool:
    """
    True when a detection may count toward sustained **person armed** (red box).

    Respects ``--gun_overlay_classes`` / profile preset: e.g. ``gun_only`` → only
    ``gun``; ``gun_and_knife`` / ``weapons`` allow high-confidence knife to arm the person.
    Smartphone and other non-weapon object classes never latch armed. Object-tier
    (orange) knife/gun boxes still require ``kind == "weapon"`` at the call site.
    """
    norm = normalize_firearm_class_display(label)
    if norm not in PERSON_ARMED_LATCH_CLASSES:
        return False
    if overlay_allowed is not None and norm not in overlay_allowed:
        return False
    return True


def is_smartphone_display_class(label: str) -> bool:
    return normalize_firearm_class_display(label) == "smartphone"


def normalize_firearm_class_display(gname: str) -> str:
    """Map YOLO class name to on-screen label (``pistol`` → ``gun``)."""
    n = str(gname or "object").strip().lower().replace("-", "_")
    if n in ("pistol", "handgun", "firearm"):
        return "gun"
    if n in ("rifle", "shotgun", "long_gun"):
        return n
    if n in ("smartphone", "cellphone", "mobile"):
        return "smartphone"
    if n in ("monedero", "wallet", "purse"):
        return "wallet"
    if n in ("billete", "bill", "banknote"):
        return "banknote"
    if n in ("tarjeta", "card"):
        return "card"
    if n in ("knife", "cuchillo", "navaja"):
        return "knife"
    return n.replace(" ", "_") or "object"


_GUN_OVERLAY_PRESETS: dict[str, frozenset[str] | None] = {
    "all": None,
    "gun_only": frozenset({"gun"}),
    "gun_and_knife": frozenset({"gun", "knife"}),
    "gun_knife_smartphone": frozenset({"gun", "knife", "smartphone"}),
    "gun_smartphone": frozenset({"gun", "smartphone"}),
    "weapons": frozenset({"gun", "knife", "rifle", "shotgun", "long_gun"}),
}


def parse_gun_overlay_classes(spec: str | None) -> frozenset[str] | None:
    """Parse overlay filter: ``all`` / presets / comma-separated display labels."""
    raw = str(spec or "all").strip()
    if not raw or raw.lower() in ("all", "*", "any"):
        return None
    key = raw.lower().replace("-", "_").replace(" ", "_")
    if key in _GUN_OVERLAY_PRESETS:
        return _GUN_OVERLAY_PRESETS[key]
    allowed: set[str] = set()
    for part in raw.split(","):
        p = part.strip()
        if p:
            allowed.add(normalize_firearm_class_display(p))
    return frozenset(allowed) if allowed else None


def dedupe_person_rows(
    rows: list[tuple[int, int, int, int, float, int | None, str, float]],
    *,
    iou_thresh: float = 0.50,
) -> list[tuple[int, int, int, int, float, int | None, str, float]]:
    """Merge overlapping person (COCO class 0) boxes; keep highest detector conf per cluster."""
    if iou_thresh <= 0 or len(rows) <= 1:
        return rows
    person_idx = [(i, r) for i, r in enumerate(rows) if r[5] in (None, 0)]
    if len(person_idx) <= 1:
        return rows
    sorted_p = sorted(person_idx, key=lambda t: -float(t[1][7]))
    keep: set[int] = set()
    kept_boxes: list[tuple[int, int, int, int]] = []
    for i, r in sorted_p:
        box = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
        if any(box_iou(box, kb) >= float(iou_thresh) for kb in kept_boxes):
            continue
        keep.add(i)
        kept_boxes.append(box)
    if len(keep) == len(person_idx):
        return rows
    return [r for i, r in enumerate(rows) if r[5] not in (None, 0) or i in keep]


def gun_overlay_class_allowed(display_label: str, allowed: frozenset[str] | None) -> bool:
    """True when ``display_label`` may be drawn (``allowed is None`` = show all)."""
    if allowed is None:
        return True
    return normalize_firearm_class_display(display_label) in allowed


def parse_gun_class_emit_min(spec: str | None) -> dict[str, float]:
    """Parse per-class emit floors, e.g. ``gun:0.02,knife:0.08`` or ``gun=0.02``."""
    raw = str(spec or "").strip()
    if not raw:
        return {}
    out: dict[str, float] = {}
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if ":" in p:
            label, val = p.split(":", 1)
        elif "=" in p:
            label, val = p.split("=", 1)
        else:
            continue
        label = normalize_firearm_class_display(label.strip())
        if not label:
            continue
        try:
            out[label] = float(val.strip())
        except ValueError:
            continue
    return out


def gun_emit_min_for_class(
    display_label: str,
    per_class: Mapping[str, float] | None,
    default: float,
) -> float:
    """Emit/draw threshold for one firearm class; falls back to ``default``."""
    if not per_class:
        return float(default)
    label = normalize_firearm_class_display(display_label)
    if label in per_class:
        return float(per_class[label])
    return float(default)


def firearm_class_display_from_detection(
    gname: str,
    yolo_cls: int | None,
    gnames: dict,
) -> str:
    if gname and str(gname).strip().lower() not in ("gun", "object", "weapon"):
        return normalize_firearm_class_display(gname)
    if yolo_cls is not None and gnames:
        raw = gnames.get(int(yolo_cls), gname)
        return normalize_firearm_class_display(str(raw))
    return normalize_firearm_class_display(gname)


def firearm_kind_for_detection(
    gc: float,
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    *,
    yolo_cls: int | None,
    gnames: dict,
    binary_sets: tuple[set[int], set[int]] | None,
    object_min: float,
    weapon_min: float,
    ref_w: int | None,
    ref_h: int | None,
) -> str:
    if binary_sets is not None and yolo_cls is not None:
        no_ids, weapon_ids = binary_sets
        if int(yolo_cls) in no_ids:
            return "object"
        if int(yolo_cls) in weapon_ids:
            return firearm_display_kind(
                gc,
                gx1,
                gy1,
                gx2,
                gy2,
                object_min=object_min,
                weapon_min=weapon_min,
                ref_w=ref_w,
                ref_h=ref_h,
            )
    return firearm_display_kind(
        gc, gx1, gy1, gx2, gy2, object_min=object_min, weapon_min=weapon_min, ref_w=ref_w, ref_h=ref_h
    )


def firearm_box_plausible(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fw: int,
    fh: int,
    max_area_frac: float,
    max_side_frac: float,
    *,
    ref_w: int | None = None,
    ref_h: int | None = None,
) -> bool:
    """Drop absurdly large false positives. If ref_w/ref_h set, limits are vs that ROI (e.g. person crop)."""
    if max_area_frac >= 1.0 and max_side_frac >= 1.0:
        return True
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    rw, rh = (fw, fh) if ref_w is None or ref_h is None else (ref_w, ref_h)
    area = bw * bh
    if max_area_frac < 1.0 and area > max_area_frac * rw * rh:
        return False
    if max_side_frac < 1.0 and (bw > max_side_frac * rw or bh > max_side_frac * rh):
        return False
    return True


def gun_box_vs_person_scale_ok(
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    px1: int,
    py1: int,
    px2: int,
    py2: int,
    *,
    max_height_frac: float = 0.45,
    max_area_frac: float = 0.12,
    max_width_frac: float = 0.70,
) -> bool:
    """
    Distance-invariant size gate vs the associated person box.

    Real handguns/rifles occupy a bounded fraction of body height/area in the image.
    Oversized "guns" (props, torso FPs, close-up toys filling the person box) fail here.
    Set any limit to ``>= 1.0`` to disable that check.
    """
    pw = int(px2) - int(px1)
    ph = int(py2) - int(py1)
    if pw <= 1 or ph <= 1:
        return True
    gw = int(gx2) - int(gx1)
    gh = int(gy2) - int(gy1)
    if gw <= 0 or gh <= 0:
        return False
    mh = float(max_height_frac)
    ma = float(max_area_frac)
    mw = float(max_width_frac)
    # Longest gun side vs person height (works for horizontal pistols and long guns).
    if mh < 1.0 and float(max(gw, gh)) > mh * float(ph):
        return False
    if mw < 1.0 and float(gw) > mw * float(pw):
        return False
    if ma < 1.0 and float(gw * gh) > ma * float(pw * ph):
        return False
    return True


def clamp_box(xyxy: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy.astype(int)
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(x1 + 1, w)
    if y2 <= y1:
        y2 = min(y1 + 1, h)
    return x1, y1, x2, y2


def box_center(x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
    return (0.5 * (float(x1) + float(x2)), 0.5 * (float(y1) + float(y2)))


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    aa = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    bb = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    return inter / (aa + bb - inter + 1e-8)


GunDet8 = tuple[float, int, int, int, int, str, int, int]
GunCand6 = tuple[float, int, int, int, int, str]


def take_top_guns_per_person(
    items: list[GunDet8],
    *,
    owner_idx: int = 6,
    max_per_person: int = 2,
    iou_thresh: float = 0.45,
) -> list[GunDet8]:
    """Keep up to ``max_per_person`` highest-conf boxes per owner, skipping heavy overlap."""
    if max_per_person <= 0 or not items:
        return items
    if max_per_person >= 99:
        return items
    by_owner: dict[int, list[GunDet8]] = {}
    for item in items:
        by_owner.setdefault(int(item[owner_idx]), []).append(item)
    out: list[GunDet8] = []
    for group in by_owner.values():
        sorted_g = sorted(group, key=lambda t: -float(t[0]))
        kept: list[GunDet8] = []
        for item in sorted_g:
            if len(kept) >= max_per_person:
                break
            box = (int(item[1]), int(item[2]), int(item[3]), int(item[4]))
            if any(
                box_iou(box, (int(o[1]), int(o[2]), int(o[3]), int(o[4]))) >= iou_thresh for o in kept
            ):
                continue
            kept.append(item)
        out.extend(kept)
    return out


def take_top_gun_candidates(
    candidates: list[GunCand6],
    *,
    max_boxes: int = 2,
    iou_thresh: float = 0.45,
) -> list[GunCand6]:
    """Keep up to ``max_boxes`` highest-conf candidates in one ROI, skipping heavy overlap."""
    if max_boxes <= 0 or not candidates:
        return []
    sorted_c = sorted(candidates, key=lambda t: -float(t[0]))
    kept: list[GunCand6] = []
    for item in sorted_c:
        if len(kept) >= max_boxes:
            break
        box = (int(item[1]), int(item[2]), int(item[3]), int(item[4]))
        if any(box_iou(box, (int(o[1]), int(o[2]), int(o[3]), int(o[4]))) >= iou_thresh for o in kept):
            continue
        kept.append(item)
    return kept


def nearest_person_ridx_for_gun(
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    person_rows: list[tuple[int, tuple[int, int, int, int]]],
) -> int:
    """
    Assign a gun box to exactly one person row.

    Prefer the person with the highest IoU overlap to the gun box (stable when someone
    walks in front and their box contains the gun center but overlaps less than the holder).
    Fall back to closest person center when overlap is negligible.
    """
    if not person_rows:
        return -1
    gun = (int(gx1), int(gy1), int(gx2), int(gy2))
    cx, cy = box_center(gx1, gy1, gx2, gy2)
    best_ridx = -1
    best_score = -1.0
    for ridx, (px1, py1, px2, py2) in person_rows:
        iou = box_iou(gun, (int(px1), int(py1), int(px2), int(py2)))
        center_inside = px1 <= cx <= px2 and py1 <= cy <= py2
        if iou < 0.02 and not center_inside:
            continue
        score = iou + (0.08 if center_inside else 0.0)
        if score > best_score:
            best_score = score
            best_ridx = ridx
    if best_ridx >= 0:
        return best_ridx

    best_d2 = float("inf")
    for ridx, (px1, py1, px2, py2) in person_rows:
        pcx, pcy = box_center(px1, py1, px2, py2)
        d2 = (cx - pcx) ** 2 + (cy - pcy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_ridx = ridx
    return best_ridx


def dedupe_gun_candidates(
    items: list[tuple[float, int, int, int, int, str, int, int]],
    *,
    iou_thresh: float = 0.45,
) -> list[tuple[float, int, int, int, int, str, int, int]]:
    """Merge overlapping gun boxes (padding overlap); keep highest conf per cluster."""
    if len(items) <= 1:
        return items
    sorted_items = sorted(items, key=lambda t: -float(t[0]))
    kept: list[tuple[float, int, int, int, int, str, int, int]] = []

    for item in sorted_items:
        gc, gx1, gy1, gx2, gy2, gnm, owner, _yolo_cls = item
        box = (gx1, gy1, gx2, gy2)
        if any(
            box_iou(box, (k[1], k[2], k[3], k[4])) >= iou_thresh and k[6] == owner
            for k in kept
        ):
            continue
        if any(box_iou(box, (k[1], k[2], k[3], k[4])) >= iou_thresh for k in kept):
            continue
        kept.append(item)
    return kept


def _is_phone_class_name(name: str) -> bool:
    return normalize_firearm_class_display(name) == "smartphone"


def _is_gun_weapon_class_name(name: str) -> bool:
    return normalize_firearm_class_display(name) in {
        "gun",
        "knife",
        "rifle",
        "shotgun",
        "long_gun",
    }


def suppress_gun_phone_conflicts(
    items: list[tuple[float, int, int, int, int, str, int, int]],
    *,
    iou_thresh: float = 0.35,
    prefer_phone_margin: float = 0.08,
) -> list[tuple[float, int, int, int, int, str, int, int]]:
    """Drop gun/knife boxes that heavily overlap a phone when the phone is competitive.

    Phones are frequently misclassified as weak pistols. When both classes fire on
    nearly the same box, keep the smartphone unless the weapon conf clearly wins.
    """
    if len(items) <= 1:
        return items
    phones = [it for it in items if _is_phone_class_name(str(it[5]))]
    weapons = [it for it in items if _is_gun_weapon_class_name(str(it[5]))]
    if not phones or not weapons:
        return items
    drop: set[int] = set()
    for wi, w in enumerate(items):
        if not _is_gun_weapon_class_name(str(w[5])):
            continue
        wbox = (int(w[1]), int(w[2]), int(w[3]), int(w[4]))
        wconf = float(w[0])
        for p in phones:
            pbox = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
            if box_iou(wbox, pbox) < float(iou_thresh):
                continue
            pconf = float(p[0])
            # Prefer phone unless weapon is clearly stronger.
            if pconf + float(prefer_phone_margin) >= wconf:
                drop.add(wi)
                break
    if not drop:
        return items
    return [it for i, it in enumerate(items) if i not in drop]


def xyxy_center_inside_any_person(
    ox1: int, oy1: int, ox2: int, oy2: int, person_boxes: list[tuple[int, int, int, int]]
) -> bool:
    """True if the center of ``(ox1..ox2)`` lies inside at least one person ``xyxy``."""
    if not person_boxes:
        return False
    mx = 0.5 * (float(ox1) + float(ox2))
    my = 0.5 * (float(oy1) + float(oy2))
    for px1, py1, px2, py2 in person_boxes:
        if px1 <= mx <= px2 and py1 <= my <= py2:
            return True
    return False


def yolo_keep_nonperson_detection(
    cid: int | None,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    det_c: float,
    args: Any,
) -> bool:
    """Extra filters for non-person COCO classes (e.g. bottle) to cut small false positives (shirt text)."""
    if cid is None or int(cid) == 0:
        return True
    if float(det_c) < float(getattr(args, "yolo_nonperson_min_conf", 0.5)):
        return False
    bw = max(0, int(x2 - x1))
    bh = max(0, int(y2 - y1))
    if min(bw, bh) < int(getattr(args, "yolo_nonperson_min_short_side_px", 44)):
        return False
    if bw * bh < int(getattr(args, "yolo_nonperson_min_area_px", 1100)):
        return False
    rap = float(getattr(args, "yolo_nonperson_max_aspect", 7.0))
    if rap > 0 and bw > 0 and bh > 0:
        ar = max(bw / float(bh), bh / float(bw))
        if ar > rap:
            return False
    return True


def gun_detection_valid(
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    fw: int,
    fh: int,
    max_area_frac: float,
    max_side_frac: float,
    gun_min_box_px: int,
    *,
    ref_w: int | None = None,
    ref_h: int | None = None,
) -> bool:
    if (gx2 - gx1) < gun_min_box_px or (gy2 - gy1) < gun_min_box_px:
        return False
    return firearm_box_plausible(
        gx1,
        gy1,
        gx2,
        gy2,
        fw,
        fh,
        max_area_frac,
        max_side_frac,
        ref_w=ref_w,
        ref_h=ref_h,
    )


def expand_xyxy_frac(
    x1: int, y1: int, x2: int, y2: int, fw: int, fh: int, pad_frac: float
) -> tuple[int, int, int, int]:
    """Pad a box by a fraction of its width/height (clamped to image)."""
    if pad_frac <= 0:
        return x1, y1, x2, y2
    bw, bh = x2 - x1, y2 - y1
    px = int(round(bw * pad_frac))
    py = int(round(bh * pad_frac))
    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(fw, x2 + px)
    ny2 = min(fh, y2 + py)
    return nx1, ny1, nx2, ny2


def expand_person_roi_for_gun(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fw: int,
    fh: int,
    pad_frac: float,
    pad_px: int,
) -> tuple[int, int, int, int]:
    """Expand person box for firearm YOLO: fractional pad, then fixed pixels per side (clamped)."""
    qx1, qy1, qx2, qy2 = expand_xyxy_frac(x1, y1, x2, y2, fw, fh, pad_frac)
    if pad_px <= 0:
        return qx1, qy1, qx2, qy2
    qx1 = max(0, qx1 - pad_px)
    qy1 = max(0, qy1 - pad_px)
    qx2 = min(fw, qx2 + pad_px)
    qy2 = min(fh, qy2 + pad_px)
    return qx1, qy1, qx2, qy2


def clip_person_upper_body(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    upper_frac: float,
) -> tuple[int, int, int, int]:
    """Keep the upper ``upper_frac`` of a person box (from head downward).

    ``upper_frac`` is the fraction of person height kept from the top (y1).
    E.g. 0.78 keeps head-through-upper-legs and drops the bottom ~22%% (feet/shoes);
    1.0 keeps the full box; lower values exclude more leg.
    """
    if upper_frac <= 0.0 or upper_frac >= 1.0:
        return x1, y1, x2, y2
    ph = max(1, y2 - y1)
    ny2 = y1 + int(round(ph * float(upper_frac)))
    ny2 = max(y1 + 1, min(y2, ny2))
    return x1, y1, x2, ny2


def gun_box_in_person_upper_body(
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    px1: int,
    py1: int,
    px2: int,
    py2: int,
    upper_frac: float,
) -> bool:
    """True when the firearm box sits in the upper portion of the person (not legs/feet)."""
    if upper_frac <= 0.0 or upper_frac >= 1.0:
        return True
    ph = max(1.0, float(py2) - float(py1))
    cutoff = float(py1) + ph * float(upper_frac)
    margin = 0.03 * ph
    if float(gy2) > cutoff + margin:
        return False
    gh = max(1.0, float(gy2) - float(gy1))
    if gh > 0.22 * ph:
        gcy = 0.5 * (float(gy1) + float(gy2))
        if gcy > cutoff:
            return False
    return True


def _box_intersection_area(
    ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int
) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    return float(iw * ih)


def _box_overlap_min_frac(
    ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int
) -> float:
    inter = _box_intersection_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
    if inter <= 0.0:
        return 0.0
    aa = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    bb = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    return inter / max(1.0, min(aa, bb))


def person_carry_zones(
    px1: int, py1: int, px2: int, py2: int
) -> list[tuple[int, int, int, int]]:
    """
    Estimated hand / carry regions from a person box (no pose model).

    Side bands cover outstretched arms; a narrow front band covers chest draw.
    """
    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)
    y_top = py1 + int(round(0.14 * ph))
    y_bot = py1 + int(round(0.74 * ph))
    side_w = max(1, int(round(0.42 * pw)))
    zones = [
        (px1, y_top, px1 + side_w, y_bot),
        (px2 - side_w, y_top, px2, y_bot),
    ]
    cw = max(1, int(round(0.30 * pw)))
    cx = (px1 + px2) // 2
    zones.append((cx - cw // 2, y_top, cx + cw // 2, py1 + int(round(0.56 * ph))))
    return zones


def gun_box_leg_gap_false_positive(
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    px1: int,
    py1: int,
    px2: int,
    py2: int,
) -> bool:
    """
    True when a tall detection in the center-lower person box is likely the between-legs gap.
    """
    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)
    gcx, gcy = box_center(gx1, gy1, gx2, gy2)
    gw = max(1, gx2 - gx1)
    gh = max(1, gy2 - gy1)
    nx = (gcx - float(px1)) / float(pw)
    ny = (gcy - float(py1)) / float(ph)
    aspect = gh / float(gw)

    if 0.28 <= nx <= 0.72 and ny >= 0.48 and aspect >= 1.6:
        return True

    hip = float(py1) + 0.46 * float(ph)
    gap_l = px1 + int(round(0.30 * pw))
    gap_r = px1 + int(round(0.70 * pw))
    if float(gy2) > hip + 0.08 * float(ph) and gh > 0.32 * float(ph):
        if gx1 < gap_r and gx2 > gap_l and aspect >= 1.45:
            return True
    return False


def gun_box_in_carry_zone(
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    px1: int,
    py1: int,
    px2: int,
    py2: int,
    *,
    min_overlap: float = 0.10,
) -> bool:
    """True when the firearm box overlaps an estimated hand / carry band."""
    if gun_box_leg_gap_false_positive(gx1, gy1, gx2, gy2, px1, py1, px2, py2):
        return False
    gcx, gcy = box_center(gx1, gy1, gx2, gy2)
    for zx1, zy1, zx2, zy2 in person_carry_zones(px1, py1, px2, py2):
        ov = _box_overlap_min_frac(gx1, gy1, gx2, gy2, zx1, zy1, zx2, zy2)
        if ov >= float(min_overlap):
            return True
        if zx1 <= gcx <= zx2 and zy1 <= gcy <= zy2:
            return True
    return False


def person_weapon_placement_ok(
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    px1: int,
    py1: int,
    px2: int,
    py2: int,
    *,
    upper_frac: float = 0.0,
    hand_zone: bool = True,
    hand_zone_min_overlap: float = 0.10,
) -> bool:
    """
    Reject leg-gap false positives; keep detections in hand / upper-carry regions.
    """
    if gun_box_leg_gap_false_positive(gx1, gy1, gx2, gy2, px1, py1, px2, py2):
        return False
    if 0.0 < upper_frac < 1.0:
        if not gun_box_in_person_upper_body(gx1, gy1, gx2, gy2, px1, py1, px2, py2, upper_frac):
            return False
    if hand_zone and not gun_box_in_carry_zone(
        gx1, gy1, gx2, gy2, px1, py1, px2, py2, min_overlap=hand_zone_min_overlap
    ):
        return False
    return True

