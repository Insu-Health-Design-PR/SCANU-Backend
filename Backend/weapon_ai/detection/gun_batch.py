"""Gun-crop TensorRT batch splitting and adaptive imgsz buckets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from weapon_ai.detection.gun_crops import PersonGunCrop


def tensorrt_batch_range(detector: Any, *, default_max: int = 8) -> tuple[int, int]:
    """Best-effort (min, max) batch from an Ultralytics/TensorRT model. CPU-safe."""
    min_b, max_b = 1, max(1, int(default_max))
    if detector is None:
        return min_b, max_b
    model = getattr(detector, "model", detector)
    for attr in ("max_batch_size", "batch", "batch_size"):
        raw = getattr(model, attr, None)
        if isinstance(raw, (int, float)) and int(raw) > 0:
            max_b = int(raw)
            break
    names = getattr(detector, "overrides", None)
    if isinstance(names, dict):
        b = names.get("batch")
        if isinstance(b, (int, float)) and int(b) > 0:
            max_b = int(b)
    return 1, max(1, max_b)


def split_batch_indices(n: int, max_batch: int) -> list[tuple[int, int]]:
    """Inclusive-exclusive slices that never exceed ``max_batch``."""
    if n <= 0:
        return []
    mb = max(1, int(max_batch))
    return [(i, min(n, i + mb)) for i in range(0, n, mb)]


def bucket_gun_imgsz(
    crop_hw: tuple[int, int],
    *,
    mode: str = "adaptive",
    default: int = 640,
    near_min_side: int = 220,
    far_max_side: int = 96,
    allowed: Sequence[int] = (512, 640, 960),
) -> int:
    """Map crop size onto a small fixed set of TensorRT-friendly squares."""
    allowed_set = {int(x) for x in allowed if int(x) > 0}
    default_i = int(default) if int(default) in allowed_set else (
        640 if 640 in allowed_set else min(allowed_set or {640})
    )
    if str(mode).lower() in {"fixed", "640", "off", "none"}:
        return default_i
    ch, cw = int(crop_hw[0]), int(crop_hw[1])
    side = min(ch, cw)
    if side <= int(far_max_side) and 960 in allowed_set:
        return 960
    if side >= int(near_min_side) and 512 in allowed_set:
        return 512
    if 640 in allowed_set:
        return 640
    return default_i


def prioritize_crop_indices(
    crops: Sequence[PersonGunCrop],
    *,
    new_ridx: set[int] | None = None,
    mmwave_ridx: set[int] | None = None,
    near_alert_ridx: set[int] | None = None,
    changed_ridx: set[int] | None = None,
) -> list[int]:
    """Stable order: high-priority people first, then remaining ridx order."""
    new_ridx = new_ridx or set()
    mmwave_ridx = mmwave_ridx or set()
    near_alert_ridx = near_alert_ridx or set()
    changed_ridx = changed_ridx or set()

    def rank(i: int) -> tuple[int, int]:
        c = crops[i]
        prio = 4
        if c.ridx in new_ridx or c.ridx in mmwave_ridx:
            prio = 0
        elif c.ridx in near_alert_ridx:
            prio = 1
        elif c.ridx in changed_ridx:
            prio = 2
        return (prio, i)

    return sorted(range(len(crops)), key=rank)
