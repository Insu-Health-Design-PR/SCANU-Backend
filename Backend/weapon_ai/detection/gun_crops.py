"""Person-ROI crop collection and batched firearm YOLO predict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from weapon_ai.detection.firearms import clamp_box, expand_person_roi_for_gun


@dataclass(frozen=True)
class PersonGunCrop:
    ridx: int
    person_xyxy: tuple[int, int, int, int]
    roi_xyxy: tuple[int, int, int, int]
    crop: np.ndarray

    @property
    def roi_wh(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.roi_xyxy
        return x2 - x1, y2 - y1


def collect_person_gun_crops(
    frame: np.ndarray,
    person_rows: Iterable[tuple[int, tuple[int, int, int, int]]],
    *,
    pad_frac: float,
    pad_px: int,
    min_box_px: int,
) -> list[PersonGunCrop]:
    """Build contiguous person-ROI crops for a single firearm YOLO call."""
    h, w = frame.shape[:2]
    out: list[PersonGunCrop] = []
    for ridx, (px1, py1, px2, py2) in person_rows:
        qx1, qy1, qx2, qy2 = expand_person_roi_for_gun(
            int(px1), int(py1), int(px2), int(py2), w, h, float(pad_frac), int(pad_px)
        )
        pr, pb = qx2 - qx1, qy2 - qy1
        if pr < int(min_box_px) or pb < int(min_box_px):
            continue
        crop = frame[qy1:qy2, qx1:qx2]
        if crop.size == 0:
            continue
        out.append(
            PersonGunCrop(
                ridx=int(ridx),
                person_xyxy=(int(px1), int(py1), int(px2), int(py2)),
                roi_xyxy=(qx1, qy1, qx2, qy2),
                crop=np.ascontiguousarray(crop),
            )
        )
    return out


def predict_gun_on_crops(
    gun_detector: Any,
    crops: list[PersonGunCrop],
    *,
    conf: float,
    imgsz: int,
    device: str,
    batched: bool = True,
) -> list[Any]:
    """Return one Ultralytics ``Results`` (or ``None``) per crop, same order as ``crops``."""
    if not crops:
        return []
    sources = [c.crop for c in crops]
    pred_kw = dict(conf=float(conf), imgsz=int(imgsz), verbose=False, device=device)
    if batched and len(sources) > 1:
        results = gun_detector.predict(source=sources, **pred_kw)
        if results is None:
            return [None] * len(sources)
        if not isinstance(results, (list, tuple)):
            results = [results]
        out = list(results)
        if len(out) < len(sources):
            out.extend([None] * (len(sources) - len(out)))
        return out[: len(sources)]
    out: list[Any] = []
    for crop in sources:
        gres = gun_detector.predict(source=crop, **pred_kw)
        out.append(gres[0] if gres else None)
    return out


def mapped_gun_boxes_from_results(
    crops: list[PersonGunCrop],
    results: list[Any],
    *,
    names: dict,
) -> list[tuple[float, int, int, int, int, str, int, int, int, int]]:
    """Map per-crop YOLO boxes back to frame coordinates.

    Returns ``(conf, gx1, gy1, gx2, gy2, name, owner_ridx, cls_id, roi_w, roi_h)``.
    Caller still applies torso / scale / emit filters.
    """
    raw: list[tuple[float, int, int, int, int, str, int, int, int, int]] = []
    for crop, gres in zip(crops, results):
        if gres is None:
            continue
        if hasattr(gres, "names") and gres.names:
            names = dict(gres.names)
        gboxes = getattr(gres, "boxes", None)
        if gboxes is None or len(gboxes) == 0:
            continue
        g_xyxy = gboxes.xyxy.cpu().numpy()
        g_cls = gboxes.cls.cpu().numpy().astype(int) if gboxes.cls is not None else None
        g_conf = gboxes.conf.cpu().numpy() if gboxes.conf is not None else None
        cw, ch = crop.crop.shape[1], crop.crop.shape[0]
        qx1, qy1, _, _ = crop.roi_xyxy
        pr, pb = crop.roi_wh
        for j, grow in enumerate(g_xyxy):
            lx1, ly1, lx2, ly2 = clamp_box(grow, cw, ch)
            gx1, gy1 = qx1 + lx1, qy1 + ly1
            gx2, gy2 = qx1 + lx2, qy1 + ly2
            cid = int(g_cls[j]) if g_cls is not None and j < len(g_cls) else 0
            gnm = names.get(cid, "gun")
            gc = float(g_conf[j]) if g_conf is not None and j < len(g_conf) else 0.0
            raw.append((gc, gx1, gy1, gx2, gy2, str(gnm), int(crop.ridx), cid, pr, pb))
    return raw
