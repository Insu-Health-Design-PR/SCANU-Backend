#!/usr/bin/env python3
"""Manual bounding-box annotations for sync_runner demo overrides.

JSON schema (version 1):

{
  "version": 1,
  "clip_id": "152636",
  "camera_front": "camera_1",
  "camera_back": "camera_2",
  "keyframes": {
    "camera_1": {
      "120": {
        "persons": [
          {"person_id": 2, "global_id": 2, "armed": true, "x1": 100, "y1": 200, "x2": 400, "y2": 800}
        ],
        "weapons": [
          {"label": "gun", "person_id": 2, "x1": 150, "y1": 300, "x2": 220, "y2": 360}
        ]
      }
    },
    "camera_2": {}
  }
}

Between keyframes, boxes are linearly interpolated per ``person_id`` / ``(person_id, label)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from weapon_ai.overlay.draw import (
    COLOR_GUN_WEAPON_BGR,
    COLOR_PERSON_ARMED_BGR,
    COLOR_PERSON_ARMED_CONCEALED_BGR,
    COLOR_PERSON_OBJECT_BGR,
    OVERLAY_RECT_GUN_WEAPON,
    OVERLAY_RECT_PERSON,
    OVERLAY_SCALE_PERSON,
    OVERLAY_THICK,
    draw_label_above_box,
)
from weapon_ai.overlay.labels import person_overlay_label


@dataclass
class ManualPerson:
    person_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    armed: bool = False
    global_id: int | None = None


@dataclass
class ManualWeapon:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = "gun"
    person_id: int | None = None


@dataclass
class ManualFrame:
    persons: list[ManualPerson]
    weapons: list[ManualWeapon]


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def _lerp_box(a: dict[str, Any], b: dict[str, Any], t: float) -> dict[str, Any]:
    out = dict(a)
    for k in ("x1", "y1", "x2", "y2"):
        out[k] = int(round(_lerp(float(a[k]), float(b[k]), t)))
    if "armed" in a or "armed" in b:
        out["armed"] = bool(a.get("armed")) if t < 0.5 else bool(b.get("armed"))
    return out


def _parse_person(raw: dict[str, Any]) -> ManualPerson:
    return ManualPerson(
        person_id=int(raw["person_id"]),
        x1=int(raw["x1"]),
        y1=int(raw["y1"]),
        x2=int(raw["x2"]),
        y2=int(raw["y2"]),
        armed=bool(raw.get("armed", False)),
        global_id=int(raw["global_id"]) if raw.get("global_id") is not None else None,
    )


def _parse_weapon(raw: dict[str, Any]) -> ManualWeapon:
    return ManualWeapon(
        x1=int(raw["x1"]),
        y1=int(raw["y1"]),
        x2=int(raw["x2"]),
        y2=int(raw["y2"]),
        label=str(raw.get("label") or "gun"),
        person_id=int(raw["person_id"]) if raw.get("person_id") is not None else None,
    )


def _parse_frame(raw: dict[str, Any]) -> ManualFrame:
    persons = [_parse_person(p) for p in (raw.get("persons") or []) if isinstance(p, dict)]
    weapons = [_parse_weapon(w) for w in (raw.get("weapons") or []) if isinstance(w, dict)]
    return ManualFrame(persons=persons, weapons=weapons)


def empty_document(*, clip_id: str = "", camera_front: str = "camera_1", camera_back: str = "camera_2") -> dict[str, Any]:
    return {
        "version": 1,
        "clip_id": str(clip_id),
        "camera_front": str(camera_front),
        "camera_back": str(camera_back),
        "ref": {
            camera_front: {"width": 0, "height": 0},
            camera_back: {"width": 0, "height": 0},
        },
        "keyframes": {camera_front: {}, camera_back: {}},
    }


def load_manual_annotations(path: Path | str) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return empty_document()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manual annotations must be a JSON object: {p}")
    if "keyframes" not in data:
        data["keyframes"] = {}
    data.setdefault("version", 1)
    return data


def save_manual_annotations(path: Path | str, data: dict[str, Any]) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _camera_keyframes(data: dict[str, Any], camera_id: str) -> dict[int, ManualFrame]:
    raw_cam = (data.get("keyframes") or {}).get(str(camera_id)) or {}
    out: dict[int, ManualFrame] = {}
    for k, v in raw_cam.items():
        try:
            fi = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            out[fi] = _parse_frame(v)
    return out


def interpolate_frame(keyframes: dict[int, ManualFrame], frame_idx: int) -> ManualFrame:
    if not keyframes:
        return ManualFrame(persons=[], weapons=[])
    keys = sorted(keyframes.keys())
    if frame_idx <= keys[0]:
        return keyframes[keys[0]]
    if frame_idx >= keys[-1]:
        return keyframes[keys[-1]]

    lo = keys[0]
    hi = keys[-1]
    for k in keys:
        if k <= frame_idx:
            lo = k
        if k >= frame_idx:
            hi = k
            break
    if lo == hi:
        return keyframes[lo]

    t = (frame_idx - lo) / max(hi - lo, 1)
    a = keyframes[lo]
    b = keyframes[hi]

    a_person = {p.person_id: p for p in a.persons}
    b_person = {p.person_id: p for p in b.persons}
    persons: list[ManualPerson] = []
    for pid in sorted(set(a_person) | set(b_person)):
        if pid in a_person and pid in b_person:
            pa, pb = a_person[pid], b_person[pid]
            persons.append(
                ManualPerson(
                    person_id=pid,
                    x1=int(round(_lerp(pa.x1, pb.x1, t))),
                    y1=int(round(_lerp(pa.y1, pb.y1, t))),
                    x2=int(round(_lerp(pa.x2, pb.x2, t))),
                    y2=int(round(_lerp(pa.y2, pb.y2, t))),
                    armed=pa.armed if t < 0.5 else pb.armed,
                    global_id=pa.global_id if pa.global_id is not None else pb.global_id,
                )
            )
        elif pid in a_person:
            persons.append(a_person[pid])
        else:
            persons.append(b_person[pid])

    def weapon_key(w: ManualWeapon) -> tuple[int | None, str]:
        return (w.person_id, w.label)

    a_weapon = {weapon_key(w): w for w in a.weapons}
    b_weapon = {weapon_key(w): w for w in b.weapons}
    weapons: list[ManualWeapon] = []
    for key in sorted(set(a_weapon) | set(b_weapon), key=lambda x: (x[0] or -1, x[1])):
        if key in a_weapon and key in b_weapon:
            wa, wb = a_weapon[key], b_weapon[key]
            weapons.append(
                ManualWeapon(
                    label=wa.label,
                    person_id=wa.person_id,
                    x1=int(round(_lerp(wa.x1, wb.x1, t))),
                    y1=int(round(_lerp(wa.y1, wb.y1, t))),
                    x2=int(round(_lerp(wa.x2, wb.x2, t))),
                    y2=int(round(_lerp(wa.y2, wb.y2, t))),
                )
            )
        elif key in a_weapon:
            weapons.append(a_weapon[key])
        else:
            weapons.append(b_weapon[key])

    return ManualFrame(persons=persons, weapons=weapons)


class ManualAnnotationStore:
    """Load keyframes and interpolate per frame / camera."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self._cache: dict[str, dict[int, ManualFrame]] = {}

    @classmethod
    def from_path(cls, path: Path | str) -> ManualAnnotationStore:
        return cls(load_manual_annotations(path))

    def has_any(self) -> bool:
        kf = self.data.get("keyframes") or {}
        return any(bool(v) for v in kf.values())

    def frame(self, camera_id: str, frame_idx: int) -> ManualFrame:
        cam = str(camera_id)
        if cam not in self._cache:
            self._cache[cam] = _camera_keyframes(self.data, cam)
        return interpolate_frame(self._cache[cam], int(frame_idx))

    def keyframe_count(self) -> int:
        total = 0
        for cam in (self.data.get("keyframes") or {}).values():
            if isinstance(cam, dict):
                total += len(cam)
        return total

    def ref_size(self, camera_id: str) -> tuple[int, int]:
        ref = (self.data.get("ref") or {}).get(str(camera_id)) or {}
        return int(ref.get("width") or 0), int(ref.get("height") or 0)


def _scale_manual_frame(frame: ManualFrame, *, frame_w: int, frame_h: int, ref_w: int, ref_h: int) -> ManualFrame:
    if ref_w <= 0 or ref_h <= 0 or (ref_w == frame_w and ref_h == frame_h):
        return frame
    sx = float(frame_w) / float(ref_w)
    sy = float(frame_h) / float(ref_h)

    def box(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        return (
            int(round(x1 * sx)),
            int(round(y1 * sy)),
            int(round(x2 * sx)),
            int(round(y2 * sy)),
        )

    persons = [
        ManualPerson(
            person_id=p.person_id,
            global_id=p.global_id,
            armed=p.armed,
            x1=b[0],
            y1=b[1],
            x2=b[2],
            y2=b[3],
        )
        for p in frame.persons
        for b in [box(p.x1, p.y1, p.x2, p.y2)]
    ]
    weapons = [
        ManualWeapon(
            label=w.label,
            person_id=w.person_id,
            x1=b[0],
            y1=b[1],
            x2=b[2],
            y2=b[3],
        )
        for w in frame.weapons
        for b in [box(w.x1, w.y1, w.x2, w.y2)]
    ]
    return ManualFrame(persons=persons, weapons=weapons)


def frame_for_draw(
    store: ManualAnnotationStore,
    camera_id: str,
    frame_idx: int,
    *,
    frame_w: int,
    frame_h: int,
) -> ManualFrame:
    mf = store.frame(camera_id, frame_idx)
    rw, rh = store.ref_size(camera_id)
    return _scale_manual_frame(mf, frame_w=frame_w, frame_h=frame_h, ref_w=rw, ref_h=rh)


def draw_manual_frame(
    frame: np.ndarray,
    manual: ManualFrame,
    *,
    ref_w: int = 0,
    ref_h: int = 0,
) -> np.ndarray:
    """Draw manual person + weapon boxes using sync_runner overlay colors."""
    h, w = frame.shape[:2]
    manual = _scale_manual_frame(manual, frame_w=w, frame_h=h, ref_w=ref_w, ref_h=ref_h)
    vis = frame.copy()
    draw_manual_overlays(vis, manual)
    return vis


def draw_manual_overlays(vis: np.ndarray, manual: ManualFrame) -> None:
    """Draw manual boxes onto an existing BGR frame (in-place)."""
    for wbox in manual.weapons:
        cv2.rectangle(
            vis, (wbox.x1, wbox.y1), (wbox.x2, wbox.y2), COLOR_GUN_WEAPON_BGR, OVERLAY_RECT_GUN_WEAPON
        )
        draw_label_above_box(
            vis,
            wbox.x1,
            wbox.y1,
            wbox.label,
            COLOR_GUN_WEAPON_BGR,
            scale=OVERLAY_SCALE_PERSON,
            thickness=OVERLAY_THICK,
        )

    weapon_person_ids = {w.person_id for w in manual.weapons}
    for p in manual.persons:
        gid = p.global_id if p.global_id is not None else p.person_id
        if p.armed and p.person_id in weapon_person_ids:
            color = COLOR_PERSON_ARMED_BGR
            concealed = False
        elif p.armed:
            color = COLOR_PERSON_ARMED_CONCEALED_BGR
            concealed = True
        else:
            color = COLOR_PERSON_OBJECT_BGR
            concealed = False
        label = person_overlay_label(
            p.person_id, global_id=gid, armed=p.armed, concealed=concealed
        )
        cv2.rectangle(vis, (p.x1, p.y1), (p.x2, p.y2), color, OVERLAY_RECT_PERSON)
        draw_label_above_box(
            vis, p.x1, p.y1, label, color, scale=OVERLAY_SCALE_PERSON, thickness=OVERLAY_THICK
        )


def frame_to_dict(frame: ManualFrame) -> dict[str, Any]:
    return {
        "persons": [
            {
                "person_id": p.person_id,
                "global_id": p.global_id,
                "armed": p.armed,
                "x1": p.x1,
                "y1": p.y1,
                "x2": p.x2,
                "y2": p.y2,
            }
            for p in frame.persons
        ],
        "weapons": [
            {
                "label": w.label,
                "person_id": w.person_id,
                "x1": w.x1,
                "y1": w.y1,
                "x2": w.x2,
                "y2": w.y2,
            }
            for w in frame.weapons
        ],
    }
