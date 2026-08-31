"""
ByteTrack for thermal inference: wraps Ultralytics' BYTETracker (same dependency as YOLO).

- ``IndexedBoxByteTracker``: generic xyxy + confidence list -> stable track IDs (local indices).
- ``ThermalByteTracker``: person-class rows only, maps **row index** -> track ID.
Association uses greedy IoU matching so displayed boxes align with tracker output.

``GunStableIdTracker`` assigns **persistent** numeric ids (overlay ``gun1``, ``gun2``) via frame-to-frame
IoU matching. That avoids visible id swaps when MOT moves a detection between score pools after the
confidence crosses the object vs weapon labeling threshold.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes ``a`` (N,4) and ``b`` (M,4) in xyxy format."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0.0, None) * np.clip(ay2 - ay1, 0.0, None)
    area_b = np.clip(bx2 - bx1, 0.0, None) * np.clip(by2 - by1, 0.0, None)
    union = area_a + area_b - inter + 1e-8
    return (inter / union).astype(np.float64)


def _greedy_assign(
    det_xyxy: np.ndarray,
    track_xyxy: np.ndarray,
    track_ids: np.ndarray,
    *,
    min_iou: float = 0.1,
) -> list[int | None]:
    """Match each detection to at most one track by descending IoU (greedy)."""
    n, m = len(det_xyxy), len(track_xyxy)
    if n == 0:
        return []
    if m == 0:
        return [None] * n
    iou = _iou_xyxy(det_xyxy.astype(np.float64), track_xyxy.astype(np.float64))
    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(m):
            pairs.append((float(iou[i, j]), i, j))
    pairs.sort(key=lambda t: -t[0])
    det_tid: list[int | None] = [None] * n
    used_det: set[int] = set()
    used_trk: set[int] = set()
    for iou_ij, i, j in pairs:
        if iou_ij < min_iou:
            break
        if i in used_det or j in used_trk:
            continue
        det_tid[i] = int(track_ids[j])
        used_det.add(i)
        used_trk.add(j)
    return det_tid


def _iou_xyxy_pair(
    ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int
) -> float:
    """IoU between two XYXY rectangles (integer pixel coords)."""
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = float(max(0, ix2 - ix1))
    ih = float(max(0, iy2 - iy1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    bb = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = aa + bb - inter + 1e-8
    return inter / union


@dataclass
class _StableGunSlot:
    sid: int
    bbox: tuple[int, int, int, int]
    missed_frames: int = 0
    object_votes: int = 0
    weapon_votes: int = 0
    last_kind: str = "object"
    object_display_id: int | None = None
    weapon_display_id: int | None = None
    weapon_earned: bool = False
    #: Person overlay id (``T1``) from ByteTrack — weapon/object numbers are scoped per person.
    person_key: str = ""

    def record_kind(self, kind: str) -> None:
        k = str(kind)
        self.last_kind = k
        if k == "weapon":
            self.weapon_votes += 1
        else:
            self.object_votes += 1

    def majority_kind(self) -> str:
        if self.weapon_votes > self.object_votes:
            return "weapon"
        if self.object_votes > self.weapon_votes:
            return "object"
        return self.last_kind if self.last_kind in ("weapon", "object") else "object"


class GunStableIdTracker:
    """
    Persist firearm overlay ids across frames regardless of MOT internal ``track_id`` churn.

    Each spatial track accumulates per-frame ``object`` vs ``weapon`` labels. Display ids are
    **per person** (``T1``): if ``T1`` already had ``weapon1``, a later gun draw for ``T1`` stays
    ``weapon1`` even if a bottle was ``object1`` in between.
    """

    def __init__(self, *, iou_threshold: float = 0.15, max_missed_frames: int = 60) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = max(1, int(max_missed_frames))
        self._next_sid = 1
        self._next_object_display = 1
        self._next_weapon_display = 1
        self._person_weapon_display: dict[str, int] = {}
        #: Tentative weapon ids (brief flicker); cleared when object track wins.
        self._person_earned_weapon_ids: dict[str, set[int]] = {}
        #: Sustained real weapon — blocks auto-unlatch for holstered / concealed.
        self._person_weapon_sustained: set[str] = set()
        self._person_object_next: dict[str, int] = {}
        self._slots: list[_StableGunSlot] = []
        self._snap_sid_to_bbox: dict[int, tuple[int, int, int, int]] = {}

    def _slot_by_sid(self, sid: int) -> _StableGunSlot | None:
        for s in self._slots:
            if s.sid == sid:
                return s
        return None

    def person_key_for_sid(self, sid: int) -> str:
        s = self._slot_by_sid(sid)
        if s is None:
            return ""
        return str(s.person_key or "").strip()

    def majority_kind(self, sid: int) -> str | None:
        s = self._slot_by_sid(sid)
        if s is None or (s.object_votes + s.weapon_votes) == 0:
            return None
        return s.majority_kind()

    def _slot_weapon_confirmed(self, slot: _StableGunSlot) -> bool:
        # Immediate: first weapon-majority frame counts (no multi-frame vote delay).
        return slot.weapon_votes >= 1 and slot.weapon_votes >= slot.object_votes

    def _slot_weapon_sustained(self, slot: _StableGunSlot) -> bool:
        return slot.weapon_votes >= 1 and slot.weapon_votes > slot.object_votes

    def _earn_weapon_for_person(self, person_key: str, slot: _StableGunSlot) -> None:
        """Assign a weapon overlay id only after the track is weapon-confirmed."""
        pk = str(person_key or "").strip()
        if not pk or not self._slot_weapon_confirmed(slot):
            return
        wid = self._weapon_display_for_person(pk, slot)
        slot.weapon_earned = True
        self._person_earned_weapon_ids.setdefault(pk, set()).add(int(wid))
        if self._slot_weapon_sustained(slot):
            self._person_weapon_sustained.add(pk)

    def _weapon_display_for_person(self, person_key: str, slot: _StableGunSlot) -> int:
        pk = str(person_key or "").strip()
        if pk and pk in self._person_weapon_display:
            wid = int(self._person_weapon_display[pk])
            slot.weapon_display_id = wid
            return wid
        if slot.weapon_display_id is not None:
            wid = int(slot.weapon_display_id)
            if pk:
                self._person_weapon_display[pk] = wid
            return wid
        wid = self._next_weapon_display
        self._next_weapon_display += 1
        slot.weapon_display_id = wid
        if pk:
            self._person_weapon_display[pk] = wid
        return wid

    def _object_display_for_person(self, person_key: str, slot: _StableGunSlot) -> int:
        pk = str(person_key or "").strip()
        if slot.object_display_id is not None:
            return int(slot.object_display_id)
        if pk:
            oid = int(self._person_object_next.get(pk, 1))
            self._person_object_next[pk] = oid + 1
        else:
            oid = self._next_object_display
            self._next_object_display += 1
        slot.object_display_id = oid
        return oid

    def display_tag(self, sid: int) -> str | None:
        """``object3`` / ``weapon2`` from vote majority; ``None`` if track unknown."""
        s = self._slot_by_sid(sid)
        if s is None or (s.object_votes + s.weapon_votes) == 0:
            return None
        mk = s.majority_kind()
        if mk == "weapon":
            if self._slot_weapon_confirmed(s):
                self._earn_weapon_for_person(s.person_key, s)
            wid = s.weapon_display_id
            if wid is None and s.person_key in self._person_weapon_display:
                wid = int(self._person_weapon_display[s.person_key])
            if wid is None:
                return None
            return f"weapon{wid}"
        oid = self._object_display_for_person(s.person_key, s)
        return f"object{oid}"

    def update(self, detections: list[tuple[int, int, int, int, str, str]]) -> dict[int, int]:
        """
        ``detections``: ``(x1, y1, x2, y2, kind, person_key)`` with ``person_key`` like ``T1``.

        Returns detection index -> internal ``sid`` for tracks with at least one labeled frame.
        """
        self._snap_sid_to_bbox = {
            s.sid: tuple(s.bbox) for s in self._slots if (s.object_votes + s.weapon_votes) > 0
        }

        matched_sid: set[int] = set()
        matched_di: set[int] = set()
        assignments: dict[int, int] = {}

        if detections:
            pairs: list[tuple[float, int, _StableGunSlot]] = []
            for di, det in enumerate(detections):
                bx1, by1, bx2, by2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                pk = str(det[5]) if len(det) > 5 else ""
                for s in self._slots:
                    iou = _iou_xyxy_pair(bx1, by1, bx2, by2, *s.bbox)
                    if iou >= self.iou_threshold:
                        score = iou + (0.12 if pk and s.person_key == pk else 0.0)
                        pairs.append((score, di, s))
            pairs.sort(key=lambda t: -t[0])
            used_slot_obj: set[int] = set()
            for _, di, s in pairs:
                if di in matched_di or id(s) in used_slot_obj:
                    continue
                used_slot_obj.add(id(s))
                matched_di.add(di)
                matched_sid.add(s.sid)
                d = detections[di]
                bx1, by1, bx2, by2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                kind = str(d[4])
                pk = str(d[5]) if len(d) > 5 else ""
                owner_pk = str(s.person_key or "").strip() or pk
                if pk and not s.person_key:
                    s.person_key = pk
                s.bbox = (bx1, by1, bx2, by2)
                s.missed_frames = 0
                s.record_kind(kind)
                if kind == "weapon" and owner_pk:
                    self._earn_weapon_for_person(owner_pk, s)
                assignments[di] = s.sid

        for s in self._slots:
            if s.sid not in matched_sid:
                s.missed_frames += 1

        self._slots = [s for s in self._slots if s.missed_frames <= self.max_missed_frames]

        for di, det in enumerate(detections):
            if di in matched_di:
                continue
            kind = str(det[4])
            if kind not in ("weapon", "object"):
                continue
            bx1, by1, bx2, by2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            pk = str(det[5]) if len(det) > 5 else ""
            sid = self._next_sid
            self._next_sid += 1
            slot = _StableGunSlot(
                sid=sid, bbox=(bx1, by1, bx2, by2), missed_frames=0, person_key=pk
            )
            slot.record_kind(kind)
            if kind == "weapon" and pk:
                self._earn_weapon_for_person(pk, slot)
            elif kind == "object":
                self._object_display_for_person(pk, slot)
            self._slots.append(slot)
            assignments[di] = sid

        return assignments

    def stable_id_for_hold_box(self, bbox: tuple[int, int, int, int]) -> int | None:
        refs: list[tuple[int, tuple[int, int, int, int]]]
        if self._snap_sid_to_bbox:
            refs = [(int(sid), tuple(bb)) for sid, bb in self._snap_sid_to_bbox.items()]
        else:
            refs = [
                (s.sid, tuple(s.bbox))
                for s in self._slots
                if (s.object_votes + s.weapon_votes) > 0
            ]

        bx1, by1, bx2, by2 = bbox
        best_sid: int | None = None
        best_i = 0.0
        for sid, ref in refs:
            iou = _iou_xyxy_pair(bx1, by1, bx2, by2, *ref)
            if iou > best_i:
                best_i = iou
                best_sid = int(sid)
        thresh = max(0.05, float(self.iou_threshold) * 0.35)
        if best_sid is None or best_i < thresh:
            return None
        return best_sid

    def display_tag_for_hold_box(self, bbox: tuple[int, int, int, int]) -> str | None:
        sid = self.stable_id_for_hold_box(bbox)
        if sid is None:
            return None
        tag = self.display_tag(sid)
        return f"{tag} hold" if tag else None

    def person_weapon_confirmed(self, person_key: str) -> bool:
        """Stable firearm track: weapon votes clearly beat object (for latch)."""
        pk = str(person_key or "").strip()
        if not pk:
            return False
        for s in self._slots:
            if s.person_key != pk:
                continue
            if self._slot_weapon_confirmed(s):
                return True
        return False

    def person_weapon_sustained(self, person_key: str) -> bool:
        """True after a real weapon track (not a brief phone/hand flicker)."""
        pk = str(person_key or "").strip()
        return bool(pk) and pk in self._person_weapon_sustained

    def person_object_confirmed(self, person_key: str) -> bool:
        """Object track won (phone / bottle / hand clutter)."""
        pk = str(person_key or "").strip()
        if not pk:
            return False
        slots = [s for s in self._slots if s.person_key == pk and (s.object_votes + s.weapon_votes) > 0]
        if not slots:
            return False
        return all(
            s.object_votes >= 4 and s.object_votes > s.weapon_votes + 1 for s in slots
        )

    def reconcile_tentative_weapon(self, person_key: str) -> None:
        """Drop brief weapon flicker when object track dominates (phone / hand)."""
        pk = str(person_key or "").strip()
        if not pk or pk in self._person_weapon_sustained:
            return
        if not self.person_track_object_majority(pk) or not self.person_object_confirmed(pk):
            return
        self._person_earned_weapon_ids.pop(pk, None)
        for s in self._slots:
            if s.person_key == pk:
                s.weapon_earned = False

    def person_track_object_majority(self, person_key: str) -> bool:
        """True when every gun track for this person is object-majority (brief weapon flicker cleared)."""
        pk = str(person_key or "").strip()
        if not pk:
            return False
        slots = [s for s in self._slots if s.person_key == pk and (s.object_votes + s.weapon_votes) > 0]
        if not slots:
            return False
        return all(s.object_votes > s.weapon_votes for s in slots)

    def person_ever_had_weapon(self, person_key: str) -> bool:
        """True only after a sustained weapon track (holstered gun stays armed concealed)."""
        return self.person_weapon_sustained(person_key)


class PersonArmedLatch:
    """
    Per-person latched armed state: once armed, stay armed when the gun is concealed.

    With ``confirm_weapon_seconds=0`` and ``confirm_weapon_frames=1``, a visible weapon
    (plus track confirm) arms on the first qualifying frame — no multi-second delay.
    """

    def __init__(
        self,
        *,
        confirm_weapon_frames: int = 1,
        confirm_weapon_seconds: float = 0.0,
        confirm_break_grace_seconds: float = 0.15,
        unlatch_object_frames: int = 18,
    ) -> None:
        self._armed: set[str] = set()
        self._peak_conf: dict[str, float] = {}
        self._weapon_streak: dict[str, int] = {}
        self._object_only_frames: dict[str, int] = {}
        self._weapon_bracket: dict[str, str] = {}
        self._confirm_started_mono: dict[str, float] = {}
        self._last_qualify_mono: dict[str, float] = {}
        self._confirm_weapon_frames = max(1, int(confirm_weapon_frames))
        self._confirm_weapon_seconds = max(0.0, float(confirm_weapon_seconds))
        self._confirm_break_grace_seconds = max(0.0, float(confirm_break_grace_seconds))
        self._unlatch_object_frames = max(1, int(unlatch_object_frames))

    def note_weapon_bracket(self, person_key: str, class_name: str) -> None:
        """Remember ``[Gun]`` / ``[Knife]`` for concealed labels (ignores phone/object)."""
        from weapon_ai.overlay.labels import armed_weapon_bracket

        pk = str(person_key or "").strip()
        if not pk:
            return
        bracket = armed_weapon_bracket(class_name)
        if bracket in ("[Gun]", "[Knife]"):
            self._weapon_bracket[pk] = bracket

    def remembered_weapon_bracket(self, person_key: str) -> str:
        pk = str(person_key or "").strip()
        return self._weapon_bracket.get(pk, "[Gun]")

    def _reset_confirm(self, pk: str) -> None:
        self._confirm_started_mono.pop(pk, None)
        self._last_qualify_mono.pop(pk, None)
        self._weapon_streak[pk] = 0

    def update(
        self,
        person_key: str,
        *,
        visible_weapon_conf: float = 0.0,
        track_confirmed_weapon: bool = False,
        track_object_majority: bool = False,
        ever_had_weapon: bool = False,
        instant_arm_threshold: float = 0.0,
        min_weapon_conf: float = 0.0,
        gun_class_frame: bool = False,
        smartphone_class_frame: bool = False,
        now_mono: float | None = None,
    ) -> None:
        del gun_class_frame, smartphone_class_frame, instant_arm_threshold
        pk = str(person_key or "").strip()
        if not pk:
            return
        now = float(time.monotonic() if now_mono is None else now_mono)
        vwc = float(visible_weapon_conf)
        min_wc = float(min_weapon_conf)
        qualifies = vwc >= min_wc and float(vwc) > 0.0

        if qualifies:
            self._weapon_streak[pk] = self._weapon_streak.get(pk, 0) + 1
            self._peak_conf[pk] = max(self._peak_conf.get(pk, 0.0), vwc)
            self._object_only_frames[pk] = 0
            self._last_qualify_mono[pk] = now
            if pk not in self._confirm_started_mono:
                self._confirm_started_mono[pk] = now
        else:
            self._weapon_streak[pk] = 0
            last_q = self._last_qualify_mono.get(pk)
            if last_q is None or (now - last_q) > self._confirm_break_grace_seconds:
                self._reset_confirm(pk)

        if pk in self._armed:
            if ever_had_weapon:
                self._object_only_frames[pk] = 0
                return
            if qualifies or track_confirmed_weapon:
                self._object_only_frames[pk] = 0
            elif track_object_majority:
                n = self._object_only_frames.get(pk, 0) + 1
                self._object_only_frames[pk] = n
                if n >= self._unlatch_object_frames:
                    self._armed.discard(pk)
                    self._peak_conf.pop(pk, None)
                    self._weapon_streak.pop(pk, None)
                    self._object_only_frames.pop(pk, None)
                    self._weapon_bracket.pop(pk, None)
                    self._reset_confirm(pk)
            return

        if not track_confirmed_weapon:
            return

        # Already earned a sustained weapon id earlier (e.g. returning from concealed).
        if ever_had_weapon and qualifies:
            self._armed.add(pk)
            return

        started = self._confirm_started_mono.get(pk)
        if started is None:
            return
        elapsed = now - started
        streak = self._weapon_streak.get(pk, 0)
        if self._confirm_weapon_seconds > 0.0:
            if elapsed >= self._confirm_weapon_seconds and qualifies:
                self._armed.add(pk)
        elif streak >= self._confirm_weapon_frames:
            self._armed.add(pk)

    def person_visual_state(self, person_key: str, *, visible_weapon: bool = False) -> str:
        """``clear`` | ``armed_gun`` | ``armed_concealed`` (no ambiguous / smartphone tags)."""
        if not self.is_armed(person_key):
            return "clear"
        return "armed_gun" if visible_weapon else "armed_concealed"

    def is_armed(self, person_key: str) -> bool:
        return str(person_key or "").strip() in self._armed

    def effective_gun_conf(self, person_key: str, frame_conf: float) -> float:
        pk = str(person_key or "").strip()
        if pk in self._armed:
            return max(float(frame_conf), self._peak_conf.get(pk, 0.0))
        return float(frame_conf)


@dataclass
class _FirearmHoldSlot:
    box: tuple
    miss: int = 0
    seen: int = 0
    published: bool = False


class FirearmOverlayHold:
    """
    Temporal hold for knife/gun overlay boxes across brief YOLO dropouts.

    Draws detections immediately (no confirm delay). After a box is seen, keeps it
    for up to ``max_hold_frames`` when the detector briefly misses it.
    """

    def __init__(
        self,
        *,
        max_hold_frames: int = 18,
        min_confirm_frames: int = 1,
        iou_threshold: float = 0.05,
        center_match_px: float = 120.0,
    ) -> None:
        self.max_hold_frames = max(0, int(max_hold_frames))
        self.min_confirm_frames = max(1, int(min_confirm_frames))
        self.iou_threshold = float(iou_threshold)
        self.center_match_px = float(center_match_px)
        self._slots: list[_FirearmHoldSlot] = []

    @staticmethod
    def _label(box: tuple) -> str:
        return str(box[4]) if len(box) > 4 else ""

    @staticmethod
    def _person_ridx(box: tuple) -> int:
        return int(box[7]) if len(box) > 7 else -1

    @staticmethod
    def _center(box: tuple) -> tuple[float, float]:
        return (0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3])))

    def _match_score(self, det: tuple, slot_box: tuple) -> float:
        dx1, dy1, dx2, dy2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        sx1, sy1, sx2, sy2 = int(slot_box[0]), int(slot_box[1]), int(slot_box[2]), int(slot_box[3])
        iou = _iou_xyxy_pair(dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2)
        dlab = self._label(det)
        dpr = self._person_ridx(det)
        slab = self._label(slot_box)
        spr = self._person_ridx(slot_box)
        same_person = dpr >= 0 and dpr == spr
        same_label = bool(dlab) and dlab == slab
        dcx, dcy = self._center(det)
        scx, scy = self._center(slot_box)
        dist = float(((dcx - scx) ** 2 + (dcy - scy) ** 2) ** 0.5)

        score = -1.0
        if iou >= self.iou_threshold:
            score = float(iou)
        elif same_person and same_label and dist <= self.center_match_px:
            # Jittery knife/gun boxes often lose IoU — still treat as same track.
            score = 0.05 + max(0.0, 1.0 - dist / max(1.0, self.center_match_px)) * 0.2
        elif same_person and dist <= self.center_match_px * 0.5:
            score = 0.02 + max(0.0, 1.0 - dist / max(1.0, self.center_match_px)) * 0.1
        else:
            return -1.0
        if same_person:
            score += 0.15
        if same_label:
            score += 0.05
        return score

    def update(self, detections: list[tuple]) -> list[tuple]:
        boxes, _held = self.update_with_held(detections)
        return boxes

    def update_with_held(self, detections: list[tuple]) -> tuple[list[tuple], set[int]]:
        """Return boxes plus indices that are held-over (not live YOLO this frame)."""
        if self.max_hold_frames <= 0 and self.min_confirm_frames <= 1:
            dets = list(detections)
            return dets, set()

        dets = list(detections)
        if not self._slots and not dets:
            return [], set()

        pairs: list[tuple[float, int, int]] = []
        for di, det in enumerate(dets):
            for si, slot in enumerate(self._slots):
                score = self._match_score(det, slot.box)
                if score < 0.0:
                    continue
                pairs.append((score, di, si))
        pairs.sort(key=lambda t: -t[0])

        matched_di: set[int] = set()
        matched_si: set[int] = set()
        for _, di, si in pairs:
            if di in matched_di or si in matched_si:
                continue
            matched_di.add(di)
            matched_si.add(si)
            slot = self._slots[si]
            slot.box = dets[di]
            slot.miss = 0
            slot.seen = min(slot.seen + 1, self.min_confirm_frames + 3)
            if slot.seen >= self.min_confirm_frames:
                slot.published = True

        for di, det in enumerate(dets):
            if di in matched_di:
                continue
            seen = 1
            published = seen >= self.min_confirm_frames
            self._slots.append(
                _FirearmHoldSlot(
                    box=det,
                    miss=0,
                    seen=seen,
                    published=published,
                )
            )
            matched_si.add(len(self._slots) - 1)

        kept: list[_FirearmHoldSlot] = []
        out: list[tuple] = []
        held_idxs: set[int] = set()
        for si, slot in enumerate(self._slots):
            if si in matched_si:
                kept.append(slot)
                if slot.published:
                    out.append(slot.box)
                continue
            if not slot.published:
                continue
            miss = slot.miss + 1
            if miss > self.max_hold_frames:
                continue
            slot.miss = miss
            kept.append(slot)
            held_idxs.add(len(out))
            out.append(slot.box)
        self._slots = kept
        return out, held_idxs


@dataclass
class _MotionGunTrack:
    tid: int
    bbox: tuple[int, int, int, int]
    hist: deque = field(default_factory=lambda: deque(maxlen=6))
    missed: int = 0
    person_key: str = ""


class WeaponPersonAssociator:
    """
    Associate each firearm/object box with the person it *moves with*, using motion.

    Instead of nearest-centroid / largest-overlap, each person and each firearm gets a
    short centroid history → velocity vector. A gun is assigned to the person whose
    **position and velocity** best agree with the gun's. This is robust to occlusion:
    when a bystander walks in front of the armed person, the bystander overlaps the gun
    box but moves with a different velocity, so the gun stays with its true carrier.

    Scoring per candidate person ``p`` for gun ``g``:
        score = spatial(iou + inside-bonus) + velocity_weight * cos(v_g, v_p)
    The velocity term only applies when *both* are moving faster than ``min_speed_px``
    (cosine is scale-free, so it survives frame-skipping); otherwise it falls back to
    pure spatial association. A previous owner gets a small hysteresis bonus so the
    assignment does not flicker between similarly-placed people.
    """

    def __init__(
        self,
        *,
        history: int = 6,
        velocity_weight: float = 0.35,
        min_speed_px: float = 1.5,
        gun_iou_match: float = 0.2,
        max_missed: int = 30,
        keep_bonus: float = 0.05,
    ) -> None:
        self._history = max(2, int(history))
        self._velocity_weight = float(velocity_weight)
        self._min_speed = float(min_speed_px)
        self._gun_iou_match = float(gun_iou_match)
        self._max_missed = max(1, int(max_missed))
        self._keep_bonus = float(keep_bonus)
        self._person_hist: dict[str, deque] = {}
        self._person_seen: dict[str, int] = {}
        self._gun_tracks: list[_MotionGunTrack] = []
        self._next_gun_tid = 1
        self._frame = 0

    @staticmethod
    def _center(b: tuple[int, int, int, int]) -> tuple[float, float]:
        return (0.5 * (float(b[0]) + float(b[2])), 0.5 * (float(b[1]) + float(b[3])))

    def _velocity(self, hist) -> tuple[float, float]:
        if not hist or len(hist) < 2:
            return (0.0, 0.0)
        x0, y0 = hist[0]
        x1, y1 = hist[-1]
        n = len(hist) - 1
        return ((x1 - x0) / n, (y1 - y0) / n)

    def _update_persons(self, persons: list[tuple[str, tuple[int, int, int, int]]]) -> None:
        self._frame += 1
        for key, bbox in persons:
            k = str(key or "").strip()
            if not k:
                continue
            h = self._person_hist.setdefault(k, deque(maxlen=self._history))
            c = self._center(bbox)
            last = h[-1] if h else None
            if last is None or abs(c[0] - last[0]) + abs(c[1] - last[1]) > 0.5:
                h.append(c)
            self._person_seen[k] = self._frame
        stale = [
            k for k, f in self._person_seen.items() if self._frame - f > self._max_missed
        ]
        for k in stale:
            self._person_hist.pop(k, None)
            self._person_seen.pop(k, None)

    def _update_guns(
        self, guns: list[tuple[int, tuple[int, int, int, int]]]
    ) -> dict[int, _MotionGunTrack]:
        pairs: list[tuple[float, int, int]] = []
        for gi, (_gidx, bbox) in enumerate(guns):
            gcx, gcy = self._center(bbox)
            # Firearm boxes are small and can move farther than their own width per
            # frame, so IoU alone drops the track. Allow a center-distance match
            # scaled by box size (like the low-IoU firearm MOT gate).
            radius = 2.0 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) + 15.0
            for ti, t in enumerate(self._gun_tracks):
                iou = _iou_xyxy_pair(*bbox, *t.bbox)
                tcx, tcy = self._center(t.bbox)
                cdist = ((gcx - tcx) ** 2 + (gcy - tcy) ** 2) ** 0.5
                if iou >= self._gun_iou_match or cdist <= radius:
                    pairs.append((iou - cdist * 1e-4, gi, ti))
        pairs.sort(key=lambda p: -p[0])
        matched_gi: set[int] = set()
        matched_ti: set[int] = set()
        assigned: dict[int, _MotionGunTrack] = {}
        for _iou, gi, ti in pairs:
            if gi in matched_gi or ti in matched_ti:
                continue
            matched_gi.add(gi)
            matched_ti.add(ti)
            gidx, bbox = guns[gi]
            t = self._gun_tracks[ti]
            c = self._center(bbox)
            last = t.hist[-1] if t.hist else None
            if last is None or abs(c[0] - last[0]) + abs(c[1] - last[1]) > 0.5:
                t.hist.append(c)
            t.bbox = bbox
            t.missed = 0
            assigned[gidx] = t
        for gi, (gidx, bbox) in enumerate(guns):
            if gi in matched_gi:
                continue
            t = _MotionGunTrack(tid=self._next_gun_tid, bbox=bbox)
            self._next_gun_tid += 1
            t.hist.append(self._center(bbox))
            self._gun_tracks.append(t)
            assigned[gidx] = t
        for ti, t in enumerate(self._gun_tracks):
            if ti not in matched_ti:
                t.missed += 1
        self._gun_tracks = [t for t in self._gun_tracks if t.missed <= self._max_missed]
        return assigned

    def associate(
        self,
        persons: list[tuple[str, tuple[int, int, int, int]]],
        guns: list[tuple[int, tuple[int, int, int, int]]],
    ) -> dict[int, str]:
        """Return ``gidx -> person_key`` (best motion+position match); ``""`` if none."""
        self._update_persons(persons)
        gun_map = self._update_guns(guns)
        out: dict[int, str] = {}
        for gidx, gbbox in guns:
            t = gun_map.get(gidx)
            gvel = self._velocity(t.hist) if t is not None else (0.0, 0.0)
            gspeed = (gvel[0] ** 2 + gvel[1] ** 2) ** 0.5
            gcx, gcy = self._center(gbbox)
            prev_key = t.person_key if t is not None else ""
            best_key = ""
            best_score = -1e9
            for key, pbbox in persons:
                k = str(key or "").strip()
                if not k:
                    continue
                iou = _iou_xyxy_pair(*gbbox, *pbbox)
                inside = pbbox[0] <= gcx <= pbbox[2] and pbbox[1] <= gcy <= pbbox[3]
                if iou <= 0.0 and not inside:
                    pcx, pcy = self._center(pbbox)
                    pdiag = (
                        (pbbox[2] - pbbox[0]) ** 2 + (pbbox[3] - pbbox[1]) ** 2
                    ) ** 0.5
                    if pdiag <= 0 or (
                        (gcx - pcx) ** 2 + (gcy - pcy) ** 2
                    ) ** 0.5 > 0.6 * pdiag:
                        continue
                spatial = iou + (0.15 if inside else 0.0)
                vterm = 0.0
                pvel = self._velocity(self._person_hist.get(k))
                pspeed = (pvel[0] ** 2 + pvel[1] ** 2) ** 0.5
                if gspeed >= self._min_speed and pspeed >= self._min_speed:
                    cos = (gvel[0] * pvel[0] + gvel[1] * pvel[1]) / (gspeed * pspeed)
                    vterm = self._velocity_weight * cos
                score = spatial + vterm
                if k == prev_key:
                    score += self._keep_bonus
                if score > best_score:
                    best_score = score
                    best_key = k
            if t is not None and best_key:
                t.person_key = best_key
            out[gidx] = best_key
        return out


@dataclass
class ByteTrackConfig:
    frame_rate: float = 30.0
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True

    @staticmethod
    def for_firearms(*, frame_rate: float = 30.0, track_buffer: int = 30) -> ByteTrackConfig:
        """Firearm MOT thresholds (less aggressive than legacy 0.01/0.05 to avoid id/score flicker)."""
        return ByteTrackConfig(
            frame_rate=frame_rate,
            track_high_thresh=0.15,
            track_low_thresh=0.08,
            new_track_thresh=0.12,
            track_buffer=track_buffer,
            match_thresh=0.48,
            fuse_score=True,
        )


class DisplayTrackIds:
    """Map Ultralytics global ``track_id`` to small 1-based display ids per stream (``T1`` vs ``gun1``)."""

    def __init__(self) -> None:
        self._by_tid: dict[int, int] = {}
        self._next = 1

    def display_num(self, track_id: int) -> int:
        tid = int(track_id)
        if tid not in self._by_tid:
            self._by_tid[tid] = self._next
            self._next += 1
        return self._by_tid[tid]


class AppearanceDisplayIds:
    """Stable Person N labels via Re-ID embeddings (survives ByteTrack id swaps).

    Each frame, raw ``track_id`` boxes are matched to stored appearance profiles
    so labels stay with the same person even when MOT reassigns track ids.
    """

    def __init__(
        self,
        *,
        sim_threshold: float = 0.62,
        profile_blend: float = 0.35,
    ) -> None:
        self._profiles: dict[int, np.ndarray] = {}
        self._next = 1
        self._sim_threshold = float(sim_threshold)
        self._profile_blend = float(profile_blend)
        self._track_hint: dict[int, int] = {}

    @staticmethod
    def _blend_profile(existing: np.ndarray, fresh: np.ndarray, alpha: float) -> np.ndarray:
        from weapon_ai.reid.embeddings import _l2_normalize

        a = float(alpha)
        mixed = (1.0 - a) * existing + a * fresh
        return _l2_normalize(mixed)

    def assign(self, items: list[tuple[int, np.ndarray | None]]) -> dict[int, int]:
        """Map raw ``track_id`` -> stable ``display_id`` for one frame."""
        from weapon_ai.reid.embeddings import cosine_similarity

        tracks = [(int(tid), emb) for tid, emb in items]
        if not tracks:
            return {}

        candidates: list[tuple[float, int, int]] = []
        for tid, emb in tracks:
            if emb is None:
                continue
            for did, prof in self._profiles.items():
                sim = cosine_similarity(emb, prof)
                if sim >= self._sim_threshold:
                    candidates.append((sim, tid, did))

        candidates.sort(key=lambda row: -row[0])
        out: dict[int, int] = {}
        assigned_tracks: set[int] = set()
        assigned_profiles: set[int] = set()

        for sim, tid, did in candidates:
            if tid in assigned_tracks or did in assigned_profiles:
                continue
            out[tid] = did
            assigned_tracks.add(tid)
            assigned_profiles.add(did)
            prof = self._profiles[did]
            emb = next(e for t, e in tracks if t == tid and e is not None)
            self._profiles[did] = self._blend_profile(prof, emb, self._profile_blend)

        for tid, emb in tracks:
            if tid in out:
                continue
            if tid in self._track_hint:
                did = self._track_hint[tid]
            else:
                did = self._next
                self._next += 1
                self._track_hint[tid] = did
            out[tid] = did
            if emb is None:
                continue
            if did in self._profiles:
                self._profiles[did] = self._blend_profile(
                    self._profiles[did], emb, self._profile_blend
                )
            else:
                self._profiles[did] = emb

        return out


class IndexedBoxByteTracker:
    """ByteTrack on an ordered list of ``(x1, y1, x2, y2, det_conf)``; returns **index** -> ``track_id``."""

    def __init__(self, cfg: ByteTrackConfig | None = None) -> None:
        self._cfg = cfg or ByteTrackConfig()
        args = IterableSimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=float(self._cfg.track_high_thresh),
            track_low_thresh=float(self._cfg.track_low_thresh),
            new_track_thresh=float(self._cfg.new_track_thresh),
            track_buffer=int(self._cfg.track_buffer),
            match_thresh=float(self._cfg.match_thresh),
            fuse_score=bool(self._cfg.fuse_score),
        )
        self._tracker = BYTETracker(args)

    def reset(self) -> None:
        self._tracker.reset()

    def predict_active(
        self,
        frame_hw: tuple[int, int],
        orig_img: np.ndarray | None,
    ) -> dict[int, tuple[int, int, int, int, float]]:
        """Advance MOT one frame with no new detections; return track_id -> (xyxy, score)."""
        h, w = int(frame_hw[0]), int(frame_hw[1])
        self._tracker.update(
            Boxes(np.zeros((0, 6), dtype=np.float32), orig_shape=(h, w)),
            orig_img,
        )
        out: dict[int, tuple[int, int, int, int, float]] = {}
        for pool in (self._tracker.tracked_stracks, self._tracker.lost_stracks):
            for track in pool:
                if not track.is_activated:
                    continue
                tid = int(track.track_id)
                if tid in out:
                    continue
                x1, y1, tw, th = track.tlwh
                x2 = int(round(float(x1) + float(tw)))
                y2 = int(round(float(y1) + float(th)))
                out[tid] = (int(round(float(x1))), int(round(float(y1))), x2, y2, float(track.score))
        return out

    def update(
        self,
        entries: list[tuple[int, int, int, int, float]],
        frame_hw: tuple[int, int],
        orig_img: np.ndarray | None,
        *,
        yolo_cls: float = 0.0,
        min_iou_match: float = 0.05,
    ) -> dict[int, int]:
        """
        ``entries`` order is preserved; return mapping **position in list** -> **track_id**.
        """
        h, w = int(frame_hw[0]), int(frame_hw[1])
        if not entries:
            return {}

        data_rows: list[list[float]] = []
        for x1, y1, x2, y2, det_conf in entries:
            dc = max(float(det_conf), 1e-4)
            data_rows.append([float(x1), float(y1), float(x2), float(y2), dc, float(yolo_cls)])

        boxes = Boxes(np.asarray(data_rows, dtype=np.float32), orig_shape=(h, w))
        tracks: np.ndarray = self._tracker.update(boxes, orig_img)
        if tracks.size == 0:
            return {}

        det_xyxy = np.asarray([[t[0], t[1], t[2], t[3]] for t in entries], dtype=np.float32)
        txy = tracks[:, :4].astype(np.float32)
        tids = tracks[:, 4]
        assigned = _greedy_assign(det_xyxy, txy, tids, min_iou=min_iou_match)
        out: dict[int, int] = {}
        for local_i, tid in enumerate(assigned):
            if tid is not None:
                out[local_i] = tid
        return out


@dataclass
class BotSortConfig:
    """BoT-SORT tuning (Kalman XYWH + optional GMC / ReID)."""

    frame_rate: float = 30.0
    track_high_thresh: float = 0.15
    track_low_thresh: float = 0.08
    new_track_thresh: float = 0.12
    track_buffer: int = 45
    match_thresh: float = 0.48
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    with_reid: bool = False
    reid_model: str = "auto"

    @staticmethod
    def for_persons(*, frame_rate: float = 30.0, track_buffer: int = 45) -> BotSortConfig:
        """Crowd-friendly BoT-SORT defaults for facing corridor person MOT."""
        return BotSortConfig(
            frame_rate=frame_rate,
            track_high_thresh=0.15,
            track_low_thresh=0.08,
            new_track_thresh=0.12,
            track_buffer=track_buffer,
            match_thresh=0.48,
            fuse_score=True,
            gmc_method="sparseOptFlow",
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=False,
        )


def _tracker_namespace(
    cfg: ByteTrackConfig | BotSortConfig,
    *,
    tracker_type: str,
) -> IterableSimpleNamespace:
    args = IterableSimpleNamespace(
        tracker_type=str(tracker_type),
        track_high_thresh=float(cfg.track_high_thresh),
        track_low_thresh=float(cfg.track_low_thresh),
        new_track_thresh=float(cfg.new_track_thresh),
        track_buffer=int(cfg.track_buffer),
        match_thresh=float(cfg.match_thresh),
        fuse_score=bool(cfg.fuse_score),
    )
    if tracker_type == "botsort":
        bcfg = cfg if isinstance(cfg, BotSortConfig) else BotSortConfig.for_persons(
            frame_rate=float(getattr(cfg, "frame_rate", 30.0)),
            track_buffer=int(getattr(cfg, "track_buffer", 45)),
        )
        args.gmc_method = str(bcfg.gmc_method)
        args.proximity_thresh = float(bcfg.proximity_thresh)
        args.appearance_thresh = float(bcfg.appearance_thresh)
        args.with_reid = bool(bcfg.with_reid)
        args.model = str(bcfg.reid_model)
    return args


class IndexedBoxBotSortTracker:
    """BoT-SORT on ordered detections; returns **index** -> ``track_id``."""

    def __init__(self, cfg: BotSortConfig | None = None) -> None:
        self._cfg = cfg or BotSortConfig.for_persons()
        args = _tracker_namespace(self._cfg, tracker_type="botsort")
        self._tracker = BOTSORT(args)

    def reset(self) -> None:
        self._tracker.reset()

    def predict_active(
        self,
        frame_hw: tuple[int, int],
        orig_img: np.ndarray | None,
    ) -> dict[int, tuple[int, int, int, int, float]]:
        h, w = int(frame_hw[0]), int(frame_hw[1])
        self._tracker.update(
            Boxes(np.zeros((0, 6), dtype=np.float32), orig_shape=(h, w)),
            orig_img,
        )
        out: dict[int, tuple[int, int, int, int, float]] = {}
        for pool in (self._tracker.tracked_stracks, self._tracker.lost_stracks):
            for track in pool:
                if not track.is_activated:
                    continue
                tid = int(track.track_id)
                if tid in out:
                    continue
                x1, y1, tw, th = track.tlwh
                x2 = int(round(float(x1) + float(tw)))
                y2 = int(round(float(y1) + float(th)))
                out[tid] = (int(round(float(x1))), int(round(float(y1))), x2, y2, float(track.score))
        return out

    def update(
        self,
        entries: list[tuple[int, int, int, int, float]],
        frame_hw: tuple[int, int],
        orig_img: np.ndarray | None,
        *,
        yolo_cls: float = 0.0,
        min_iou_match: float = 0.05,
    ) -> dict[int, int]:
        h, w = int(frame_hw[0]), int(frame_hw[1])
        if not entries:
            return {}

        data_rows: list[list[float]] = []
        for x1, y1, x2, y2, det_conf in entries:
            dc = max(float(det_conf), 1e-4)
            data_rows.append([float(x1), float(y1), float(x2), float(y2), dc, float(yolo_cls)])

        boxes = Boxes(np.asarray(data_rows, dtype=np.float32), orig_shape=(h, w))
        tracks: np.ndarray = self._tracker.update(boxes, orig_img)
        if tracks.size == 0:
            return {}

        det_xyxy = np.asarray([[t[0], t[1], t[2], t[3]] for t in entries], dtype=np.float32)
        txy = tracks[:, :4].astype(np.float32)
        tids = tracks[:, 4]
        assigned = _greedy_assign(det_xyxy, txy, tids, min_iou=min_iou_match)
        out: dict[int, int] = {}
        for local_i, tid in enumerate(assigned):
            if tid is not None:
                out[local_i] = tid
        return out


def make_indexed_box_tracker(
    *,
    tracker_type: str = "bytetrack",
    frame_rate: float = 30.0,
    track_buffer: int = 45,
) -> IndexedBoxByteTracker | IndexedBoxBotSortTracker:
    kind = str(tracker_type or "bytetrack").strip().lower()
    if kind in {"botsort", "bot-sort", "bot_sort"}:
        return IndexedBoxBotSortTracker(
            BotSortConfig.for_persons(frame_rate=float(frame_rate), track_buffer=int(track_buffer))
        )
    return IndexedBoxByteTracker(
        ByteTrackConfig(
            frame_rate=float(frame_rate),
            track_high_thresh=0.15,
            track_low_thresh=0.08,
            new_track_thresh=0.12,
            track_buffer=int(track_buffer),
            match_thresh=0.48,
            fuse_score=True,
        )
    )


class ThermalByteTracker:
    """Stateful ByteTrack over person-class rows; call ``update`` once per displayed frame."""

    def __init__(self, cfg: ByteTrackConfig | None = None) -> None:
        self._inner = IndexedBoxByteTracker(cfg)

    def reset(self) -> None:
        self._inner.reset()

    def predict_person_rows(
        self,
        frame_hw: tuple[int, int],
        orig_img: np.ndarray | None,
        track_meta: dict[int, tuple[float, int | None, str, float]],
    ) -> tuple[
        list[tuple[int, int, int, int, float, int | None, str, float]],
        dict[int, int],
    ]:
        """Between YOLO frames: propagate boxes via ByteTrack Kalman prediction."""
        active = self._inner.predict_active(frame_hw, orig_img)
        rows: list[tuple[int, int, int, int, float, int | None, str, float]] = []
        row_track: dict[int, int] = {}
        for tid in sorted(active.keys()):
            meta = track_meta.get(tid)
            if meta is None:
                continue
            x1, y1, x2, y2, det_c = active[tid]
            prob, cid, ytag, _stored_det = meta
            ridx = len(rows)
            rows.append((x1, y1, x2, y2, float(prob), cid, str(ytag), float(det_c)))
            row_track[ridx] = tid
        return rows, row_track

    def update(
        self,
        rows: list[tuple[int, int, int, int, float, int | None, str, float]],
        frame_hw: tuple[int, int],
        orig_img: np.ndarray | None,
    ) -> dict[int, int]:
        """Return mapping **global row index** -> **track_id** for matched persons only."""
        entries: list[tuple[int, int, int, int, float]] = []
        ridx_map: list[int] = []
        for ridx, r in enumerate(rows):
            x1, y1, x2, y2, _p, cid, _tag, det_conf = r
            if cid is not None and int(cid) != 0:
                continue
            entries.append((x1, y1, x2, y2, float(det_conf)))
            ridx_map.append(ridx)
        local = self._inner.update(entries, frame_hw, orig_img, yolo_cls=0.0, min_iou_match=0.05)
        return {ridx_map[i]: tid for i, tid in local.items()}
