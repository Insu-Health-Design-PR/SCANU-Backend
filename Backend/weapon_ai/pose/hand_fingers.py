"""MediaPipe hand landmarks: finger skeleton + gun grip / trigger / shooting cues."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from weapon_ai.detection.firearms import box_center as _box_center

# MediaPipe HandLandmarker 21-point topology.
_WRIST = 0
_THUMB_CMC, _THUMB_MCP, _THUMB_IP, _THUMB_TIP = 1, 2, 3, 4
_INDEX_MCP, _INDEX_PIP, _INDEX_DIP, _INDEX_TIP = 5, 6, 7, 8
_MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_DIP, _MIDDLE_TIP = 9, 10, 11, 12
_RING_MCP, _RING_PIP, _RING_DIP, _RING_TIP = 13, 14, 15, 16
_PINKY_MCP, _PINKY_PIP, _PINKY_DIP, _PINKY_TIP = 17, 18, 19, 20

_HAND_CONNECTIONS = (
    (_WRIST, _THUMB_CMC),
    (_THUMB_CMC, _THUMB_MCP),
    (_THUMB_MCP, _THUMB_IP),
    (_THUMB_IP, _THUMB_TIP),
    (_WRIST, _INDEX_MCP),
    (_INDEX_MCP, _INDEX_PIP),
    (_INDEX_PIP, _INDEX_DIP),
    (_INDEX_DIP, _INDEX_TIP),
    (_WRIST, _MIDDLE_MCP),
    (_MIDDLE_MCP, _MIDDLE_PIP),
    (_MIDDLE_PIP, _MIDDLE_DIP),
    (_MIDDLE_DIP, _MIDDLE_TIP),
    (_WRIST, _RING_MCP),
    (_RING_MCP, _RING_PIP),
    (_RING_PIP, _RING_DIP),
    (_RING_DIP, _RING_TIP),
    (_WRIST, _PINKY_MCP),
    (_PINKY_MCP, _PINKY_PIP),
    (_PINKY_PIP, _PINKY_DIP),
    (_PINKY_DIP, _PINKY_TIP),
    (_INDEX_MCP, _MIDDLE_MCP),
    (_MIDDLE_MCP, _RING_MCP),
    (_RING_MCP, _PINKY_MCP),
)

_COLOR_FINGER_BGR = (80, 255, 80)
_COLOR_INDEX_BGR = (0, 255, 255)
_COLOR_TRIGGER_BGR = (0, 0, 255)
_COLOR_SHOOT_BGR = (0, 80, 255)

DEFAULT_HAND_MODEL = Path(__file__).resolve().parents[2] / "trained_models" / "pose" / "hand_landmarker.task"


@dataclass(frozen=True)
class HandLandmarks:
    """Full-frame 21 hand landmarks as (x, y, z) in pixels (z unused for 2D cues)."""

    landmarks: np.ndarray  # shape (21, 3)
    handedness: str = ""  # "Left" | "Right" | ""
    score: float = 1.0

    def point(self, idx: int) -> tuple[float, float]:
        return float(self.landmarks[idx, 0]), float(self.landmarks[idx, 1])

    def wrist(self) -> tuple[float, float]:
        return self.point(_WRIST)

    def index_tip(self) -> tuple[float, float]:
        return self.point(_INDEX_TIP)

    def index_mcp(self) -> tuple[float, float]:
        return self.point(_INDEX_MCP)

    def palm_center(self) -> tuple[float, float]:
        idxs = (_WRIST, _INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP)
        xs = [float(self.landmarks[i, 0]) for i in idxs]
        ys = [float(self.landmarks[i, 1]) for i in idxs]
        return sum(xs) / len(xs), sum(ys) / len(ys)


GripState = str  # "none" | "grip" | "finger_on_trigger" | "shooting"


@dataclass(frozen=True)
class HandGunCue:
    """Per-hand relationship to a nearby firearm box."""

    state: GripState
    boost: float
    hand: HandLandmarks
    gun_box: tuple[int, int, int, int] | None = None


def default_hand_model_path() -> Path:
    return DEFAULT_HAND_MODEL


class HandFingerEstimator:
    """MediaPipe HandLandmarker wrapper; crops around person wrists / upper body."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        max_num_hands: int = 4,
        min_detection_confidence: float = 0.25,
        min_tracking_confidence: float = 0.25,
    ) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        path = Path(model_path) if model_path else DEFAULT_HAND_MODEL
        if not path.is_file():
            raise FileNotFoundError(f"Hand landmarker model not found: {path}")
        base = mp_python.BaseOptions(model_asset_path=str(path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max(1, int(max_num_hands)),
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=float(min_tracking_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

    def _detect_rgb(self, rgb: np.ndarray) -> list[HandLandmarks]:
        from mediapipe import Image as MpImage
        from mediapipe import ImageFormat

        if rgb is None or rgb.size == 0:
            return []
        h, w = rgb.shape[:2]
        # MediaPipe is more reliable when the hand fills a decent portion of the crop.
        # Upscale small ROIs so distant hands still resolve.
        scale = 1.0
        work = rgb
        min_side = min(h, w)
        if min_side < 256:
            scale = 256.0 / float(min_side)
            work = cv2.resize(
                rgb,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_LINEAR,
            )
        wh, ww = work.shape[:2]
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(work))
        result = self._landmarker.detect(mp_image)
        hands: list[HandLandmarks] = []
        if not result.hand_landmarks:
            return hands
        handedness_list = result.handedness or []
        inv = 1.0 / scale if scale != 1.0 else 1.0
        for i, lm_list in enumerate(result.hand_landmarks):
            pts = np.zeros((21, 3), dtype=np.float32)
            for j, lm in enumerate(lm_list[:21]):
                pts[j, 0] = float(lm.x) * ww * inv
                pts[j, 1] = float(lm.y) * wh * inv
                pts[j, 2] = float(getattr(lm, "z", 0.0) or 0.0)
            label = ""
            score = 1.0
            if i < len(handedness_list) and handedness_list[i]:
                cat = handedness_list[i][0]
                label = str(getattr(cat, "category_name", "") or "")
                score = float(getattr(cat, "score", 1.0) or 1.0)
            hands.append(HandLandmarks(landmarks=pts, handedness=label, score=score))
        return hands

    def estimate_for_persons(
        self,
        frame_bgr: np.ndarray,
        person_rows: list[tuple[int, tuple[int, int, int, int]]],
        *,
        wrist_hints: dict[int, list[tuple[float, float]]] | None = None,
        pad_frac: float = 0.18,
        min_box_px: int = 40,
        max_persons: int = 0,
        full_frame_fallback: bool = False,
    ) -> dict[int, list[HandLandmarks]]:
        """
        Detect finger skeletons per person.

        Prefer cropping around YOLO-pose wrists when available; else upper-body ROI.
        Full-frame MediaPipe is off by default (slow).
        """
        if not person_rows:
            return {}
        rows = list(person_rows)
        if max_persons > 0 and len(rows) > max_persons:
            rows = sorted(
                rows,
                key=lambda r: (r[1][2] - r[1][0]) * (r[1][3] - r[1][1]),
                reverse=True,
            )[: int(max_persons)]
        h, w = frame_bgr.shape[:2]
        out: dict[int, list[HandLandmarks]] = {}
        for ridx, (px1, py1, px2, py2) in rows:
            pw, ph = px2 - px1, py2 - py1
            if pw < min_box_px or ph < min_box_px:
                continue
            crops: list[tuple[int, int, int, int]] = []
            hints = (wrist_hints or {}).get(int(ridx), [])
            if hints:
                for wx, wy in hints[:2]:
                    side = max(int(0.55 * max(pw, ph)), 140)
                    side = min(side, 320)
                    cx, cy = int(wx), int(wy)
                    crops.append(
                        (
                            max(0, cx - side // 2),
                            max(0, cy - side // 2),
                            min(w, cx + side // 2),
                            min(h, cy + side // 2),
                        )
                    )
            else:
                pad_x = int(round(pad_frac * pw))
                pad_y = int(round(pad_frac * ph))
                uy2 = py1 + int(round(0.72 * ph))
                crops.append(
                    (
                        max(0, px1 - pad_x),
                        max(0, py1 - pad_y),
                        min(w, px2 + pad_x),
                        min(h, uy2 + pad_y),
                    )
                )

            found: list[HandLandmarks] = []
            seen_centers: list[tuple[float, float]] = []
            for cx1, cy1, cx2, cy2 in crops:
                if (cx2 - cx1) < 48 or (cy2 - cy1) < 48:
                    continue
                crop = frame_bgr[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                ch, cw = crop.shape[:2]
                work = crop
                scale = 1.0
                if max(ch, cw) > 256:
                    scale = 256.0 / float(max(ch, cw))
                    work = cv2.resize(
                        crop,
                        (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
                        interpolation=cv2.INTER_AREA,
                    )
                rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
                inv = 1.0 / scale if scale != 1.0 else 1.0
                for hand in self._detect_rgb(rgb):
                    lm = hand.landmarks.copy()
                    lm[:, 0] = lm[:, 0] * inv + float(cx1)
                    lm[:, 1] = lm[:, 1] * inv + float(cy1)
                    mapped = HandLandmarks(
                        landmarks=lm, handedness=hand.handedness, score=hand.score
                    )
                    pcx, pcy = mapped.palm_center()
                    if not (
                        px1 - 0.25 * pw <= pcx <= px2 + 0.25 * pw
                        and py1 - 0.20 * ph <= pcy <= py2 + 0.15 * ph
                    ):
                        continue
                    dup = False
                    for sx, sy in seen_centers:
                        if float(np.hypot(pcx - sx, pcy - sy)) < 28.0:
                            dup = True
                            break
                    if dup:
                        continue
                    seen_centers.append((pcx, pcy))
                    found.append(mapped)
                    if len(found) >= 2:
                        break
                if len(found) >= 2:
                    break
            if found:
                out[int(ridx)] = found

        if not full_frame_fallback:
            return out

        missing = [r for r in rows if int(r[0]) not in out]
        if missing:
            rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            fh, fw = rgb_full.shape[:2]
            scale = 1.0
            work = rgb_full
            if max(fh, fw) > 960:
                scale = 960.0 / float(max(fh, fw))
                work = cv2.resize(
                    rgb_full,
                    (max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            ff_hands = self._detect_rgb(work)
            inv = 1.0 / scale if scale != 1.0 else 1.0
            for hand in ff_hands:
                lm = hand.landmarks.copy()
                lm[:, 0] *= inv
                lm[:, 1] *= inv
                mapped = HandLandmarks(
                    landmarks=lm, handedness=hand.handedness, score=hand.score
                )
                pcx, pcy = mapped.palm_center()
                best_ridx = -1
                best_dist = 1e18
                for ridx, (px1, py1, px2, py2) in missing:
                    pw, ph = max(1, px2 - px1), max(1, py2 - py1)
                    if not (
                        px1 - 0.30 * pw <= pcx <= px2 + 0.30 * pw
                        and py1 - 0.25 * ph <= pcy <= py2 + 0.20 * ph
                    ):
                        continue
                    cx = 0.5 * (px1 + px2)
                    cy = 0.45 * (py1 + py2)
                    d = float(np.hypot(pcx - cx, pcy - cy))
                    if d < best_dist:
                        best_dist = d
                        best_ridx = int(ridx)
                if best_ridx < 0:
                    continue
                bucket = out.setdefault(best_ridx, [])
                dup = False
                for existing in bucket:
                    ex, ey = existing.palm_center()
                    if float(np.hypot(pcx - ex, pcy - ey)) < 36.0:
                        dup = True
                        break
                if not dup and len(bucket) < 2:
                    bucket.append(mapped)

        return out



def scale_person_hands(
    hands_by_ridx: dict[int, list[HandLandmarks]],
    sx: float,
    sy: float,
) -> dict[int, list[HandLandmarks]]:
    if sx == 1.0 and sy == 1.0:
        return hands_by_ridx
    out: dict[int, list[HandLandmarks]] = {}
    for ridx, hands in hands_by_ridx.items():
        scaled: list[HandLandmarks] = []
        for hand in hands:
            lm = hand.landmarks.copy()
            lm[:, 0] *= float(sx)
            lm[:, 1] *= float(sy)
            scaled.append(HandLandmarks(landmarks=lm, handedness=hand.handedness, score=hand.score))
        out[ridx] = scaled
    return out


def _unit(vx: float, vy: float) -> tuple[float, float]:
    n = float(np.hypot(vx, vy))
    if n < 1e-6:
        return 0.0, 0.0
    return vx / n, vy / n


def _cosine(ax: float, ay: float, bx: float, by: float) -> float:
    au = _unit(ax, ay)
    bu = _unit(bx, by)
    return float(au[0] * bu[0] + au[1] * bu[1])


def _point_in_box(x: float, y: float, x1: int, y1: int, x2: int, y2: int, pad: float = 0.0) -> bool:
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def _index_extended(hand: HandLandmarks) -> bool:
    """True when index tip is farther from wrist than MCP (roughly straight finger)."""
    wx, wy = hand.wrist()
    mx, my = hand.index_mcp()
    tx, ty = hand.index_tip()
    return float(np.hypot(tx - wx, ty - wy)) > float(np.hypot(mx - wx, my - wy)) * 1.08


def _index_curl_ratio(hand: HandLandmarks) -> float:
    """0 ≈ extended, 1 ≈ curled (tip close to MCP relative to chain length)."""
    mx, my = hand.index_mcp()
    px, py = hand.point(_INDEX_PIP)
    dx, dy = hand.point(_INDEX_DIP)
    tx, ty = hand.index_tip()
    chain = (
        float(np.hypot(px - mx, py - my))
        + float(np.hypot(dx - px, dy - py))
        + float(np.hypot(tx - dx, ty - dy))
    )
    tip_mcp = float(np.hypot(tx - mx, ty - my))
    if chain < 1e-3:
        return 0.0
    # Extended tip_mcp ≈ chain; curled tip_mcp << chain.
    return float(np.clip(1.0 - tip_mcp / chain, 0.0, 1.0))


def classify_hand_gun_cue(
    hand: HandLandmarks,
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    *,
    grip_pad_px: float = 28.0,
    trigger_radius_px: float = 36.0,
    shoot_align_min: float = 0.72,
    boost_grip: float = 0.12,
    boost_trigger: float = 0.18,
    boost_shoot: float = 0.25,
    arms_down: bool = False,
    arm_posture: str | None = None,
) -> HandGunCue:
    """
    Classify hand vs firearm.

    When pose arm posture is known (``arm_posture`` / ``arms_down``):
      arms up   → shooting (if near gun)
      arms down → grip / holding (if near gun)
    Otherwise fall back to finger heuristics.
    """
    gcx, gcy = _box_center(gx1, gy1, gx2, gy2)
    gw = max(1, gx2 - gx1)
    gh = max(1, gy2 - gy1)
    gun_long = (float(gw), 0.0) if gw >= gh else (0.0, float(gh))
    if gw >= gh:
        gun_long = (float(gw), 0.0)

    wx, wy = hand.wrist()
    pcx, pcy = hand.palm_center()
    tx, ty = hand.index_tip()
    mx, my = hand.index_mcp()

    palm_near = _point_in_box(pcx, pcy, gx1, gy1, gx2, gy2, pad=grip_pad_px) or (
        float(np.hypot(pcx - gcx, pcy - gcy)) <= max(gw, gh) * 0.65
    )
    wrist_near = _point_in_box(wx, wy, gx1, gy1, gx2, gy2, pad=grip_pad_px * 1.2)
    tip_dist = float(np.hypot(tx - gcx, ty - gcy))
    tip_near = tip_dist <= max(trigger_radius_px, 0.45 * max(gw, gh))
    tip_in_gun = _point_in_box(tx, ty, gx1, gy1, gx2, gy2, pad=trigger_radius_px * 0.35)
    near_gun = palm_near or wrist_near or tip_near
    gun_box = (gx1, gy1, gx2, gy2)

    posture = str(arm_posture or "").strip().lower()
    if posture in ("down", "arms_down") or (not posture and arms_down):
        if near_gun:
            return HandGunCue(state="grip", boost=float(boost_grip), hand=hand, gun_box=gun_box)
        return HandGunCue(state="none", boost=0.0, hand=hand, gun_box=None)
    if posture in ("up", "arms_up", "raised", "shooting"):
        if near_gun:
            return HandGunCue(
                state="shooting",
                boost=float(boost_shoot),
                hand=hand,
                gun_box=gun_box,
            )
        return HandGunCue(state="none", boost=0.0, hand=hand, gun_box=None)

    # Trigger region heuristic: middle/lower half of gun box (side-view pistol).
    trigger_y1 = gy1 + int(0.35 * gh)
    tip_in_trigger_band = tip_in_gun and (ty >= trigger_y1)
    curl = _index_curl_ratio(hand)
    extended = _index_extended(hand)
    idx_dir = (tx - mx, ty - my)
    align = abs(_cosine(idx_dir[0], idx_dir[1], gun_long[0], gun_long[1]))
    far_x = gx2 if abs(gx2 - wx) >= abs(gx1 - wx) else gx1
    far_y = gy2 if abs(gy2 - wy) >= abs(gy1 - wy) else gy1
    align_barrel = abs(_cosine(idx_dir[0], idx_dir[1], far_x - wx, far_y - wy))
    align = max(align, align_barrel)

    if (palm_near or wrist_near) and extended and align >= shoot_align_min and tip_near:
        return HandGunCue(
            state="shooting",
            boost=float(boost_shoot),
            hand=hand,
            gun_box=gun_box,
        )
    if (palm_near or wrist_near or tip_near) and (tip_in_trigger_band or (tip_in_gun and curl >= 0.22)):
        return HandGunCue(
            state="finger_on_trigger",
            boost=float(boost_trigger),
            hand=hand,
            gun_box=gun_box,
        )
    if near_gun:
        return HandGunCue(state="grip", boost=float(boost_grip), hand=hand, gun_box=gun_box)
    return HandGunCue(state="none", boost=0.0, hand=hand, gun_box=None)


def best_hand_gun_cue(
    hands: list[HandLandmarks],
    gun_boxes: list[tuple[int, int, int, int]],
    **kwargs,
) -> HandGunCue | None:
    """Pick the strongest hand↔gun cue among candidates."""
    rank = {"none": 0, "grip": 1, "finger_on_trigger": 2, "shooting": 3}
    best: HandGunCue | None = None
    for hand in hands:
        for gx1, gy1, gx2, gy2 in gun_boxes:
            cue = classify_hand_gun_cue(hand, gx1, gy1, gx2, gy2, **kwargs)
            if cue.state == "none":
                continue
            if best is None:
                best = cue
                continue
            if rank[cue.state] > rank[best.state] or (
                rank[cue.state] == rank[best.state] and cue.boost > best.boost
            ):
                best = cue
    return best


def gun_finger_confidence_boost(
    base_conf: float,
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    hands: list[HandLandmarks] | None,
    *,
    enabled: bool = True,
    arms_down: bool = False,
    arm_posture: str | None = None,
    **kwargs,
) -> tuple[float, GripState]:
    """Boost firearm confidence from finger / grip alignment; return (conf, best_state)."""
    if not enabled or not hands:
        return float(base_conf), "none"
    cue = best_hand_gun_cue(
        hands,
        [(gx1, gy1, gx2, gy2)],
        arms_down=bool(arms_down),
        arm_posture=arm_posture,
        **kwargs,
    )
    if cue is None or cue.state == "none":
        return float(base_conf), "none"
    return min(1.0, float(base_conf) + float(cue.boost)), cue.state


def grip_state_label(state: GripState) -> str:
    if state == "shooting":
        return "shooting"
    if state == "finger_on_trigger":
        return "holding"
    if state == "grip":
        return "holding"
    return ""


def draw_hand_fingers(
    frame: np.ndarray,
    hands: list[HandLandmarks],
    *,
    cue: HandGunCue | None = None,
) -> None:
    """Draw 21-point finger skeletons; highlight index when trigger/shooting."""
    highlight = cue.state if cue is not None else "none"
    for hand in hands:
        is_focus = cue is not None and hand is cue.hand
        for a, b in _HAND_CONNECTIONS:
            ax, ay = hand.point(a)
            bx, by = hand.point(b)
            color = _COLOR_FINGER_BGR
            thick = 3
            if is_focus and highlight == "shooting":
                color = _COLOR_SHOOT_BGR
                thick = 4
            elif is_focus and highlight == "finger_on_trigger":
                color = _COLOR_TRIGGER_BGR
                thick = 4
            elif a in (_INDEX_MCP, _INDEX_PIP, _INDEX_DIP, _INDEX_TIP) or b in (
                _INDEX_MCP,
                _INDEX_PIP,
                _INDEX_DIP,
                _INDEX_TIP,
            ):
                color = _COLOR_INDEX_BGR
                thick = 3
            cv2.line(
                frame,
                (int(ax), int(ay)),
                (int(bx), int(by)),
                color,
                thick,
                lineType=cv2.LINE_AA,
            )
        for i in range(21):
            x, y = hand.point(i)
            r = 4 if i != _INDEX_TIP else 6
            c = _COLOR_INDEX_BGR
            if is_focus and highlight == "shooting":
                c = _COLOR_SHOOT_BGR
            elif is_focus and highlight == "finger_on_trigger":
                c = _COLOR_TRIGGER_BGR
            elif i == _INDEX_TIP:
                c = _COLOR_INDEX_BGR
            else:
                c = _COLOR_FINGER_BGR
            cv2.circle(frame, (int(x), int(y)), r, c, -1, lineType=cv2.LINE_AA)
        if is_focus and highlight in ("finger_on_trigger", "shooting"):
            tx, ty = hand.index_tip()
            tag = "TRIGGER" if highlight == "finger_on_trigger" else "SHOOTING"
            cv2.putText(
                frame,
                tag,
                (int(tx) + 6, int(ty) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                _COLOR_TRIGGER_BGR if highlight == "finger_on_trigger" else _COLOR_SHOOT_BGR,
                2,
                lineType=cv2.LINE_AA,
            )


def grip_state_label(state: GripState) -> str:
    if state == "shooting":
        return "shooting"
    if state == "finger_on_trigger":
        return "holding"
    if state == "grip":
        return "holding"
    return ""
