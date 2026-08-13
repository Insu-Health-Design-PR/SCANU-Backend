"""YOLO pose: wrist-aware gun confidence boost and on-frame skeleton overlay."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from weapon_ai.detection.firearms import box_center as _box_center

# COCO 17-keypoint indices (Ultralytics YOLO pose).
_KPT_NOSE = 0
_KPT_L_EYE, _KPT_R_EYE = 1, 2
_KPT_L_EAR, _KPT_R_EAR = 3, 4
_KPT_L_SHOULDER, _KPT_R_SHOULDER = 5, 6
_KPT_L_ELBOW, _KPT_R_ELBOW = 7, 8
_KPT_L_WRIST, _KPT_R_WRIST = 9, 10
_KPT_L_HIP, _KPT_R_HIP = 11, 12
_KPT_L_KNEE, _KPT_R_KNEE = 13, 14
_KPT_L_ANKLE, _KPT_R_ANKLE = 15, 16

# Full COCO body (torso + head + legs).
_SKELETON_BODY = (
    (_KPT_L_EAR, _KPT_L_EYE),
    (_KPT_R_EAR, _KPT_R_EYE),
    (_KPT_L_EYE, _KPT_NOSE),
    (_KPT_R_EYE, _KPT_NOSE),
    (_KPT_L_SHOULDER, _KPT_R_SHOULDER),
    (_KPT_L_SHOULDER, _KPT_L_HIP),
    (_KPT_R_SHOULDER, _KPT_R_HIP),
    (_KPT_L_HIP, _KPT_R_HIP),
    (_KPT_NOSE, _KPT_L_SHOULDER),
    (_KPT_NOSE, _KPT_R_SHOULDER),
    (_KPT_L_HIP, _KPT_L_KNEE),
    (_KPT_L_KNEE, _KPT_L_ANKLE),
    (_KPT_R_HIP, _KPT_R_KNEE),
    (_KPT_R_KNEE, _KPT_R_ANKLE),
)

# Arm / hand chain (emphasized).
_SKELETON_ARMS = (
    (_KPT_L_SHOULDER, _KPT_L_ELBOW),
    (_KPT_L_ELBOW, _KPT_L_WRIST),
    (_KPT_R_SHOULDER, _KPT_R_ELBOW),
    (_KPT_R_ELBOW, _KPT_R_WRIST),
)

_WRIST_IDXS = (_KPT_L_WRIST, _KPT_R_WRIST)
_ELBOW_IDXS = (_KPT_L_ELBOW, _KPT_R_ELBOW)
_JOINT_DOT_IDXS = (
    _KPT_NOSE,
    _KPT_L_SHOULDER,
    _KPT_R_SHOULDER,
    _KPT_L_HIP,
    _KPT_R_HIP,
    _KPT_L_KNEE,
    _KPT_R_KNEE,
    _KPT_L_ANKLE,
    _KPT_R_ANKLE,
)

_COLOR_BODY_BGR = (220, 220, 220)
_COLOR_LEG_BGR = (180, 180, 180)
_COLOR_ARM_BGR = (255, 220, 0)
_COLOR_WRIST_BGR = (0, 180, 255)
_COLOR_WRIST_NEAR_GUN_BGR = (0, 0, 255)
_COLOR_JOINT_BGR = (240, 240, 240)


@dataclass(frozen=True)
class PersonPose:
    """Full-frame COCO keypoints for one person (x, y, conf) per joint."""

    keypoints: np.ndarray  # shape (17, 3)

    def wrist_points(self, min_conf: float) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        for idx in _WRIST_IDXS:
            x, y, c = self.keypoints[idx]
            if float(c) >= float(min_conf):
                out.append((float(x), float(y), float(c)))
        return out

    def any_arm_raised(self, min_conf: float = 0.35) -> bool | None:
        """
        Whether either arm looks raised / ready to aim.

        Returns:
          True  — at least one arm raised (wrist in upper torso or above)
          False — observed arms hang low (arms down)
          None  — not enough keypoints to judge
        """
        kp = self.keypoints
        judged = 0
        raised = False
        for sh_i, el_i, wr_i, hip_i in (
            (_KPT_L_SHOULDER, _KPT_L_ELBOW, _KPT_L_WRIST, _KPT_L_HIP),
            (_KPT_R_SHOULDER, _KPT_R_ELBOW, _KPT_R_WRIST, _KPT_R_HIP),
        ):
            sx, sy, sc = kp[sh_i]
            wx, wy, wc = kp[wr_i]
            _hx, hy, hc = kp[hip_i]
            if float(sc) < float(min_conf) or float(wc) < float(min_conf):
                continue
            judged += 1
            # Raised / aiming: wrist at chest height or higher (upper ~50% of torso).
            if float(hc) >= float(min_conf) and float(hy) > float(sy) + 8.0:
                torso_h = float(hy - sy)
                if float(wy) <= float(sy) + 0.50 * torso_h:
                    raised = True
                    break
                continue
            # No hip: wrist at/above shoulder, or not hanging past a low elbow.
            if float(wy) <= float(sy) + 20.0:
                raised = True
                break
            _ex, ey, ec = kp[el_i]
            if float(ec) >= float(min_conf) and float(wy) < float(ey) - 6.0:
                # Wrist above elbow → not a hanging arm.
                raised = True
                break
        if judged == 0:
            return None
        return raised

    def arms_down(self, min_conf: float = 0.35) -> bool:
        """True when pose says arms hang low (not aiming / shooting posture)."""
        raised = self.any_arm_raised(min_conf)
        return raised is False

    def arm_gun_state(self, min_conf: float = 0.35) -> str | None:
        """
        Arms-up → shooting, arms-down → holding (grip).

        Returns ``\"shooting\"``, ``\"grip\"`` (holding), or ``None`` if pose is unclear.
        """
        raised = self.any_arm_raised(min_conf)
        if raised is True:
            return "shooting"
        if raised is False:
            return "grip"
        return None


def _box_iou(
    ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int
) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    aa = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    bb = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    union = aa + bb - inter
    return inter / max(1.0, union)


class PoseEstimator:
    """Lazy YOLO-pose wrapper; matches detections to existing person boxes."""

    def __init__(self, model_path: str, *, device: int | str = 0) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self._device = device
        self._model_path = str(model_path)

    def estimate_for_persons(
        self,
        frame: np.ndarray,
        person_rows: list[tuple[int, tuple[int, int, int, int]]],
        *,
        min_box_px: int = 24,
        imgsz: int = 640,
        max_persons: int = 0,
        crop_infer: bool = True,
        crop_pad_frac: float = 0.12,
    ) -> dict[int, PersonPose]:
        if not person_rows:
            return {}
        rows = list(person_rows)
        if max_persons > 0 and len(rows) > max_persons:
            rows = sorted(
                rows,
                key=lambda r: (r[1][2] - r[1][0]) * (r[1][3] - r[1][1]),
                reverse=True,
            )[: int(max_persons)]
        # Per-person crops give much cleaner skeletons than full-frame matching,
        # especially on multi-cam / crowded frames.
        if crop_infer and len(rows) <= 12:
            return self._estimate_from_crops(
                frame,
                rows,
                min_box_px=min_box_px,
                imgsz=imgsz,
                crop_pad_frac=crop_pad_frac,
            )
        return self._estimate_from_full_frame(
            frame,
            rows,
            min_box_px=min_box_px,
            imgsz=imgsz,
        )

    def _estimate_from_crops(
        self,
        frame: np.ndarray,
        rows: list[tuple[int, tuple[int, int, int, int]]],
        *,
        min_box_px: int,
        imgsz: int,
        crop_pad_frac: float,
    ) -> dict[int, PersonPose]:
        h, w = frame.shape[:2]
        poses_by_ridx: dict[int, PersonPose] = {}
        pad = max(0.0, float(crop_pad_frac))
        for ridx, (px1, py1, px2, py2) in rows:
            bw, bh = px2 - px1, py2 - py1
            if bw < min_box_px or bh < min_box_px:
                continue
            dx = int(round(bw * pad))
            dy = int(round(bh * pad))
            x1 = max(0, px1 - dx)
            y1 = max(0, py1 - dy)
            x2 = min(w, px2 + dx)
            y2 = min(h, py2 + dy)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # Keep pose imgsz proportional to crop; Ultralytics needs multiple of 32.
            raw = int(max(320, min(int(imgsz) or 640, max(crop.shape[0], crop.shape[1]))))
            local_imgsz = max(320, int(round(raw / 32.0) * 32))
            try:
                res = self._model.predict(
                    source=crop,
                    conf=0.20,
                    imgsz=local_imgsz,
                    verbose=False,
                    device=self._device,
                )
            except Exception:
                continue
            if not res:
                continue
            r0 = res[0]
            kpts_obj = getattr(r0, "keypoints", None)
            boxes = getattr(r0, "boxes", None)
            if kpts_obj is None:
                continue
            try:
                kpts_data = kpts_obj.data.cpu().numpy()
            except Exception:
                continue
            if kpts_data is None or len(kpts_data) == 0:
                continue
            best_j = 0
            if boxes is not None and len(boxes) > 0:
                try:
                    det_xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                except Exception:
                    det_xyxy, confs = None, None
                if det_xyxy is not None and len(det_xyxy):
                    # Prefer detection whose center is nearest crop center / highest conf.
                    cx = 0.5 * (crop.shape[1] - 1)
                    cy = 0.5 * (crop.shape[0] - 1)
                    best_score = -1.0
                    for j in range(min(len(det_xyxy), len(kpts_data))):
                        dx1, dy1, dx2, dy2 = det_xyxy[j]
                        dcx = 0.5 * (dx1 + dx2)
                        dcy = 0.5 * (dy1 + dy2)
                        dist = float(np.hypot(dcx - cx, dcy - cy))
                        conf = float(confs[j]) if confs is not None and j < len(confs) else 0.5
                        score = conf - 0.001 * dist
                        if score > best_score:
                            best_score = score
                            best_j = j
            kp = np.asarray(kpts_data[best_j], dtype=np.float32)
            if kp.ndim != 2 or kp.shape[0] < 1:
                continue
            if kp.shape[0] < 17:
                pad_k = np.zeros((17, 3), dtype=np.float32)
                cols = min(3, kp.shape[1])
                pad_k[: kp.shape[0], :cols] = kp[:, :cols]
                kp = pad_k
            else:
                kp = kp[:17, :3].copy()
            kp[:, 0] = np.clip(kp[:, 0] + float(x1), 0, w - 1)
            kp[:, 1] = np.clip(kp[:, 1] + float(y1), 0, h - 1)
            # Drop near-zero conf joints explicitly.
            kp[kp[:, 2] < 0.05, 2] = 0.0
            if float(np.count_nonzero(kp[:, 2] >= 0.25)) < 4:
                continue
            poses_by_ridx[int(ridx)] = PersonPose(keypoints=kp)
        return poses_by_ridx

    def _estimate_from_full_frame(
        self,
        frame: np.ndarray,
        rows: list[tuple[int, tuple[int, int, int, int]]],
        *,
        min_box_px: int,
        imgsz: int,
    ) -> dict[int, PersonPose]:
        h, w = frame.shape[:2]
        try:
            res = self._model.predict(
                source=frame,
                conf=0.20,
                imgsz=max(320, int(imgsz) or 640),
                verbose=False,
                device=self._device,
            )
        except Exception:
            return {}
        if not res:
            return {}
        r0 = res[0]
        kpts_obj = getattr(r0, "keypoints", None)
        boxes = getattr(r0, "boxes", None)
        if kpts_obj is None or boxes is None or len(boxes) == 0:
            return {}
        try:
            kpts_data = kpts_obj.data.cpu().numpy()
        except Exception:
            return {}
        try:
            det_xyxy = boxes.xyxy.cpu().numpy()
        except Exception:
            det_xyxy = None

        poses_by_ridx: dict[int, PersonPose] = {}
        used_det: set[int] = set()
        for ridx, (px1, py1, px2, py2) in rows:
            if (px2 - px1) < min_box_px or (py2 - py1) < min_box_px:
                continue
            best_score = -1.0
            best_j = -1
            pcx = 0.5 * (px1 + px2)
            pcy = 0.5 * (py1 + py2)
            if det_xyxy is not None:
                for j in range(min(len(det_xyxy), len(kpts_data))):
                    if j in used_det:
                        continue
                    dx1, dy1, dx2, dy2 = det_xyxy[j]
                    iou = _box_iou(px1, py1, px2, py2, int(dx1), int(dy1), int(dx2), int(dy2))
                    dcx = 0.5 * (dx1 + dx2)
                    dcy = 0.5 * (dy1 + dy2)
                    # Combine IoU with center proximity (IoU alone fails on loose person boxes).
                    dist = float(np.hypot(dcx - pcx, dcy - pcy))
                    diag = float(np.hypot(px2 - px1, py2 - py1)) + 1e-3
                    score = float(iou) + 0.35 * max(0.0, 1.0 - dist / diag)
                    if score > best_score:
                        best_score = score
                        best_j = j
            else:
                for j in range(len(kpts_data)):
                    if j in used_det:
                        continue
                    kp = kpts_data[j]
                    vis = kp[kp[:, 2] > 0.2]
                    if vis.size == 0:
                        continue
                    mx = float(vis[:, 0].mean())
                    my = float(vis[:, 1].mean())
                    if px1 <= mx <= px2 and py1 <= my <= py2:
                        best_j = j
                        best_score = 1.0
                        break
            if best_j < 0 or best_score < 0.12:
                continue
            used_det.add(best_j)
            kp = np.asarray(kpts_data[best_j], dtype=np.float32)
            if kp.shape[0] < 17:
                pad = np.zeros((17, 3), dtype=np.float32)
                pad[: kp.shape[0]] = kp[:, :3] if kp.shape[1] >= 3 else np.pad(kp, ((0, 0), (0, 1)))
                kp = pad
            else:
                kp = kp[:17, :3].copy()
            kp[:, 0] = np.clip(kp[:, 0], 0, w - 1)
            kp[:, 1] = np.clip(kp[:, 1], 0, h - 1)
            poses_by_ridx[int(ridx)] = PersonPose(keypoints=kp)
        return poses_by_ridx


def scale_person_poses(
    poses: dict[int, PersonPose],
    sx: float,
    sy: float,
) -> dict[int, PersonPose]:
    if sx == 1.0 and sy == 1.0:
        return poses
    out: dict[int, PersonPose] = {}
    for ridx, pose in poses.items():
        kp = pose.keypoints.copy()
        kp[:, 0] *= float(sx)
        kp[:, 1] *= float(sy)
        out[ridx] = PersonPose(keypoints=kp)
    return out


def _dist_point_to_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-6:
        return float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return float(np.hypot(px - cx, py - cy))


def gun_near_pose_hands(
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    pose: PersonPose | None,
    *,
    hand_radius_px: float = 100.0,
    min_kpt_conf: float = 0.35,
) -> bool:
    if pose is None:
        return False
    gcx, gcy = _box_center(gx1, gy1, gx2, gy2)
    radius = max(8.0, float(hand_radius_px))
    for wx, wy, _wc in pose.wrist_points(min_kpt_conf):
        if float(np.hypot(gcx - wx, gcy - wy)) <= radius:
            return True
    kp = pose.keypoints
    for elbow_i, wrist_i in ((_KPT_L_ELBOW, _KPT_L_WRIST), (_KPT_R_ELBOW, _KPT_R_WRIST)):
        ex, ey, ec = kp[elbow_i]
        wx, wy, wc = kp[wrist_i]
        if float(ec) < min_kpt_conf or float(wc) < min_kpt_conf:
            continue
        if _dist_point_to_segment(gcx, gcy, float(ex), float(ey), float(wx), float(wy)) <= radius * 0.65:
            return True
    return False


def gun_pose_confidence_boost(
    base_conf: float,
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    pose: PersonPose | None,
    *,
    hand_radius_px: float = 100.0,
    boost_max: float = 0.20,
    min_kpt_conf: float = 0.35,
) -> float:
    """Raise firearm confidence when the box sits near estimated wrist / forearm keypoints."""
    if pose is None or boost_max <= 0.0:
        return float(base_conf)
    gcx, gcy = _box_center(gx1, gy1, gx2, gy2)
    radius = max(8.0, float(hand_radius_px))
    best_prox = 0.0
    for wx, wy, wc in pose.wrist_points(min_kpt_conf):
        dist = float(np.hypot(gcx - wx, gcy - wy))
        if dist <= radius:
            best_prox = max(best_prox, 1.0 - dist / radius)
    kp = pose.keypoints
    for elbow_i, wrist_i in ((_KPT_L_ELBOW, _KPT_L_WRIST), (_KPT_R_ELBOW, _KPT_R_WRIST)):
        ex, ey, ec = kp[elbow_i]
        wx, wy, wc = kp[wrist_i]
        if float(ec) < min_kpt_conf or float(wc) < min_kpt_conf:
            continue
        dist = _dist_point_to_segment(gcx, gcy, float(ex), float(ey), float(wx), float(wy))
        arm_radius = radius * 0.65
        if dist <= arm_radius:
            best_prox = max(best_prox, 0.75 * (1.0 - dist / arm_radius))
    if best_prox <= 0.0:
        return float(base_conf)
    boosted = float(base_conf) + float(boost_max) * best_prox
    return min(1.0, boosted)


def draw_person_pose(
    frame: np.ndarray,
    pose: PersonPose,
    *,
    min_kpt_conf: float = 0.35,
    gun_boxes_for_person: list[tuple[int, int, int, int]] | None = None,
    hand_radius_px: float = 100.0,
) -> None:
    """Draw full COCO body skeleton with emphasized arm/hand chains."""
    kp = pose.keypoints
    h, w = frame.shape[:2]
    min_c = float(min_kpt_conf)

    def _ok(i: int) -> bool:
        return float(kp[i, 2]) >= min_c

    def _pt(i: int) -> tuple[int, int] | None:
        if not _ok(i):
            return None
        x, y = int(round(float(kp[i, 0]))), int(round(float(kp[i, 1])))
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return x, y

    def _bone_ok(a: int, b: int) -> bool:
        """Reject absurdly long bones (bad matches / hallucination)."""
        pa, pb = _pt(a), _pt(b)
        if not pa or not pb:
            return False
        dist = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
        # Relative to torso size when available.
        ls, rs = _pt(_KPT_L_SHOULDER), _pt(_KPT_R_SHOULDER)
        lh, rh = _pt(_KPT_L_HIP), _pt(_KPT_R_HIP)
        span = 0.0
        if ls and rh:
            span = max(span, float(np.hypot(ls[0] - rh[0], ls[1] - rh[1])))
        if rs and lh:
            span = max(span, float(np.hypot(rs[0] - lh[0], rs[1] - lh[1])))
        if span <= 1.0:
            span = float(max(h, w)) * 0.35
        return dist <= 1.35 * span

    leg_bones = {
        (_KPT_L_HIP, _KPT_L_KNEE),
        (_KPT_L_KNEE, _KPT_L_ANKLE),
        (_KPT_R_HIP, _KPT_R_KNEE),
        (_KPT_R_KNEE, _KPT_R_ANKLE),
    }
    for a, b in _SKELETON_BODY:
        if not _bone_ok(a, b):
            continue
        pa, pb = _pt(a), _pt(b)
        color = _COLOR_LEG_BGR if (a, b) in leg_bones else _COLOR_BODY_BGR
        thick = 2 if (a, b) in leg_bones else 2
        cv2.line(frame, pa, pb, color, thick, lineType=cv2.LINE_AA)

    for a, b in _SKELETON_ARMS:
        if not _bone_ok(a, b):
            continue
        pa, pb = _pt(a), _pt(b)
        cv2.line(frame, pa, pb, _COLOR_ARM_BGR, 3, lineType=cv2.LINE_AA)

    near_gun = False
    if gun_boxes_for_person:
        for gx1, gy1, gx2, gy2 in gun_boxes_for_person:
            if gun_near_pose_hands(
                gx1, gy1, gx2, gy2, pose,
                hand_radius_px=hand_radius_px,
                min_kpt_conf=min_kpt_conf,
            ):
                near_gun = True
                break

    for idx in _JOINT_DOT_IDXS:
        p = _pt(idx)
        if p:
            cv2.circle(frame, p, 3, _COLOR_JOINT_BGR, -1, lineType=cv2.LINE_AA)

    for idx in _WRIST_IDXS:
        p = _pt(idx)
        if not p:
            continue
        color = _COLOR_WRIST_NEAR_GUN_BGR if near_gun else _COLOR_WRIST_BGR
        cv2.circle(frame, p, 5, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, p, 7, color, 1, lineType=cv2.LINE_AA)

    for idx in _ELBOW_IDXS:
        p = _pt(idx)
        if p:
            cv2.circle(frame, p, 4, _COLOR_ARM_BGR, -1, lineType=cv2.LINE_AA)
