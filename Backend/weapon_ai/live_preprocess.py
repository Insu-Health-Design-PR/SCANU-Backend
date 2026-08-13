"""Live-frame downscale / box remap for high-res capture on server-class GPUs."""

from __future__ import annotations

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _resize_bgr(frame: np.ndarray, width: int, height: int, *, use_gpu: bool) -> np.ndarray:
    fw = int(frame.shape[1]) if frame is not None and frame.ndim >= 2 else 0
    fh = int(frame.shape[0]) if frame is not None and frame.ndim >= 2 else 0
    width = int(width)
    height = int(height)
    if fw < 1 or fh < 1 or width < 1 or height < 1:
        return frame
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    if (
        use_gpu
        and torch is not None
        and F is not None
        and torch.cuda.is_available()
    ):
        try:
            t = torch.from_numpy(frame).to(device="cuda", non_blocking=True)
            t = t.permute(2, 0, 1).unsqueeze(0).float()
            t = F.interpolate(t, size=(int(height), int(width)), mode="bilinear", align_corners=False)
            out = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
            if out.flags["C_CONTIGUOUS"]:
                return out
            return np.ascontiguousarray(out)
        except Exception:
            pass
    out = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(out)


def resize_bgr_to(
    frame: np.ndarray,
    width: int,
    height: int,
    *,
    use_gpu: bool,
) -> np.ndarray:
    """Resize to exact width×height (panel / fixed stream size)."""
    return _resize_bgr(frame, int(width), int(height), use_gpu=use_gpu)


def downscale_to_max_width(
    frame: np.ndarray,
    max_width: int,
    *,
    use_gpu: bool,
) -> tuple[np.ndarray, float, float]:
    """Return (resized_frame, scale_x, scale_y) mapping resized coords -> source coords."""
    h, w = frame.shape[:2]
    mw = int(max_width)
    if mw <= 0 or w <= mw:
        return frame, 1.0, 1.0
    nh = max(1, int(round(h * (mw / float(w)))))
    small = _resize_bgr(frame, mw, nh, use_gpu=use_gpu)
    sx = float(w) / float(small.shape[1])
    sy = float(h) / float(small.shape[0])
    return small, sx, sy


def scale_person_rows(
    rows: list[tuple[int, int, int, int, float, int | None, str, float]],
    sx: float,
    sy: float,
) -> list[tuple[int, int, int, int, float, int | None, str, float]]:
    if sx == 1.0 and sy == 1.0:
        return rows
    out: list[tuple[int, int, int, int, float, int | None, str, float]] = []
    for x1, y1, x2, y2, prob, cid, tag, det_c in rows:
        out.append(
            (
                int(round(x1 * sx)),
                int(round(y1 * sy)),
                int(round(x2 * sx)),
                int(round(y2 * sy)),
                prob,
                cid,
                tag,
                det_c,
            )
        )
    return out


def resize_bgr_max_width(
    frame: np.ndarray,
    max_width: int,
    *,
    use_gpu: bool,
) -> np.ndarray:
    """Downscale for IPC/WebRTC when capture resolution exceeds stream budget."""
    small, _, _ = downscale_to_max_width(frame, max_width, use_gpu=use_gpu)
    return small


def scale_gun_boxes(
    boxes: list[tuple[int, int, int, int, str, str, float, int, str]],
    sx: float,
    sy: float,
) -> list[tuple[int, int, int, int, str, str, float, int, str]]:
    if sx == 1.0 and sy == 1.0:
        return boxes
    out: list[tuple[int, int, int, int, str, str, float, int, str]] = []
    for x1, y1, x2, y2, glabel, gkind, gconf, pridx, gname in boxes:
        out.append(
            (
                int(round(x1 * sx)),
                int(round(y1 * sy)),
                int(round(x2 * sx)),
                int(round(y2 * sy)),
                glabel,
                gkind,
                gconf,
                pridx,
                gname,
            )
        )
    return out
