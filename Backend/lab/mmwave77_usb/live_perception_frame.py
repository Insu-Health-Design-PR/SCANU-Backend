"""Render a single human-perception JPEG frame (Adrian lab style) for UI live_frame.

Adapted from ``lab.mmwave77_usb.perception_video`` — matplotlib Agg → JPEG bytes,
without FFMpegWriter. Suitable for MJPEG / ``live_mmwave.jpg``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def render_perception_jpeg(
    *,
    points_xyz: np.ndarray,
    track: dict[str, Any] | None = None,
    anomalies: list[dict[str, Any]] | None = None,
    screening_state: str = "background",
    title: str = "mmWave · human perception",
    width: int = 960,
    height: int = 720,
) -> bytes:
    """
    Draw 3D + top/side panels with body (blue) vs anomaly (orange/red) colors.

    ``points_xyz`` shape (N, 3+) in meters. Optional track / anomalies match
    ``perception.jsonl`` field shapes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        pts = np.zeros((0, 3), dtype=np.float32)
    else:
        pts = pts[:, :3]

    body_mask = np.zeros(len(pts), dtype=bool)
    if track is not None and len(pts):
        center = np.asarray(track.get("position_m") or [0, 0, 0], dtype=np.float32)
        extent = np.asarray(track.get("observed_extent_m") or [0.6, 0.6, 1.6], dtype=np.float32)
        half = np.maximum(0.5 * extent + np.asarray([0.25, 0.25, 0.3], dtype=np.float32),
                          np.asarray([0.3, 0.3, 0.35], dtype=np.float32))
        body_mask = np.all(np.abs(pts - center) <= half, axis=1)

    anomaly_xyz = np.zeros((0, 3), dtype=np.float32)
    anomaly_colors: list[str] = []
    if anomalies:
        rows = []
        for item in anomalies:
            pos = item.get("position_m") or item.get("xyz_m")
            if not pos or len(pos) < 3:
                continue
            rows.append([float(pos[0]), float(pos[1]), float(pos[2])])
            persistent = bool(item.get("persistent") or item.get("is_persistent"))
            anomaly_colors.append("#ff4057" if persistent else "#ffc247")
        if rows:
            anomaly_xyz = np.asarray(rows, dtype=np.float32)

    dpi = 100
    fig_w, fig_h = width / dpi, height / dpi
    figure = plt.figure(figsize=(fig_w, fig_h), facecolor="#040a12")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.6, 1.0), height_ratios=(1.0, 1.0),
                               hspace=0.28, wspace=0.22)
    figure.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.88)
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_top = figure.add_subplot(grid[0, 1])
    axis_side = figure.add_subplot(grid[1, 1])

    figure.suptitle(title, x=0.05, ha="left", color="#f3f7fb", fontsize=13)
    figure.text(
        0.05, 0.02,
        f"state: {screening_state} · experimental radar evidence (not weapon confirmation)",
        color="#ffcc80", fontsize=8,
    )

    axis_3d.set_facecolor("#07111e")
    scene = ~body_mask
    if len(pts):
        if scene.any():
            axis_3d.scatter(pts[scene, 0], pts[scene, 1], pts[scene, 2],
                            s=5, c="#6e7f90", alpha=0.15, edgecolors="none")
        if body_mask.any():
            axis_3d.scatter(pts[body_mask, 0], pts[body_mask, 1], pts[body_mask, 2],
                            s=14, c="#4bb3fd", alpha=0.75, edgecolors="none")
    if len(anomaly_xyz):
        axis_3d.scatter(
            anomaly_xyz[:, 0], anomaly_xyz[:, 1], anomaly_xyz[:, 2],
            s=80, c=anomaly_colors or "#ffc247", marker="D",
            edgecolors="#fff4d6", linewidths=0.6,
        )
    axis_3d.scatter([0], [0], [0], marker="^", s=60, c="#f5f8fb")
    axis_3d.set_xlabel("X (m)", color="#c9d5e2", fontsize=8)
    axis_3d.set_ylabel("Y (m)", color="#c9d5e2", fontsize=8)
    axis_3d.set_zlabel("Z (m)", color="#c9d5e2", fontsize=8)
    axis_3d.tick_params(colors="#9aafc2", labelsize=6)
    axis_3d.set_xlim(-2.5, 2.5)
    axis_3d.set_ylim(0.0, 5.0)
    axis_3d.set_zlim(-0.5, 2.2)

    for ax, title_s, xs, ys in (
        (axis_top, "Top (X–Y)",
         (pts[:, 0], pts[:, 1]) if len(pts) else (np.array([]), np.array([])),
         None),
        (axis_side, "Side (Y–Z)",
         (pts[:, 1], pts[:, 2]) if len(pts) else (np.array([]), np.array([])),
         None),
    ):
        ax.set_facecolor("#07111e")
        ax.set_title(title_s, loc="left", color="#f3f7fb", fontsize=9)
        ax.tick_params(colors="#a9b8c8", labelsize=6)
        ax.grid(True, color="#29405a", alpha=0.4, linewidth=0.5)
        if len(pts):
            if ax is axis_top:
                if scene.any():
                    ax.scatter(pts[scene, 0], pts[scene, 1], s=4, c="#6e7f90", alpha=0.2)
                if body_mask.any():
                    ax.scatter(pts[body_mask, 0], pts[body_mask, 1], s=10, c="#4bb3fd", alpha=0.8)
                if len(anomaly_xyz):
                    ax.scatter(anomaly_xyz[:, 0], anomaly_xyz[:, 1], s=40, c=anomaly_colors or "#ffc247",
                               marker="D", edgecolors="#fff4d6")
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(0.0, 5.0)
            else:
                if scene.any():
                    ax.scatter(pts[scene, 1], pts[scene, 2], s=4, c="#6e7f90", alpha=0.2)
                if body_mask.any():
                    ax.scatter(pts[body_mask, 1], pts[body_mask, 2], s=10, c="#4bb3fd", alpha=0.8)
                if len(anomaly_xyz):
                    ax.scatter(anomaly_xyz[:, 1], anomaly_xyz[:, 2], s=40, c=anomaly_colors or "#ffc247",
                               marker="D", edgecolors="#fff4d6")
                ax.set_xlim(0.0, 5.0)
                ax.set_ylim(-0.5, 2.2)

    figure.canvas.draw()
    w, h = figure.canvas.get_width_height()
    buf = np.asarray(figure.canvas.buffer_rgba())
    rgb = buf[:, :, :3].copy()
    plt.close(figure)

    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("JPEG encode failed for mmWave perception frame")
    return bytes(encoded)


def render_status_jpeg(message: str, *, width: int = 960, height: int = 540) -> bytes:
    """Placeholder JPEG when radar is idle / missing."""
    import cv2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (18, 17, 12)
    y = 60
    for line in message.split("\n"):
        cv2.putText(img, line[:90], (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
        y += 32
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return bytes(encoded) if ok else b""
