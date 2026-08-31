#!/usr/bin/env python3
"""Render a combined dual-sensor video: cameras + fused point cloud + evidence.

Layout (1080p, dark theme):
  Top row:    Camera A | Fused point cloud (3D, A frame) | Camera B
  Bottom row: Top view | Side view | Elevation-azimuth | A/B/fused timeline
              | Combined evidence timeline

Each rendered frame corresponds to one matched radar window. Camera frames are
sought from the remuxed (real-host-time) mp4s using each sensor's
host_monotonic_ns alignment, so the cameras and the fused cloud stay on the
same timeline.

This is laboratory evidence visualization, not a material or weapon
classification.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lab.dual_mmwave77_stereo.point_cloud_fusion import (
    PerceptionSpec,
    _fuse_window,
    _load_clutter,
    _load_clutter_explicit,
    _window_anomalies_at,
    load_session,
    matched_window_pairs,
    window_center_monotonic_ns,
)
from lab.mmwave77_usb.artifacts import sha256_file

DUAL_FUSION_VIDEO_SCHEMA = "scanu_lab_dual_point_cloud_fusion_video_v1"


def _window_anomalies(session, index: int) -> list[dict[str, Any]]:
    return _window_anomalies_at(session, index)


def _load_camera_timestamps(camera_frames_path: Path) -> list[int]:
    timestamps: list[int] = []
    with camera_frames_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            timestamps.append(int(json.loads(line)["host_monotonic_ns"]))
    if not timestamps:
        raise ValueError(f"no camera frames in {camera_frames_path}")
    return timestamps


def _camera_seek_s(
    camera_timestamps_ns: list[int],
    radar_monotonic_ns: int,
) -> float:
    """Seconds into the remuxed camera mp4 for the nearest radar-time frame.

    Selecting the nearest recorded camera timestamp (rather than always the
    first timestamp after the radar sample) keeps the visual association error
    bounded by approximately half a camera period.  The returned value is a
    position in the camera file, not a radar-to-camera latency.
    """
    if not camera_timestamps_ns:
        return 0.0
    import bisect

    i = bisect.bisect_left(camera_timestamps_ns, radar_monotonic_ns)
    candidates = [max(0, min(i, len(camera_timestamps_ns) - 1))]
    if i > 0:
        candidates.append(i - 1)
    i = min(candidates, key=lambda candidate: abs(camera_timestamps_ns[candidate] - radar_monotonic_ns))
    camera_ns = camera_timestamps_ns[i]
    return (camera_ns - camera_timestamps_ns[0]) / 1e9


def _camera_alignment_error_ms(
    camera_timestamps_ns: list[int],
    radar_monotonic_ns: int,
) -> float:
    """Signed nearest-frame host-time residual, for transparent labeling."""
    if not camera_timestamps_ns:
        return float("nan")
    import bisect

    i = bisect.bisect_left(camera_timestamps_ns, radar_monotonic_ns)
    candidates = [max(0, min(i, len(camera_timestamps_ns) - 1))]
    if i > 0:
        candidates.append(i - 1)
    nearest = min(candidates, key=lambda candidate: abs(camera_timestamps_ns[candidate] - radar_monotonic_ns))
    return (camera_timestamps_ns[nearest] - radar_monotonic_ns) / 1e6


def _style_2d(axis, title: str) -> None:
    # Every animation frame redraws the panel.  Clearing prevents old lines
    # and legends from being retained as duplicate visual evidence.
    axis.clear()
    axis.set_facecolor("#07111e")
    axis.set_title(title, loc="left", color="#f3f7fb", fontsize=9, pad=7)
    axis.tick_params(colors="#a9b8c8", labelsize=6)
    axis.grid(True, color="#29405a", alpha=0.38, linewidth=0.5)
    for spine in axis.spines.values():
        spine.set_color("#29405a")


def _reported_point_sizes(points: np.ndarray) -> np.ndarray:
    """Small display sizes for actual post-CFAR returns, scaled by SNR.

    This is deliberately only a marker-size mapping.  It preserves each
    reported point rather than snapping points onto a voxel grid or inflating
    them into an artificial solid surface.
    """
    if not len(points):
        return np.empty(0, dtype=np.float32)
    snr = np.nan_to_num(points[:, 4], nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(snr, [10.0, 90.0])
    scale = (snr - lo) / max(float(hi - lo), 1e-6)
    return (2.0 + 4.5 * np.clip(scale, 0.0, 1.0)).astype(np.float32)


def _unique_legend(axis) -> None:
    """Avoid duplicate legend entries when A and B emit the same evidence."""
    handles, labels = axis.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    if unique:
        axis.legend(
            unique.values(), unique.keys(), loc="upper left", fontsize=6,
            facecolor="#091522", edgecolor="#29405a", labelcolor="#d7e2ec",
        )


def _person_mask(points: np.ndarray, centroid: list[float] | None, extent: list[float] | None) -> np.ndarray:
    if centroid is None or extent is None or not len(points):
        return np.zeros(len(points), dtype=bool)
    half = np.maximum(
        0.5 * np.asarray(extent, dtype=np.float32)
        + np.asarray([0.25, 0.25, 0.3], dtype=np.float32),
        np.asarray([0.3, 0.3, 0.35], dtype=np.float32),
    )
    center = np.asarray(centroid, dtype=np.float32)
    return np.all(np.abs(points[:, :3] - center) <= half, axis=1)


def _azimuth_elevation(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(points):
        return np.empty(0), np.empty(0)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    azimuth = np.degrees(np.arctan2(x, y))
    elevation = np.degrees(np.arctan2(z, np.hypot(x, y)))
    return azimuth, elevation


def _track_aligned_temporal_voxels(
    windows: list[dict[str, Any]],
    index: int,
    *,
    history_windows: int,
    voxel_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate *reported* person points in the current track frame.

    This intentionally works only on the post-CFAR points already present in
    the session.  It improves temporal readability by translating each prior
    person cluster to the current fused centroid, then returns voxel centres
    and a recency-weighted observation count.  It never interpolates a dense
    radar image or claims additional physical resolution.
    """
    if history_windows < 0 or voxel_m <= 0:
        raise ValueError("history_windows must be non-negative and voxel_m positive")
    current = windows[index]["row"]
    current_centroid = current.get("fused_centroid_m")
    if current_centroid is None:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
    current_center = np.asarray(current_centroid, dtype=np.float32)
    samples: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    start = max(0, index - history_windows)
    for source_index in range(start, index + 1):
        source = windows[source_index]
        source_centroid = source["row"].get("fused_centroid_m")
        if source_centroid is None:
            continue
        source_points = source["fused"]
        mask = _person_mask(
            source_points,
            source_centroid,
            source["row"].get("fused_extent_m"),
        )
        if not np.any(mask):
            continue
        aligned = (
            source_points[mask, :3]
            - np.asarray(source_centroid, dtype=np.float32)
            + current_center
        )
        age = index - source_index
        samples.append(aligned)
        weights.append(np.full(len(aligned), 1.0 / (1.0 + age), dtype=np.float32))
    if not samples:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)

    xyz = np.concatenate(samples, axis=0)
    weight = np.concatenate(weights, axis=0)
    keys = np.floor(xyz / voxel_m).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    voxel_weight = np.zeros(len(unique), dtype=np.float32)
    np.add.at(voxel_weight, inverse, weight)
    # A voxel centre is a display location, not an inferred return.
    centres = (unique.astype(np.float32) + 0.5) * voxel_m
    return centres, voxel_weight


def _anomaly_colors(anomalies: list[dict[str, Any]]) -> list[str]:
    """Yellow for unverified reflective anomalies, never a material label."""
    colors: list[str] = []
    for item in anomalies:
        colors.append("#ffc247")
    return colors


def _plot_anomalies(
    axis,
    centers: list[list[float]],
    anomalies: list[dict[str, Any]],
    *,
    marker: str = "D",
    size: float = 42,
    transform: bool = False,
) -> None:
    if not centers:
        return
    pts = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    colors = _anomaly_colors(anomalies)
    axis.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        marker=marker,
        s=size,
        c=colors,
        alpha=0.92,
        edgecolors="#fff3c4",
        linewidths=0.3,
        label="Single-view reflective candidate",
    )


def _corroborated_reflective_candidates(
    centers_a: list[list[float]],
    anomalies_a: list[dict[str, Any]],
    centers_b: list[list[float]],
    anomalies_b: list[dict[str, Any]],
    *,
    gate_m: float,
) -> list[dict[str, Any]]:
    """Associate persistent reflective candidates from both sensor views.

    This is deliberately a geometric visualization gate, not a material
    detector: each source has already reported a persistent body-associated
    SNR anomaly, and the two transformed locations must be close enough to be
    shown as one orange candidate.  A candidate from only one sensor remains
    yellow.  No point is invented and no material conclusion is made.
    """
    if gate_m <= 0:
        raise ValueError("dual-view reflectivity gate must be positive")

    def persistent_pairs(centers: list[list[float]], anomalies: list[dict[str, Any]]):
        result: list[tuple[np.ndarray, dict[str, Any]]] = []
        for center, anomaly in zip(centers, anomalies):
            if not bool(anomaly.get("persistent_body_associated_anomaly", False)):
                continue
            vector = np.asarray(center, dtype=np.float32)
            if vector.shape == (3,) and np.all(np.isfinite(vector)):
                result.append((vector, anomaly))
        return result

    a_items = persistent_pairs(centers_a, anomalies_a)
    b_items = persistent_pairs(centers_b, anomalies_b)
    possible: list[tuple[float, int, int]] = []
    for ia, (a_center, _) in enumerate(a_items):
        for ib, (b_center, _) in enumerate(b_items):
            distance = float(np.linalg.norm(a_center - b_center))
            if distance <= gate_m:
                possible.append((distance, ia, ib))

    used_a: set[int] = set()
    used_b: set[int] = set()
    candidates: list[dict[str, Any]] = []
    for distance, ia, ib in sorted(possible):
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        a_center, a_anomaly = a_items[ia]
        b_center, b_anomaly = b_items[ib]
        candidates.append(
            {
                "center_m": ((a_center + b_center) / 2.0).tolist(),
                "association_distance_m": distance,
                "score_a": float(a_anomaly.get("reflective_anomaly_score", 0.0)),
                "score_b": float(b_anomaly.get("reflective_anomaly_score", 0.0)),
            }
        )
    return candidates


def _plot_dual_view_candidates(axis, candidates: list[dict[str, Any]], *, x_index: int | None = None) -> None:
    """Render corroborated reflectivity locations in orange, without labels."""
    if not candidates:
        return
    centers = np.asarray([item["center_m"] for item in candidates], dtype=np.float32)
    if x_index is None:
        axis.scatter(
            centers[:, 0], centers[:, 1], centers[:, 2], marker="D", s=85,
            c="#ff8a3d", edgecolors="#fff2c7", linewidths=0.8, alpha=0.98,
            label="Dual-view reflective candidate",
        )
    else:
        axis.scatter(
            centers[:, x_index], centers[:, x_index + 1], marker="D", s=48,
            c="#ff8a3d", edgecolors="#fff2c7", linewidths=0.5, alpha=0.98,
        )


def _candidate_point_mask(
    points: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    radius_m: float = 0.20,
) -> np.ndarray:
    """Return reported points close to a dual-view candidate centre."""
    if not len(points) or not candidates:
        return np.zeros(len(points), dtype=bool)
    centers = np.asarray([item["center_m"] for item in candidates], dtype=np.float32)
    distances = np.linalg.norm(points[:, None, :3] - centers[None, :, :], axis=2)
    return np.any(distances <= radius_m, axis=1)


def _plot_anomalies_2d(
    axis,
    centers: list[list[float]],
    anomalies: list[dict[str, Any]],
    *,
    x_index: int,
    y_index: int,
) -> None:
    if not centers:
        return
    pts = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    colors = _anomaly_colors(anomalies)
    axis.scatter(
        pts[:, x_index],
        pts[:, y_index],
        marker="D",
        s=30,
        c=colors,
        alpha=0.92,
        edgecolors="#fff3c4",
        linewidths=0.2,
    )


def render_dual_fusion_video(
    session_a_dir: Path,
    session_b_dir: Path,
    *,
    distance_m: float,
    clock_offset_b_minus_a_s: float,
    window_tolerance_s: float,
    calibration_a: Path | None,
    calibration_b: Path | None,
    camera_a: Path,
    camera_b: Path | None,
    camera_a_frames: Path,
    camera_b_frames: Path | None,
    output_path: Path,
    fps: int = 5,
    dpi: int = 100,
    temporal_history_windows: int = 1,
    temporal_voxel_m: float = 0.12,
    point_cloud_style: str = "observed_points",
    dual_reflectivity_gate_m: float = 0.35,
    overwrite: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if point_cloud_style not in {"observed_points", "observed_sources", "temporal_density"}:
        raise ValueError("point_cloud_style must be observed_points, observed_sources or temporal_density")
    if dual_reflectivity_gate_m <= 0:
        raise ValueError("dual_reflectivity_gate_m must be positive")

    session_a = load_session(session_a_dir)
    session_b = load_session(session_b_dir)
    clutter_a = (
        _load_clutter_explicit(calibration_a)
        if calibration_a is not None
        else _load_clutter(session_a_dir)
    )
    clutter_b = (
        _load_clutter_explicit(calibration_b)
        if calibration_b is not None
        else _load_clutter(session_b_dir)
    )

    pairs = matched_window_pairs(
        session_a, session_b, clock_offset_b_minus_a_s, window_tolerance_s
    )
    if not pairs:
        raise ValueError("no matched window pairs; nothing to render")

    camera_a_timestamps = _load_camera_timestamps(camera_a_frames)
    cap_a = cv2.VideoCapture(str(camera_a))
    if not cap_a.isOpened():
        raise ValueError("could not open camera A video")
    if camera_b is not None and camera_b_frames is not None:
        camera_b_timestamps = _load_camera_timestamps(camera_b_frames)
        cap_b = cv2.VideoCapture(str(camera_b))
        if not cap_b.isOpened():
            raise ValueError("could not open camera B video")
    else:
        camera_b_timestamps = []
        cap_b = None

    # Pre-compute all fused windows with the exact report logic (track carried).
    spec = PerceptionSpec()
    fused_track: Any = None
    windows: list[dict[str, Any]] = []
    for i_a, i_b in pairs:
        row, pts_a, pts_b, fused, fused_track = _fuse_window(
            session_a,
            session_b,
            i_a,
            i_b,
            distance_m,
            clutter_a,
            clutter_b,
            15,
            spec,
            fused_track,
        )
        # Each perception row summarizes ten radar frames.  Associate the
        # camera with the temporal center of that evidence window.
        radar_center_a_ns = window_center_monotonic_ns(
            session_a, session_a.window_frame_start[i_a]
        )
        radar_center_b_ns = window_center_monotonic_ns(
            session_b, session_b.window_frame_start[i_b]
        )
        row["camera_a_seek_s"] = _camera_seek_s(
            camera_a_timestamps, radar_center_a_ns
        )
        row["camera_a_alignment_error_ms"] = _camera_alignment_error_ms(
            camera_a_timestamps, radar_center_a_ns
        )
        if camera_b_timestamps:
            row["camera_b_seek_s"] = _camera_seek_s(
                camera_b_timestamps, radar_center_b_ns
            )
            row["camera_b_alignment_error_ms"] = _camera_alignment_error_ms(
                camera_b_timestamps, radar_center_b_ns
            )
        else:
            row["camera_b_seek_s"] = None
            row["camera_b_alignment_error_ms"] = None
        windows.append(
            {
                "row": row,
                "pts_a": pts_a,
                "pts_b": pts_b,
                "fused": fused,
                "anomalies_a": _window_anomalies(session_a, i_a),
                "anomalies_b": _window_anomalies(session_b, i_b),
            }
        )

    n = len(windows)
    rows = [w["row"] for w in windows]
    time_s = np.asarray([r["time_s"] for r in rows], dtype=np.float32)
    a_points = np.asarray([r["a_person_points"] for r in rows], dtype=np.float32)
    b_points = np.asarray([r["b_person_points"] for r in rows], dtype=np.float32)
    fused_points = np.asarray([r["fused_person_points"] for r in rows], dtype=np.float32)
    track_conf = np.asarray([r["fused_track_confidence"] for r in rows], dtype=np.float32)
    track_present = np.asarray([r["fused_track_present"] for r in rows], dtype=bool)

    metadata_path = output_path.with_suffix(".metadata.json")
    if not overwrite and (output_path.exists() or metadata_path.exists()):
        raise ValueError(
            f"video output already exists: {output_path}; use --overwrite intentionally"
        )

    figure = plt.figure(figsize=(19.2, 10.8), facecolor="#040a12")
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.28, 0.72),
        hspace=0.16,
    )
    top = grid[0].subgridspec(1, 3, width_ratios=(1.0, 1.15, 1.0), wspace=0.06)
    bottom = grid[1].subgridspec(1, 5, wspace=0.34)
    figure.subplots_adjust(left=0.035, right=0.985, bottom=0.06, top=0.90)

    axis_cam_a = figure.add_subplot(top[0])
    axis_3d = figure.add_subplot(top[1], projection="3d")
    axis_cam_b = figure.add_subplot(top[2])
    axis_top = figure.add_subplot(bottom[0])
    axis_side = figure.add_subplot(bottom[1])
    axis_elev_az = figure.add_subplot(bottom[2])
    axis_timeline = figure.add_subplot(bottom[3])
    axis_combined = figure.add_subplot(bottom[4])

    figure.suptitle(
        "Two facing AWR1843BOOST · fused point-cloud evidence",
        x=0.035,
        ha="left",
        color="#f3f7fb",
        fontsize=17,
    )
    status_text = figure.text(
        0.035, 0.925, "", color="#a9bfd3", fontsize=9, ha="left"
    )
    figure.text(
        0.035,
        0.022,
        "Experimental radar evidence · not a confirmed material or weapon classification",
        color="#ffcc80",
        fontsize=9,
        weight="bold",
    )
    figure.text(
        0.5,
        0.022,
        "B transformed into A's frame (x'=-x, y'=D-y, z'=z) · D = "
        f"{distance_m:.2f} m",
        color="#879bad",
        fontsize=8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=6500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    with writer.saving(figure, str(output_path), dpi=dpi):
        for index, window in enumerate(windows):
            row = window["row"]
            pts_a = window["pts_a"]
            pts_b = window["pts_b"]
            fused = window["fused"]

            centroid = row["fused_centroid_m"]
            extent = row["fused_extent_m"]
            person_mask = _person_mask(fused, centroid, extent)
            scene_mask = ~person_mask
            dual_candidates = _corroborated_reflective_candidates(
                row["a_anomaly_centers"],
                window["anomalies_a"],
                row["b_anomaly_centers"],
                window["anomalies_b"],
                gate_m=dual_reflectivity_gate_m,
            )
            temporal_xyz, temporal_weight = _track_aligned_temporal_voxels(
                windows,
                index,
                history_windows=temporal_history_windows,
                voxel_m=temporal_voxel_m,
            )

            # --- Cameras ---------------------------------------------------
            for axis, cap, seek in (
                (axis_cam_a, cap_a, row["camera_a_seek_s"]),
                (axis_cam_b, cap_b, row["camera_b_seek_s"]),
            ):
                axis.clear()
                axis.set_axis_off()
                if cap is not None and seek is not None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, seek * 1000.0)
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        axis.imshow(frame_rgb)
                    else:
                        axis.set_facecolor("#050b14")
                        axis.text(
                            0.5,
                            0.5,
                            "camera frame unavailable",
                            transform=axis.transAxes,
                            ha="center",
                            va="center",
                            color="#9dafbf",
                            fontsize=9,
                        )
                else:
                    axis.set_facecolor("#050b14")
                    axis.text(
                        0.5,
                        0.5,
                        "camera B unavailable\n(device busy)",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                        color="#9dafbf",
                        fontsize=9,
                    )
            axis_cam_a.set_title(
                f"Camera A  ·  radar-time aligned ({row['camera_a_alignment_error_ms']:+.0f} ms)",
                loc="left",
                color="#f3f7fb",
                fontsize=9,
            )
            if row["camera_b_seek_s"] is not None:
                axis_cam_b.set_title(
                    f"Camera B  ·  radar-time aligned ({row['camera_b_alignment_error_ms']:+.0f} ms)",
                    loc="left",
                    color="#f3f7fb",
                    fontsize=9,
                )
            else:
                axis_cam_b.set_title(
                    "Camera B  ·  unavailable",
                    loc="left",
                    color="#f3f7fb",
                    fontsize=9,
                )

            # --- 3D fused cloud ---------------------------------------------
            axis_3d.clear()
            axis_3d.set_facecolor("#07111e")
            cloud_title = (
                "Fused observed volume · sensor-source provenance"
                if point_cloud_style == "observed_sources"
                else "Fused observed volume · small reported returns"
                if point_cloud_style == "observed_points"
                else "Fused observed volume · track-aligned temporal density"
            )
            axis_3d.set_title(
                cloud_title,
                loc="left",
                color="#f3f7fb",
                fontsize=11,
                pad=10,
            )
            if len(fused):
                axis_3d.scatter(
                    fused[scene_mask, 0],
                    fused[scene_mask, 1],
                    fused[scene_mask, 2],
                    s=4,
                    c="#6e7f90",
                    alpha=0.10,
                    edgecolors="none",
                    label="Scene returns",
                )
            if point_cloud_style == "observed_points":
                observed_person = fused[person_mask]
                if len(observed_person):
                    axis_3d.scatter(
                        observed_person[:, 0], observed_person[:, 1], observed_person[:, 2],
                        s=_reported_point_sizes(observed_person),
                        c=observed_person[:, 4], cmap="Blues", vmin=0, vmax=40,
                        alpha=0.64, edgecolors="none", label="Person reported returns",
                    )
            elif point_cloud_style == "observed_sources":
                a_person_mask = _person_mask(pts_a, centroid, extent)
                b_person_mask = _person_mask(pts_b, centroid, extent)
                if np.any(a_person_mask):
                    axis_3d.scatter(
                        pts_a[a_person_mask, 0], pts_a[a_person_mask, 1], pts_a[a_person_mask, 2],
                        s=12, c="#4bb3fd", alpha=0.72, edgecolors="none",
                        label="Person returns · A",
                    )
                if np.any(b_person_mask):
                    axis_3d.scatter(
                        pts_b[b_person_mask, 0], pts_b[b_person_mask, 1], pts_b[b_person_mask, 2],
                        s=12, c="#c58cff", alpha=0.72, edgecolors="none",
                        label="Person returns · B (transformed)",
                    )
            elif len(temporal_xyz):
                axis_3d.scatter(
                    temporal_xyz[:, 0], temporal_xyz[:, 1], temporal_xyz[:, 2],
                    s=8 + 16 * np.clip(temporal_weight, 0, 2),
                    c=temporal_weight, cmap="Blues", vmin=0, vmax=2,
                    alpha=0.76, edgecolors="none",
                    label="Track-aligned reported voxels",
                )
            reflective_points = _candidate_point_mask(fused, dual_candidates)
            if np.any(reflective_points):
                axis_3d.scatter(
                    fused[reflective_points, 0], fused[reflective_points, 1], fused[reflective_points, 2],
                    s=24, c="#ff8a3d", alpha=0.95, edgecolors="#fff2c7", linewidths=0.25,
                    label="Corroborating reported returns",
                )
            axis_3d.scatter([0], [0], [1.016], marker="^", s=70, c="#f5f8fb", label="Sensor A")
            axis_3d.scatter([0], [distance_m], [1.016], marker="^", s=70, c="#ffc247", label="Sensor B")
            _plot_anomalies(
                axis_3d,
                row["a_anomaly_centers"],
                window["anomalies_a"],
                marker="D",
                size=52,
            )
            _plot_dual_view_candidates(axis_3d, dual_candidates)
            _plot_anomalies(
                axis_3d,
                row["b_anomaly_centers"],
                window["anomalies_b"],
                marker="D",
                size=52,
            )
            if centroid is not None:
                axis_3d.scatter(
                    [centroid[0]], [centroid[1]], [centroid[2]],
                    marker="x", s=90, c="#38e5c5", linewidths=1.6,
                    label="Fused centroid",
                )
            plane_x, plane_y = np.meshgrid(
                np.linspace(-2.5, 2.5, 2), np.linspace(0, distance_m, 2)
            )
            axis_3d.plot_surface(
                plane_x, plane_y, np.zeros_like(plane_x),
                color="#284158", alpha=0.06, shade=False,
            )
            axis_3d.set(
                xlim=(-2.5, 2.5),
                ylim=(0, distance_m),
                zlim=(-1.6, 1.6),
                xlabel="x (m)",
                ylabel="y (m)",
                zlabel="z (m)",
            )
            axis_3d.view_init(elev=20, azim=-60)
            axis_3d.tick_params(colors="#9aafc2", labelsize=6)
            axis_3d.xaxis.label.set_color("#c9d5e2")
            axis_3d.yaxis.label.set_color("#c9d5e2")
            axis_3d.zaxis.label.set_color("#c9d5e2")
            _unique_legend(axis_3d)

            # --- Top view ----------------------------------------------------
            _style_2d(axis_top, "Top view · fused x-y")
            if len(fused):
                axis_top.scatter(fused[scene_mask, 0], fused[scene_mask, 1], s=3, c="#718191", alpha=0.12)
            if point_cloud_style == "observed_points":
                observed_person = fused[person_mask]
                axis_top.scatter(
                    observed_person[:, 0], observed_person[:, 1],
                    s=_reported_point_sizes(observed_person), c=observed_person[:, 4],
                    cmap="Blues", vmin=0, vmax=40, alpha=0.64,
                )
            elif point_cloud_style == "observed_sources":
                a_person_mask = _person_mask(pts_a, centroid, extent)
                b_person_mask = _person_mask(pts_b, centroid, extent)
                axis_top.scatter(pts_a[a_person_mask, 0], pts_a[a_person_mask, 1], s=8, c="#4bb3fd", alpha=0.72)
                axis_top.scatter(pts_b[b_person_mask, 0], pts_b[b_person_mask, 1], s=8, c="#c58cff", alpha=0.72)
            elif len(temporal_xyz):
                axis_top.scatter(temporal_xyz[:, 0], temporal_xyz[:, 1], s=8 + 12 * np.clip(temporal_weight, 0, 2), c=temporal_weight, cmap="Blues", vmin=0, vmax=2, alpha=0.76)
            if np.any(reflective_points):
                axis_top.scatter(fused[reflective_points, 0], fused[reflective_points, 1], s=16, c="#ff8a3d", alpha=0.95)
            axis_top.scatter([0], [0], marker="^", s=35, c="#ffffff")
            axis_top.scatter([0], [distance_m], marker="^", s=35, c="#ffc247")
            _plot_anomalies_2d(
                axis_top,
                row["a_anomaly_centers"],
                window["anomalies_a"],
                x_index=0,
                y_index=1,
            )
            _plot_dual_view_candidates(axis_top, dual_candidates, x_index=0)
            _plot_anomalies_2d(
                axis_top,
                row["b_anomaly_centers"],
                window["anomalies_b"],
                x_index=0,
                y_index=1,
            )
            if centroid is not None:
                axis_top.scatter([centroid[0]], [centroid[1]], marker="x", s=60, c="#38e5c5")
            axis_top.set(xlim=(-2.5, 2.5), ylim=(0, distance_m), xlabel="x (m)", ylabel="y (m)")

            # --- Side view -----------------------------------------------------
            _style_2d(axis_side, "Side view · fused y-z")
            if len(fused):
                axis_side.scatter(fused[scene_mask, 1], fused[scene_mask, 2], s=3, c="#718191", alpha=0.12)
            if point_cloud_style == "observed_points":
                observed_person = fused[person_mask]
                axis_side.scatter(
                    observed_person[:, 1], observed_person[:, 2],
                    s=_reported_point_sizes(observed_person), c=observed_person[:, 4],
                    cmap="Blues", vmin=0, vmax=40, alpha=0.64,
                )
            elif point_cloud_style == "observed_sources":
                a_person_mask = _person_mask(pts_a, centroid, extent)
                b_person_mask = _person_mask(pts_b, centroid, extent)
                axis_side.scatter(pts_a[a_person_mask, 1], pts_a[a_person_mask, 2], s=8, c="#4bb3fd", alpha=0.72)
                axis_side.scatter(pts_b[b_person_mask, 1], pts_b[b_person_mask, 2], s=8, c="#c58cff", alpha=0.72)
            elif len(temporal_xyz):
                axis_side.scatter(temporal_xyz[:, 1], temporal_xyz[:, 2], s=8 + 12 * np.clip(temporal_weight, 0, 2), c=temporal_weight, cmap="Blues", vmin=0, vmax=2, alpha=0.76)
            if np.any(reflective_points):
                axis_side.scatter(fused[reflective_points, 1], fused[reflective_points, 2], s=16, c="#ff8a3d", alpha=0.95)
            _plot_anomalies_2d(
                axis_side,
                row["a_anomaly_centers"],
                window["anomalies_a"],
                x_index=1,
                y_index=2,
            )
            _plot_dual_view_candidates(axis_side, dual_candidates, x_index=1)
            _plot_anomalies_2d(
                axis_side,
                row["b_anomaly_centers"],
                window["anomalies_b"],
                x_index=1,
                y_index=2,
            )
            if centroid is not None:
                axis_side.scatter([centroid[1]], [centroid[2]], marker="x", s=60, c="#38e5c5")
            axis_side.set(xlim=(0, distance_m), ylim=(-1.6, 1.6), xlabel="y (m)", ylabel="z (m)")

            # --- Elevation-azimuth ----------------------------------------------
            _style_2d(axis_elev_az, "Elevation-azimuth · fused returns")
            if len(fused):
                az, el = _azimuth_elevation(fused)
                axis_elev_az.scatter(az, el, s=3, c="#4bb3fd", alpha=0.45)
            axis_elev_az.set(xlim=(-60, 60), ylim=(-30, 30), xlabel="Azimuth (deg)", ylabel="Elevation (deg)")
            axis_elev_az.axvline(0, color="#29405a", linewidth=0.6)
            axis_elev_az.axhline(0, color="#29405a", linewidth=0.6)

            # --- A/B/fused timeline ----------------------------------------------
            _style_2d(axis_timeline, "Person-cluster points · A / B / fused")
            axis_timeline.plot(time_s[: index + 1], a_points[: index + 1], color="#8ab4ff", linewidth=1.3, label="Sensor A")
            axis_timeline.plot(time_s[: index + 1], b_points[: index + 1], color="#ffc247", linewidth=1.3, label="Sensor B")
            axis_timeline.plot(time_s[: index + 1], fused_points[: index + 1], color="#4bb3fd", linewidth=1.8, label="Fused")
            axis_timeline.axvline(time_s[index], color="#f3f7fb", linewidth=0.9, alpha=0.7)
            axis_timeline.set(
                xlim=(0, max(time_s[-1], 1.0)),
                ylim=(0, max(float(fused_points.max()), 50) * 1.05),
                xlabel="Time (s)",
                ylabel="Points",
            )
            axis_timeline.legend(
                loc="upper left", fontsize=6, facecolor="#091522",
                edgecolor="#29405a", labelcolor="#d7e2ec",
            )

            # --- Combined evidence timeline ---------------------------------------
            _style_2d(axis_combined, "Combined evidence · fused track")
            axis_combined.plot(time_s[: index + 1], track_conf[: index + 1], color="#38e5c5", linewidth=1.6, label="Fused track confidence")
            axis_combined.fill_between(
                time_s[: index + 1],
                0,
                track_conf[: index + 1],
                color="#38e5c5",
                alpha=0.15,
            )
            axis_combined.axvline(time_s[index], color="#f3f7fb", linewidth=0.9, alpha=0.7)
            axis_combined.axhline(0.68, color="#ff4057", linestyle="--", linewidth=0.8, alpha=0.7)
            axis_combined.set(
                xlim=(0, max(time_s[-1], 1.0)),
                ylim=(0, 1.02),
                xlabel="Time (s)",
                ylabel="Uncalibrated score",
            )
            axis_combined.legend(
                loc="upper left", fontsize=6, facecolor="#091522",
                edgecolor="#29405a", labelcolor="#d7e2ec",
            )

            present = (
                "PRESENT" if row["fused_track_present"]
                else "no fused track"
            )
            status_text.set_text(
                f"Window {index + 1}/{n} · time {row['time_s']:.2f} s · "
                f"fused {row['fused_person_points']} pts / A {row['a_person_points']} / "
                f"B {row['b_person_points']} · {point_cloud_style.replace('_', ' ')} · "
                f"dual-view reflective {len(dual_candidates)} · track conf "
                f"{row['fused_track_confidence']:.2f} · {present}"
            )
            writer.grab_frame()

    cap_a.release()
    if cap_b is not None:
        cap_b.release()
    plt.close(figure)

    metadata = {
        "schema_version": DUAL_FUSION_VIDEO_SCHEMA,
        "experimental": True,
        "material_confirmed": False,
        "weapon_classification": False,
        "sensor_distance_m": distance_m,
        "clock_offset_b_minus_a_s_applied": clock_offset_b_minus_a_s,
        "matched_window_pairs": n,
        "source_session_a": str(session_a_dir),
        "source_session_b": str(session_b_dir),
        "source_camera_a": str(camera_a),
        "source_camera_b": str(camera_b) if camera_b is not None else None,
        "anomaly_windows_a": int(sum(1 for r in rows if r["a_screening_state"] == "suspicious_metal")),
        "anomaly_windows_b": int(sum(1 for r in rows if r["b_screening_state"] == "suspicious_metal")),
        "anomaly_windows_either": int(
            sum(
                1
                for r in rows
                if r["a_screening_state"] == "suspicious_metal"
                or r["b_screening_state"] == "suspicious_metal"
            )
        ),
        "output_video": str(output_path),
        "output_video_sha256": sha256_file(output_path),
        "windows": n,
        "fps": fps,
        "point_cloud_style": point_cloud_style,
        "dual_view_reflective_visualization": {
            "enabled": True,
            "association_gate_m": dual_reflectivity_gate_m,
            "requires_persistent_source_anomaly": True,
            "requires_two_view_spatial_association": True,
            "does_not_confirm_material": True,
            "candidate_windows": int(
                sum(
                    bool(_corroborated_reflective_candidates(
                        r["a_anomaly_centers"],
                        windows[index]["anomalies_a"],
                        r["b_anomaly_centers"],
                        windows[index]["anomalies_b"],
                        gate_m=dual_reflectivity_gate_m,
                    ))
                    for index, r in enumerate(rows)
                )
            ),
        },
        "camera_alignment_error_ms": {
            "a_median": float(np.median([r["camera_a_alignment_error_ms"] for r in rows])),
            "a_max_abs": float(np.max(np.abs([r["camera_a_alignment_error_ms"] for r in rows]))),
            "b_median": (
                float(np.median([r["camera_b_alignment_error_ms"] for r in rows if r["camera_b_alignment_error_ms"] is not None]))
                if any(r["camera_b_alignment_error_ms"] is not None for r in rows) else None
            ),
            "b_max_abs": (
                float(np.max(np.abs([r["camera_b_alignment_error_ms"] for r in rows if r["camera_b_alignment_error_ms"] is not None])))
                if any(r["camera_b_alignment_error_ms"] is not None for r in rows) else None
            ),
        },
        "temporal_display": {
            "track_aligned_reported_points_only": True,
            "history_windows": temporal_history_windows,
            "voxel_m": temporal_voxel_m,
            "does_not_add_physical_resolution": True,
        },
        "duration_s": n / fps,
        "resolution_px": [int(round(19.2 * dpi)), int(round(10.8 * dpi))],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "fused person cluster is descriptive sparse-radar evidence, not anatomical reconstruction",
            "camera frames are host-receipt aligned (no hardware sync with the radar)",
            "scores are uncalibrated engineering evidence rankings",
            "video does not confirm material or classify a firearm",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output_path, metadata_path, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a combined dual-sensor evidence video (cameras + fused point cloud + graphs)"
    )
    parser.add_argument("--session-a", required=True, type=Path)
    parser.add_argument("--session-b", required=True, type=Path)
    parser.add_argument("--calibration-a", type=Path)
    parser.add_argument("--calibration-b", type=Path)
    parser.add_argument("--distance-m", required=True, type=float)
    parser.add_argument("--clock-offset-b-minus-a-s", type=float, default=0.0)
    parser.add_argument("--window-tolerance-s", type=float, default=0.5)
    parser.add_argument("--camera-a", required=True, type=Path)
    parser.add_argument("--camera-b", type=Path)
    parser.add_argument("--camera-a-frames", required=True, type=Path)
    parser.add_argument("--camera-b-frames", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--temporal-history-windows", type=int, default=1)
    parser.add_argument("--temporal-voxel-m", type=float, default=0.12)
    parser.add_argument(
        "--point-cloud-style",
        choices=("observed_points", "observed_sources", "temporal_density"),
        default="observed_points",
        help="observed_points draws small real returns; observed_sources preserves A/B provenance; temporal_density is a labeled display accumulation",
    )
    parser.add_argument(
        "--dual-reflectivity-gate-m",
        type=float,
        default=0.35,
        help="maximum A/B transformed-center separation for an orange dual-view reflective candidate",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        output, metadata_path, metadata = render_dual_fusion_video(
            args.session_a,
            args.session_b,
            distance_m=args.distance_m,
            clock_offset_b_minus_a_s=args.clock_offset_b_minus_a_s,
            window_tolerance_s=args.window_tolerance_s,
            calibration_a=args.calibration_a,
            calibration_b=args.calibration_b,
            camera_a=args.camera_a,
            camera_b=args.camera_b,
            camera_a_frames=args.camera_a_frames,
            camera_b_frames=args.camera_b_frames,
            output_path=args.output,
            fps=args.fps,
            dpi=args.dpi,
            temporal_history_windows=args.temporal_history_windows,
            temporal_voxel_m=args.temporal_voxel_m,
            point_cloud_style=args.point_cloud_style,
            dual_reflectivity_gate_m=args.dual_reflectivity_gate_m,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "video": str(output),
                "metadata": str(metadata_path),
                "windows": metadata["matched_window_pairs"],
                "duration_s": metadata["duration_s"],
                "sha256": metadata["output_video_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
