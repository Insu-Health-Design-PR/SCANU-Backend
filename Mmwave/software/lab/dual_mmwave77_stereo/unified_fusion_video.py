#!/usr/bin/env python3
"""Unified dual-sensor fusion video: one point cloud from both radars.

Layout (1080p, dark theme, same visual style as dual_fusion_video.py):
  Top row:     Camera A | Camera B
  Middle:      ONE fused point cloud (both sensors in a single world frame)
  Bottom row:  3 charts: Top view · Side view · Fused timeline (fused only)

Both sensors' post-CFAR points are transformed into sensor A's frame
(B: x'=-x, y'=D-y, z'=z) and drawn in the SAME scatter — the "unified" view.
Anomalies stay as yellow/red diamonds on the fused axes.

Produces four videos:
  1. <prefix>_unified.mp4        — cameras + fused cloud + 3 charts
  2. <prefix>_cameras_only.mp4   — only the two camera panels
  3. <prefix>_radar_only.mp4     — only the unified point cloud
  4. <prefix>_triptych.mp4       — Camera A | fused cloud | Camera B

This is laboratory evidence visualization, not a material or weapon
classification.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lab.dual_mmwave77_stereo.dual_fusion_video import (
    _azimuth_elevation,
    _camera_alignment_error_ms,
    _candidate_point_mask,
    _camera_seek_s,
    _corroborated_reflective_candidates,
    _load_camera_timestamps,
    _person_mask,
    _plot_anomalies,
    _plot_anomalies_2d,
    _plot_dual_view_candidates,
    _reported_point_sizes,
    _style_2d,
    _track_aligned_temporal_voxels,
    _unique_legend,
)
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
from lab.mmwave77_usb.perception_video import (
    _ellipsoid,
    _profile_sources,
    _window_profile,
)

UNIFIED_VIDEO_SCHEMA = "scanu_lab_dual_unified_fusion_video_v1"
TORSO_CORE_LATERAL_M = 0.20
TORSO_CORE_DEPTH_M = 0.18
TORSO_CORE_VERTICAL_M = 0.25
OFF_CORE_MIN_SCORE = 0.75
HUMAN_FOCUS_HALF_EXTENT_M = np.asarray([0.70, 0.75, 1.15], dtype=np.float32)


def _human_focus_limits(center: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return a person-scale display crop; this does not alter measurements."""
    center = np.asarray(center, dtype=np.float32)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("human-focus center must be a finite x/y/z coordinate")
    return tuple(
        (float(center[index] - HUMAN_FOCUS_HALF_EXTENT_M[index]),
         float(center[index] + HUMAN_FOCUS_HALF_EXTENT_M[index]))
        for index in range(3)
    )


def _human_focus_sizes(
    points: np.ndarray,
    *,
    density_weight: float = 1.0,
) -> np.ndarray:
    """Use SNR only to make displayed post-CFAR returns easier to inspect."""
    if not len(points):
        return np.empty((0,), dtype=np.float32)
    snr = np.nan_to_num(points[:, 4], nan=0.0, posinf=40.0, neginf=0.0)
    # A bounded display mapping avoids one bright return visually obscuring a person.
    if not np.isfinite(density_weight) or density_weight <= 0.0:
        raise ValueError("density_weight must be finite and positive")
    return (
        (12.0 + 16.0 * np.clip(snr / 40.0, 0.0, 1.0))
        * float(density_weight)
    ).astype(np.float32)


def _recent_track_returns(
    windows: list[dict[str, Any]],
    index: int,
    *,
    history_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original track-associated TLVs from a short recent history.

    This intentionally avoids voxel centres.  Older reported points only get a
    lower display alpha; neither their position nor their density is invented.
    """
    if history_windows <= 0:
        raise ValueError("history_windows must be positive")
    coordinates: list[np.ndarray] = []
    ages: list[np.ndarray] = []
    start = max(0, index - history_windows + 1)
    for source_index in range(start, index + 1):
        observed_a, observed_b, _ = _source_track_points(windows[source_index])
        points = (
            np.concatenate([observed_a, observed_b], axis=0)
            if len(observed_b)
            else observed_a
        )
        if len(points):
            coordinates.append(points[:, :3])
            ages.append(np.full(len(points), index - source_index, dtype=np.int16))
    if not coordinates:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int16),
        )
    return np.concatenate(coordinates, axis=0), np.concatenate(ages, axis=0)


def _balanced_density_weights(count_a: int, count_b: int) -> tuple[float, float]:
    """Keep the denser UART stream from visually dominating a fused display."""
    positive = [count for count in (count_a, count_b) if count > 0]
    if not positive:
        return 1.0, 1.0
    reference = float(min(positive))
    # Square-root scaling preserves the fact that one sensor reported more
    # returns while avoiding a 2x stream density becoming a 2x visual mass.
    return (
        float(np.sqrt(reference / max(count_a, 1))),
        float(np.sqrt(reference / max(count_b, 1))),
    )


def _persistent_single_view_anomalies(
    windows: list[dict[str, Any]],
    index: int,
    *,
    source: str,
    gate_m: float = 0.30,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Require a candidate in two adjacent 0.5 s windows before display.

    It is a visualization persistence gate only.  It does not determine a
    material class and cannot recover a return absent from the UART stream.
    """
    if source not in {"a", "b"}:
        raise ValueError("source must be 'a' or 'b'")
    row = windows[index]["row"]
    centers, anomalies = _off_core_anomalies(
        row[f"{source}_anomaly_centers"], windows[index][f"anomalies_{source}"]
    )
    if index == 0:
        return [], []
    previous = windows[index - 1]
    previous_centers, _ = _off_core_anomalies(
        previous["row"][f"{source}_anomaly_centers"],
        previous[f"anomalies_{source}"],
    )
    if not previous_centers:
        return [], []
    previous_xyz = np.asarray(previous_centers, dtype=np.float32)
    kept = [
        position
        for position, anomaly in zip(centers, anomalies)
        if np.any(np.linalg.norm(previous_xyz - np.asarray(position, dtype=np.float32), axis=1) <= gate_m)
    ]
    kept_anomalies = [
        anomaly
        for position, anomaly in zip(centers, anomalies)
        if np.any(np.linalg.norm(previous_xyz - np.asarray(position, dtype=np.float32), axis=1) <= gate_m)
    ]
    return kept, kept_anomalies


def _load_source_tracks(perception_path: Path) -> list[dict[str, Any] | None]:
    """Load the original per-radar tracks in perception-window order."""
    tracks: list[dict[str, Any] | None] = []
    with perception_path.open() as stream:
        for line in stream:
            if not line.strip() or '"window_index"' not in line:
                continue
            tracks.append(json.loads(line).get("track"))
    return tracks


def _transform_track_b_to_a(
    track: dict[str, Any] | None,
    distance_m: float,
) -> dict[str, Any] | None:
    """Transform B's original track box into A's world frame."""
    if track is None:
        return None
    transformed = dict(track)
    x, y, z = (float(value) for value in track["position_m"])
    transformed["position_m"] = [-x, distance_m - y, z]
    transformed["observed_extent_m"] = [
        float(value) for value in track["observed_extent_m"]
    ]
    return transformed


def _source_track_mask(
    points: np.ndarray,
    track: dict[str, Any] | None,
) -> np.ndarray:
    """Select only returns associated with one radar's original track box."""
    if track is None or not len(points):
        return np.zeros(len(points), dtype=bool)
    center = np.asarray(track["position_m"], dtype=np.float32)
    extent = np.asarray(track["observed_extent_m"], dtype=np.float32)
    half_extent = np.maximum(
        0.5 * extent + np.asarray([0.25, 0.25, 0.30], dtype=np.float32),
        np.asarray([0.30, 0.30, 0.35], dtype=np.float32),
    )
    return np.all(np.abs(points[:, :3] - center) <= half_extent, axis=1)


def _source_track_points(window: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return A points, B points, and their mask in the concatenated cloud."""
    mask_a = _source_track_mask(window["pts_a"], window.get("track_a"))
    mask_b = _source_track_mask(window["pts_b"], window.get("track_b"))
    source_mask = np.concatenate([mask_a, mask_b])
    return window["pts_a"][mask_a], window["pts_b"][mask_b], source_mask


def _classic_track_geometry(
    observed_a: np.ndarray,
    observed_b: np.ndarray,
    fallback_center: list[float] | None,
    fallback_extent: list[float] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Describe the fused observed body without voxelizing or adding points."""
    observed = (
        np.concatenate([observed_a, observed_b], axis=0)
        if len(observed_b)
        else observed_a
    )
    if len(observed) >= 4:
        xyz = observed[:, :3]
        center = np.median(xyz, axis=0).astype(np.float32)
        lo, hi = np.percentile(xyz, [5.0, 95.0], axis=0)
        extent = np.maximum(
            np.asarray(hi - lo, dtype=np.float32),
            np.asarray([0.10, 0.10, 0.20], dtype=np.float32),
        )
        return center, extent
    if fallback_center is None or fallback_extent is None:
        return None, None
    center = np.asarray(fallback_center, dtype=np.float32)
    extent = np.asarray(fallback_extent, dtype=np.float32)
    if center.shape != (3,) or extent.shape != (3,):
        return None, None
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(extent)):
        return None, None
    return center, np.maximum(extent, 0.0)


def _classic_anomaly_scores(window: dict[str, Any]) -> tuple[float, float, float]:
    """Return source and combined uncalibrated reflectivity evidence scores."""
    score_a = max(
        (float(item.get("reflective_anomaly_score", 0.0)) for item in window["anomalies_a"]),
        default=0.0,
    )
    score_b = max(
        (float(item.get("reflective_anomaly_score", 0.0)) for item in window["anomalies_b"]),
        default=0.0,
    )
    return score_a, score_b, max(score_a, score_b)


def _classic_anomaly_markers(
    window: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return transformed measured anomaly locations and classic colors/sizes."""
    centers = [
        *window["row"]["a_anomaly_centers"],
        *window["row"]["b_anomaly_centers"],
    ]
    anomalies = [*window["anomalies_a"], *window["anomalies_b"]]
    if not centers:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.float32),
        )
    scores = np.asarray(
        [float(item.get("reflective_anomaly_score", 0.0)) for item in anomalies],
        dtype=np.float32,
    )
    persistent = np.asarray(
        [bool(item.get("persistent_body_associated_anomaly", False)) for item in anomalies],
        dtype=bool,
    )
    colors = np.where(persistent, "#ff4057", "#ffc247")
    return np.asarray(centers, dtype=np.float32).reshape(-1, 3), colors, 90.0 + 90.0 * scores


def _off_core_anomalies(
    centers: list[list[float]],
    anomalies: list[dict[str, Any]],
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Keep strong off-core anomalies in the handheld-object view.

    Empty-room calibration cannot model a participant's normal torso return.
    This display gate therefore suppresses the track core while retaining a
    strong lateral/vertical/depth anomaly even before it becomes persistent.
    Single-view candidates stay yellow; the downstream dual-view association
    independently requires persistence before it can draw orange evidence.
    This remains a display heuristic, not a material classifier, and may
    suppress an object carried directly at the torso centre.
    """
    kept_centers: list[list[float]] = []
    kept_anomalies: list[dict[str, Any]] = []
    for center, anomaly in zip(centers, anomalies):
        if float(anomaly.get("reflective_anomaly_score", 0.0)) < OFF_CORE_MIN_SCORE:
            continue
        relative = np.asarray(anomaly.get("relative_to_track_m", []), dtype=np.float32)
        if relative.shape != (3,) or not np.all(np.isfinite(relative)):
            continue
        outside_core = (
            abs(float(relative[0])) >= TORSO_CORE_LATERAL_M
            or abs(float(relative[1])) >= TORSO_CORE_DEPTH_M
            or abs(float(relative[2])) >= TORSO_CORE_VERTICAL_M
        )
        if outside_core:
            kept_centers.append(center)
            kept_anomalies.append(anomaly)
    return kept_centers, kept_anomalies


def _render_figure(
    *,
    windows: list[dict[str, Any]],
    time_s: np.ndarray,
    fused_points: np.ndarray,
    track_conf: np.ndarray,
    distance_m: float,
    cap_a,
    cap_b,
    camera_a_timestamps: list[int],
    camera_b_timestamps: list[int],
    fps: int,
    dpi: int,
    output_main: Path,
    output_cameras: Path,
    output_radar: Path,
    output_triptych: Path,
    output_classic: Path,
    profiles_a: dict[str, Any] | None,
    profiles_b: dict[str, Any] | None,
    temporal_history_windows: int,
    temporal_voxel_m: float,
    dual_reflectivity_gate_m: float,
    human_focus: bool,
    triptych_only: bool,
    classic_human_centric: bool,
    classic_only: bool,
    overwrite: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    n = len(windows)
    rows = [w["row"] for w in windows]
    classic_scores = np.asarray(
        [_classic_anomaly_scores(window) for window in windows],
        dtype=np.float32,
    )

    def _make_figure():
        figure = plt.figure(figsize=(19.2, 10.8), facecolor="#040a12")
        grid = figure.add_gridspec(
            3,
            1,
            height_ratios=(1.0, 1.35, 0.9),
            hspace=0.28,
        )
        top = grid[0].subgridspec(1, 2, wspace=0.04)
        middle = grid[1].subgridspec(1, 1)
        bottom = grid[2].subgridspec(1, 3, wspace=0.34)
        figure.subplots_adjust(left=0.045, right=0.965, bottom=0.055, top=0.90)
        return figure, top, middle, bottom

    # --- main figure: cameras + unified cloud + 3 charts -------------------
    figure, top, middle, bottom = _make_figure()
    axis_cam_a = figure.add_subplot(top[0])
    axis_cam_b = figure.add_subplot(top[1])
    axis_3d = figure.add_subplot(middle[0], projection="3d")
    axis_top = figure.add_subplot(bottom[0])
    axis_side = figure.add_subplot(bottom[1])
    axis_timeline = figure.add_subplot(bottom[2])

    figure.suptitle(
        "Two facing AWR1843BOOST · unified point cloud (both sensors)",
        x=0.035,
        ha="left",
        color="#f3f7fb",
        fontsize=17,
    )
    status_text = figure.text(0.035, 0.925, "", color="#a9bfd3", fontsize=9, ha="left")
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
        "ONE fused cloud in A's frame (x'=-x, y'=D-y, z'=z) · D = "
        f"{distance_m:.2f} m",
        color="#879bad",
        fontsize=8,
    )

    # --- cameras-only figure ------------------------------------------------
    figure_cam, top_cam, _, _ = _make_figure()
    cam_only_a = figure_cam.add_subplot(top_cam[0])
    cam_only_b = figure_cam.add_subplot(top_cam[1])
    figure_cam.suptitle(
        "Two facing AWR1843BOOST · cameras only",
        x=0.035, ha="left", color="#f3f7fb", fontsize=17,
    )

    # --- radar-only figure ----------------------------------------------------
    figure_rad, _, middle_rad, _ = _make_figure()
    axis_3d_rad = figure_rad.add_subplot(middle_rad[0], projection="3d")
    figure_rad.suptitle(
        "Two facing AWR1843BOOST · unified point cloud only",
        x=0.035, ha="left", color="#f3f7fb", fontsize=17,
    )
    figure_rad.text(
        0.5,
        0.022,
        "ONE fused cloud in A's frame (x'=-x, y'=D-y, z'=z) · D = "
        f"{distance_m:.2f} m",
        color="#879bad", fontsize=8, ha="center",
    )

    # --- synchronized triptych: Camera A | fused radar | Camera B ---------
    # This uses the same canvas as the other exports: 1920x1080 at 100 dpi or
    # 3840x2160 at the complete-demo preset's 200 dpi.
    figure_trip = plt.figure(figsize=(19.2, 10.8), facecolor="#040a12")
    trip_grid = figure_trip.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 1.0),
        wspace=0.025,
    )
    figure_trip.subplots_adjust(left=0.015, right=0.985, bottom=0.075, top=0.90)
    trip_cam_a = figure_trip.add_subplot(trip_grid[0])
    trip_radar = figure_trip.add_subplot(trip_grid[1], projection="3d")
    trip_cam_b = figure_trip.add_subplot(trip_grid[2])
    figure_trip.suptitle(
        "Two facing AWR1843BOOST · synchronized cameras and fused radar",
        x=0.015,
        ha="left",
        color="#f3f7fb",
        fontsize=18,
    )
    trip_status = figure_trip.text(
        0.015, 0.925, "", color="#a9bfd3", fontsize=9, ha="left",
    )
    figure_trip.text(
        0.015,
        0.022,
        "Experimental post-CFAR radar evidence · not a confirmed material or weapon classification",
        color="#ffcc80",
        fontsize=9,
        weight="bold",
    )
    figure_trip.text(
        0.985,
        0.022,
        "Camera A  |  geometrically fused radar A+B  |  Camera B",
        color="#879bad",
        fontsize=8,
        ha="right",
    )

    # --- classic fused dashboard ------------------------------------------
    # This intentionally mirrors the proven single-radar human-centric
    # renderer. It uses one measured A+B cloud and does not voxelize,
    # interpolate, or synthesize body returns.
    figure_classic = plt.figure(figsize=(19.2, 10.8), facecolor="#040a12")
    classic_grid = figure_classic.add_gridspec(
        3,
        3,
        width_ratios=(1.3, 1.3, 1.0),
        height_ratios=(1.0, 1.0, 0.78),
        hspace=0.35,
        wspace=0.22,
    )
    figure_classic.subplots_adjust(
        left=0.04, right=0.98, bottom=0.09, top=0.88
    )
    classic_3d = figure_classic.add_subplot(classic_grid[0:2, 0:2], projection="3d")
    classic_top = figure_classic.add_subplot(classic_grid[0, 2])
    classic_side = figure_classic.add_subplot(classic_grid[1, 2])
    classic_profile = figure_classic.add_subplot(classic_grid[2, 0])
    classic_world = figure_classic.add_subplot(classic_grid[2, 1])
    classic_timeline = figure_classic.add_subplot(classic_grid[2, 2])
    figure_classic.suptitle(
        "Two facing AWR1843BOOST · classic human-centric fused perception",
        x=0.04,
        ha="left",
        color="#f3f7fb",
        fontsize=18,
    )
    classic_status = figure_classic.text(
        0.04, 0.915, "", color="#a9bfd3", fontsize=10, ha="left"
    )
    figure_classic.text(
        0.04,
        0.027,
        "Experimental reflective evidence · Not confirmed metal or weapon classification",
        color="#ffcc80",
        fontsize=10,
        weight="bold",
    )
    figure_classic.text(
        0.61,
        0.027,
        "Measured sparse post-CFAR TLVs · no interpolation or added physical resolution",
        color="#879bad",
        fontsize=8,
    )

    def _draw_cameras(axis_a, axis_b, row) -> None:
        for axis, cap, seek in (
            (axis_a, cap_a, row["camera_a_seek_s"]),
            (axis_b, cap_b, row["camera_b_seek_s"]),
        ):
            axis.clear()
            axis.set_axis_off()
            if cap is not None and seek is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, seek * 1000.0)
                ok, frame = cap.read()
                if ok and frame is not None:
                    axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    axis.set_facecolor("#050b14")
                    axis.text(0.5, 0.5, "camera frame unavailable",
                              transform=axis.transAxes, ha="center", va="center",
                              color="#9dafbf", fontsize=9)
            else:
                axis.set_facecolor("#050b14")
                axis.text(0.5, 0.5, "camera unavailable\n(device busy)",
                          transform=axis.transAxes, ha="center", va="center",
                          color="#9dafbf", fontsize=9)
        axis_a.set_title(
            f"Camera A  ·  radar-window aligned "
            f"({row['camera_a_alignment_error_ms']:+.0f} ms)",
            loc="left", color="#f3f7fb", fontsize=9,
        )
        axis_b.set_title(
            f"Camera B  ·  radar-window aligned "
            f"({row['camera_b_alignment_error_ms']:+.0f} ms)"
            if row["camera_b_seek_s"] is not None
            else "Camera B  ·  unavailable",
            loc="left", color="#f3f7fb", fontsize=9,
        )

    def _draw_cloud(
        axis,
        window,
        *,
        show_charts_footer: bool,
        person_zoom: bool = False,
        human_focus_style: bool = False,
    ) -> None:
        row = window["row"]
        fused = window["fused"]
        centroid = row["fused_centroid_m"]
        extent = row["fused_extent_m"]
        observed_a, observed_b, person_mask = _source_track_points(window)
        observed_person = (
            np.concatenate([observed_a, observed_b], axis=0)
            if len(observed_b)
            else observed_a
        )
        scene_mask = ~person_mask
        index = windows.index(window)
        # A facing radar can lose a handheld return as the participant turns,
        # self-occludes, or changes aspect. Yellow therefore reports a strong
        # off-core observation from either source at the current instant.
        # Orange remains the stricter, optional A+B persistent association.
        a_centers, a_anomalies = _off_core_anomalies(
            row["a_anomaly_centers"], window["anomalies_a"]
        )
        b_centers, b_anomalies = _off_core_anomalies(
            row["b_anomaly_centers"], window["anomalies_b"]
        )
        dual_candidates = _corroborated_reflective_candidates(
            a_centers, a_anomalies,
            b_centers, b_anomalies,
            gate_m=dual_reflectivity_gate_m,
        )
        if human_focus_style:
            temporal_xyz, temporal_age = _recent_track_returns(
                windows, index, history_windows=temporal_history_windows,
            )
            # Current raw points get 1.0 and the preceding half-second gets
            # 0.25. These are alpha values, not a signal-strength estimate.
            temporal_weight = np.where(temporal_age == 0, 1.0, 0.25).astype(np.float32)
        else:
            temporal_xyz, temporal_weight = _track_aligned_temporal_voxels(
                windows, index,
                history_windows=temporal_history_windows, voxel_m=temporal_voxel_m,
            )
        axis.clear()
        axis.set_facecolor("#07111e")
        axis.set_title(
            "Unified point cloud · original track returns from A + B"
            + (" · person-centered temporal view" if human_focus_style else "")
            + (" · person-centered view" if person_zoom and not human_focus_style else ""),
            loc="left", color="#f3f7fb", fontsize=11, pad=10,
        )
        if human_focus_style and len(temporal_xyz):
            weights = np.asarray(temporal_weight, dtype=np.float32)
            axis.scatter(
                temporal_xyz[:, 0], temporal_xyz[:, 1], temporal_xyz[:, 2],
                s=8.0, c="#5aa7de", alpha=0.12, edgecolors="none",
                label="Recent reported returns (≤ 1 s)",
            )
        if len(fused):
            axis.scatter(fused[scene_mask, 0], fused[scene_mask, 1], fused[scene_mask, 2],
                         s=4, c="#6e7f90", alpha=0.10 if not human_focus_style else 0.04,
                         edgecolors="none", label="Scene returns")
        density_a, density_b = _balanced_density_weights(len(observed_a), len(observed_b))
        if len(observed_a):
            axis.scatter(
                observed_a[:, 0], observed_a[:, 1], observed_a[:, 2],
                s=(_human_focus_sizes(observed_a, density_weight=density_a)
                   if human_focus_style else 7),
                c="#55b7ff", alpha=0.82 if human_focus_style else 0.72, edgecolors="none",
                label="Track returns · sensor A",
            )
        if len(observed_b):
            axis.scatter(
                observed_b[:, 0], observed_b[:, 1], observed_b[:, 2],
                s=(_human_focus_sizes(observed_b, density_weight=density_b)
                   if human_focus_style else 7),
                c="#b0e3ff", alpha=0.70 if human_focus_style else 0.58, edgecolors="none",
                label="Track returns · sensor B",
            )
        reflective_points = _candidate_point_mask(fused, dual_candidates)
        reflective_points &= person_mask
        if np.any(reflective_points):
            axis.scatter(fused[reflective_points, 0], fused[reflective_points, 1], fused[reflective_points, 2],
                         s=28 if human_focus_style else 7, c="#ff8a3d", alpha=0.92,
                         edgecolors="none", label="Corroborating reported returns")
        if not person_zoom:
            axis.scatter([0], [0], [1.016], marker="^", s=70, c="#f5f8fb", label="Sensor A")
            axis.scatter([0], [distance_m], [1.016], marker="^", s=70, c="#ffc247", label="Sensor B")
        if human_focus_style:
            # Yellow is a single-radar reflective excess. Orange requires a
            # persistent association from both views. Neither label claims a
            # material identity, and both remain visible in the clean triptych.
            _plot_anomalies(axis, a_centers, a_anomalies, marker="D", size=105)
            _plot_anomalies(axis, b_centers, b_anomalies, marker="D", size=105)
            _plot_dual_view_candidates(axis, dual_candidates)
        else:
            _plot_anomalies(axis, a_centers, a_anomalies, marker="D", size=42)
            _plot_anomalies(axis, b_centers, b_anomalies, marker="D", size=42)
            _plot_dual_view_candidates(axis, dual_candidates)
        if centroid is not None and not human_focus_style:
            index = windows.index(window)
            history = np.asarray(
                [
                    candidate["row"]["fused_centroid_m"]
                    for candidate in windows[max(0, index - 12): index + 1]
                    if candidate["row"]["fused_centroid_m"] is not None
                ],
                dtype=np.float32,
            )
            if len(history) >= 2:
                axis.plot(
                    history[:, 0], history[:, 1], history[:, 2],
                    color="#38e5c5", linewidth=2.0, alpha=0.85,
                    label="Fused track history",
                )
            axis.scatter([centroid[0]], [centroid[1]], [centroid[2]],
                         marker="x", s=90, c="#38e5c5", linewidths=1.6, label="Fused centroid")
        if not human_focus_style:
            plane_x, plane_y = np.meshgrid(np.linspace(-2.5, 2.5, 2), np.linspace(0, distance_m, 2))
            axis.plot_surface(plane_x, plane_y, np.zeros_like(plane_x),
                              color="#284158", alpha=0.06, shade=False)
        if person_zoom and centroid is not None:
            center = (
                np.median(observed_person[:, :3], axis=0)
                if len(observed_person)
                else np.asarray(centroid, dtype=np.float32)
            )
            if human_focus_style:
                xlim, ylim, zlim = _human_focus_limits(center)
                x_half, y_half, z_half = (float(value) for value in HUMAN_FOCUS_HALF_EXTENT_M)
            else:
                x_half, y_half, z_half = 1.15, 1.15, 1.35
                xlim = (center[0] - x_half, center[0] + x_half)
                ylim = (center[1] - y_half, center[1] + y_half)
                zlim = (center[2] - z_half, center[2] + z_half)
            axis.set(
                xlim=xlim, ylim=ylim, zlim=zlim,
                xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
            )
            axis.set_box_aspect((2.0 * x_half, 2.0 * y_half, 2.0 * z_half))
        else:
            axis.set(xlim=(-2.5, 2.5), ylim=(0, distance_m), zlim=(-1.6, 1.6),
                     xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
        axis.view_init(elev=14 if human_focus_style else 20, azim=-68 if human_focus_style else -60)
        axis.tick_params(colors="#9aafc2", labelsize=6)
        axis.xaxis.label.set_color("#c9d5e2")
        axis.yaxis.label.set_color("#c9d5e2")
        axis.zaxis.label.set_color("#c9d5e2")
        if human_focus_style:
            # Matplotlib's default opaque panes read like a physical cube.  Keep
            # the coordinate crop, but remove that visual enclosure so only
            # observed post-CFAR returns carry visual weight.
            for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
                pane.set_facecolor((0.02, 0.05, 0.09, 0.0))
                pane.set_edgecolor((0.18, 0.30, 0.42, 0.22))
            axis.set_box_aspect((1.40, 1.50, 2.30))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_zticks([])
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.set_zlabel("")
            axis.grid(False)
            axis.text2D(
                0.02, 0.02,
                "original post-CFAR returns · ≤ 1 s display history · no added physical resolution",
                transform=axis.transAxes, color="#9dafbf", fontsize=6,
            )
        _unique_legend(axis)
        return temporal_xyz, temporal_weight, dual_candidates, reflective_points

    def _draw_charts(window, temporal_xyz, temporal_weight) -> None:
        row = window["row"]
        fused = window["fused"]
        centroid = row["fused_centroid_m"]
        index = windows.index(window)
        a_centers, a_anomalies = _off_core_anomalies(
            row["a_anomaly_centers"], window["anomalies_a"]
        )
        b_centers, b_anomalies = _off_core_anomalies(
            row["b_anomaly_centers"], window["anomalies_b"]
        )
        dual_candidates = _corroborated_reflective_candidates(
            a_centers, a_anomalies,
            b_centers, b_anomalies,
            gate_m=dual_reflectivity_gate_m,
        )
        reflective_points = _candidate_point_mask(fused, dual_candidates)
        # Top view · fused x-y
        _style_2d(axis_top, "Top view · unified x-y")
        if len(fused):
            axis_top.scatter(fused[:, 0], fused[:, 1], s=3, c="#718191", alpha=0.12)
        person_mask = _person_mask(fused, row["fused_centroid_m"], row["fused_extent_m"])
        observed_person = fused[person_mask]
        if len(observed_person):
            axis_top.scatter(observed_person[:, 0], observed_person[:, 1],
                             s=_reported_point_sizes(observed_person), c=observed_person[:, 4],
                             cmap="Blues", vmin=0, vmax=40, alpha=0.64)
        axis_top.scatter([0], [0], marker="^", s=35, c="#ffffff")
        axis_top.scatter([0], [distance_m], marker="^", s=35, c="#ffc247")
        _plot_anomalies_2d(axis_top, a_centers, a_anomalies,
                           x_index=0, y_index=1)
        _plot_anomalies_2d(axis_top, b_centers, b_anomalies,
                           x_index=0, y_index=1)
        if np.any(reflective_points):
            axis_top.scatter(fused[reflective_points, 0], fused[reflective_points, 1], s=7, c="#ff8a3d", alpha=0.90)
        _plot_dual_view_candidates(axis_top, dual_candidates, x_index=0)
        if centroid is not None:
            axis_top.scatter([centroid[0]], [centroid[1]], marker="x", s=60, c="#38e5c5")
        axis_top.set(xlim=(-2.5, 2.5), ylim=(0, distance_m), xlabel="x (m)", ylabel="y (m)")

        # Side view · fused y-z
        _style_2d(axis_side, "Side view · unified y-z")
        if len(fused):
            axis_side.scatter(fused[:, 1], fused[:, 2], s=3, c="#718191", alpha=0.12)
        if len(observed_person):
            axis_side.scatter(observed_person[:, 1], observed_person[:, 2],
                              s=_reported_point_sizes(observed_person), c=observed_person[:, 4],
                              cmap="Blues", vmin=0, vmax=40, alpha=0.64)
        _plot_anomalies_2d(axis_side, a_centers, a_anomalies,
                           x_index=1, y_index=2)
        _plot_anomalies_2d(axis_side, b_centers, b_anomalies,
                           x_index=1, y_index=2)
        if np.any(reflective_points):
            axis_side.scatter(fused[reflective_points, 1], fused[reflective_points, 2], s=7, c="#ff8a3d", alpha=0.90)
        _plot_dual_view_candidates(axis_side, dual_candidates, x_index=1)
        if centroid is not None:
            axis_side.scatter([centroid[1]], [centroid[2]], marker="x", s=60, c="#38e5c5")
        axis_side.set(xlim=(0, distance_m), ylim=(-1.6, 1.6), xlabel="y (m)", ylabel="z (m)")

        # Timeline · fused only
        _style_2d(axis_timeline, "Person-cluster points · unified (fused)")
        axis_timeline.plot(time_s[: index + 1], fused_points[: index + 1],
                           color="#4bb3fd", linewidth=2.0, label="Unified cloud")
        axis_timeline.axvline(time_s[index], color="#f3f7fb", linewidth=0.9, alpha=0.7)
        axis_timeline.set(
            xlim=(0, max(float(time_s[-1]), 1.0)),
            ylim=(0, max(float(fused_points.max()), 50) * 1.05),
            xlabel="Time (s)", ylabel="Points",
        )
        axis_timeline.legend(loc="upper left", fontsize=6, facecolor="#091522",
                             edgecolor="#29405a", labelcolor="#d7e2ec")
        # Fused track confidence on the same axis (right side, secondary)
        ax2 = axis_timeline.twinx()
        ax2.clear()
        ax2.plot(time_s[: index + 1], track_conf[: index + 1],
                 color="#38e5c5", linewidth=1.2, alpha=0.8)
        ax2.set_ylim(0, 1.02)
        ax2.tick_params(colors="#38e5c5", labelsize=5)
        ax2.set_ylabel("track conf", color="#38e5c5", fontsize=7)

    def _draw_classic_frame(index: int, window: dict[str, Any]) -> None:
        """Draw the established six-panel dashboard with one fused cloud."""
        from matplotlib.patches import Ellipse, Rectangle

        row = window["row"]
        fused = window["fused"]
        observed_a, observed_b, person_mask = _source_track_points(window)
        scene_mask = ~person_mask
        observed = (
            np.concatenate([observed_a, observed_b], axis=0)
            if len(observed_b)
            else observed_a
        )
        center, extent = _classic_track_geometry(
            observed_a,
            observed_b,
            row["fused_centroid_m"],
            row["fused_extent_m"],
        )
        anomaly_xyz, anomaly_colors, anomaly_sizes = _classic_anomaly_markers(window)
        for axis in (
            classic_3d,
            classic_top,
            classic_side,
            classic_profile,
            classic_world,
            classic_timeline,
        ):
            axis.clear()

        classic_3d.set_facecolor("#07111e")
        classic_3d.set_title(
            "Observed fused radar volume · global human track",
            loc="left", color="#f3f7fb", fontsize=12, pad=12,
        )
        if len(fused):
            classic_3d.scatter(
                fused[scene_mask, 0], fused[scene_mask, 1], fused[scene_mask, 2],
                s=5, c="#6e7f90", alpha=0.12, edgecolors="none",
                label="Scene returns",
            )
        if len(observed):
            classic_3d.scatter(
                observed[:, 0], observed[:, 1], observed[:, 2],
                s=12, c="#4bb3fd", alpha=0.68, edgecolors="none",
                label="Fused track-associated returns",
            )
        classic_3d.scatter(
            [0], [0], [0], marker="^", s=85, c="#f5f8fb", label="Sensor A"
        )
        classic_3d.scatter(
            [0], [distance_m], [0], marker="^", s=85, c="#ffc247", label="Sensor B"
        )
        if center is not None and extent is not None:
            ex, ey, ez = _ellipsoid(center, extent)
            classic_3d.plot_wireframe(
                ex, ey, ez, rstride=2, cstride=2, color="#59c7ff",
                alpha=0.42, linewidth=0.6,
            )
            history = np.asarray(
                [
                    candidate["row"]["fused_centroid_m"]
                    for candidate in windows[max(0, index - 12): index + 1]
                    if candidate["row"]["fused_centroid_m"] is not None
                ],
                dtype=np.float32,
            )
            if len(history) >= 2:
                classic_3d.plot(
                    history[:, 0], history[:, 1], history[:, 2],
                    color="#38e5c5", linewidth=2.0, alpha=0.85,
                    label="Fused track history",
                )
        if len(anomaly_xyz):
            classic_3d.scatter(
                anomaly_xyz[:, 0], anomaly_xyz[:, 1], anomaly_xyz[:, 2],
                s=anomaly_sizes, c=anomaly_colors, marker="D",
                edgecolors="#fff4d6", linewidths=0.8,
                label="Reflective anomaly",
            )
        plane_x, plane_y = np.meshgrid(
            np.linspace(-2.2, 2.2, 2), np.linspace(0, distance_m, 2)
        )
        classic_3d.plot_surface(
            plane_x, plane_y, np.zeros_like(plane_x),
            color="#284158", alpha=0.08, shade=False,
        )
        classic_3d.set(
            xlim=(-2.2, 2.2), ylim=(0, distance_m), zlim=(-1.6, 1.6),
            xlabel="Right / left x (m)", ylabel="World depth y (m)",
            zlabel="Relative elevation z (m)",
        )
        classic_3d.view_init(elev=23, azim=-58)
        classic_3d.tick_params(colors="#9aafc2", labelsize=7)
        classic_3d.xaxis.label.set_color("#c9d5e2")
        classic_3d.yaxis.label.set_color("#c9d5e2")
        classic_3d.zaxis.label.set_color("#c9d5e2")
        _unique_legend(classic_3d)

        _style_2d(classic_top, "Top view · fused person location")
        if len(fused):
            classic_top.scatter(
                fused[scene_mask, 0], fused[scene_mask, 1], s=4,
                c="#718191", alpha=0.12,
            )
        if len(observed):
            classic_top.scatter(
                observed[:, 0], observed[:, 1], s=12,
                c="#4bb3fd", alpha=0.70,
            )
        classic_top.scatter([0, 0], [0, distance_m], marker="^", s=45,
                            c=["#ffffff", "#ffc247"])
        if center is not None and extent is not None:
            classic_top.add_patch(Ellipse(
                (float(center[0]), float(center[1])),
                width=max(float(extent[0]) + 0.5, 0.6),
                height=max(float(extent[1]) + 0.5, 0.6),
                facecolor="#2ea8ff", edgecolor="#8ad8ff",
                alpha=0.18, linewidth=1.4,
            ))
            classic_top.text(
                float(center[0]), float(center[1]), " GLOBAL TRACK",
                color="#c8efff", fontsize=7, weight="bold",
            )
        if len(anomaly_xyz):
            classic_top.scatter(
                anomaly_xyz[:, 0], anomaly_xyz[:, 1],
                s=0.52 * anomaly_sizes, c=anomaly_colors, marker="D",
                edgecolors="#fff4d6", linewidths=0.5,
            )
        classic_top.set(
            xlim=(-2.2, 2.2), ylim=(0, distance_m),
            xlabel="Right / left x (m)", ylabel="World depth y (m)",
        )

        _style_2d(classic_side, "Side view · fused vertical extent")
        if len(fused):
            classic_side.scatter(
                fused[scene_mask, 1], fused[scene_mask, 2], s=4,
                c="#718191", alpha=0.12,
            )
        if len(observed):
            classic_side.scatter(
                observed[:, 1], observed[:, 2], s=12,
                c="#4bb3fd", alpha=0.70,
            )
        if center is not None and extent is not None:
            width = max(float(extent[1]) + 0.5, 0.6)
            height = max(float(extent[2]) + 0.6, 0.7)
            classic_side.add_patch(Rectangle(
                (float(center[1]) - width / 2.0, float(center[2]) - height / 2.0),
                width, height, facecolor="#2ea8ff", edgecolor="#8ad8ff",
                alpha=0.18, linewidth=1.4,
            ))
        if len(anomaly_xyz):
            classic_side.scatter(
                anomaly_xyz[:, 1], anomaly_xyz[:, 2],
                s=0.52 * anomaly_sizes, c=anomaly_colors, marker="D",
                edgecolors="#fff4d6", linewidths=0.5,
            )
        classic_side.set(
            xlim=(0, distance_m), ylim=(-1.6, 1.6),
            xlabel="World depth y (m)", ylabel="Relative elevation z (m)",
        )

        _style_2d(classic_profile, "Range profiles · A/B vs empty-room baselines")
        profile_rows = (
            ("A current", "#4bb3fd", profiles_a, int(window["frame_start_a"])),
            ("B current", "#b0e3ff", profiles_b, int(window["frame_start_b"])),
        )
        profile_drawn = False
        for label, color, sources, frame_start in profile_rows:
            current, baseline, _ = _window_profile(sources, frame_start, frame_start + 9)
            if current is not None:
                classic_profile.plot(
                    np.arange(len(current)), current, color=color,
                    linewidth=1.05, label=label,
                )
                profile_drawn = True
            if baseline is not None:
                classic_profile.plot(
                    np.arange(len(baseline)), baseline, color=color,
                    linewidth=0.8, linestyle="--", alpha=0.45,
                    label=label.replace("current", "empty baseline"),
                )
        if profile_drawn:
            classic_profile.legend(
                loc="upper right", fontsize=6, facecolor="#091522",
                edgecolor="#29405a", labelcolor="#d7e2ec",
            )
        else:
            classic_profile.text(
                0.5, 0.5, "Range profiles unavailable",
                transform=classic_profile.transAxes, ha="center", va="center",
                color="#9dafbf", fontsize=9,
            )
        classic_profile.set(xlabel="Range bin", ylabel="TLV magnitude")

        _style_2d(classic_world, "Measured sparse returns · fused world x-y")
        if len(fused):
            histogram, x_edges, y_edges = np.histogram2d(
                fused[:, 0], fused[:, 1], bins=(44, 38),
                range=((-2.2, 2.2), (0, distance_m)),
            )
            classic_world.pcolormesh(
                x_edges, y_edges, np.log1p(histogram.T),
                shading="auto", cmap="magma", vmin=0,
            )
        if len(anomaly_xyz):
            classic_world.scatter(
                anomaly_xyz[:, 0], anomaly_xyz[:, 1], s=32,
                c=anomaly_colors, marker="D", edgecolors="#fff4d6",
                linewidths=0.4,
            )
        classic_world.set(
            xlim=(-2.2, 2.2), ylim=(0, distance_m),
            xlabel="World x (m)", ylabel="World y (m)",
        )

        _style_2d(classic_timeline, "Evidence timeline · fused track")
        timeline = np.arange(index + 1)
        classic_timeline.plot(
            timeline, track_conf[: index + 1], color="#4bb3fd",
            linewidth=1.7, label="Track confidence",
        )
        classic_timeline.plot(
            timeline, classic_scores[: index + 1, 0], color="#7ec8ff",
            linewidth=1.0, alpha=0.75, label="Radar A anomaly",
        )
        classic_timeline.plot(
            timeline, classic_scores[: index + 1, 1], color="#ffc247",
            linewidth=1.0, alpha=0.75, label="Radar B anomaly",
        )
        classic_timeline.plot(
            timeline, classic_scores[: index + 1, 2], color="#ff8a3d",
            linewidth=1.5, label="Either-view evidence",
        )
        classic_timeline.axhline(
            0.68, color="#ff4057", linestyle="--", linewidth=0.8, alpha=0.7,
        )
        classic_timeline.set(
            xlim=(0, max(len(windows) - 1, 1)), ylim=(0, 1.02),
            xlabel="Window", ylabel="Uncalibrated score",
        )
        classic_timeline.legend(
            loc="upper left", fontsize=6, facecolor="#091522",
            edgecolor="#29405a", labelcolor="#d7e2ec",
        )

        max_score = float(classic_scores[index, 2])
        status = "PRESENT" if row["fused_track_present"] else "no observed global track"
        classic_status.set_text(
            f"Window {index + 1}/{len(windows)} · time {row['time_s']:.2f} s · "
            f"A {len(observed_a)} / B {len(observed_b)} track points · "
            f"confidence {row['fused_track_confidence']:.2f} · "
            f"max reflective evidence {max_score:.2f} · {status}"
        )

    import cv2  # noqa: E402 - needed inside render

    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=6500,
                                    extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    writer_cam = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=5000,
                                        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    writer_rad = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=5000,
                                        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    writer_trip = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=12000,
                                         extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    writer_classic = animation.FFMpegWriter(
        fps=fps, codec="libx264", bitrate=6500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    output_main.parent.mkdir(parents=True, exist_ok=True)

    def _draw_triptych_frame(index: int, window: dict[str, Any]) -> None:
        row = window["row"]
        _draw_cameras(trip_cam_a, trip_cam_b, row)
        _draw_cloud(
            trip_radar,
            window,
            show_charts_footer=False,
            person_zoom=True,
            human_focus_style=human_focus,
        )
        present = "PRESENT" if row["fused_track_present"] else "no fused track"
        trip_status.set_text(
            f"Matched window {index + 1}/{n} · radar time {row['time_s']:.2f} s · "
            f"A+B fused points {row['fused_person_points']} · "
            f"track confidence {row['fused_track_confidence']:.2f} · {present}"
        )

    if classic_only:
        with writer_classic.saving(figure_classic, str(output_classic), dpi=dpi):
            for index, window in enumerate(windows):
                _draw_classic_frame(index, window)
                writer_classic.grab_frame()
    elif triptych_only:
        if classic_human_centric:
            with writer_trip.saving(figure_trip, str(output_triptych), dpi=dpi), \
                 writer_classic.saving(figure_classic, str(output_classic), dpi=dpi):
                for index, window in enumerate(windows):
                    _draw_triptych_frame(index, window)
                    _draw_classic_frame(index, window)
                    writer_trip.grab_frame()
                    writer_classic.grab_frame()
        else:
            with writer_trip.saving(figure_trip, str(output_triptych), dpi=dpi):
                for index, window in enumerate(windows):
                    _draw_triptych_frame(index, window)
                    writer_trip.grab_frame()
    else:
        if classic_human_centric:
            with writer.saving(figure, str(output_main), dpi=dpi), \
                 writer_cam.saving(figure_cam, str(output_cameras), dpi=dpi), \
                 writer_rad.saving(figure_rad, str(output_radar), dpi=dpi), \
                 writer_trip.saving(figure_trip, str(output_triptych), dpi=dpi), \
                 writer_classic.saving(figure_classic, str(output_classic), dpi=dpi):
                for index, window in enumerate(windows):
                    row = window["row"]
                    _draw_cameras(axis_cam_a, axis_cam_b, row)
                    _draw_cameras(cam_only_a, cam_only_b, row)
                    temporal_xyz, temporal_weight, dual_candidates, _ = _draw_cloud(
                        axis_3d, window, show_charts_footer=True
                    )
                    _draw_cloud(axis_3d_rad, window, show_charts_footer=False)
                    _draw_triptych_frame(index, window)
                    _draw_classic_frame(index, window)
                    _draw_charts(window, temporal_xyz, temporal_weight)
                    present = "PRESENT" if row["fused_track_present"] else "no fused track"
                    status_text.set_text(
                        f"Window {index + 1}/{n} · time {row['time_s']:.2f} s · "
                        f"unified {row['fused_person_points']} pts · reported points · "
                        f"dual-view reflective {len(dual_candidates)} · "
                        f"track conf {row['fused_track_confidence']:.2f} · {present}"
                    )
                    writer.grab_frame()
                    writer_cam.grab_frame()
                    writer_rad.grab_frame()
                    writer_trip.grab_frame()
                    writer_classic.grab_frame()
        else:
            with writer.saving(figure, str(output_main), dpi=dpi), \
                 writer_cam.saving(figure_cam, str(output_cameras), dpi=dpi), \
                 writer_rad.saving(figure_rad, str(output_radar), dpi=dpi), \
                 writer_trip.saving(figure_trip, str(output_triptych), dpi=dpi):
                for index, window in enumerate(windows):
                    row = window["row"]
                    _draw_cameras(axis_cam_a, axis_cam_b, row)
                    _draw_cameras(cam_only_a, cam_only_b, row)
                    temporal_xyz, temporal_weight, dual_candidates, _ = _draw_cloud(
                        axis_3d, window, show_charts_footer=True
                    )
                    _draw_cloud(axis_3d_rad, window, show_charts_footer=False)
                    _draw_triptych_frame(index, window)
                    _draw_charts(window, temporal_xyz, temporal_weight)
                    present = "PRESENT" if row["fused_track_present"] else "no fused track"
                    status_text.set_text(
                        f"Window {index + 1}/{n} · time {row['time_s']:.2f} s · "
                        f"unified {row['fused_person_points']} pts · reported points · "
                        f"dual-view reflective {len(dual_candidates)} · "
                        f"track conf {row['fused_track_confidence']:.2f} · {present}"
                    )
                    writer.grab_frame()
                    writer_cam.grab_frame()
                    writer_rad.grab_frame()
                    writer_trip.grab_frame()

    if cap_a is not None:
        cap_a.release()
    if cap_b is not None:
        cap_b.release()
    plt.close(figure)
    plt.close(figure_cam)
    plt.close(figure_rad)
    plt.close(figure_trip)
    plt.close(figure_classic)


def render_unified_videos(
    session_a_dir: Path,
    session_b_dir: Path,
    *,
    distance_m: float,
    clock_offset_b_minus_a_s: float,
    window_tolerance_s: float,
    calibration_a: Path | None,
    calibration_b: Path | None,
    camera_a: Path | None,
    camera_b: Path | None,
    camera_a_frames: Path | None,
    camera_b_frames: Path | None,
    output_prefix: Path,
    fps: int = 5,
    dpi: int = 100,
    temporal_history_windows: int = 1,
    temporal_voxel_m: float = 0.12,
    dual_reflectivity_gate_m: float = 0.35,
    human_focus: bool = False,
    triptych_only: bool = False,
    classic_human_centric: bool = False,
    classic_only: bool = False,
    overwrite: bool = False,
) -> list[tuple[Path, Path, dict[str, Any]]]:
    import cv2  # noqa: E402

    if dual_reflectivity_gate_m <= 0:
        raise ValueError("dual_reflectivity_gate_m must be positive")
    if classic_only and not classic_human_centric:
        raise ValueError("classic_only requires classic_human_centric")
    if not classic_only and (camera_a is None or camera_a_frames is None):
        raise ValueError("camera A video and timestamps are required unless classic_only is enabled")
    session_a = load_session(session_a_dir)
    session_b = load_session(session_b_dir)
    source_tracks_a = _load_source_tracks(session_a_dir / "perception.jsonl")
    source_tracks_b = _load_source_tracks(session_b_dir / "perception.jsonl")
    if len(source_tracks_a) != len(session_a.window_index):
        raise ValueError("sensor A track rows do not match perception windows")
    if len(source_tracks_b) != len(session_b.window_index):
        raise ValueError("sensor B track rows do not match perception windows")
    clutter_a = (_load_clutter_explicit(calibration_a) if calibration_a is not None
                 else _load_clutter(session_a_dir))
    clutter_b = (_load_clutter_explicit(calibration_b) if calibration_b is not None
                 else _load_clutter(session_b_dir))
    summary_a_path = session_a_dir / "perception_summary.json"
    summary_b_path = session_b_dir / "perception_summary.json"
    summary_a = json.loads(summary_a_path.read_text()) if summary_a_path.is_file() else {}
    summary_b = json.loads(summary_b_path.read_text()) if summary_b_path.is_file() else {}
    profiles_a = _profile_sources(session_a_dir, summary_a, calibration_a)
    profiles_b = _profile_sources(session_b_dir, summary_b, calibration_b)

    pairs = matched_window_pairs(session_a, session_b, clock_offset_b_minus_a_s, window_tolerance_s)
    if not pairs:
        raise ValueError("no matched window pairs; nothing to render")

    if classic_only:
        camera_a_timestamps = []
        cap_a = None
    else:
        assert camera_a is not None and camera_a_frames is not None
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

    spec = PerceptionSpec()
    fused_track: Any = None
    windows: list[dict[str, Any]] = []
    for i_a, i_b in pairs:
        row, pts_a, pts_b, fused, fused_track = _fuse_window(
            session_a, session_b, i_a, i_b, distance_m, clutter_a, clutter_b,
            15, spec, fused_track,
        )
        radar_center_a_ns = window_center_monotonic_ns(
            session_a, session_a.window_frame_start[i_a]
        )
        radar_center_b_ns = window_center_monotonic_ns(
            session_b, session_b.window_frame_start[i_b]
        )
        row["camera_a_seek_s"] = (
            _camera_seek_s(camera_a_timestamps, radar_center_a_ns)
            if camera_a_timestamps else None
        )
        row["camera_a_alignment_error_ms"] = (
            _camera_alignment_error_ms(camera_a_timestamps, radar_center_a_ns)
            if camera_a_timestamps else None
        )
        row["camera_b_seek_s"] = (
            _camera_seek_s(camera_b_timestamps, radar_center_b_ns)
            if camera_b_timestamps else None
        )
        row["camera_b_alignment_error_ms"] = (
            _camera_alignment_error_ms(
                camera_b_timestamps, radar_center_b_ns
            )
            if camera_b_timestamps else None
        )
        windows.append({
            "row": row, "pts_a": pts_a, "pts_b": pts_b, "fused": fused,
            "frame_start_a": session_a.window_frame_start[i_a],
            "frame_start_b": session_b.window_frame_start[i_b],
            "track_a": source_tracks_a[i_a],
            "track_b": _transform_track_b_to_a(source_tracks_b[i_b], distance_m),
            "anomalies_a": _window_anomalies_at(session_a, i_a),
            "anomalies_b": _window_anomalies_at(session_b, i_b),
        })

    n = len(windows)
    rows = [w["row"] for w in windows]
    time_s = np.asarray([r["time_s"] for r in rows], dtype=np.float32)
    fused_points = np.asarray([r["fused_person_points"] for r in rows], dtype=np.float32)
    track_conf = np.asarray([r["fused_track_confidence"] for r in rows], dtype=np.float32)

    output_main = output_prefix.with_name(output_prefix.name + "_unified.mp4")
    output_cameras = output_prefix.with_name(output_prefix.name + "_cameras_only.mp4")
    output_radar = output_prefix.with_name(output_prefix.name + "_radar_only.mp4")
    output_triptych = output_prefix.with_name(output_prefix.name + "_triptych.mp4")
    output_classic = output_prefix.with_name(
        output_prefix.name + "_classic_fused_dashboard.mp4"
    )
    outputs = [output_classic] if classic_only else (
        [output_triptych, *([output_classic] if classic_human_centric else [])]
        if triptych_only
        else [
            output_main,
            output_cameras,
            output_radar,
            output_triptych,
            *([output_classic] if classic_human_centric else []),
        ]
    )
    if not overwrite and any(path.exists() for path in outputs):
        raise ValueError(f"video output already exists: {[str(p) for p in outputs]}; use --overwrite intentionally")

    _render_figure(
        windows=windows,
        time_s=time_s,
        fused_points=fused_points,
        track_conf=track_conf,
        distance_m=distance_m,
        cap_a=cap_a,
        cap_b=cap_b,
        camera_a_timestamps=camera_a_timestamps,
        camera_b_timestamps=camera_b_timestamps,
        fps=fps,
        dpi=dpi,
        output_main=output_main,
        output_cameras=output_cameras,
        output_radar=output_radar,
        output_triptych=output_triptych,
        output_classic=output_classic,
        profiles_a=profiles_a,
        profiles_b=profiles_b,
        temporal_history_windows=temporal_history_windows,
        temporal_voxel_m=temporal_voxel_m,
        dual_reflectivity_gate_m=dual_reflectivity_gate_m,
        human_focus=human_focus,
        triptych_only=triptych_only,
        classic_human_centric=classic_human_centric,
        classic_only=classic_only,
        overwrite=overwrite,
    )

    results: list[tuple[Path, Path, dict[str, Any]]] = []
    output_layouts = {
        output_main: (
            "engineering_dashboard",
            [int(round(19.2 * dpi)), int(round(10.8 * dpi))],
        ),
        output_cameras: (
            "cameras_only",
            [int(round(19.2 * dpi)), int(round(10.8 * dpi))],
        ),
        output_radar: (
            "fused_radar_only",
            [int(round(19.2 * dpi)), int(round(10.8 * dpi))],
        ),
        output_triptych: (
            "camera_a_fused_radar_camera_b",
            [int(round(19.2 * dpi)), int(round(10.8 * dpi))],
        ),
        output_classic: (
            "classic_human_centric_fused_dashboard",
            [int(round(19.2 * dpi)), int(round(10.8 * dpi))],
        ),
    }
    for output in outputs:
        layout_name, resolution_px = output_layouts[output]
        is_classic = layout_name == "classic_human_centric_fused_dashboard"
        metadata = {
            "schema_version": UNIFIED_VIDEO_SCHEMA,
            "experimental": True,
            "material_confirmed": False,
            "weapon_classification": False,
            "sensor_distance_m": distance_m,
            "clock_offset_b_minus_a_s_applied": clock_offset_b_minus_a_s,
            "matched_window_pairs": n,
            "unified_point_cloud": "A + B transformed into A frame, one scatter",
            "human_focus_display": human_focus,
            "classic_human_centric_display": is_classic,
            "temporal_display": {
                "history_windows": 1 if is_classic else temporal_history_windows,
                "voxel_m": None if is_classic else temporal_voxel_m,
                "source": (
                    "current original track-associated post-CFAR returns only"
                    if is_classic else
                    "original track-associated post-CFAR returns with alpha-only aging"
                    if human_focus else "track-aligned voxelized reported returns"
                ),
                "sensor_density_balanced_for_display": bool(human_focus and not is_classic),
                "does_not_add_physical_resolution": True,
            },
            "layout": layout_name,
            "camera_alignment_error_ms": {
                "a_median": (float(
                    np.median([r["camera_a_alignment_error_ms"] for r in rows])
                ) if camera_a_timestamps else None),
                "a_max_abs": (float(
                    np.max(np.abs([r["camera_a_alignment_error_ms"] for r in rows]))
                ) if camera_a_timestamps else None),
                "b_median": (
                    float(np.median([r["camera_b_alignment_error_ms"] for r in rows]))
                    if camera_b_timestamps else None
                ),
                "b_max_abs": (
                    float(np.max(np.abs([r["camera_b_alignment_error_ms"] for r in rows])))
                    if camera_b_timestamps else None
                ),
            },
            "dual_view_reflective_visualization": {
                "association_gate_m": dual_reflectivity_gate_m,
                "requires_persistent_source_anomaly": True,
                "requires_two_view_spatial_association": True,
                "handheld_demo_torso_core_suppression_m": {
                    "lateral": TORSO_CORE_LATERAL_M,
                    "depth": TORSO_CORE_DEPTH_M,
                    "vertical": TORSO_CORE_VERTICAL_M,
                },
                "off_core_single_view_min_score": OFF_CORE_MIN_SCORE,
                "single_view_off_core_candidate_may_be_transient": True,
                "human_focus_single_view_is_current_observation": bool(human_focus),
                "torso_core_suppression_may_hide_center_carried_objects": True,
                "human_focus_colors": {
                    "yellow": "single-view reflective excess",
                    "orange": "persistent two-view reflective association",
                },
                "classic_colors": {
                    "yellow": "transient source reflective anomaly",
                    "red": "persistent source reflective anomaly",
                },
                "does_not_confirm_material": True,
            },
            "source_session_a": str(session_a_dir),
            "source_session_b": str(session_b_dir),
            "source_camera_a": str(camera_a) if camera_a is not None else None,
            "source_camera_b": str(camera_b) if camera_b is not None else None,
            "output_video": str(output),
            "output_video_sha256": sha256_file(output),
            "duration_s": round(n / fps, 3),
            "resolution_px": resolution_px,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "limitations": [
                "one fused cloud is a descriptive overlay of two post-CFAR TLV pipelines, not coherent ADC fusion",
                "yellow/red markers are uncalibrated reflective anomalies, not confirmed metal",
                "camera frames are host-receipt aligned (no hardware sync with the radar)",
                "scores are uncalibrated engineering evidence rankings",
                "video does not confirm material or classify a firearm",
            ],
        }
        metadata_path = output.with_suffix(output.suffix + ".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        results.append((output, metadata_path, metadata))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified dual-sensor videos, including Camera A | fused radar A+B | Camera B"
    )
    parser.add_argument("--session-a", required=True, type=Path)
    parser.add_argument("--session-b", required=True, type=Path)
    parser.add_argument("--calibration-a", type=Path)
    parser.add_argument("--calibration-b", type=Path)
    parser.add_argument("--distance-m", required=True, type=float)
    parser.add_argument("--clock-offset-b-minus-a-s", type=float, default=0.0)
    parser.add_argument("--window-tolerance-s", type=float, default=0.5)
    parser.add_argument("--camera-a", type=Path)
    parser.add_argument("--camera-b", type=Path)
    parser.add_argument("--camera-a-frames", type=Path)
    parser.add_argument("--camera-b-frames", type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--temporal-history-windows", type=int, default=1)
    parser.add_argument("--temporal-voxel-m", type=float, default=0.12)
    parser.add_argument("--dual-reflectivity-gate-m", type=float, default=0.35)
    parser.add_argument(
        "--human-focus",
        action="store_true",
        help="Use a person-scale display crop with recent temporal returns; display only.",
    )
    parser.add_argument(
        "--only-triptych",
        action="store_true",
        help="Render only Camera A | fused radar A+B | Camera B",
    )
    parser.add_argument(
        "--classic-human-centric",
        action="store_true",
        help=(
            "Also render the classic six-panel fused dashboard using only "
            "measured post-CFAR points."
        ),
    )
    parser.add_argument(
        "--only-classic",
        action="store_true",
        help="Render only the classic fused radar dashboard; no camera files are opened.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    results = render_unified_videos(
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
        output_prefix=args.output_prefix,
        fps=args.fps,
        dpi=args.dpi,
        temporal_history_windows=args.temporal_history_windows,
        temporal_voxel_m=args.temporal_voxel_m,
        dual_reflectivity_gate_m=args.dual_reflectivity_gate_m,
        human_focus=args.human_focus,
        triptych_only=args.only_triptych,
        classic_human_centric=args.classic_human_centric,
        classic_only=args.only_classic,
        overwrite=args.overwrite,
    )
    print(json.dumps([{"output": str(path), "metadata": str(meta)} for path, meta, _ in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
