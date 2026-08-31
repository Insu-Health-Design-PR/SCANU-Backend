#!/usr/bin/env python3
"""Render an operator-readable human-centric AWR1843 perception video."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lab.mmwave77_usb.artifacts import sha256_file
from lab.mmwave77_usb.perception import (
    PERCEPTION_SCHEMA,
    _clutter_membership,
    _load_clutter,
    _load_frames,
    _point_array,
)


PERCEPTION_VIDEO_SCHEMA = "scanu_lab_awr1843_human_perception_video_v1"
MINIMAL_CUBE_VIDEO_SCHEMA = "scanu_lab_awr1843_observed_cube_video_v1"


def _load_perception(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
            if row.get("schema_version") != PERCEPTION_SCHEMA:
                raise ValueError(
                    f"unexpected perception schema on {path}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"perception output is empty: {path}")
    return rows


def _style_2d(axis, title: str) -> None:
    axis.set_facecolor("#07111e")
    axis.set_title(title, loc="left", color="#f3f7fb", fontsize=10, pad=8)
    axis.tick_params(colors="#a9b8c8", labelsize=7)
    axis.grid(True, color="#29405a", alpha=0.38, linewidth=0.55)
    for spine in axis.spines.values():
        spine.set_color("#29405a")


def _body_mask(points: np.ndarray, track: dict[str, Any] | None) -> np.ndarray:
    if track is None or not len(points):
        return np.zeros(len(points), dtype=bool)
    center = np.asarray(track["position_m"], dtype=np.float32)
    extent = np.asarray(track["observed_extent_m"], dtype=np.float32)
    half_extent = np.maximum(
        0.5 * extent + np.asarray([0.25, 0.25, 0.3], dtype=np.float32),
        np.asarray([0.3, 0.3, 0.35], dtype=np.float32),
    )
    return np.all(np.abs(points[:, :3] - center) <= half_extent, axis=1)


def _ellipsoid(center: np.ndarray, extent: np.ndarray) -> tuple[np.ndarray, ...]:
    radii = np.maximum(
        0.5 * np.asarray(extent, dtype=np.float32)
        + np.asarray([0.25, 0.25, 0.3], dtype=np.float32),
        np.asarray([0.3, 0.3, 0.35], dtype=np.float32),
    )
    u = np.linspace(0, 2 * np.pi, 22)
    v = np.linspace(-np.pi / 2, np.pi / 2, 12)
    x = center[0] + radii[0] * np.outer(np.cos(u), np.cos(v))
    y = center[1] + radii[1] * np.outer(np.sin(u), np.cos(v))
    z = center[2] + radii[2] * np.outer(np.ones_like(u), np.sin(v))
    return x, y, z


def _profile_sources(
    session: Path,
    summary: dict[str, Any],
    calibration_session: Path | None = None,
) -> dict[str, Any] | None:
    current_path = session / "range_profiles.npz"
    if not current_path.is_file():
        return None
    with np.load(current_path) as current:
        result = {
            "profiles": np.asarray(current["profiles"], dtype=np.float32),
            "lengths": np.asarray(current["profile_lengths"], dtype=np.int64),
            "frame_numbers": np.asarray(current["frame_number"], dtype=np.uint32),
        }
    calibration_text = (
        str(calibration_session)
        if calibration_session is not None
        else summary.get("calibration_session")
    )
    baseline_path = (
        Path(calibration_text) / "empty_room_baseline.npz"
        if calibration_text
        else None
    )
    if baseline_path is not None and baseline_path.is_file():
        with np.load(baseline_path) as baseline:
            result["baseline_median"] = np.asarray(
                baseline["range_median"], dtype=np.float32
            )
            result["baseline_scale"] = np.asarray(
                baseline["range_robust_scale"], dtype=np.float32
            )
    person_reference_text = summary.get("person_reference_session")
    person_reference_path = (
        Path(person_reference_text) / "range_profiles.npz"
        if person_reference_text
        else None
    )
    if person_reference_path is not None and person_reference_path.is_file():
        with np.load(person_reference_path) as reference:
            values = np.asarray(reference["profiles"], dtype=np.float32)
            lengths = np.asarray(reference["profile_lengths"], dtype=np.int64)
        complete = lengths == values.shape[1]
        if np.any(complete):
            result["person_reference_median"] = np.median(
                values[complete], axis=0
            ).astype(np.float32)
    return result


def _window_profile(
    sources: dict[str, Any] | None,
    frame_start: int,
    frame_end: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if sources is None:
        return None, None, None
    mask = (
        (sources["frame_numbers"] >= frame_start)
        & (sources["frame_numbers"] <= frame_end)
        & (sources["lengths"] == sources["profiles"].shape[1])
    )
    if not np.any(mask):
        return (
            None,
            sources.get("baseline_median"),
            sources.get("person_reference_median"),
        )
    current = np.nanmedian(sources["profiles"][mask], axis=0).astype(np.float32)
    return (
        current,
        sources.get("baseline_median"),
        sources.get("person_reference_median"),
    )


def _display_anomalies(row: dict[str, Any]) -> list[dict[str, Any]]:
    anomalies = list(row.get("anomalies", []))
    if not bool((row.get("quality") or {}).get("person_reference_calibrated")):
        return anomalies
    return [item for item in anomalies if item.get("person_reference_excess")]


def _anomaly_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anomalies = _display_anomalies(row)
    if not anomalies:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=bool),
        )
    return (
        np.asarray([item["center_m"] for item in anomalies], dtype=np.float32),
        np.asarray(
            [item["reflective_anomaly_score"] for item in anomalies],
            dtype=np.float32,
        ),
        np.asarray(
            [item["persistent_body_associated_anomaly"] for item in anomalies],
            dtype=bool,
        ),
    )


def _thorax_anomaly_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep anomaly markers in the tracked torso/chest envelope only."""
    xyz, scores, persistent = _anomaly_arrays(row)
    track = row.get("track")
    if track is None or not len(xyz):
        return xyz, scores, persistent
    center = np.asarray(track.get("position_m", []), dtype=np.float32)
    extent = np.asarray(track.get("observed_extent_m", []), dtype=np.float32)
    if center.shape != (3,) or extent.shape != (3,) or not np.all(np.isfinite(center)):
        return xyz, scores, persistent
    half = np.maximum(0.65 * extent, np.asarray([0.35, 0.35, 0.40], dtype=np.float32))
    relative = np.abs(xyz - center)
    keep = np.all(relative <= half, axis=1)
    # Tight vertical chest band: suppress lower-body/floor returns while
    # retaining the central torso region. This is a display gate, not anatomy.
    keep &= (xyz[:, 2] - center[2] >= -0.35) & (xyz[:, 2] - center[2] <= 0.35)
    return xyz[keep], scores[keep], persistent[keep]


def render_perception_video(
    perception_path: Path,
    *,
    output_path: Path | None = None,
    frames_path: Path | None = None,
    calibration_session: Path | None = None,
    fps: int = 5,
    dpi: int = 100,
    overwrite: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    perception_path = perception_path.expanduser().resolve()
    if not perception_path.is_file():
        raise ValueError(f"perception output does not exist: {perception_path}")
    session = perception_path.parent
    summary_path = perception_path.with_name("perception_summary.json")
    if not summary_path.is_file():
        raise ValueError(f"perception summary does not exist: {summary_path}")
    summary = json.loads(summary_path.read_text())
    frames_path = (
        frames_path.expanduser().resolve()
        if frames_path is not None
        else session / "frames.jsonl"
    )
    if not frames_path.is_file():
        raise ValueError(f"decoded frames do not exist: {frames_path}")
    if fps <= 0 or dpi <= 0:
        raise ValueError("fps and dpi must be greater than zero")
    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else session
        / (
            "human_perception_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_en.mp4"
        )
    )
    metadata_path = output_path.with_suffix(".metadata.json")
    if not overwrite and (output_path.exists() or metadata_path.exists()):
        raise ValueError(
            f"video output already exists: {output_path}; use --overwrite intentionally"
        )

    rows = _load_perception(perception_path)
    frames, _ = _load_frames(frames_path)
    calibration_path = (
        calibration_session.expanduser().resolve()
        if calibration_session is not None
        else (
            Path(summary["calibration_session"])
            if summary.get("calibration_session")
            else None
        )
    )
    clutter = _load_clutter(calibration_path) if calibration_path is not None else None
    profiles = _profile_sources(session, summary, calibration_path)
    track_confidence = np.asarray(
        [
            float(row["track"]["track_confidence"]) if row.get("track") else 0.0
            for row in rows
        ],
        dtype=np.float32,
    )
    anomaly_scores = np.asarray(
        [
            max(
                (
                    float(item["reflective_anomaly_score"])
                    for item in _display_anomalies(row)
                ),
                default=0.0,
            )
            for row in rows
        ],
        dtype=np.float32,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Ellipse, Rectangle

    figure = plt.figure(figsize=(19.2, 10.8), facecolor="#040a12")
    grid = figure.add_gridspec(
        3,
        3,
        width_ratios=(1.3, 1.3, 1.0),
        height_ratios=(1.0, 1.0, 0.78),
        hspace=0.35,
        wspace=0.22,
    )
    figure.subplots_adjust(left=0.04, right=0.98, bottom=0.09, top=0.88)
    axis_3d = figure.add_subplot(grid[0:2, 0:2], projection="3d")
    axis_top = figure.add_subplot(grid[0, 2])
    axis_side = figure.add_subplot(grid[1, 2])
    axis_profile = figure.add_subplot(grid[2, 0])
    axis_ra = figure.add_subplot(grid[2, 1])
    axis_timeline = figure.add_subplot(grid[2, 2])
    figure.suptitle(
        "AWR1843BOOST · human-centric radar perception",
        x=0.04,
        ha="left",
        color="#f3f7fb",
        fontsize=18,
    )
    status_text = figure.text(
        0.04, 0.915, "", color="#a9bfd3", fontsize=10, ha="left"
    )
    figure.text(
        0.04,
        0.027,
        "Experimental radar evidence · Not a confirmed material or weapon classification",
        color="#ffcc80",
        fontsize=10,
        weight="bold",
    )
    figure.text(
        0.64,
        0.027,
        "Sparse post-CFAR TLVs · displayed bins do not add physical resolution",
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
        for index, row in enumerate(rows):
            points, _ = _point_array(
                frames, int(row["frame_start"]), int(row["frame_end"])
            )
            clutter_mask = _clutter_membership(points, clutter)
            track = row.get("track")
            observed_track = (
                track
                if track is not None and track.get("observed_this_window")
                else None
            )
            body_mask = _body_mask(points, observed_track) & ~clutter_mask
            scene_mask = ~body_mask
            anomaly_xyz, anomaly_score, anomaly_persistent = _anomaly_arrays(row)
            anomaly_colors = np.where(
                anomaly_persistent, "#ff4057", "#ffc247"
            )
            for axis in (
                axis_3d,
                axis_top,
                axis_side,
                axis_profile,
                axis_ra,
                axis_timeline,
            ):
                axis.clear()

            axis_3d.set_facecolor("#07111e")
            axis_3d.set_title(
                "Observed radar volume · human track",
                loc="left",
                color="#f3f7fb",
                fontsize=12,
                pad=12,
            )
            if len(points):
                axis_3d.scatter(
                    points[scene_mask, 0],
                    points[scene_mask, 1],
                    points[scene_mask, 2],
                    s=5,
                    c="#6e7f90",
                    alpha=0.12,
                    edgecolors="none",
                    label="Scene returns",
                )
                axis_3d.scatter(
                    points[body_mask, 0],
                    points[body_mask, 1],
                    points[body_mask, 2],
                    s=12,
                    c="#4bb3fd",
                    alpha=0.68,
                    edgecolors="none",
                    label="Track-associated returns",
                )
            axis_3d.scatter(
                [0], [0], [0], marker="^", s=85, c="#f5f8fb", label="Radar"
            )
            if observed_track is not None:
                center = np.asarray(observed_track["position_m"], dtype=np.float32)
                extent = np.asarray(
                    observed_track["observed_extent_m"], dtype=np.float32
                )
                ex, ey, ez = _ellipsoid(center, extent)
                axis_3d.plot_wireframe(
                    ex,
                    ey,
                    ez,
                    rstride=2,
                    cstride=2,
                    color="#59c7ff",
                    alpha=0.42,
                    linewidth=0.6,
                )
                history = np.asarray(row.get("track_history_m", []), dtype=np.float32)
                if len(history) >= 2:
                    axis_3d.plot(
                        history[:, 0],
                        history[:, 1],
                        history[:, 2],
                        color="#38e5c5",
                        linewidth=2.0,
                        alpha=0.85,
                    )
            if len(anomaly_xyz):
                axis_3d.scatter(
                    anomaly_xyz[:, 0],
                    anomaly_xyz[:, 1],
                    anomaly_xyz[:, 2],
                    s=90 + 90 * anomaly_score,
                    c=anomaly_colors,
                    marker="D",
                    edgecolors="#fff4d6",
                    linewidths=0.8,
                    label="Reflective anomaly",
                )
            plane_x, plane_y = np.meshgrid(
                np.linspace(-3.5, 3.5, 2), np.linspace(0, 5.0, 2)
            )
            axis_3d.plot_surface(
                plane_x,
                plane_y,
                np.zeros_like(plane_x),
                color="#284158",
                alpha=0.08,
                shade=False,
            )
            axis_3d.set(
                xlim=(-3.5, 3.5),
                ylim=(0, 5.0),
                zlim=(-1.8, 1.8),
                xlabel="Right / left x (m)",
                ylabel="Depth y (m)",
                zlabel="Relative elevation z (m)",
            )
            axis_3d.view_init(elev=23, azim=-58)
            axis_3d.tick_params(colors="#9aafc2", labelsize=7)
            axis_3d.xaxis.label.set_color("#c9d5e2")
            axis_3d.yaxis.label.set_color("#c9d5e2")
            axis_3d.zaxis.label.set_color("#c9d5e2")
            axis_3d.legend(
                loc="upper left",
                fontsize=7,
                facecolor="#091522",
                edgecolor="#29405a",
                labelcolor="#d7e2ec",
            )

            _style_2d(axis_top, "Top view · person location")
            if len(points):
                axis_top.scatter(
                    points[scene_mask, 0],
                    points[scene_mask, 1],
                    s=4,
                    c="#718191",
                    alpha=0.12,
                )
                axis_top.scatter(
                    points[body_mask, 0],
                    points[body_mask, 1],
                    s=12,
                    c="#4bb3fd",
                    alpha=0.7,
                )
            axis_top.scatter([0], [0], marker="^", s=45, c="#ffffff")
            if observed_track is not None:
                center = np.asarray(observed_track["position_m"])
                extent = np.asarray(observed_track["observed_extent_m"])
                axis_top.add_patch(
                    Ellipse(
                        (center[0], center[1]),
                        width=max(float(extent[0]) + 0.5, 0.6),
                        height=max(float(extent[1]) + 0.5, 0.6),
                        facecolor="#2ea8ff",
                        edgecolor="#8ad8ff",
                        alpha=0.18,
                        linewidth=1.4,
                    )
                )
                axis_top.text(
                    center[0],
                    center[1],
                    f" TRACK {observed_track['track_id']}",
                    color="#c8efff",
                    fontsize=7,
                    weight="bold",
                )
            if len(anomaly_xyz):
                axis_top.scatter(
                    anomaly_xyz[:, 0],
                    anomaly_xyz[:, 1],
                    s=45 + 50 * anomaly_score,
                    c=anomaly_colors,
                    marker="D",
                    edgecolors="#fff4d6",
                    linewidths=0.5,
                )
            axis_top.set(
                xlim=(-3.5, 3.5),
                ylim=(0, 5.0),
                xlabel="Right / left x (m)",
                ylabel="Depth y (m)",
            )

            _style_2d(axis_side, "Side view · observed vertical extent")
            if len(points):
                axis_side.scatter(
                    points[scene_mask, 1],
                    points[scene_mask, 2],
                    s=4,
                    c="#718191",
                    alpha=0.12,
                )
                axis_side.scatter(
                    points[body_mask, 1],
                    points[body_mask, 2],
                    s=12,
                    c="#4bb3fd",
                    alpha=0.7,
                )
            if observed_track is not None:
                center = np.asarray(observed_track["position_m"])
                extent = np.asarray(observed_track["observed_extent_m"])
                axis_side.add_patch(
                    Rectangle(
                        (
                            center[1] - max(float(extent[1]) + 0.5, 0.6) / 2,
                            center[2] - max(float(extent[2]) + 0.6, 0.7) / 2,
                        ),
                        max(float(extent[1]) + 0.5, 0.6),
                        max(float(extent[2]) + 0.6, 0.7),
                        facecolor="#2ea8ff",
                        edgecolor="#8ad8ff",
                        alpha=0.18,
                        linewidth=1.4,
                    )
                )
            if len(anomaly_xyz):
                axis_side.scatter(
                    anomaly_xyz[:, 1],
                    anomaly_xyz[:, 2],
                    s=45 + 50 * anomaly_score,
                    c=anomaly_colors,
                    marker="D",
                    edgecolors="#fff4d6",
                    linewidths=0.5,
                )
            axis_side.set(
                xlim=(0, 5.0),
                ylim=(-1.8, 1.8),
                xlabel="Depth y (m)",
                ylabel="Relative elevation z (m)",
            )

            current_profile, baseline_profile, person_reference_profile = _window_profile(
                profiles, int(row["frame_start"]), int(row["frame_end"])
            )
            _style_2d(
                axis_profile,
                (
                    "Range profile · current vs empty-room and person references"
                    if person_reference_profile is not None
                    else "Range profile · current vs empty-room baseline"
                    if baseline_profile is not None
                    else "Range profile · external baseline unavailable"
                ),
            )
            if current_profile is not None:
                bins = np.arange(len(current_profile))
                axis_profile.plot(
                    bins,
                    current_profile,
                    color="#4bb3fd",
                    linewidth=1.1,
                    label="Current",
                )
                if baseline_profile is not None:
                    axis_profile.plot(
                        bins,
                        baseline_profile,
                        color="#8b98a5",
                        linewidth=1.0,
                        alpha=0.85,
                        label="Empty-room median",
                    )
                if person_reference_profile is not None:
                    axis_profile.plot(
                        bins,
                        person_reference_profile,
                        color="#38e5c5",
                        linewidth=1.0,
                        alpha=0.9,
                        label="Matched person-only median",
                    )
                axis_profile.legend(
                    loc="upper right",
                    fontsize=7,
                    facecolor="#091522",
                    edgecolor="#29405a",
                    labelcolor="#d7e2ec",
                )
                axis_profile.set_xlim(0, len(current_profile) - 1)
            else:
                axis_profile.text(
                    0.5,
                    0.5,
                    "Full range profiles unavailable",
                    transform=axis_profile.transAxes,
                    ha="center",
                    va="center",
                    color="#9dafbf",
                    fontsize=9,
                )
            axis_profile.set(xlabel="Range bin", ylabel="TLV magnitude")

            _style_2d(axis_ra, "Measured sparse bins · range–azimuth")
            if len(points):
                point_range = np.linalg.norm(points[:, :3], axis=1)
                point_azimuth = np.degrees(np.arctan2(points[:, 0], points[:, 1]))
                histogram, az_edges, range_edges = np.histogram2d(
                    point_azimuth,
                    point_range,
                    bins=(40, 32),
                    range=((-50, 50), (0, 5.0)),
                )
                axis_ra.pcolormesh(
                    az_edges,
                    range_edges,
                    np.log1p(histogram.T),
                    shading="auto",
                    cmap="magma",
                    vmin=0,
                )
            if len(anomaly_xyz):
                anomaly_range = np.linalg.norm(anomaly_xyz, axis=1)
                anomaly_azimuth = np.degrees(
                    np.arctan2(anomaly_xyz[:, 0], anomaly_xyz[:, 1])
                )
                axis_ra.scatter(
                    anomaly_azimuth,
                    anomaly_range,
                    s=30,
                    c=anomaly_colors,
                    marker="D",
                    edgecolors="#fff4d6",
                    linewidths=0.4,
                )
            axis_ra.set(
                xlim=(-50, 50),
                ylim=(0, 5.0),
                xlabel="Azimuth (deg)",
                ylabel="Range (m)",
            )

            _style_2d(axis_timeline, "Evidence timeline")
            timeline_x = np.arange(index + 1)
            axis_timeline.plot(
                timeline_x,
                track_confidence[: index + 1],
                color="#4bb3fd",
                linewidth=1.7,
                label="Track confidence",
            )
            axis_timeline.plot(
                timeline_x,
                anomaly_scores[: index + 1],
                color="#ffc247",
                linewidth=1.5,
                label="Max anomaly score",
            )
            axis_timeline.axhline(
                0.68, color="#ff4057", linestyle="--", linewidth=0.8, alpha=0.7
            )
            axis_timeline.set(
                xlim=(0, max(len(rows) - 1, 1)),
                ylim=(0, 1.02),
                xlabel="Window",
                ylabel="Uncalibrated score",
            )
            axis_timeline.legend(
                loc="upper left",
                fontsize=7,
                facecolor="#091522",
                edgecolor="#29405a",
                labelcolor="#d7e2ec",
            )

            track_text = (
                f"Track {observed_track['track_id']} · confidence "
                f"{observed_track['track_confidence']:.2f} · "
                f"range {np.linalg.norm(observed_track['position_m']):.2f} m"
                if observed_track is not None
                else "No observed human return"
            )
            anomaly_text = (
                f"max anomaly {float(anomaly_score.max()):.2f}"
                if len(anomaly_score)
                else "no body-associated anomaly"
            )
            status_text.set_text(
                f"Window {index + 1}/{len(rows)} · frames "
                f"{row['frame_start']}–{row['frame_end']} · {track_text} · "
                f"{anomaly_text} · state: {row['screening_state']}"
            )
            writer.grab_frame()
    plt.close(figure)

    metadata = {
        "schema_version": PERCEPTION_VIDEO_SCHEMA,
        "experimental": True,
        "material_confirmed": False,
        "weapon_classification": False,
        "source_perception": str(perception_path),
        "source_perception_sha256": sha256_file(perception_path),
        "source_frames": str(frames_path),
        "source_frames_sha256": sha256_file(frames_path),
        "calibration_session_used": (
            str(calibration_path) if calibration_path is not None else None
        ),
        "person_reference_session_used": summary.get("person_reference_session"),
        "person_reference_calibrated": bool(
            summary.get("person_reference_calibrated")
        ),
        "output_video": str(output_path),
        "output_video_sha256": sha256_file(output_path),
        "windows": len(rows),
        "fps": fps,
        "duration_s": len(rows) / fps,
        "resolution_px": [int(round(19.2 * dpi)), int(round(10.8 * dpi))],
        "interpolated_heatmap": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "body volume is an observed sparse-radar track, not anatomical reconstruction",
            "scores are uncalibrated engineering evidence rankings",
            "matched person-reference subtraction requires repeatable position and motion",
            "range-azimuth panel bins measured point TLVs without Gaussian interpolation",
            "video does not confirm material or classify a firearm",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output_path, metadata_path, metadata


def render_observed_cube_video(
    perception_path: Path,
    *,
    output_path: Path,
    frames_path: Path | None = None,
    calibration_session: Path | None = None,
    fps: int = 5,
    dpi: int = 100,
    overwrite: bool = False,
    include_peaks: bool = True,
    show_reflectivity_cues: bool = True,
    emphasize_chest_cues: bool = False,
    suppress_cues_first_s: float = 0.0,
    hold_chest_cue_s: float = 0.0,
) -> tuple[Path, Path, dict[str, Any]]:
    """Render the established perception cloud with only its 3D cube.

    This is intentionally a layout change, not a new point-cloud algorithm:
    it uses the same track-associated body mask, clutter membership, anomaly
    rows, axes, and observed ellipsoid as :func:`render_perception_video`.
    """

    perception_path = perception_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not perception_path.is_file():
        raise ValueError(f"perception output does not exist: {perception_path}")
    session = perception_path.parent
    summary_path = session / "perception_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"perception summary does not exist: {summary_path}")
    if output_path.exists() and not overwrite:
        raise ValueError(f"output already exists: {output_path}; use --overwrite intentionally")
    if fps <= 0 or dpi <= 0 or suppress_cues_first_s < 0 or hold_chest_cue_s < 0:
        raise ValueError("fps and dpi must be greater than zero")
    frames_path = frames_path.expanduser().resolve() if frames_path is not None else session / "frames.jsonl"
    if not frames_path.is_file():
        raise ValueError(f"decoded frames do not exist: {frames_path}")
    summary = json.loads(summary_path.read_text())
    calibration_path = calibration_session.expanduser().resolve() if calibration_session is not None else (
        Path(summary["calibration_session"]) if summary.get("calibration_session") else None
    )
    rows = _load_perception(perception_path)
    frames, _ = _load_frames(frames_path)
    clutter = _load_clutter(calibration_path) if calibration_path is not None else None
    profile_sources = _profile_sources(session, summary, calibration_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    figure = plt.figure(figsize=(19.2, 10.8), facecolor="#040a12")
    grid = figure.add_gridspec(4, 1, height_ratios=(3.4, 0.9, 0.01, 0.01), hspace=0.10) if include_peaks else None
    axis = figure.add_subplot(grid[0], projection="3d") if grid is not None else figure.add_subplot(111, projection="3d")
    peaks_axis = figure.add_subplot(grid[1]) if grid is not None else None
    figure.subplots_adjust(left=0.05, right=0.97, bottom=0.08, top=0.88)
    figure.suptitle("AWR1843BOOST · observed radar volume", x=0.035, ha="left", color="#f3f7fb", fontsize=18)
    status = figure.text(0.035, 0.915, "", color="#a9bfd3", fontsize=10, ha="left")
    figure.text(
        0.035, 0.026,
        "Blue: track-associated observed returns · Yellow/red: reflective anomaly evidence",
        color="#d4e0ea", fontsize=9,
    )
    figure.text(
        0.965, 0.026,
        "Experimental radar evidence · not a confirmed material or weapon classification",
        color="#ffcc80", fontsize=9, ha="right", weight="bold",
    )
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264", bitrate=6500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_chest_relative: np.ndarray | None = None
    last_chest_age_s = float("inf")
    last_display_track: dict[str, Any] | None = None
    last_track_age_s = float("inf")
    last_body_relative: np.ndarray | None = None
    last_body_age_s = float("inf")
    with writer.saving(figure, str(output_path), dpi=dpi):
        for index, row in enumerate(rows):
            axis.clear()
            if peaks_axis is not None:
                peaks_axis.clear()
            points, _ = _point_array(frames, int(row["frame_start"]), int(row["frame_end"]))
            clutter_mask = _clutter_membership(points, clutter)
            track = row.get("track")
            observed_track = track if track is not None and track.get("observed_this_window") else None
            if observed_track is not None:
                last_display_track = observed_track
                last_track_age_s = 0.0
            elif track is not None and last_display_track is not None and last_track_age_s < 3.0:
                # Use the predicted track for a short display-only coast. The
                # points remain current TLV returns; only the association is held.
                last_display_track = track
            elif last_display_track is not None and last_track_age_s < 3.0:
                pass
            else:
                last_display_track = None
            display_track = last_display_track
            last_track_age_s += 1.0 / fps
            body_mask = _body_mask(points, display_track) & ~clutter_mask
            display_body_points = points[body_mask]
            if display_track is not None and len(display_body_points):
                center = np.asarray(display_track["position_m"], dtype=np.float32)
                last_body_relative = display_body_points[:, :3] - center
                last_body_age_s = 0.0
            elif (
                display_track is not None
                and last_body_relative is not None
                and last_body_age_s < 3.0
            ):
                # Display-only continuity for short association gaps. These
                # are held prior returns, never new radar measurements.
                center = np.asarray(display_track["position_m"], dtype=np.float32)
                display_body_points = center + last_body_relative
            else:
                display_body_points = np.empty((0, 3), dtype=np.float32)
            last_body_age_s += 1.0 / fps
            scene_mask = ~body_mask
            if (index / fps) < suppress_cues_first_s:
                anomaly_xyz = np.empty((0, 3), dtype=np.float32)
                anomaly_score = np.empty(0, dtype=np.float32)
                anomaly_persistent = np.empty(0, dtype=bool)
            else:
                anomaly_xyz, anomaly_score, anomaly_persistent = _thorax_anomaly_arrays(row)
                if len(anomaly_xyz) and display_track is not None:
                    track_center = np.asarray(display_track["position_m"], dtype=np.float32)
                    last_chest_relative = np.median(anomaly_xyz, axis=0) - track_center
                    last_chest_age_s = 0.0
                elif (
                    not len(anomaly_xyz)
                    and display_track is not None
                    and last_chest_relative is not None
                    and last_chest_age_s < hold_chest_cue_s
                ):
                    track_center = np.asarray(display_track["position_m"], dtype=np.float32)
                    anomaly_xyz = (track_center + last_chest_relative).reshape(1, 3)
                    anomaly_score = np.asarray([0.5], dtype=np.float32)
                    anomaly_persistent = np.asarray([True], dtype=bool)
                last_chest_age_s += 1.0 / fps
            anomaly_colors = (
                np.full(len(anomaly_xyz), "#ffd25a", dtype=object)
                if emphasize_chest_cues
                else np.where(anomaly_persistent, "#ff4057", "#ffc247")
            )

            axis.set_facecolor("#07111e")
            axis.set_title("Observed radar volume · human track", loc="left", color="#f3f7fb", fontsize=12, pad=12)
            if len(points):
                axis.scatter(points[scene_mask, 0], points[scene_mask, 1], points[scene_mask, 2], s=5, c="#6e7f90", alpha=0.12, edgecolors="none", label="Scene returns")
                axis.scatter(points[body_mask, 0], points[body_mask, 1], points[body_mask, 2], s=12, c="#4bb3fd", alpha=0.68, edgecolors="none", label="Track-associated returns")
            if len(display_body_points) and not len(points[body_mask]):
                axis.scatter(
                    display_body_points[:, 0], display_body_points[:, 1], display_body_points[:, 2],
                    s=12, c="#70c9ff", alpha=0.42, edgecolors="none",
                    label="Held track points · visual continuity",
                )
            axis.scatter([0], [0], [0], marker="^", s=85, c="#f5f8fb", label="Radar")
            if display_track is not None:
                center = np.asarray(display_track["position_m"], dtype=np.float32)
                extent = np.asarray(display_track["observed_extent_m"], dtype=np.float32)
                ex, ey, ez = _ellipsoid(center, extent)
                axis.plot_wireframe(ex, ey, ez, rstride=2, cstride=2, color="#59c7ff", alpha=0.42, linewidth=0.6)
            if show_reflectivity_cues and len(anomaly_xyz):
                marker_size = (150 + 150 * anomaly_score) if emphasize_chest_cues else (90 + 90 * anomaly_score)
                axis.scatter(anomaly_xyz[:, 0], anomaly_xyz[:, 1], anomaly_xyz[:, 2], s=marker_size, c=anomaly_colors, marker="D", edgecolors="#fff4d6", linewidths=0.8, label="Chest reflectivity cue")
            plane_x, plane_y = np.meshgrid(np.linspace(-3.5, 3.5, 2), np.linspace(0, 5.0, 2))
            axis.plot_surface(plane_x, plane_y, np.zeros_like(plane_x), color="#284158", alpha=0.08, shade=False)
            axis.set(xlim=(-3.5, 3.5), ylim=(0, 5.0), zlim=(-1.8, 1.8), xlabel="Right / left x (m)", ylabel="Depth y (m)", zlabel="Relative elevation z (m)")
            axis.view_init(elev=23, azim=-58)
            axis.tick_params(colors="#9aafc2", labelsize=7)
            axis.xaxis.label.set_color("#c9d5e2")
            axis.yaxis.label.set_color("#c9d5e2")
            axis.zaxis.label.set_color("#c9d5e2")
            handles, labels = axis.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            axis.legend(unique.values(), unique.keys(), loc="upper left", fontsize=8, facecolor="#091522", edgecolor="#29405a", labelcolor="#d7e2ec")
            current_profile, baseline_profile, _ = _window_profile(
                profile_sources,
                int(row["frame_start"]), int(row["frame_end"]),
            )
            if peaks_axis is not None:
                peaks_axis.set_facecolor("#07111e")
                peaks_axis.set_title("Reflectivity peaks · current minus empty-room baseline", loc="left", color="#f3f7fb", fontsize=10, pad=5)
            if peaks_axis is not None and current_profile is not None:
                bins = np.arange(len(current_profile))
                reference = baseline_profile if baseline_profile is not None and len(baseline_profile) == len(current_profile) else np.zeros_like(current_profile)
                residual = current_profile - reference
                peaks_axis.plot(bins, residual, color="#4bb3fd", linewidth=1.0, label="Residual magnitude")
                if len(residual) >= 3:
                    local = np.flatnonzero((residual[1:-1] >= residual[:-2]) & (residual[1:-1] >= residual[2:])) + 1
                    if len(local):
                        top = local[np.argsort(residual[local])[-5:]]
                        peaks_axis.scatter(top, residual[top], s=20, c="#ffbf45", edgecolors="#fff4d6", linewidths=0.4, label="Peak")
                peaks_axis.axhline(0.0, color="#62778b", linewidth=0.6)
                peaks_axis.set_xlim(0, len(residual) - 1)
                peaks_axis.legend(loc="upper right", fontsize=7, facecolor="#091522", edgecolor="#29405a", labelcolor="#d7e2ec")
            elif peaks_axis is not None:
                peaks_axis.text(0.5, 0.5, "Range profile unavailable", transform=peaks_axis.transAxes, ha="center", va="center", color="#9dafbf")
            if peaks_axis is not None:
                peaks_axis.set_xlabel("Range bin", color="#c9d5e2")
                peaks_axis.set_ylabel("Residual magnitude", color="#c9d5e2")
                peaks_axis.tick_params(colors="#9aafc2", labelsize=7)
                peaks_axis.grid(True, color="#29405a", alpha=0.32, linewidth=0.5)
            state = str(row.get("screening_state", "background"))
            status.set_text(f"Window {index + 1}/{len(rows)} · frames {row['frame_start']}–{row['frame_end']} · state: {state}")
            writer.grab_frame()
    plt.close(figure)
    metadata = {
        "schema_version": MINIMAL_CUBE_VIDEO_SCHEMA,
        "experimental": True,
        "material_confirmed": False,
        "weapon_classification": False,
        "source_perception": str(perception_path),
        "source_perception_sha256": sha256_file(perception_path),
        "source_frames": str(frames_path),
        "source_frames_sha256": sha256_file(frames_path),
        "calibration_session_used": str(calibration_path) if calibration_path is not None else None,
        "output_video": str(output_path),
        "output_video_sha256": sha256_file(output_path),
        "windows": len(rows), "fps": fps, "duration_s": len(rows) / fps,
        "resolution_px": [int(round(19.2 * dpi)), int(round(10.8 * dpi))],
        "layout": "3d_cube_plus_reflection_peaks_no_top_side_or_trajectory" if include_peaks else "3d_cube_only_no_top_side_or_trajectory",
        "reflectivity_cues_shown": show_reflectivity_cues,
        "chest_cues_emphasized": emphasize_chest_cues,
        "cues_suppressed_first_s": suppress_cues_first_s,
        "chest_cue_hold_s": hold_chest_cue_s,
        "held_cues_are_visual_continuity_only": hold_chest_cue_s > 0,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "uses processed sparse post-CFAR TLVs, not dense imagery or raw ADC",
            "body volume is an observed track, not anatomical reconstruction",
            "reflective anomaly markers do not confirm material, object, or firearm identity",
        ],
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output_path, metadata_path, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a 1080p human-centric AWR1843 perception video"
    )
    parser.add_argument("--perception", required=True, type=Path)
    parser.add_argument("--frames", type=Path)
    parser.add_argument(
        "--calibration-session",
        type=Path,
        help="local calibration path when a copied session was relocated",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--minimal-cube-only", action="store_true", help="render the established point cloud as a 3D cube only")
    parser.add_argument("--cube-without-graph", action="store_true", help="render only the established 3D cube")
    parser.add_argument("--hide-reflectivity-cues", action="store_true", help="omit yellow/red anomaly markers from the cube")
    parser.add_argument("--emphasize-chest-cues", action="store_true", help="make existing thorax cues larger and yellow")
    parser.add_argument("--suppress-cues-first-s", type=float, default=0.0)
    parser.add_argument("--hold-chest-cue-s", type=float, default=0.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        renderer = render_observed_cube_video if (args.minimal_cube_only or args.cube_without_graph) else render_perception_video
        render_kwargs = {
            "output_path": args.output, "frames_path": args.frames,
            "calibration_session": args.calibration_session, "fps": args.fps,
            "dpi": args.dpi, "overwrite": args.overwrite,
        }
        if renderer is render_observed_cube_video:
            render_kwargs["include_peaks"] = not args.cube_without_graph
            render_kwargs["show_reflectivity_cues"] = not args.hide_reflectivity_cues
            render_kwargs["emphasize_chest_cues"] = args.emphasize_chest_cues
            render_kwargs["suppress_cues_first_s"] = args.suppress_cues_first_s
            render_kwargs["hold_chest_cue_s"] = args.hold_chest_cue_s
        output, metadata_path, metadata = renderer(args.perception, **render_kwargs)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "video": str(output),
                "metadata": str(metadata_path),
                "windows": metadata["windows"],
                "duration_s": metadata["duration_s"],
                "sha256": metadata["output_video_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
