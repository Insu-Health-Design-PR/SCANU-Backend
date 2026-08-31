#!/usr/bin/env python3
"""Render one AWR1843 TLV stream for a raised-hand reflectivity experiment.

The renderer is deliberately single-radar.  It does not register, merge, or
cross-validate another radar.  Red and yellow are reflectivity evidence
rankings in a raised-hand region of interest; neither color identifies a
material, an object, or a weapon.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lab.mmwave77_usb.artifacts import sha256_file
from lab.mmwave77_usb.perception import (
    _clutter_membership,
    _dbscan_labels,
    _load_clutter,
    _load_frames,
    _point_array,
)
from lab.mmwave77_usb.perception_video import _load_perception


HANDHELD_POINT_CLOUD_VIDEO_SCHEMA = "scanu_lab_awr1843_raised_hand_video_v1"


@dataclass(frozen=True)
class RaisedHandSpec:
    """Conservative, display-only filter for an elevated hand experiment."""

    minimum_elevation_above_track_m: float = 0.35
    lateral_gate_m: float = 1.10
    depth_gate_m: float = 1.00
    minimum_snr_db: float = 18.0
    local_snr_mad_scale: float = 1.0
    cluster_radius_m: float = 0.25
    cluster_min_points: int = 1
    persistence_windows: int = 2

    def validate(self) -> None:
        for name, value in (
            ("minimum_elevation_above_track_m", self.minimum_elevation_above_track_m),
            ("lateral_gate_m", self.lateral_gate_m),
            ("depth_gate_m", self.depth_gate_m),
            ("minimum_snr_db", self.minimum_snr_db),
            ("local_snr_mad_scale", self.local_snr_mad_scale),
            ("cluster_radius_m", self.cluster_radius_m),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.cluster_min_points < 1 or self.persistence_windows < 2:
            raise ValueError("cluster_min_points must be at least 1 and persistence_windows at least 2")


@dataclass
class _CueTrack:
    relative_position: np.ndarray
    hits: int
    last_window: int


def person_and_raised_hand_masks(
    points: np.ndarray,
    track: dict[str, Any] | None,
    spec: RaisedHandSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return broad person envelope and compact elevated-reflector candidates.

    The upper region is intentionally broader than the torso association used
    by the legacy anomaly scorer, so a hand raised above the estimated torso is
    visible to this *visualization*.  It remains a signal cue, not a detector.
    """

    spec.validate()
    person = np.zeros(len(points), dtype=bool)
    candidate = np.zeros(len(points), dtype=bool)
    if track is None or not len(points):
        return person, candidate
    try:
        center = np.asarray(track["position_m"], dtype=np.float32)
        extent = np.asarray(track["observed_extent_m"], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return person, candidate
    if center.shape != (3,) or extent.shape != (3,) or not np.all(np.isfinite(center)):
        return person, candidate

    half = np.maximum(0.5 * extent + np.asarray([0.45, 0.45, 0.45]), [0.55, 0.55, 0.65])
    # Preserve the visible hand/arm trail while keeping remote scene returns gray.
    half[2] = max(float(half[2]), 1.55)
    person = np.all(np.abs(points[:, :3] - center) <= half, axis=1)

    relative = points[:, :3] - center
    elevated = (
        (np.abs(points[:, 0] - center[0]) <= spec.lateral_gate_m)
        & (np.abs(points[:, 1] - center[1]) <= spec.depth_gate_m)
        & (points[:, 2] >= center[2] + spec.minimum_elevation_above_track_m)
    )
    # Do not colour the compact torso core as a hand-held cue.  This is a
    # geometric exclusion only; it does not infer anatomy or material.
    core_half = np.maximum(0.22 * extent + np.asarray([0.08, 0.08, 0.10]), [0.12, 0.12, 0.16])
    outside_core = np.sum((relative / core_half) ** 2, axis=1) > 1.0
    # Use the torso core as the local reference.  The earlier version used the
    # expanded envelope, which includes the raised hand and can therefore set
    # its own threshold high enough to suppress every hand return.
    torso_core = (
        (np.abs(points[:, 0] - center[0]) <= 0.50)
        & (np.abs(points[:, 1] - center[1]) <= 0.50)
        & (np.abs(points[:, 2] - center[2]) <= 0.45)
    )
    local = points[torso_core, 4]
    if len(local) < spec.cluster_min_points:
        return person, candidate
    local_median = float(np.median(local))
    local_mad = max(float(np.median(np.abs(local - local_median))), 1.0)
    threshold = max(spec.minimum_snr_db, local_median + spec.local_snr_mad_scale * 1.4826 * local_mad)
    candidate = elevated & outside_core & (points[:, 4] >= threshold)
    return person, candidate


def _persistent_mask(
    points: np.ndarray,
    candidate: np.ndarray,
    track: dict[str, Any] | None,
    cue_tracks: list[_CueTrack],
    window_index: int,
    spec: RaisedHandSpec,
) -> tuple[np.ndarray, list[_CueTrack]]:
    persistent = np.zeros(len(points), dtype=bool)
    if track is None or np.count_nonzero(candidate) < spec.cluster_min_points:
        return persistent, [item for item in cue_tracks if window_index - item.last_window <= 3]
    center = np.asarray(track["position_m"], dtype=np.float32)
    indexes = np.flatnonzero(candidate)
    labels = _dbscan_labels(points[indexes, :3], radius_m=spec.cluster_radius_m, min_points=spec.cluster_min_points)
    kept = [item for item in cue_tracks if window_index - item.last_window <= 3]
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        cluster_indexes = indexes[labels == label]
        relative = np.median(points[cluster_indexes, :3], axis=0) - center
        matches = [item for item in kept if np.linalg.norm(item.relative_position - relative) <= 0.25]
        if matches:
            cue = min(matches, key=lambda item: np.linalg.norm(item.relative_position - relative))
            cue.relative_position = (0.65 * cue.relative_position + 0.35 * relative).astype(np.float32)
            cue.hits += 1
            cue.last_window = window_index
        else:
            cue = _CueTrack(relative_position=relative.astype(np.float32), hits=1, last_window=window_index)
            kept.append(cue)
        if cue.hits >= spec.persistence_windows:
            persistent[cluster_indexes] = True
    return persistent, kept


def render_handheld_point_cloud_video(
    perception_path: Path,
    *,
    output_path: Path,
    calibration_session: Path | None = None,
    fps: int = 2,
    dpi: int = 200,
    overwrite: bool = False,
    spec: RaisedHandSpec = RaisedHandSpec(),
    minimal: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write a point-cloud-only English video for one independently processed radar."""

    perception_path = perception_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not perception_path.is_file():
        raise ValueError(f"perception output does not exist: {perception_path}")
    if output_path.exists() and not overwrite:
        raise ValueError(f"output already exists: {output_path}; use --overwrite intentionally")
    if fps <= 0 or dpi <= 0:
        raise ValueError("fps and dpi must be greater than zero")
    spec.validate()
    session = perception_path.parent
    frames_path = session / "frames.jsonl"
    summary_path = session / "perception_summary.json"
    if not frames_path.is_file() or not summary_path.is_file():
        raise ValueError("perception session must contain frames.jsonl and perception_summary.json")
    summary = json.loads(summary_path.read_text())
    calibration_path = calibration_session or (
        Path(summary["calibration_session"]) if summary.get("calibration_session") else None
    )
    clutter = _load_clutter(calibration_path) if calibration_path is not None else None
    rows = _load_perception(perception_path)
    frames, _ = _load_frames(frames_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    figure = plt.figure(figsize=(19.2, 10.8), facecolor="#040a12")
    axis = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.88)
    figure.suptitle(
        "AWR1843BOOST · observed 3D radar volume" if minimal else "AWR1843BOOST · independent raised-hand point-cloud evidence",
        x=0.035, ha="left", color="#f3f7fb", fontsize=18,
    )
    status = figure.text(0.035, 0.915, "", color="#a9bfd3", fontsize=10, ha="left")
    figure.text(
        0.035, 0.025,
        "Blue: person-associated returns · Yellow: elevated high-reflectivity cue · Red: persistent elevated cue",
        color="#d4e0ea", fontsize=9,
    )
    figure.text(
        0.965, 0.025,
        "Experimental radar evidence only · colors do not confirm metal, an object, or a weapon",
        color="#ffcc80", fontsize=9, ha="right", weight="bold",
    )
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264", bitrate=6500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    cue_tracks: list[_CueTrack] = []
    # A short trace makes motion legible while retaining the distinction between
    # the current frame (solid blue) and previously reported returns (faint
    # blue).  It is not interpolation, upsampling, or a denser radar product.
    recent_person_returns: deque[np.ndarray] = deque(maxlen=4)
    red_points = 0
    yellow_points = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(figure, str(output_path), dpi=dpi):
        for index, row in enumerate(rows):
            axis.clear()
            points, _ = _point_array(frames, int(row["frame_start"]), int(row["frame_end"]))
            clutter_mask = _clutter_membership(points, clutter)
            visible = ~clutter_mask
            person, candidate = person_and_raised_hand_masks(points, row.get("track"), spec)
            person &= visible
            candidate &= visible
            persistent, cue_tracks = _persistent_mask(points, candidate, row.get("track"), cue_tracks, index, spec)
            persistent &= candidate
            suspect = candidate & ~persistent
            scene = visible & ~person & ~candidate
            red_points += int(np.count_nonzero(persistent))
            yellow_points += int(np.count_nonzero(suspect))

            axis.set_facecolor("#07111e")
            axis.set_title("Single-radar observed point cloud" if minimal else "Single-radar point cloud · no cross-radar fusion", loc="left", color="#f3f7fb", fontsize=12, pad=12)
            if len(points):
                axis.scatter(points[scene, 0], points[scene, 1], points[scene, 2], s=2.0, c="#697b8c", alpha=0.08, edgecolors="none", label="Scene returns")
                if not minimal and recent_person_returns:
                    trail = np.concatenate(tuple(recent_person_returns), axis=0)
                    axis.scatter(trail[:, 0], trail[:, 1], trail[:, 2], s=2.2, c="#5bc0ff", alpha=0.12, edgecolors="none", label="Recent observed person trace")
                axis.scatter(points[person & ~candidate, 0], points[person & ~candidate, 1], points[person & ~candidate, 2], s=5.5, c="#3da9f5", alpha=0.62, edgecolors="none", label="Person-associated returns")
                axis.scatter(points[suspect, 0], points[suspect, 1], points[suspect, 2], s=16, c="#ffd25a", alpha=0.92, edgecolors="#fff2c7", linewidths=0.20, label="Elevated high-reflectivity cue")
                axis.scatter(points[persistent, 0], points[persistent, 1], points[persistent, 2], s=24, c="#ff4258", alpha=0.96, edgecolors="#ffe4e8", linewidths=0.30, label="Persistent elevated cue")
            axis.scatter([0], [0], [0], marker="^", s=90, c="#f5f8fb", label="Radar")
            track = row.get("track")
            history = np.asarray(row.get("track_history_m", []), dtype=np.float32)
            if not minimal and len(history) >= 2 and history.ndim == 2 and history.shape[1] == 3:
                axis.plot(history[:, 0], history[:, 1], history[:, 2], color="#37e0ca", linewidth=1.6, alpha=0.85, label="Track path")
            if track is not None:
                center = np.asarray(track.get("position_m", []), dtype=np.float32)
                if center.shape == (3,) and np.all(np.isfinite(center)):
                    if not minimal:
                        axis.scatter([center[0]], [center[1]], [center[2]], marker="x", s=70, c="#d1fbff", linewidths=1.5, label="Track center")
                    try:
                        extent = np.asarray(track.get("observed_extent_m", []), dtype=np.float32)
                    except (TypeError, ValueError):
                        extent = np.empty(0, dtype=np.float32)
                    if extent.shape == (3,) and np.all(np.isfinite(extent)):
                        x_half = max(1.10, float(extent[0]) * 0.55 + 0.80)
                        y_half = max(1.10, float(extent[1]) * 0.55 + 0.80)
                        z_half = max(1.35, float(extent[2]) * 0.55 + 0.70)
                        axis.set_xlim(float(center[0] - x_half), float(center[0] + x_half))
                        axis.set_ylim(max(0.0, float(center[1] - y_half)), float(center[1] + y_half))
                        axis.set_zlim(float(center[2] - z_half), float(center[2] + z_half))
                    else:
                        axis.set(xlim=(-3.5, 3.5), ylim=(0, 5.0), zlim=(-1.8, 2.4))
                else:
                    axis.set(xlim=(-3.5, 3.5), ylim=(0, 5.0), zlim=(-1.8, 2.4))
            else:
                axis.set(xlim=(-3.5, 3.5), ylim=(0, 5.0), zlim=(-1.8, 2.4))
            axis.set(xlabel="Right / left x (m)", ylabel="Depth y (m)", zlabel="Relative elevation z (m)")
            axis.set_box_aspect((1.0, 1.0, 1.25))
            axis.set_proj_type("ortho")
            axis.view_init(elev=22, azim=-58)
            axis.tick_params(colors="#9aafc2", labelsize=7)
            axis.xaxis.label.set_color("#c9d5e2")
            axis.yaxis.label.set_color("#c9d5e2")
            axis.zaxis.label.set_color("#c9d5e2")
            handles, labels = axis.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            axis.legend(unique.values(), unique.keys(), loc="upper left", fontsize=8, facecolor="#091522", edgecolor="#29405a", labelcolor="#d7e2ec")
            status.set_text(f"Window {index + 1}/{len(rows)} · reported returns {len(points)} · elevated cues yellow {np.count_nonzero(suspect)} / red {np.count_nonzero(persistent)}")
            writer.grab_frame()
            if len(points) and not minimal:
                recent_person_returns.append(points[person & ~candidate, :3].copy())
    plt.close(figure)
    metadata = {
        "schema_version": HANDHELD_POINT_CLOUD_VIDEO_SCHEMA,
        "experimental": True,
        "material_confirmed": False,
        "weapon_classification": False,
        "fusion_used": False,
        "source_perception": str(perception_path),
        "source_perception_sha256": sha256_file(perception_path),
        "source_frames": str(frames_path),
        "source_frames_sha256": sha256_file(frames_path),
        "output_video": str(output_path),
        "output_video_sha256": sha256_file(output_path),
        "windows": len(rows),
        "fps": fps,
        "duration_s": len(rows) / fps,
        "resolution_px": [int(19.2 * dpi), int(10.8 * dpi)],
        "raised_hand_spec": spec.__dict__,
        "minimal_cube_only": minimal,
        "yellow_elevated_cue_points": yellow_points,
        "red_persistent_elevated_cue_points": red_points,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "processed post-CFAR TLVs are sparse radar returns, not dense imagery",
            "red and yellow are single-radar reflectivity cues, not confirmed metal or object identity",
            "the raised-hand region is a display ROI; it is not anatomical localization",
            "this renderer does not use cross-radar fusion or material classification",
        ],
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output_path, metadata_path, metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a single-radar raised-hand point-cloud evidence video")
    parser.add_argument("--perception", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-session", type=Path)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--minimal", action="store_true", help="render only the 3D cube; omit track path and return trail")
    parser.add_argument("--minimum-elevation-above-track-m", type=float, default=RaisedHandSpec.minimum_elevation_above_track_m)
    parser.add_argument("--persistence-windows", type=int, default=RaisedHandSpec.persistence_windows)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output, metadata, result = render_handheld_point_cloud_video(
            args.perception, output_path=args.output, calibration_session=args.calibration_session,
            fps=args.fps, dpi=args.dpi, overwrite=args.overwrite, minimal=args.minimal,
            spec=RaisedHandSpec(
                minimum_elevation_above_track_m=args.minimum_elevation_above_track_m,
                persistence_windows=args.persistence_windows,
            ),
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps({"ok": True, "video": str(output), "metadata": str(metadata), "windows": result["windows"], "material_confirmed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
