#!/usr/bin/env python3
"""Render raw AWR1843 points and highlight metal-like candidate voxels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


METAL_VIDEO_SCHEMA = "scanu_lab_awr1843_metal_like_video_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frame_points(frames_path: Path) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = {}
    with frames_path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on {frames_path}:{line_number}: {exc}"
                ) from exc
            if row.get("parse_ok") and isinstance(row.get("points"), list):
                frames[int(row["frame_number"])] = row["points"]
    if not frames:
        raise ValueError(f"no parsed point frames in {frames_path}")
    return frames


def _raw_window_points(
    frames: dict[int, list[dict]],
    frame_start: int,
    frame_end: int,
    range_edges: np.ndarray,
    azimuth_edges: np.ndarray,
    elevation_edges: np.ndarray,
    score_cube: np.ndarray,
    candidate_cube: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map raw Cartesian point occurrences back to scored RAE voxels."""

    rows = [
        point
        for frame_number in range(frame_start, frame_end + 1)
        for point in frames.get(frame_number, [])
    ]
    if not rows:
        empty = np.empty(0, dtype=np.float32)
        return {
            "x": empty,
            "y": empty,
            "z": empty,
            "score": empty,
            "candidate": np.empty(0, dtype=bool),
        }
    values: list[tuple[float, float, float]] = []
    for point in rows:
        try:
            xyz = (float(point["x"]), float(point["y"]), float(point["z"]))
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in xyz):
            values.append(xyz)
    if not values:
        empty = np.empty(0, dtype=np.float32)
        return {
            "x": empty,
            "y": empty,
            "z": empty,
            "score": empty,
            "candidate": np.empty(0, dtype=bool),
        }

    xyz = np.asarray(values, dtype=np.float32)
    x, y, z = xyz.T
    range_m = np.sqrt(x * x + y * y + z * z)
    azimuth = np.degrees(np.arctan2(x, y))
    elevation = np.degrees(np.arctan2(z, np.hypot(x, y)))
    indexes = [
        np.searchsorted(edges, value, side="right") - 1
        for edges, value in (
            (range_edges, range_m),
            (azimuth_edges, azimuth),
            (elevation_edges, elevation),
        )
    ]
    valid = np.ones(len(xyz), dtype=bool)
    for index, edges in zip(indexes, (range_edges, azimuth_edges, elevation_edges)):
        valid &= (index >= 0) & (index < len(edges) - 1)
    x, y, z = x[valid], y[valid], z[valid]
    range_index, azimuth_index, elevation_index = (
        index[valid] for index in indexes
    )
    score = score_cube[range_index, azimuth_index, elevation_index]
    candidate = candidate_cube[range_index, azimuth_index, elevation_index]
    return {
        "x": x,
        "y": y,
        "z": z,
        "score": score.astype(np.float32),
        "candidate": candidate.astype(bool),
    }


def _style(axis, title: str) -> None:
    axis.set_title(title, loc="left", color="#f4f7fb", fontsize=11)
    axis.set_facecolor("#07111e")
    axis.tick_params(colors="#a9b8c8", labelsize=8)
    axis.grid(True, color="#29405a", alpha=0.45, linewidth=0.6)


def render_metal_like_video(
    map_path: Path,
    output_path: Path | None = None,
    *,
    frames_path: Path | None = None,
    fps: int = 5,
    dpi: int = 100,
    overwrite: bool = False,
) -> tuple[Path, Path, dict]:
    map_path = map_path.expanduser().resolve()
    if not map_path.is_file():
        raise ValueError(f"metal-like map does not exist: {map_path}")
    frames_path = (
        frames_path.expanduser().resolve()
        if frames_path is not None
        else map_path.with_name("frames.jsonl")
    )
    if not frames_path.is_file():
        raise ValueError(f"raw decoded frames do not exist: {frames_path}")
    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else map_path.with_name("metal_like_point_cloud_en.mp4")
    )
    metadata_path = output_path.with_suffix(".metadata.json")
    if not overwrite and (output_path.exists() or metadata_path.exists()):
        raise ValueError(
            f"output already exists: {output_path}; use --overwrite intentionally"
        )
    if fps <= 0 or dpi <= 0:
        raise ValueError("fps and dpi must be greater than zero")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, colors

    with np.load(map_path) as source:
        scores = np.asarray(source["metal_like_score"])
        candidates = np.asarray(source["candidate_mask"], dtype=bool)
        range_edges = np.asarray(source["range_edges_m"])
        azimuth_edges = np.asarray(source["azimuth_edges_deg"])
        elevation_edges = np.asarray(source["elevation_edges_deg"])
        frame_start = np.asarray(source["frame_start"])
        frame_end = np.asarray(source["frame_end"])
    if scores.shape != candidates.shape or scores.ndim != 4:
        raise ValueError("score and candidate arrays must share [window, R, A, E]")
    frames = _load_frame_points(frames_path)

    range_max = float(range_edges[-1])
    x_limit = range_max * math.sin(
        math.radians(float(np.abs(azimuth_edges).max()))
    ) * 1.06
    z_limit = range_max * math.sin(
        math.radians(float(np.abs(elevation_edges).max()))
    ) * 1.12
    norm = colors.Normalize(vmin=0.68, vmax=1.0)
    cmap = plt.get_cmap("autumn_r")
    figure = plt.figure(figsize=(12.8, 7.2), facecolor="#050b14")
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.42, 1), hspace=0.34, wspace=0.22
    )
    figure.subplots_adjust(left=0.04, right=0.98, bottom=0.18, top=0.86)
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_top = figure.add_subplot(grid[0, 1])
    axis_side = figure.add_subplot(grid[1, 1])
    figure.suptitle(
        "AWR1843BOOST · 3D environment and reflective candidates",
        color="#f4f7fb",
        fontsize=16,
        x=0.04,
        ha="left",
    )
    status = figure.text(0.04, 0.925, "", color="#9eb1c5", fontsize=9)
    figure.text(0.68, 0.925, "● environment returns", color="#8f9cac", fontsize=9)
    figure.text(
        0.82, 0.925, "◆ metal-like (unverified)", color="#ffb000", fontsize=9
    )
    figure.text(
        0.47,
        0.025,
        "Reflectivity ranking only · not confirmed metal or weapon",
        color="#9eb1c5",
        fontsize=8,
        ha="center",
    )
    color_axis = figure.add_axes((0.46, 0.095, 0.26, 0.016))
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=color_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Heuristic reflectivity score", color="#c9d5e2", fontsize=8)
    colorbar.ax.tick_params(colors="#a9b8c8", labelsize=7)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=4200,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    total_returns = 0
    total_candidate_returns = 0

    with writer.saving(figure, str(output_path), dpi=dpi):
        for index in range(len(scores)):
            points = _raw_window_points(
                frames,
                int(frame_start[index]),
                int(frame_end[index]),
                range_edges,
                azimuth_edges,
                elevation_edges,
                scores[index],
                candidates[index],
            )
            candidate = points["candidate"]
            total_returns += len(candidate)
            total_candidate_returns += int(np.count_nonzero(candidate))
            candidate_colors = cmap(norm(points["score"][candidate]))
            for axis in (axis_3d, axis_top, axis_side):
                axis.clear()

            _style(axis_3d, "Full 3D point cloud")
            axis_3d.scatter(
                points["x"], points["y"], points["z"],
                s=7, c="#8f9cac", alpha=0.24, edgecolors="none",
            )
            axis_3d.scatter(
                points["x"][candidate], points["y"][candidate], points["z"][candidate],
                s=24, c=candidate_colors, alpha=0.88,
                marker="D", edgecolors="#fff3c4", linewidths=0.25,
            )
            axis_3d.scatter([0], [0], [0], marker="^", s=48, c="#ffffff")
            axis_3d.set(
                xlim=(-x_limit, x_limit),
                ylim=(0, range_max),
                zlim=(-z_limit, z_limit),
                xlabel="Right / left x (m)",
                ylabel="Depth (m)",
                zlabel="Elevation z (m)",
            )
            axis_3d.view_init(elev=22, azim=-58)
            for pane in (
                axis_3d.xaxis.pane,
                axis_3d.yaxis.pane,
                axis_3d.zaxis.pane,
            ):
                pane.set_facecolor((0.03, 0.07, 0.12, 1))

            _style(axis_top, "Top view · azimuth and depth")
            axis_top.scatter(
                points["x"], points["y"],
                s=7, c="#8f9cac", alpha=0.24, edgecolors="none",
            )
            axis_top.scatter(
                points["x"][candidate], points["y"][candidate],
                s=24, c=candidate_colors, marker="D",
                alpha=0.88, edgecolors="#fff3c4", linewidths=0.25,
            )
            axis_top.scatter([0], [0], marker="^", s=38, c="#ffffff")
            axis_top.set(
                xlim=(-x_limit, x_limit),
                ylim=(0, range_max),
                xlabel="Right / left x (m)",
                ylabel="Depth (m)",
            )

            _style(axis_side, "Side view · depth and elevation")
            axis_side.scatter(
                points["y"], points["z"],
                s=7, c="#8f9cac", alpha=0.24, edgecolors="none",
            )
            axis_side.scatter(
                points["y"][candidate], points["z"][candidate],
                s=24, c=candidate_colors, marker="D",
                alpha=0.88, edgecolors="#fff3c4", linewidths=0.25,
            )
            axis_side.scatter([0], [0], marker="^", s=38, c="#ffffff")
            axis_side.set(
                xlim=(0, range_max),
                ylim=(-z_limit, z_limit),
                xlabel="Depth (m)",
                ylabel="Elevation z (m)",
            )
            for axis in (axis_3d, axis_top, axis_side):
                axis.xaxis.label.set_color("#c9d5e2")
                axis.yaxis.label.set_color("#c9d5e2")
            axis_3d.zaxis.label.set_color("#c9d5e2")
            status.set_text(
                f"Window {index + 1}/{len(scores)} · frames "
                f"{int(frame_start[index])}–{int(frame_end[index])} · "
                f"{int(np.count_nonzero(candidate))} highlighted of "
                f"{len(candidate)} raw point returns"
            )
            writer.grab_frame()
    plt.close(figure)

    metadata = {
        "schema_version": METAL_VIDEO_SCHEMA,
        "experimental": True,
        "material_confirmed": False,
        "weapon_classification": False,
        "rendering_source": "raw_cartesian_point_occurrences",
        "source_map": str(map_path),
        "source_map_sha256": _sha256(map_path),
        "source_frames": str(frames_path),
        "source_frames_sha256": _sha256(frames_path),
        "output_video": str(output_path),
        "output_video_sha256": _sha256(output_path),
        "frames": int(len(scores)),
        "fps": fps,
        "duration_s": float(len(scores) / fps),
        "resolution_px": [int(round(12.8 * dpi)), int(round(7.2 * dpi))],
        "total_windowed_point_returns": total_returns,
        "total_highlighted_point_returns": total_candidate_returns,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "highlight means persistent high reflectivity, not confirmed metal",
            "raw point occurrences repeat across overlapping temporal windows",
            "video does not infer or display a weapon class",
            "controlled labeled calibration is required before classification",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output_path, metadata_path, metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render raw environment points plus metal-like highlights"
    )
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output, metadata_path, metadata = render_metal_like_video(
            args.map,
            args.output,
            frames_path=args.frames,
            fps=args.fps,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps({
        "ok": True,
        "video": str(output),
        "metadata": str(metadata_path),
        "raw_point_returns": metadata["total_windowed_point_returns"],
        "highlighted_point_returns": metadata["total_highlighted_point_returns"],
        "sha256": metadata["output_video_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
