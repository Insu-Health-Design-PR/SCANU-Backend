#!/usr/bin/env python3
"""Render an experimental AWR1843 sparse RAE cube as an MP4 map."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


VIDEO_SCHEMA_VERSION = "scanu_lab_awr1843_rae_video_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _occupied_cartesian(
    hit_count: np.ndarray,
    ranges: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
) -> dict[str, np.ndarray]:
    occupied = np.argwhere(hit_count > 0)
    if not len(occupied):
        empty = np.empty(0, dtype=np.float32)
        return {key: empty for key in ("x", "y", "z", "elevation", "count")}
    range_m = ranges[occupied[:, 0]]
    azimuth_deg = azimuths[occupied[:, 1]]
    elevation_deg = elevations[occupied[:, 2]]
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    horizontal_m = range_m * np.cos(elevation_rad)
    return {
        "x": (horizontal_m * np.sin(azimuth_rad)).astype(np.float32),
        "y": (horizontal_m * np.cos(azimuth_rad)).astype(np.float32),
        "z": (range_m * np.sin(elevation_rad)).astype(np.float32),
        "elevation": elevation_deg.astype(np.float32),
        "count": hit_count[tuple(occupied.T)].astype(np.float32),
    }


def _style_axis(axis, title: str) -> None:
    axis.set_title(title, loc="left", color="#f4f7fb", fontsize=11)
    axis.set_facecolor("#07111e")
    axis.tick_params(colors="#a9b8c8", labelsize=8)
    axis.grid(True, color="#29405a", alpha=0.45, linewidth=0.6)


def render_cube_video(
    cube_path: Path,
    output_path: Path | None = None,
    *,
    fps: int = 5,
    dpi: int = 100,
    overwrite: bool = False,
) -> tuple[Path, Path, dict]:
    if fps <= 0 or dpi <= 0:
        raise ValueError("fps and dpi must be greater than zero")
    cube_path = cube_path.expanduser().resolve()
    if not cube_path.is_file():
        raise ValueError(f"cube does not exist: {cube_path}")
    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else cube_path.with_name("rae_map.mp4")
    )
    metadata_path = output_path.with_suffix(".metadata.json")
    if not overwrite and (output_path.exists() or metadata_path.exists()):
        raise ValueError(
            f"output already exists: {output_path}; use --overwrite intentionally"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, colors

    with np.load(cube_path) as cube:
        hits = np.asarray(cube["hit_count"])
        ranges = np.asarray(cube["range_centers_m"])
        azimuths = np.asarray(cube["azimuth_centers_deg"])
        elevations = np.asarray(cube["elevation_centers_deg"])
        frame_start = np.asarray(cube["frame_start"])
        frame_end = np.asarray(cube["frame_end"])
    if hits.ndim != 4 or hits.shape[0] == 0 or not np.any(hits):
        raise ValueError("cube must contain occupied [window, range, azimuth, elevation] voxels")

    range_max = float(np.max(ranges))
    x_limit = range_max * math.sin(math.radians(float(np.max(np.abs(azimuths))))) * 1.06
    z_limit = range_max * math.sin(math.radians(float(np.max(np.abs(elevations))))) * 1.12
    norm = colors.Normalize(vmin=float(elevations.min()), vmax=float(elevations.max()))
    cmap = plt.get_cmap("turbo")

    figure = plt.figure(figsize=(12.8, 7.2), facecolor="#050b14")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.42, 1), hspace=0.34, wspace=0.22)
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_top = figure.add_subplot(grid[0, 1])
    axis_side = figure.add_subplot(grid[1, 1])
    figure.suptitle(
        "AWR1843BOOST · mapa 3D disperso",
        color="#f4f7fb",
        fontsize=16,
        x=0.04,
        ha="left",
    )
    status = figure.text(0.04, 0.925, "", color="#9eb1c5", fontsize=9)
    color_axis = figure.add_axes((0.45, 0.075, 0.28, 0.018))
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=color_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Elevation (degrees)", color="#c9d5e2", fontsize=9)
    colorbar.ax.tick_params(colors="#a9b8c8", labelsize=8)

    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=3500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    total_observations = 0
    with writer.saving(figure, str(output_path), dpi=dpi):
        for index, window in enumerate(hits):
            points = _occupied_cartesian(window, ranges, azimuths, elevations)
            observations = int(points["count"].sum())
            total_observations += observations
            sizes = 10 + 18 * np.log1p(points["count"])
            rgba = cmap(norm(points["elevation"]))
            rgba[:, 3] = np.clip(0.42 + 0.12 * np.log1p(points["count"]), 0.42, 0.95)
            for axis in (axis_3d, axis_top, axis_side):
                axis.clear()

            _style_axis(axis_3d, "x / depth / elevation volume")
            axis_3d.scatter(
                points["x"], points["y"], points["z"],
                s=sizes, c=rgba, edgecolors="none", depthshade=False,
            )
            axis_3d.scatter([0], [0], [0], marker="^", s=55, c="#ffffff")
            axis_3d.set(
                xlim=(-x_limit, x_limit), ylim=(0, range_max),
                zlim=(-z_limit, z_limit),
                xlabel="right x (m)", ylabel="depth (m)", zlabel="z height (m)",
            )
            axis_3d.view_init(elev=22, azim=-58)
            for pane in (axis_3d.xaxis.pane, axis_3d.yaxis.pane, axis_3d.zaxis.pane):
                pane.set_facecolor((0.03, 0.07, 0.12, 1))

            _style_axis(axis_top, "Top view · azimuth and depth")
            axis_top.scatter(points["x"], points["y"], s=sizes * 0.7, c=rgba, edgecolors="none")
            axis_top.scatter([0], [0], marker="^", s=45, c="#ffffff")
            axis_top.set(
                xlim=(-x_limit, x_limit), ylim=(0, range_max),
                xlabel="right x (m)", ylabel="depth (m)",
            )
            axis_top.set_aspect("equal", adjustable="box")

            _style_axis(axis_side, "Side view · depth and elevation")
            axis_side.scatter(points["y"], points["z"], s=sizes * 0.7, c=rgba, edgecolors="none")
            axis_side.scatter([0], [0], marker="^", s=45, c="#ffffff")
            axis_side.set(
                xlim=(0, range_max), ylim=(-z_limit, z_limit),
                xlabel="depth (m)", ylabel="z height (m)",
            )
            for axis in (axis_3d, axis_top, axis_side):
                axis.xaxis.label.set_color("#c9d5e2")
                axis.yaxis.label.set_color("#c9d5e2")
            axis_3d.zaxis.label.set_color("#c9d5e2")
            status.set_text(
                f"Ventana {index + 1}/{len(hits)} · frames "
                f"{int(frame_start[index])}–{int(frame_end[index])} · "
                f"{observations} observaciones"
            )
            writer.grab_frame()
    plt.close(figure)

    metadata = {
        "schema_version": VIDEO_SCHEMA_VERSION,
        "experimental": True,
        "canonical_training_compatible": False,
        "source_cube": str(cube_path),
        "source_cube_sha256": _sha256(cube_path),
        "output_video": str(output_path),
        "output_video_sha256": _sha256(output_path),
        "fps": fps,
        "frames": int(len(hits)),
        "duration_s": float(len(hits) / fps),
        "resolution_px": [int(round(12.8 * dpi)), int(round(7.2 * dpi))],
        "total_windowed_point_observations": total_observations,
        "color_encoding": "elevation_deg",
        "size_encoding": "log1p_hit_count",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "sparse processed TLV detections, not a dense raw-ADC radar cube",
            "points repeat across overlapping temporal windows",
            "experimental laboratory evidence, not production detector output",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output_path, metadata_path, metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render an AWR1843 sparse RAE cube as MP4")
    parser.add_argument("--cube", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output, metadata_path, metadata = render_cube_video(
            args.cube, args.output, fps=args.fps, dpi=args.dpi, overwrite=args.overwrite
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps({
        "ok": True, "video": str(output), "metadata": str(metadata_path),
        "frames": metadata["frames"], "fps": metadata["fps"],
        "duration_s": metadata["duration_s"],
        "sha256": metadata["output_video_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
