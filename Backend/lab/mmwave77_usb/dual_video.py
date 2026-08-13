#!/usr/bin/env python3
"""Render two uncalibrated AWR1843 sparse cubes in one comparison video."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .video import _occupied_cartesian


DUAL_VIDEO_SCHEMA = "scanu_lab_awr1843_dual_overlay_video_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _style(axis, title: str) -> None:
    axis.set_title(title, loc="left", color="#f4f7fb", fontsize=10)
    axis.set_facecolor("#07111e")
    axis.tick_params(colors="#a9b8c8", labelsize=8)
    axis.grid(True, color="#29405a", alpha=0.45, linewidth=0.6)


def render_dual_video(
    overlay_path: Path,
    output_path: Path | None = None,
    *,
    fps: int = 5,
    dpi: int = 100,
    overwrite: bool = False,
) -> tuple[Path, Path, dict]:
    if fps <= 0 or dpi <= 0:
        raise ValueError("fps and dpi must be greater than zero")
    overlay_path = overlay_path.expanduser().resolve()
    if not overlay_path.is_file():
        raise ValueError(f"dual overlay does not exist: {overlay_path}")
    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else overlay_path.with_name("dual_rae_map.mp4")
    )
    metadata_path = output_path.with_suffix(".metadata.json")
    if not overwrite and (output_path.exists() or metadata_path.exists()):
        raise ValueError(
            f"output already exists: {output_path}; use --overwrite intentionally"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    with np.load(overlay_path) as overlay:
        hits_a = np.asarray(overlay["hit_count_a"])
        hits_b = np.asarray(overlay["hit_count_b"])
        ranges = np.asarray(overlay["range_centers_m"])
        azimuths = np.asarray(overlay["azimuth_centers_deg"])
        elevations = np.asarray(overlay["elevation_centers_deg"])
        frames_a = np.asarray(overlay["frame_start_a"])
        frames_b = np.asarray(overlay["frame_start_b"])
    if hits_a.shape != hits_b.shape or hits_a.ndim != 4:
        raise ValueError("dual hit-count arrays must share [window, range, azimuth, elevation]")

    range_max = float(ranges.max())
    x_limit = range_max * math.sin(math.radians(float(np.abs(azimuths).max()))) * 1.06
    z_limit = range_max * math.sin(math.radians(float(np.abs(elevations).max()))) * 1.12
    color_a = "#25d6e6"
    color_b = "#ff9a3d"
    figure = plt.figure(figsize=(12.8, 7.2), facecolor="#050b14")
    grid = figure.add_gridspec(2, 2, hspace=0.34, wspace=0.25)
    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    axis_top = figure.add_subplot(grid[0, 1])
    axis_side_a = figure.add_subplot(grid[1, 0])
    axis_side_b = figure.add_subplot(grid[1, 1])
    figure.suptitle(
        "Dos AWR1843BOOST · comparación 3D paralela",
        color="#f4f7fb",
        fontsize=16,
        x=0.04,
        ha="left",
    )
    status = figure.text(0.04, 0.925, "", color="#9eb1c5", fontsize=9)
    figure.text(0.70, 0.925, "● Sensor A", color=color_a, fontsize=9)
    figure.text(0.78, 0.925, "● Sensor B", color=color_b, fontsize=9)
    figure.text(
        0.87, 0.925, "sin calibración", color="#c9d5e2", fontsize=9
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=3800,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    with writer.saving(figure, str(output_path), dpi=dpi):
        for index in range(len(hits_a)):
            points_a = _occupied_cartesian(
                hits_a[index], ranges, azimuths, elevations
            )
            points_b = _occupied_cartesian(
                hits_b[index], ranges, azimuths, elevations
            )
            size_a = 8 + 14 * np.log1p(points_a["count"])
            size_b = 8 + 14 * np.log1p(points_b["count"])
            for axis in (axis_3d, axis_top, axis_side_a, axis_side_b):
                axis.clear()

            _style(axis_3d, "Superposición 3D provisional")
            axis_3d.scatter(
                points_a["x"], points_a["y"], points_a["z"],
                s=size_a, c=color_a, alpha=0.62, edgecolors="none", label="A",
            )
            axis_3d.scatter(
                points_b["x"], points_b["y"], points_b["z"],
                s=size_b, c=color_b, alpha=0.62, edgecolors="none", label="B",
            )
            axis_3d.scatter([0], [0], [0], marker="^", s=45, c="#ffffff")
            axis_3d.set(
                xlim=(-x_limit, x_limit), ylim=(0, range_max),
                zlim=(-z_limit, z_limit),
                xlabel="x (m)", ylabel="profundidad (m)", zlabel="z (m)",
            )
            axis_3d.view_init(elev=21, azim=-58)
            for pane in (axis_3d.xaxis.pane, axis_3d.yaxis.pane, axis_3d.zaxis.pane):
                pane.set_facecolor((0.03, 0.07, 0.12, 1))

            _style(axis_top, "Vista superior superpuesta")
            axis_top.scatter(
                points_a["x"], points_a["y"], s=size_a, c=color_a,
                alpha=0.62, edgecolors="none",
            )
            axis_top.scatter(
                points_b["x"], points_b["y"], s=size_b, c=color_b,
                alpha=0.62, edgecolors="none",
            )
            axis_top.scatter([0], [0], marker="^", s=40, c="#ffffff")
            axis_top.set(
                xlim=(-x_limit, x_limit), ylim=(0, range_max),
                xlabel="x derecha (m)", ylabel="profundidad (m)",
            )

            for axis, points, sizes, color, label in (
                (axis_side_a, points_a, size_a, color_a, "Sensor A · lateral"),
                (axis_side_b, points_b, size_b, color_b, "Sensor B · lateral"),
            ):
                _style(axis, label)
                axis.scatter(
                    points["y"], points["z"], s=sizes, c=color,
                    alpha=0.66, edgecolors="none",
                )
                axis.scatter([0], [0], marker="^", s=40, c="#ffffff")
                axis.set(
                    xlim=(0, range_max), ylim=(-z_limit, z_limit),
                    xlabel="profundidad (m)", ylabel="altura z (m)",
                )
            for axis in (axis_3d, axis_top, axis_side_a, axis_side_b):
                axis.xaxis.label.set_color("#c9d5e2")
                axis.yaxis.label.set_color("#c9d5e2")
            axis_3d.zaxis.label.set_color("#c9d5e2")
            status.set_text(
                f"Ventana {index + 1}/{len(hits_a)} · "
                f"A frame {int(frames_a[index])}, {int(points_a['count'].sum())} obs · "
                f"B frame {int(frames_b[index])}, {int(points_b['count'].sum())} obs"
            )
            writer.grab_frame()
    plt.close(figure)

    metadata = {
        "schema_version": DUAL_VIDEO_SCHEMA,
        "experimental": True,
        "extrinsic_calibrated": False,
        "fusion_mode": "identity_overlay_comparison_only",
        "source_overlay": str(overlay_path),
        "source_overlay_sha256": _sha256(overlay_path),
        "output_video": str(output_path),
        "output_video_sha256": _sha256(output_path),
        "frames": int(len(hits_a)),
        "fps": fps,
        "duration_s": float(len(hits_a) / fps),
        "resolution_px": [int(round(12.8 * dpi)), int(round(7.2 * dpi))],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "sensor origins are overlaid because baseline and extrinsics are not measured",
            "comparison is not coherent aperture processing or calibrated metric fusion",
            "processed sparse TLV detections, not raw ADC",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output_path, metadata_path, metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a dual AWR1843 comparison MP4")
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output, metadata_path, metadata = render_dual_video(
            args.overlay,
            args.output,
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
        "sha256": metadata["output_video_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
