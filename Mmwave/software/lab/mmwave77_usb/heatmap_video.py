#!/usr/bin/env python3
"""Render paper-style heatmaps from sparse AWR1843 detected-point TLVs.

Gaussian splatting here is only a visualization operation. It does not
reconstruct a dense complex radar cube or add physical sensor resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


HEATMAP_VIDEO_SCHEMA = "scanu_lab_awr1843_sparse_heatmap_video_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        raise ValueError("sigma_bins must be greater than zero")
    radius = max(1, int(np.ceil(3.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    return kernel / kernel.sum()


def _smooth_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    return np.apply_along_axis(
        lambda row: np.convolve(
            np.pad(row, (radius, radius), mode="constant"),
            kernel,
            mode="valid",
        ),
        axis,
        values,
    )


def _smooth_2d(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    kernel = _gaussian_kernel(sigma_bins)
    smoothed = _smooth_axis(np.asarray(values, dtype=np.float32), kernel, 0)
    return _smooth_axis(smoothed, kernel, 1).astype(np.float32)


def sparse_heatmaps(
    hit_count: np.ndarray,
    snr_mean_db: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    sigma_bins: float = 1.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized range/azimuth and azimuth/elevation maps.

    Inputs use ``[range, azimuth, elevation]`` ordering. Outputs use
    ``[range, azimuth]`` and ``[elevation, azimuth]`` for direct plotting.
    """

    hits = np.asarray(hit_count, dtype=np.float32)
    snr = np.asarray(snr_mean_db, dtype=np.float32)
    candidates = np.asarray(candidate_mask, dtype=bool)
    if hits.shape != snr.shape or hits.shape != candidates.shape or hits.ndim != 3:
        raise ValueError("inputs must share [range, azimuth, elevation]")

    occupied = hits > 0
    weighted = np.log1p(hits) * np.log1p(np.maximum(snr, 0.0)) * occupied
    ra = _smooth_2d(weighted.sum(axis=2), sigma_bins)
    ae = _smooth_2d(weighted.sum(axis=0).T, sigma_bins)
    ra_candidates = _smooth_2d((hits * candidates).sum(axis=2), sigma_bins)
    ae_candidates = _smooth_2d((hits * candidates).sum(axis=0).T, sigma_bins)

    for values in (ra, ae, ra_candidates, ae_candidates):
        peak = (
            float(np.percentile(values[values > 0], 99.0))
            if np.any(values > 0)
            else 0.0
        )
        if peak > 0:
            values /= peak
        np.clip(values, 0.0, 1.0, out=values)
    return ra, ae, ra_candidates, ae_candidates


def render_sparse_heatmap_video(
    map_path: Path,
    output_path: Path | None = None,
    *,
    fps: int = 5,
    dpi: int = 100,
    sigma_bins: float = 1.15,
    overwrite: bool = False,
) -> tuple[Path, Path, dict]:
    map_path = map_path.expanduser().resolve()
    if not map_path.is_file():
        raise ValueError(f"metal-like map does not exist: {map_path}")
    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else map_path.with_name("metal_like_heatmaps_en.mp4")
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
    from matplotlib import animation

    with np.load(map_path) as source:
        hits = np.asarray(source["hit_count"])
        snr = np.asarray(source["snr_mean_db"])
        candidates = np.asarray(source["candidate_mask"], dtype=bool)
        range_edges = np.asarray(source["range_edges_m"])
        azimuth_edges = np.asarray(source["azimuth_edges_deg"])
        elevation_edges = np.asarray(source["elevation_edges_deg"])
        frame_start = np.asarray(source["frame_start"])
        frame_end = np.asarray(source["frame_end"])
    if hits.shape != snr.shape or hits.shape != candidates.shape or hits.ndim != 4:
        raise ValueError("map arrays must share [window, range, azimuth, elevation]")

    figure = plt.figure(figsize=(12.8, 7.2), facecolor="#050b14")
    grid = figure.add_gridspec(1, 2, width_ratios=(2.5, 1), wspace=0.16)
    figure.subplots_adjust(left=0.07, right=0.97, bottom=0.17, top=0.82)
    axis_ra = figure.add_subplot(grid[0, 0])
    axis_ae = figure.add_subplot(grid[0, 1])
    figure.suptitle(
        "AWR1843BOOST · sparse radar heatmaps",
        color="#f4f7fb",
        fontsize=16,
        x=0.07,
        ha="left",
    )
    status = figure.text(0.07, 0.885, "", color="#9eb1c5", fontsize=9)
    figure.text(
        0.07,
        0.045,
        "Gaussian interpolation of detected-point TLVs · not a dense ADC radar cube",
        color="#9eb1c5",
        fontsize=8,
    )
    figure.text(
        0.69,
        0.045,
        "White contour: metal-like heuristic · unverified",
        color="#f4f7fb",
        fontsize=8,
    )
    color_axis = figure.add_axes((0.34, 0.09, 0.34, 0.018))
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=4200,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    with writer.saving(figure, str(output_path), dpi=dpi):
        colorbar = None
        for index in range(len(hits)):
            ra, ae, ra_candidates, ae_candidates = sparse_heatmaps(
                hits[index],
                snr[index],
                candidates[index],
                sigma_bins=sigma_bins,
            )
            for axis in (axis_ra, axis_ae):
                axis.clear()
                axis.set_facecolor("#07111e")
                axis.tick_params(colors="#a9b8c8", labelsize=8)

            image_ra = axis_ra.imshow(
                ra,
                origin="lower",
                aspect="auto",
                interpolation="bilinear",
                cmap="turbo",
                vmin=0.0,
                vmax=1.0,
                extent=(
                    float(azimuth_edges[0]),
                    float(azimuth_edges[-1]),
                    float(range_edges[0]),
                    float(range_edges[-1]),
                ),
            )
            axis_ra.contour(
                np.linspace(azimuth_edges[0], azimuth_edges[-1], ra.shape[1]),
                np.linspace(range_edges[0], range_edges[-1], ra.shape[0]),
                ra_candidates,
                levels=[0.38],
                colors=["#ffffff"],
                linewidths=1.0,
            )
            axis_ra.set(
                title="Range–azimuth intensity",
                xlabel="Azimuth angle (deg)",
                ylabel="Range (m)",
            )
            axis_ae.imshow(
                ae,
                origin="lower",
                aspect="auto",
                interpolation="bilinear",
                cmap="turbo",
                vmin=0.0,
                vmax=1.0,
                extent=(
                    float(azimuth_edges[0]),
                    float(azimuth_edges[-1]),
                    float(elevation_edges[0]),
                    float(elevation_edges[-1]),
                ),
            )
            axis_ae.contour(
                np.linspace(azimuth_edges[0], azimuth_edges[-1], ae.shape[1]),
                np.linspace(elevation_edges[0], elevation_edges[-1], ae.shape[0]),
                ae_candidates,
                levels=[0.38],
                colors=["#ffffff"],
                linewidths=1.0,
            )
            axis_ae.set(
                title="Azimuth–elevation intensity",
                xlabel="Azimuth angle (deg)",
                ylabel="Elevation angle (deg)",
            )
            for axis in (axis_ra, axis_ae):
                axis.title.set_color("#f4f7fb")
                axis.xaxis.label.set_color("#c9d5e2")
                axis.yaxis.label.set_color("#c9d5e2")
            status.set_text(
                f"Window {index + 1}/{len(hits)} · frames "
                f"{int(frame_start[index])}–{int(frame_end[index])}"
            )
            if colorbar is None:
                colorbar = figure.colorbar(
                    image_ra, cax=color_axis, orientation="horizontal"
                )
                colorbar.set_label(
                    "Normalized sparse-return intensity",
                    color="#c9d5e2",
                    fontsize=8,
                )
                colorbar.ax.tick_params(colors="#a9b8c8", labelsize=7)
            writer.grab_frame()
    plt.close(figure)

    metadata = {
        "schema_version": HEATMAP_VIDEO_SCHEMA,
        "experimental": True,
        "dense_adc_cube": False,
        "interpolated_visualization": True,
        "material_confirmed": False,
        "weapon_classification": False,
        "source_map": str(map_path),
        "source_map_sha256": _sha256(map_path),
        "output_video": str(output_path),
        "output_video_sha256": _sha256(output_path),
        "windows": int(len(hits)),
        "fps": int(fps),
        "duration_s": float(len(hits) / fps),
        "resolution_px": [int(12.8 * dpi), int(7.2 * dpi)],
        "gaussian_sigma_bins": float(sigma_bins),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "heatmaps interpolate sparse detected-point TLVs for display",
            "interpolation does not add physical resolution or reconstruct raw ADC",
            "white contours are an unverified reflectivity heuristic",
            "video does not infer metal material or a weapon class",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output_path, metadata_path, metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render paper-style heatmaps from sparse AWR1843 TLVs"
    )
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--sigma-bins", type=float, default=1.15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output, metadata_path, metadata = render_sparse_heatmap_video(
            args.map,
            args.output,
            fps=args.fps,
            dpi=args.dpi,
            sigma_bins=args.sigma_bins,
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
