#!/usr/bin/env python3
"""Build an experimental sparse range-azimuth-elevation cube from TI TLVs.

This module bins already-detected Cartesian points emitted by the xWR18xx
out-of-box firmware.  It does not reconstruct a dense complex radar cube and
must not be treated as raw-ADC or canonical SCAN-U training evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


CUBE_SCHEMA_VERSION = "scanu_lab_awr1843_sparse_rae_v1"


@dataclass(frozen=True)
class CubeSpec:
    range_min_m: float = 0.25
    range_max_m: float = 8.64
    azimuth_min_deg: float = -50.0
    azimuth_max_deg: float = 50.0
    elevation_min_deg: float = -20.0
    elevation_max_deg: float = 20.0
    range_bins: int = 64
    azimuth_bins: int = 48
    elevation_bins: int = 24
    window_frames: int = 10
    stride_frames: int = 5

    def validate(self) -> None:
        bounds = (
            ("range", self.range_min_m, self.range_max_m),
            ("azimuth", self.azimuth_min_deg, self.azimuth_max_deg),
            ("elevation", self.elevation_min_deg, self.elevation_max_deg),
        )
        for name, lower, upper in bounds:
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"invalid {name} bounds: {lower}, {upper}")
        for name, value in (
            ("range_bins", self.range_bins),
            ("azimuth_bins", self.azimuth_bins),
            ("elevation_bins", self.elevation_bins),
            ("window_frames", self.window_frames),
            ("stride_frames", self.stride_frames),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be greater than zero")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frames(path: Path) -> tuple[list[dict], int]:
    frames: list[dict] = []
    rejected_rows = 0
    with path.open() as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict) or not row.get("parse_ok", False):
                rejected_rows += 1
                continue
            points = row.get("points")
            if not isinstance(points, list):
                rejected_rows += 1
                continue
            frames.append(row)
    if not frames:
        raise ValueError(f"no successfully parsed TLV frames in {path}")
    return frames, rejected_rows


def _window_starts(frame_count: int, window_frames: int, stride_frames: int) -> list[int]:
    if frame_count <= window_frames:
        return [0]
    starts = list(range(0, frame_count - window_frames + 1, stride_frames))
    last = frame_count - window_frames
    if starts[-1] != last:
        starts.append(last)
    return starts


def _edges(spec: CubeSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.linspace(spec.range_min_m, spec.range_max_m, spec.range_bins + 1),
        np.linspace(
            spec.azimuth_min_deg, spec.azimuth_max_deg, spec.azimuth_bins + 1
        ),
        np.linspace(
            spec.elevation_min_deg,
            spec.elevation_max_deg,
            spec.elevation_bins + 1,
        ),
    )


def _centers(edges: np.ndarray) -> np.ndarray:
    return ((edges[:-1] + edges[1:]) * 0.5).astype(np.float32)


def build_sparse_cube(
    frames: Sequence[dict],
    spec: CubeSpec,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Accumulate detected points into temporal range/azimuth/elevation voxels."""

    spec.validate()
    starts = _window_starts(len(frames), spec.window_frames, spec.stride_frames)
    shape = (
        len(starts),
        spec.range_bins,
        spec.azimuth_bins,
        spec.elevation_bins,
    )
    hit_count_accumulator = np.zeros(shape, dtype=np.uint32)
    snr_sum = np.zeros(shape, dtype=np.float32)
    doppler_sum = np.zeros(shape, dtype=np.float32)
    doppler_abs_max = np.zeros(shape, dtype=np.float32)
    range_edges, azimuth_edges, elevation_edges = _edges(spec)
    accepted_points = 0
    dropped_nonfinite = 0
    dropped_out_of_bounds = 0
    clipped_hit_counts = 0
    frame_start: list[int] = []
    frame_end: list[int] = []

    for window_index, start in enumerate(starts):
        stop = min(start + spec.window_frames, len(frames))
        window = frames[start:stop]
        frame_start.append(int(window[0].get("frame_number", start)))
        frame_end.append(int(window[-1].get("frame_number", stop - 1)))

        for frame in window:
            for point in frame["points"]:
                try:
                    x = float(point["x"])
                    y = float(point["y"])
                    z = float(point["z"])
                    doppler = float(point.get("doppler", 0.0))
                    snr = float(point.get("snr", 0.0))
                except (KeyError, TypeError, ValueError):
                    dropped_nonfinite += 1
                    continue
                if not all(math.isfinite(value) for value in (x, y, z, doppler, snr)):
                    dropped_nonfinite += 1
                    continue

                range_m = math.sqrt(x * x + y * y + z * z)
                azimuth_deg = math.degrees(math.atan2(x, y))
                elevation_deg = math.degrees(math.atan2(z, math.hypot(x, y)))
                values = (range_m, azimuth_deg, elevation_deg)
                edges = (range_edges, azimuth_edges, elevation_edges)
                indexes: list[int] = []
                outside = False
                for value, axis_edges in zip(values, edges):
                    if value < axis_edges[0] or value > axis_edges[-1]:
                        outside = True
                        break
                    index = int(np.searchsorted(axis_edges, value, side="right") - 1)
                    indexes.append(min(index, len(axis_edges) - 2))
                if outside:
                    dropped_out_of_bounds += 1
                    continue

                voxel = (window_index, indexes[0], indexes[1], indexes[2])
                hit_count_accumulator[voxel] += 1
                snr_sum[voxel] += snr
                doppler_sum[voxel] += doppler
                doppler_abs_max[voxel] = max(
                    doppler_abs_max[voxel], abs(doppler)
                )
                accepted_points += 1

    clipped_hit_counts = int(
        np.maximum(
            hit_count_accumulator.astype(np.int64) - np.iinfo(np.uint16).max,
            0,
        ).sum()
    )
    hit_count = np.minimum(
        hit_count_accumulator, np.iinfo(np.uint16).max
    ).astype(np.uint16)
    nonempty = hit_count_accumulator > 0
    snr_mean = np.zeros(shape, dtype=np.float32)
    doppler_mean = np.zeros(shape, dtype=np.float32)
    np.divide(snr_sum, hit_count_accumulator, out=snr_mean, where=nonempty)
    np.divide(doppler_sum, hit_count_accumulator, out=doppler_mean, where=nonempty)

    arrays = {
        "hit_count": hit_count,
        "snr_mean_db": snr_mean,
        "doppler_mean_mps": doppler_mean,
        "doppler_abs_max_mps": doppler_abs_max,
        "range_edges_m": range_edges.astype(np.float32),
        "range_centers_m": _centers(range_edges),
        "azimuth_edges_deg": azimuth_edges.astype(np.float32),
        "azimuth_centers_deg": _centers(azimuth_edges),
        "elevation_edges_deg": elevation_edges.astype(np.float32),
        "elevation_centers_deg": _centers(elevation_edges),
        "frame_start": np.asarray(frame_start, dtype=np.uint32),
        "frame_end": np.asarray(frame_end, dtype=np.uint32),
    }
    stats = {
        "input_frames": len(frames),
        "windows": len(starts),
        "accepted_point_observations": accepted_points,
        "dropped_nonfinite_points": dropped_nonfinite,
        "dropped_out_of_bounds_points": dropped_out_of_bounds,
        "clipped_hit_counts": clipped_hit_counts,
        "nonempty_voxels": int(np.count_nonzero(nonempty)),
    }
    return arrays, stats


def build_session_cube(
    session: Path,
    output: Path | None,
    spec: CubeSpec,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path, dict]:
    session = session.expanduser().resolve()
    frames_path = session / "frames.jsonl"
    if not frames_path.is_file():
        raise ValueError(f"session has no frames.jsonl: {session}")
    output_path = (
        output.expanduser().resolve()
        if output is not None
        else session / "rae_cube_tlv.npz"
    )
    metadata_path = output_path.with_suffix(".metadata.json")
    if not overwrite and (output_path.exists() or metadata_path.exists()):
        raise ValueError(
            f"output already exists: {output_path}; use --overwrite intentionally"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames, rejected_rows = _load_frames(frames_path)
    arrays, stats = build_sparse_cube(frames, spec)
    np.savez_compressed(output_path, **arrays)
    metadata = {
        "schema_version": CUBE_SCHEMA_VERSION,
        "experimental": True,
        "representation": "sparse_tlv_point_accumulation",
        "dense_adc_cube": False,
        "canonical_training_compatible": False,
        "source_session": str(session),
        "source_frames": str(frames_path),
        "source_frames_sha256": _sha256(frames_path),
        "output_npz": str(output_path),
        "output_npz_sha256": _sha256(output_path),
        "axis_order": ["window", "range", "azimuth", "elevation"],
        "coordinate_convention": {
            "x": "right_m",
            "y": "forward_m",
            "z": "up_m",
            "azimuth": "atan2(x,y)_deg",
            "elevation": "atan2(z,hypot(x,y))_deg",
        },
        "spec": spec.__dict__,
        "statistics": {**stats, "rejected_frame_rows": rejected_rows},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "voxels contain accumulated detected points, not dense complex ADC samples",
            "empty voxels mean no reported detection, not proven empty physical space",
            "angular accuracy and resolution are limited by firmware, antenna geometry, calibration, multipath, and SNR",
            "experimental laboratory artifact; not compatible with canonical SCAN-U training",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output_path, metadata_path, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an experimental sparse range/azimuth/elevation cube from AWR1843 TLVs"
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--range-min-m", type=float, default=0.25)
    parser.add_argument("--range-max-m", type=float, default=8.64)
    parser.add_argument("--azimuth-min-deg", type=float, default=-50.0)
    parser.add_argument("--azimuth-max-deg", type=float, default=50.0)
    parser.add_argument("--elevation-min-deg", type=float, default=-20.0)
    parser.add_argument("--elevation-max-deg", type=float, default=20.0)
    parser.add_argument("--range-bins", type=int, default=64)
    parser.add_argument("--azimuth-bins", type=int, default=48)
    parser.add_argument("--elevation-bins", type=int, default=24)
    parser.add_argument("--window-frames", type=int, default=10)
    parser.add_argument("--stride-frames", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    spec = CubeSpec(
        range_min_m=args.range_min_m,
        range_max_m=args.range_max_m,
        azimuth_min_deg=args.azimuth_min_deg,
        azimuth_max_deg=args.azimuth_max_deg,
        elevation_min_deg=args.elevation_min_deg,
        elevation_max_deg=args.elevation_max_deg,
        range_bins=args.range_bins,
        azimuth_bins=args.azimuth_bins,
        elevation_bins=args.elevation_bins,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
    )
    try:
        output_path, metadata_path, metadata = build_session_cube(
            args.session, args.output, spec, overwrite=args.overwrite
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "cube": str(output_path),
                "metadata": str(metadata_path),
                **metadata["statistics"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
