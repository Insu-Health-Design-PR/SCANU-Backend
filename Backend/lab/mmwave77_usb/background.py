#!/usr/bin/env python3
"""Build a robust empty-room baseline for AWR1843 processed TLV evidence."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lab.mmwave77_usb.artifacts import sha256_file, ti_configuration_signature


BACKGROUND_SCHEMA = "scanu_lab_awr1843_empty_room_baseline_v1"


@dataclass(frozen=True)
class BackgroundSpec:
    min_frames: int = 40
    near_field_m: float = 0.5
    static_occupancy_threshold: float = 0.75
    minimum_scale: float = 1.0

    def validate(self) -> None:
        if self.min_frames < 2:
            raise ValueError("min_frames must be at least 2")
        if not math.isfinite(self.near_field_m) or self.near_field_m <= 0:
            raise ValueError("near_field_m must be greater than zero")
        if not 0.0 < self.static_occupancy_threshold <= 1.0:
            raise ValueError("static_occupancy_threshold must be in (0, 1]")
        if not math.isfinite(self.minimum_scale) or self.minimum_scale <= 0:
            raise ValueError("minimum_scale must be greater than zero")


def _load_profiles(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"profile artifact does not exist: {path}")
    with np.load(path) as source:
        profiles = np.asarray(source["profiles"], dtype=np.float32)
        lengths = np.asarray(source["profile_lengths"], dtype=np.int64)
        frame_numbers = np.asarray(source["frame_number"], dtype=np.uint32)
    if profiles.ndim != 2 or len(profiles) != len(lengths):
        raise ValueError(f"invalid profile artifact shape: {path}")
    if len(profiles) != len(frame_numbers):
        raise ValueError(f"profile/frame identity mismatch: {path}")
    if not len(profiles):
        raise ValueError(f"profile artifact is empty: {path}")
    return profiles, lengths, frame_numbers


def _dominant_profile_bins(lengths: np.ndarray) -> int:
    valid_lengths = lengths[lengths > 0]
    if not len(valid_lengths):
        raise ValueError("profile artifact has no present profiles")
    values, counts = np.unique(valid_lengths, return_counts=True)
    return int(values[np.argmax(counts)])


def robust_profile_statistics(
    profiles: np.ndarray,
    *,
    minimum_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Return robust per-bin empty-room statistics."""

    values = np.asarray(profiles, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("profiles must be [frame, range_bin] with at least 2 frames")
    median = np.median(values, axis=0).astype(np.float32)
    mad = np.median(np.abs(values - median), axis=0).astype(np.float32)
    p05, p95 = np.percentile(values, (5.0, 95.0), axis=0).astype(np.float32)
    robust_scale = np.maximum.reduce(
        (
            1.4826 * mad,
            0.1 * np.maximum(p95 - p05, 0.0),
            np.full_like(median, float(minimum_scale)),
        )
    ).astype(np.float32)
    return {
        "median": median,
        "mad": mad,
        "p05": p05,
        "p95": p95,
        "robust_scale": robust_scale,
    }


def range_profile_residual(
    current_profiles: np.ndarray,
    baseline_median: np.ndarray,
    baseline_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return residual and robust z-score against an empty-room baseline."""

    current = np.asarray(current_profiles, dtype=np.float32)
    median = np.asarray(baseline_median, dtype=np.float32).reshape(-1)
    scale = np.asarray(baseline_scale, dtype=np.float32).reshape(-1)
    if current.shape[-1] != len(median) or len(scale) != len(median):
        raise ValueError("current profile and baseline bin counts do not match")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("baseline scale must be finite and greater than zero")
    residual = current - median
    return residual.astype(np.float32), (residual / scale).astype(np.float32)


def _rae_baseline(
    cube_path: Path,
    spec: BackgroundSpec,
) -> dict[str, np.ndarray]:
    with np.load(cube_path) as source:
        hits = np.asarray(source["hit_count"], dtype=np.float32)
        snr = np.asarray(source["snr_mean_db"], dtype=np.float32)
        range_centers = np.asarray(source["range_centers_m"], dtype=np.float32)
        azimuth_centers = np.asarray(
            source["azimuth_centers_deg"], dtype=np.float32
        )
        elevation_centers = np.asarray(
            source["elevation_centers_deg"], dtype=np.float32
        )
    if hits.ndim != 4 or hits.shape != snr.shape:
        raise ValueError("cube hit_count and snr_mean_db must share [window,R,A,E]")
    occupied = hits > 0
    occupancy_fraction = occupied.mean(axis=0).astype(np.float32)
    hit_median = np.median(hits, axis=0).astype(np.float32)
    snr_nan = np.where(occupied, snr, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        snr_median = np.nanmedian(snr_nan, axis=0).astype(np.float32)
    snr_median = np.nan_to_num(snr_median, nan=0.0)
    near_range = range_centers < spec.near_field_m
    near_field_mask = np.broadcast_to(
        near_range[:, None, None], occupancy_fraction.shape
    ).copy()
    static_voxel_mask = occupancy_fraction >= spec.static_occupancy_threshold
    clutter_mask = near_field_mask | static_voxel_mask
    range_occupancy_fraction = occupied.any(axis=(2, 3)).mean(axis=0).astype(
        np.float32
    )
    persistent_range_mask = (
        range_occupancy_fraction >= spec.static_occupancy_threshold
    )
    return {
        "occupancy_fraction": occupancy_fraction,
        "hit_median": hit_median,
        "snr_median_db": snr_median,
        "near_field_mask": near_field_mask.astype(np.uint8),
        "static_voxel_mask": static_voxel_mask.astype(np.uint8),
        "clutter_mask": clutter_mask.astype(np.uint8),
        "range_occupancy_fraction": range_occupancy_fraction,
        "persistent_range_mask": persistent_range_mask.astype(np.uint8),
        "range_centers_m": range_centers,
        "azimuth_centers_deg": azimuth_centers,
        "elevation_centers_deg": elevation_centers,
    }


def build_empty_room_baseline(
    session: Path,
    *,
    condition: str,
    spec: BackgroundSpec = BackgroundSpec(),
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    """Create range-profile and optional sparse-RAE empty-room baselines."""

    spec.validate()
    if condition != "empty_room":
        raise ValueError(
            "baseline requires explicit condition='empty_room'; "
            "do not learn background while a participant is present"
        )
    session = session.expanduser().resolve()
    range_path = session / "range_profiles.npz"
    noise_path = session / "noise_profiles.npz"
    range_profiles, range_lengths, range_frame_numbers = _load_profiles(range_path)
    noise_profiles, noise_lengths, noise_frame_numbers = _load_profiles(noise_path)
    range_bins = _dominant_profile_bins(range_lengths)
    noise_bins = _dominant_profile_bins(noise_lengths)
    if range_bins != noise_bins:
        raise ValueError(
            f"range/noise profile bin mismatch: {range_bins} vs {noise_bins}"
        )
    range_by_frame = {
        int(frame_number): profile[:range_bins]
        for frame_number, profile, length in zip(
            range_frame_numbers, range_profiles, range_lengths
        )
        if int(length) == range_bins
        and np.all(np.isfinite(profile[:range_bins]))
    }
    noise_by_frame = {
        int(frame_number): profile[:noise_bins]
        for frame_number, profile, length in zip(
            noise_frame_numbers, noise_profiles, noise_lengths
        )
        if int(length) == noise_bins
        and np.all(np.isfinite(profile[:noise_bins]))
    }
    common_frames = [
        int(frame_number)
        for frame_number in range_frame_numbers
        if int(frame_number) in range_by_frame
        and int(frame_number) in noise_by_frame
    ]
    usable_frames = len(common_frames)
    if usable_frames < spec.min_frames:
        raise ValueError(
            f"empty-room baseline has {usable_frames} complete frames; "
            f"requires at least {spec.min_frames}"
        )
    range_values = np.stack(
        [range_by_frame[frame_number] for frame_number in common_frames]
    )
    noise_values = np.stack(
        [noise_by_frame[frame_number] for frame_number in common_frames]
    )
    range_stats = robust_profile_statistics(
        range_values, minimum_scale=spec.minimum_scale
    )
    noise_stats = robust_profile_statistics(
        noise_values, minimum_scale=spec.minimum_scale
    )
    # The noise TLV is stored as a separate reference rather than silently
    # mixed into the empirical variability denominator.
    profile_baseline_path = session / "empty_room_baseline.npz"
    clutter_path = session / "clutter_mask.npz"
    quality_path = session / "calibration_quality.json"
    manifest_path = session / "calibration_manifest.json"
    outputs = (
        profile_baseline_path,
        clutter_path,
        quality_path,
        manifest_path,
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            f"calibration output already exists: {existing[0]}; "
            "use --overwrite intentionally"
        )

    np.savez_compressed(
        profile_baseline_path,
        range_median=range_stats["median"],
        range_mad=range_stats["mad"],
        range_p05=range_stats["p05"],
        range_p95=range_stats["p95"],
        range_robust_scale=range_stats["robust_scale"],
        noise_median=noise_stats["median"],
        noise_mad=noise_stats["mad"],
        noise_p05=noise_stats["p05"],
        noise_p95=noise_stats["p95"],
        noise_robust_scale=noise_stats["robust_scale"],
        source_frame_number=np.asarray(common_frames, dtype=np.uint32),
    )

    cube_path = session / "rae_cube_tlv.npz"
    rae_available = cube_path.is_file()
    rae_arrays: dict[str, np.ndarray]
    if rae_available:
        rae_arrays = _rae_baseline(cube_path, spec)
    else:
        rae_arrays = {
            "occupancy_fraction": np.empty((0, 0, 0), dtype=np.float32),
            "hit_median": np.empty((0, 0, 0), dtype=np.float32),
            "snr_median_db": np.empty((0, 0, 0), dtype=np.float32),
            "near_field_mask": np.empty((0, 0, 0), dtype=np.uint8),
            "static_voxel_mask": np.empty((0, 0, 0), dtype=np.uint8),
            "clutter_mask": np.empty((0, 0, 0), dtype=np.uint8),
            "range_occupancy_fraction": np.empty(0, dtype=np.float32),
            "persistent_range_mask": np.empty(0, dtype=np.uint8),
            "range_centers_m": np.empty(0, dtype=np.float32),
            "azimuth_centers_deg": np.empty(0, dtype=np.float32),
            "elevation_centers_deg": np.empty(0, dtype=np.float32),
        }
    np.savez_compressed(clutter_path, **rae_arrays)

    capture_quality_path = session / "capture_quality.json"
    transport_quality = (
        json.loads(capture_quality_path.read_text())
        if capture_quality_path.is_file()
        else None
    )
    profile_presence = usable_frames / max(len(range_profiles), 1)
    configuration_signature = ti_configuration_signature(session)
    quality = {
        "schema_version": BACKGROUND_SCHEMA,
        "status": (
            "ready"
            if rae_available and configuration_signature is not None
            else "partial"
        ),
        "condition": condition,
        "usable_profile_frames": usable_frames,
        "required_profile_frames": spec.min_frames,
        "profile_presence_fraction": profile_presence,
        "profile_bins": range_bins,
        "rae_baseline_available": rae_available,
        "transport_quality": transport_quality,
        "checks": {
            "enough_frames": usable_frames >= spec.min_frames,
            "range_noise_bins_match": range_bins == noise_bins,
            "all_range_statistics_finite": bool(
                all(np.all(np.isfinite(values)) for values in range_stats.values())
            ),
            "all_noise_statistics_finite": bool(
                all(np.all(np.isfinite(values)) for values in noise_stats.values())
            ),
            "ti_configuration_signature_available": (
                configuration_signature is not None
            ),
        },
        "limitations": [
            "empty-room calibration removes stable scene response; it does not identify material",
            "sparse RAE occupancy represents post-CFAR detections, not proven empty space",
            "calibration is invalid after sensor or major scene geometry moves",
        ],
    }
    quality_path.write_text(json.dumps(quality, indent=2) + "\n")

    manifest = {
        "schema_version": BACKGROUND_SCHEMA,
        "experimental": True,
        "canonical_training_compatible": False,
        "calibration_id": (
            f"awr1843-empty-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        ),
        "condition": condition,
        "source_session": str(session),
        "source_range_profiles": str(range_path),
        "source_range_profiles_sha256": sha256_file(range_path),
        "source_noise_profiles": str(noise_path),
        "source_noise_profiles_sha256": sha256_file(noise_path),
        "source_cube": str(cube_path) if cube_path.is_file() else None,
        "source_cube_sha256": sha256_file(cube_path) if cube_path.is_file() else None,
        "ti_configuration_signature": configuration_signature,
        "spec": asdict(spec),
        "outputs": {
            "empty_room_baseline": str(profile_baseline_path),
            "empty_room_baseline_sha256": sha256_file(profile_baseline_path),
            "clutter_mask": str(clutter_path),
            "clutter_mask_sha256": sha256_file(clutter_path),
            "calibration_quality": str(quality_path),
            "calibration_quality_sha256": sha256_file(quality_path),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return (
        profile_baseline_path,
        clutter_path,
        quality_path,
        manifest_path,
        manifest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a robust AWR1843 empty-room profile and clutter baseline"
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--condition", required=True, choices=("empty_room",))
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--near-field-m", type=float, default=0.5)
    parser.add_argument("--static-occupancy-threshold", type=float, default=0.75)
    parser.add_argument("--minimum-scale", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    spec = BackgroundSpec(
        min_frames=args.min_frames,
        near_field_m=args.near_field_m,
        static_occupancy_threshold=args.static_occupancy_threshold,
        minimum_scale=args.minimum_scale,
    )
    try:
        baseline, clutter, quality, manifest_path, manifest = (
            build_empty_room_baseline(
                args.session,
                condition=args.condition,
                spec=spec,
                overwrite=args.overwrite,
            )
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "baseline": str(baseline),
                "clutter_mask": str(clutter),
                "quality": str(quality),
                "manifest": str(manifest_path),
                "calibration_id": manifest["calibration_id"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
