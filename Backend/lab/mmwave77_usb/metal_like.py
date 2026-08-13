#!/usr/bin/env python3
"""Score high-reflectivity persistent voxels as experimental metal-like candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


METAL_LIKE_SCHEMA = "scanu_lab_awr1843_metal_like_v1"


@dataclass(frozen=True)
class MetalLikeSpec:
    min_snr_db: float = 12.0
    min_hits: int = 2
    score_threshold: float = 0.68
    snr_midpoint_db: float = 14.0
    snr_scale_db: float = 3.0
    persistence_scale_hits: float = 3.0

    def validate(self) -> None:
        if self.min_hits <= 0:
            raise ValueError("min_hits must be greater than zero")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between zero and one")
        if self.snr_scale_db <= 0 or self.persistence_scale_hits <= 0:
            raise ValueError("score scales must be greater than zero")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def score_metal_like(
    hit_count: np.ndarray,
    snr_mean_db: np.ndarray,
    spec: MetalLikeSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return heuristic score and candidate mask for RAE voxels.

    The score is a reflectivity/persistence ranking, not a material probability.
    """

    spec.validate()
    hits = np.asarray(hit_count, dtype=np.float32)
    snr = np.asarray(snr_mean_db, dtype=np.float32)
    if hits.shape != snr.shape or hits.ndim != 4:
        raise ValueError("hit_count and snr_mean_db must share [window, R, A, E]")
    occupied = hits > 0
    absolute_snr = _sigmoid(
        (snr - spec.snr_midpoint_db) / spec.snr_scale_db
    )
    persistence = 1.0 - np.exp(-hits / spec.persistence_scale_hits)
    contrast = np.full(hits.shape, 0.5, dtype=np.float32)

    for window in range(hits.shape[0]):
        for range_index in range(hits.shape[1]):
            mask = occupied[window, range_index]
            values = snr[window, range_index][mask]
            if len(values) < 4:
                continue
            median = float(np.median(values))
            mad = max(float(np.median(np.abs(values - median))), 1.0)
            contrast[window, range_index] = _sigmoid(
                (snr[window, range_index] - median) / (1.4826 * mad)
            )

    score = (
        0.55 * absolute_snr
        + 0.30 * persistence
        + 0.15 * contrast
    ).astype(np.float32)
    eligible = (
        occupied
        & (hits >= spec.min_hits)
        & (snr >= spec.min_snr_db)
    )
    score = np.where(eligible, score, 0.0).astype(np.float32)
    candidate = score >= spec.score_threshold
    return score, candidate


def build_metal_like_map(
    cube_path: Path,
    output_path: Path | None,
    spec: MetalLikeSpec,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path, dict]:
    cube_path = cube_path.expanduser().resolve()
    if not cube_path.is_file():
        raise ValueError(f"cube does not exist: {cube_path}")
    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else cube_path.with_name("metal_like_map.npz")
    )
    metadata_path = output_path.with_suffix(".metadata.json")
    if not overwrite and (output_path.exists() or metadata_path.exists()):
        raise ValueError(
            f"output already exists: {output_path}; use --overwrite intentionally"
        )

    with np.load(cube_path) as cube:
        hits = np.asarray(cube["hit_count"])
        snr = np.asarray(cube["snr_mean_db"])
        score, candidate = score_metal_like(hits, snr, spec)
        names = (
            "range_edges_m",
            "range_centers_m",
            "azimuth_edges_deg",
            "azimuth_centers_deg",
            "elevation_edges_deg",
            "elevation_centers_deg",
            "frame_start",
            "frame_end",
        )
        arrays = {
            "hit_count": hits,
            "snr_mean_db": snr,
            "metal_like_score": score,
            "candidate_mask": candidate.astype(np.uint8),
            **{name: np.asarray(cube[name]) for name in names},
        }
    np.savez_compressed(output_path, **arrays)
    occupied = hits > 0
    per_window = candidate.reshape(len(candidate), -1).sum(axis=1)
    metadata = {
        "schema_version": METAL_LIKE_SCHEMA,
        "experimental": True,
        "material_confirmed": False,
        "weapon_classification": False,
        "canonical_training_compatible": False,
        "source_cube": str(cube_path),
        "source_cube_sha256": _sha256(cube_path),
        "output_map": str(output_path),
        "output_map_sha256": _sha256(output_path),
        "spec": asdict(spec),
        "statistics": {
            "windows": int(len(hits)),
            "occupied_voxels": int(np.count_nonzero(occupied)),
            "candidate_voxels": int(np.count_nonzero(candidate)),
            "candidate_fraction_of_occupied": (
                float(np.count_nonzero(candidate) / np.count_nonzero(occupied))
                if np.count_nonzero(occupied)
                else 0.0
            ),
            "candidate_voxels_per_window": per_window.astype(int).tolist(),
        },
        "score_definition": {
            "absolute_snr_weight": 0.55,
            "temporal_persistence_weight": 0.30,
            "same_range_robust_contrast_weight": 0.15,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "score is a heuristic reflectivity ranking, not probability of metal",
            "strong nonmetal reflectors can be highlighted and metal can be missed",
            "no weapon class is inferred from radar voxels",
            "requires controlled labeled captures before model training or safety use",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output_path, metadata_path, metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score persistent high-reflectivity AWR1843 voxels as metal-like"
    )
    parser.add_argument("--cube", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-snr-db", type=float, default=12.0)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.68)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    spec = MetalLikeSpec(
        min_snr_db=args.min_snr_db,
        min_hits=args.min_hits,
        score_threshold=args.score_threshold,
    )
    try:
        output, metadata_path, metadata = build_metal_like_map(
            args.cube, args.output, spec, overwrite=args.overwrite
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps({
        "ok": True,
        "map": str(output),
        "metadata": str(metadata_path),
        **metadata["statistics"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
