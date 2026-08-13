"""Auditable signal artifacts for experimental AWR1843 USB captures.

The xWR18xx demo UART exposes more evidence than the sparse point cloud.  This
module preserves the complete range/noise profiles alongside frame identity
and timing without claiming that the profiles are raw ADC data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SIGNAL_ARTIFACT_SCHEMA = "scanu_lab_awr1843_signal_artifacts_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ti_configuration_signature(session: Path) -> str | None:
    """Hash normalized TI CLI commands while excluding variable responses."""

    path = session / "configuration.json"
    if not path.is_file():
        return None
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid TI configuration audit: {path}: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError(f"TI configuration audit must contain a list: {path}")
    commands = [
        " ".join(str(row.get("command", "")).strip().split())
        for row in rows
        if isinstance(row, dict) and str(row.get("command", "")).strip()
    ]
    if not commands:
        return None
    payload = json.dumps(commands, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass
class SignalFrameRecord:
    """One successfully decoded TI frame and its full 1D signal profiles."""

    frame_number: int
    sensor_cycles: int
    host_monotonic_ns: int
    host_utc: str
    range_profile: np.ndarray | None = None
    noise_profile: np.ndarray | None = None
    tlv_types: tuple[int, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)


def _profile_matrix(
    records: list[SignalFrameRecord],
    attribute: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profiles = [
        np.asarray(getattr(record, attribute), dtype=np.float32).reshape(-1)
        if getattr(record, attribute) is not None
        else np.empty(0, dtype=np.float32)
        for record in records
    ]
    lengths = np.asarray([len(profile) for profile in profiles], dtype=np.uint16)
    width = int(lengths.max()) if len(lengths) else 0
    matrix = np.full((len(records), width), np.nan, dtype=np.float32)
    present = lengths > 0
    for index, profile in enumerate(profiles):
        if len(profile):
            matrix[index, : len(profile)] = profile
    return matrix, lengths, present


def _frame_identity(records: list[SignalFrameRecord]) -> dict[str, np.ndarray]:
    return {
        "frame_number": np.asarray(
            [record.frame_number for record in records], dtype=np.uint32
        ),
        "sensor_cycles": np.asarray(
            [record.sensor_cycles for record in records], dtype=np.uint64
        ),
        "host_monotonic_ns": np.asarray(
            [record.host_monotonic_ns for record in records], dtype=np.int64
        ),
        "host_utc": np.asarray(
            [record.host_utc for record in records], dtype=np.str_
        ),
    }


def _sequence_quality(frame_numbers: np.ndarray) -> dict[str, Any]:
    numbers = np.asarray(frame_numbers, dtype=np.int64)
    if not len(numbers):
        return {
            "first_frame": None,
            "last_frame": None,
            "sequence_gaps": 0,
            "missing_frames": 0,
            "duplicate_or_reversed_frames": 0,
            "monotonic": False,
        }
    deltas = np.diff(numbers)
    positive_gaps = deltas[deltas > 1]
    return {
        "first_frame": int(numbers[0]),
        "last_frame": int(numbers[-1]),
        "sequence_gaps": int(len(positive_gaps)),
        "missing_frames": int(np.sum(positive_gaps - 1)),
        "duplicate_or_reversed_frames": int(np.count_nonzero(deltas <= 0)),
        "monotonic": bool(np.all(deltas > 0)) if len(deltas) else True,
    }


def write_signal_artifacts(
    session: Path,
    records: Iterable[SignalFrameRecord],
    *,
    source: str,
    transport_quality: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write complete profile arrays and a capture-quality report."""

    session = session.expanduser().resolve()
    rows = list(records)
    if not rows:
        raise ValueError("no successfully decoded signal records")
    outputs = {
        "range": session / "range_profiles.npz",
        "noise": session / "noise_profiles.npz",
        "range_metadata": session / "range_profiles.metadata.json",
        "noise_metadata": session / "noise_profiles.metadata.json",
        "quality": session / "capture_quality.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            f"signal artifact already exists: {existing[0]}; use --overwrite intentionally"
        )

    identity = _frame_identity(rows)
    range_matrix, range_lengths, range_present = _profile_matrix(
        rows, "range_profile"
    )
    noise_matrix, noise_lengths, noise_present = _profile_matrix(
        rows, "noise_profile"
    )
    raw_path = session / "raw_uart.bin"
    source_hash = sha256_file(raw_path) if raw_path.is_file() else None
    common = {
        **identity,
        "tlv_type_mask": np.asarray(
            [
                sum(1 << int(tlv_type) for tlv_type in record.tlv_types)
                for record in rows
            ],
            dtype=np.uint32,
        ),
    }
    np.savez_compressed(
        outputs["range"],
        profiles=range_matrix,
        profile_lengths=range_lengths,
        profile_present=range_present,
        **common,
    )
    np.savez_compressed(
        outputs["noise"],
        profiles=noise_matrix,
        profile_lengths=noise_lengths,
        profile_present=noise_present,
        **common,
    )

    def profile_metadata(
        kind: str,
        output: Path,
        matrix: np.ndarray,
        lengths: np.ndarray,
        present: np.ndarray,
    ) -> dict[str, Any]:
        nonzero_lengths = lengths[lengths > 0]
        return {
            "schema_version": SIGNAL_ARTIFACT_SCHEMA,
            "experimental": True,
            "canonical_training_compatible": False,
            "source": source,
            "source_raw_uart": str(raw_path) if raw_path.is_file() else None,
            "source_raw_uart_sha256": source_hash,
            "profile_kind": kind,
            "representation": "xwr18xx_demo_tlv_uint16_converted_to_float32",
            "output": str(output),
            "output_sha256": sha256_file(output),
            "frame_count": int(len(rows)),
            "frames_with_profile": int(np.count_nonzero(present)),
            "profile_bins_min": (
                int(nonzero_lengths.min()) if len(nonzero_lengths) else 0
            ),
            "profile_bins_max": int(matrix.shape[1]),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "limitations": [
                "profile is processed xWR18xx demo output, not complex raw ADC",
                "host timestamps describe UART receipt, not hardware acquisition",
            ],
        }

    range_metadata = profile_metadata(
        "range", outputs["range"], range_matrix, range_lengths, range_present
    )
    noise_metadata = profile_metadata(
        "noise", outputs["noise"], noise_matrix, noise_lengths, noise_present
    )
    outputs["range_metadata"].write_text(
        json.dumps(range_metadata, indent=2) + "\n"
    )
    outputs["noise_metadata"].write_text(
        json.dumps(noise_metadata, indent=2) + "\n"
    )

    frame_numbers = identity["frame_number"]
    stats_keys = sorted({key for record in rows for key in record.stats})
    quality = {
        "schema_version": SIGNAL_ARTIFACT_SCHEMA,
        "experimental": True,
        "source": source,
        "source_raw_uart": str(raw_path) if raw_path.is_file() else None,
        "source_raw_uart_sha256": source_hash,
        "decoded_frames": int(len(rows)),
        "sequence": _sequence_quality(frame_numbers),
        "range_profiles": {
            "present_frames": int(np.count_nonzero(range_present)),
            "missing_frames": int(len(rows) - np.count_nonzero(range_present)),
            "consistent_length": bool(
                len(set(int(value) for value in range_lengths if value > 0)) <= 1
            ),
            "lengths": sorted(
                set(int(value) for value in range_lengths if value > 0)
            ),
        },
        "noise_profiles": {
            "present_frames": int(np.count_nonzero(noise_present)),
            "missing_frames": int(len(rows) - np.count_nonzero(noise_present)),
            "consistent_length": bool(
                len(set(int(value) for value in noise_lengths if value > 0)) <= 1
            ),
            "lengths": sorted(
                set(int(value) for value in noise_lengths if value > 0)
            ),
        },
        "stats_fields": stats_keys,
        "transport": dict(transport_quality or {}),
        "outputs": {
            "range_profiles": str(outputs["range"]),
            "noise_profiles": str(outputs["noise"]),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "quality report validates capture integrity, not material classification",
            "USB TLVs contain processed detections and profiles, not raw ADC samples",
        ],
    }
    outputs["quality"].write_text(json.dumps(quality, indent=2) + "\n")
    return {
        "range_profiles": outputs["range"],
        "noise_profiles": outputs["noise"],
        "capture_quality": outputs["quality"],
        "quality": quality,
    }
