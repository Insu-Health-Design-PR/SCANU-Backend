#!/usr/bin/env python3
"""Rebuild AWR1843 TLV artifacts from a preserved ``raw_uart.bin`` file."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from lab.mmwave77_usb.artifacts import sha256_file, write_signal_artifacts
from lab.mmwave77_usb.runner import TiTlvFramer, _decode_frame
from layer1_sensor_hub.radar.radar_constants import FRAME_HEADER_SIZE
from layer1_sensor_hub.radar.tlv_parser import TLVParser
from layer1_sensor_hub.radar.uart_source import FrameHeader


REPROCESS_SCHEMA = "scanu_lab_awr1843_uart_reprocess_v1"


def _original_timestamps(frames_path: Path) -> dict[int, tuple[int, str]]:
    timestamps: dict[int, tuple[int, str]] = {}
    if not frames_path.is_file():
        return timestamps
    with frames_path.open() as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on {frames_path}:{line_number}: {exc}"
                ) from exc
            try:
                frame_number = int(row["frame_number"])
                monotonic_ns = int(row["host_monotonic_ns"])
                host_utc = str(row["host_utc"])
            except (KeyError, TypeError, ValueError):
                continue
            timestamps.setdefault(frame_number, (monotonic_ns, host_utc))
    return timestamps


def reprocess_session(
    session: Path,
    *,
    output_frames: Path | None = None,
    chunk_bytes: int = 65_536,
    overwrite: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Decode preserved UART bytes without opening or configuring a sensor."""

    session = session.expanduser().resolve()
    raw_path = session / "raw_uart.bin"
    if not raw_path.is_file():
        raise ValueError(f"session has no raw_uart.bin: {session}")
    if chunk_bytes < 64:
        raise ValueError("chunk_bytes must be at least 64")
    output_frames = (
        output_frames.expanduser().resolve()
        if output_frames is not None
        else session / "frames_reprocessed.jsonl"
    )
    manifest_path = session / "reprocessing_manifest.json"
    existing = [path for path in (output_frames, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            f"reprocessed output already exists: {existing[0]}; "
            "use --overwrite intentionally"
        )

    timestamps = _original_timestamps(session / "frames.jsonl")
    framer = TiTlvFramer()
    parser = TLVParser()
    records = []
    rows: list[dict[str, Any]] = []
    row_parse_errors = 0
    with raw_path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            for frame in framer.feed(chunk):
                # Read the frame number before decoding so original UART receipt
                # timing can be retained when frames.jsonl is available.
                header = FrameHeader.from_bytes(frame[:FRAME_HEADER_SIZE])
                timing = timestamps.get(int(header.frame_number))
                row, record = _decode_frame(
                    frame,
                    parser,
                    host_monotonic_ns=timing[0] if timing else -1,
                    host_utc=timing[1] if timing else "",
                )
                rows.append(row)
                if row.get("parse_ok") and record is not None:
                    records.append(record)
                else:
                    row_parse_errors += 1

    if not rows:
        raise ValueError(f"no TI TLV frames recovered from {raw_path}")
    output_frames.parent.mkdir(parents=True, exist_ok=True)
    with output_frames.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    signal_outputs = write_signal_artifacts(
        session,
        records,
        source="raw_uart_reprocessing",
        transport_quality={
            "raw_uart_bytes": raw_path.stat().st_size,
            "raw_uart_sha256": sha256_file(raw_path),
            "framed_packets": framer.frames,
            "row_parse_errors": row_parse_errors,
            "internal_tlv_parse_errors": int(getattr(parser, "_parse_errors", 0)),
            "invalid_headers": framer.invalid_headers,
            "discarded_bytes": framer.discarded_bytes,
            "trailing_buffer_bytes": len(framer.buffer),
        },
        overwrite=overwrite,
    )
    manifest = {
        "schema_version": REPROCESS_SCHEMA,
        "experimental": True,
        "sensor_opened": False,
        "canonical_training_compatible": False,
        "source_raw_uart": str(raw_path),
        "source_raw_uart_sha256": sha256_file(raw_path),
        "source_frames_timestamps": (
            str(session / "frames.jsonl")
            if (session / "frames.jsonl").is_file()
            else None
        ),
        "output_frames": str(output_frames),
        "output_frames_sha256": sha256_file(output_frames),
        "range_profiles": str(signal_outputs["range_profiles"]),
        "noise_profiles": str(signal_outputs["noise_profiles"]),
        "capture_quality": str(signal_outputs["capture_quality"]),
        "statistics": {
            "recovered_frames": len(rows),
            "successfully_decoded_frames": len(records),
            "row_parse_errors": row_parse_errors,
            "invalid_headers": framer.invalid_headers,
            "discarded_bytes": framer.discarded_bytes,
            "trailing_buffer_bytes": len(framer.buffer),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "reprocessing reconstructs processed demo TLVs, not raw ADC",
            "missing host timestamps cannot be recreated from raw UART bytes",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output_frames, manifest_path, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reprocess preserved AWR1843 raw UART into auditable TLV artifacts"
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output-frames", type=Path)
    parser.add_argument("--chunk-bytes", type=int, default=65_536)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        output_frames, manifest_path, manifest = reprocess_session(
            args.session,
            output_frames=args.output_frames,
            chunk_bytes=args.chunk_bytes,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "frames": str(output_frames),
                "manifest": str(manifest_path),
                **manifest["statistics"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
