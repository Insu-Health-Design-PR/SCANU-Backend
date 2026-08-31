"""Cross-node comparison for the two-radar facing ("stereo") mmWave lab test.

Independent, diagnostic-only module. It does not modify or read any canonical
Layer 1-8 contract and its output cannot enter schema-V6 training.

It answers a narrowly scoped question: for two independently captured and
independently processed AWR1843 lab sessions (one per facing radar, each
already run through the existing, unmodified `lab.mmwave77_usb.perception`
pipeline), how often did each side's `screening_state` agree at the same
wall-clock instant? This is a descriptive comparison of two independent
processed-TLV, post-CFAR pipelines. It is not a fusion of raw returns, not a
triangulation of a shared position, and not a validated accuracy measurement:
there is no controlled ground-truth label in this lab capture.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "scanu_lab_dual_mmwave77_stereo_fusion_v1"

_FLAGGED_STATES = ("person", "suspicious_metal")


def _parse_utc(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _load_frame_times(session: Path) -> dict[int, float]:
    path = session / "frames.jsonl"
    if not path.is_file():
        raise RuntimeError(f"missing frames.jsonl in {session}")
    out: dict[int, float] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["frame_number"])] = _parse_utc(row["host_utc"])
    if not out:
        raise RuntimeError(f"frames.jsonl in {session} contained no rows")
    return out


def _load_windows(session: Path) -> list[dict[str, Any]]:
    path = session / "perception.jsonl"
    if not path.is_file():
        raise RuntimeError(f"missing perception.jsonl in {session}")
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class TimedWindow:
    window_index: int
    absolute_utc_s: float
    screening_state: str
    scene_state: str
    track_present: bool


def _timed_windows(session: Path, applied_offset_s: float) -> list[TimedWindow]:
    frame_times = _load_frame_times(session)
    windows = _load_windows(session)
    out: list[TimedWindow] = []
    for row in windows:
        start = int(row["frame_start"])
        end = int(row["frame_end"])
        candidates = [frame_times[f] for f in range(start, end) if f in frame_times]
        if not candidates:
            continue
        absolute = (sum(candidates) / len(candidates)) + applied_offset_s
        out.append(
            TimedWindow(
                window_index=int(row["window_index"]),
                absolute_utc_s=absolute,
                screening_state=str(row["screening_state"]),
                scene_state=str(row["scene_state"]),
                track_present=row.get("track") is not None,
            )
        )
    return out


def _match(
    windows_a: list[TimedWindow],
    windows_b: list[TimedWindow],
    tolerance_s: float,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    used_b: set[int] = set()
    for window_a in windows_a:
        best_idx: int | None = None
        best_dt = tolerance_s
        for idx, window_b in enumerate(windows_b):
            if idx in used_b:
                continue
            dt = abs(window_b.absolute_utc_s - window_a.absolute_utc_s)
            if dt <= best_dt:
                best_idx = idx
                best_dt = dt
        if best_idx is not None:
            used_b.add(best_idx)
            window_b = windows_b[best_idx]
            matches.append(
                {
                    "window_a": window_a.window_index,
                    "window_b": window_b.window_index,
                    "time_delta_s": round(
                        window_b.absolute_utc_s - window_a.absolute_utc_s, 4
                    ),
                    "screening_state_a": window_a.screening_state,
                    "screening_state_b": window_b.screening_state,
                    "agree": window_a.screening_state == window_b.screening_state,
                }
            )
    matches.sort(key=lambda item: item["window_a"])
    return matches


def run_fusion(
    session_a: Path,
    session_b: Path,
    *,
    clock_offset_b_minus_a_s: float,
    tolerance_s: float,
) -> dict[str, Any]:
    """Compare two independent single-radar perception timelines.

    `clock_offset_b_minus_a_s` is host B's clock minus host A's clock, in
    seconds, as measured externally (see `scripts/measure_clock_offset.sh`).
    It is subtracted from B's timestamps so both timelines are expressed on
    host A's clock reference before matching.
    """

    windows_a = _timed_windows(session_a, applied_offset_s=0.0)
    windows_b = _timed_windows(session_b, applied_offset_s=-clock_offset_b_minus_a_s)
    matches = _match(windows_a, windows_b, tolerance_s)

    agreeing = sum(1 for row in matches if row["agree"])
    both_flagged = sum(
        1
        for row in matches
        if row["screening_state_a"] in _FLAGGED_STATES
        and row["screening_state_b"] in _FLAGGED_STATES
    )
    either_flagged = sum(
        1
        for row in matches
        if row["screening_state_a"] in _FLAGGED_STATES
        or row["screening_state_b"] in _FLAGGED_STATES
    )
    both_suspicious_metal = sum(
        1
        for row in matches
        if row["screening_state_a"] == "suspicious_metal"
        and row["screening_state_b"] == "suspicious_metal"
    )
    either_suspicious_metal = sum(
        1
        for row in matches
        if row["screening_state_a"] == "suspicious_metal"
        or row["screening_state_b"] == "suspicious_metal"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "experimental": True,
        "not_a_ground_truth_accuracy_measurement": True,
        "session_a": str(session_a),
        "session_b": str(session_b),
        "clock_offset_b_minus_a_s_applied": clock_offset_b_minus_a_s,
        "window_match_tolerance_s": tolerance_s,
        "windows_a_total": len(windows_a),
        "windows_b_total": len(windows_b),
        "matched_window_pairs": len(matches),
        "unmatched_windows_a": len(windows_a) - len(matches),
        "unmatched_windows_b": len(windows_b) - len(matches),
        "screening_state_agreement_fraction": (
            round(agreeing / len(matches), 4) if matches else None
        ),
        "both_flagged_person_or_suspicious_metal": both_flagged,
        "either_flagged_person_or_suspicious_metal": either_flagged,
        "both_flagged_suspicious_metal": both_suspicious_metal,
        "either_flagged_suspicious_metal": either_suspicious_metal,
        "matches": matches,
        "limitations": [
            "descriptive comparison of two independently processed, "
            "post-CFAR single-radar pipelines; not a raw-return fusion or "
            "triangulation of a shared position",
            "no controlled ground-truth label exists in this lab capture, "
            "so agreement/union/intersection counts are not accuracy, "
            "precision, or recall",
            "screening_state is experimental lab output "
            "(lab.mmwave77_usb.perception), not a canonical Layer 5 fusion "
            "decision",
            "cross-host time alignment depends on the externally supplied "
            "clock_offset_b_minus_a_s measurement; verify it was measured "
            "close in time to this capture",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two independent single-radar AWR1843 lab perception "
            "outputs captured by facing sensors on separate hosts."
        )
    )
    parser.add_argument("--session-a", required=True, type=Path)
    parser.add_argument("--session-b", required=True, type=Path)
    parser.add_argument(
        "--clock-offset-b-minus-a-s",
        type=float,
        default=0.0,
        help="host B clock minus host A clock, in seconds, measured externally",
    )
    parser.add_argument("--window-tolerance-s", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not args.session_a.is_dir():
        print(f"error: session-a does not exist: {args.session_a}", file=sys.stderr)
        return 2
    if not args.session_b.is_dir():
        print(f"error: session-b does not exist: {args.session_b}", file=sys.stderr)
        return 2
    report = run_fusion(
        args.session_a,
        args.session_b,
        clock_offset_b_minus_a_s=args.clock_offset_b_minus_a_s,
        tolerance_s=args.window_tolerance_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "matches"}, indent=2))
    return 0 if report["matched_window_pairs"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
