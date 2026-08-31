#!/usr/bin/env python3
"""Render the full 226e909e evidence video method (5 charts, both sensors).

Uses exactly `lab.dual_mmwave77_stereo.dual_fusion_video.render_dual_fusion_video`
(the method added by commit 226e909e): camera A | fused 3D cloud | camera B on
top, and five charts below (top view, side view, elevation-azimuth, A/B/fused
timeline, combined evidence timeline), with anomaly diamonds, screening state
and fused track confidence.

This is a thin wrapper so the full method can be invoked consistently on the
server and locally. It is laboratory evidence visualization, not a material or
weapon classification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from lab.dual_mmwave77_stereo.dual_fusion_video import render_dual_fusion_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the full dual-fusion evidence video (226e909e method, 5 charts)"
    )
    parser.add_argument("--session-a", required=True, type=Path)
    parser.add_argument("--session-b", required=True, type=Path)
    parser.add_argument("--calibration-a", type=Path)
    parser.add_argument("--calibration-b", type=Path)
    parser.add_argument("--distance-m", required=True, type=float)
    parser.add_argument("--clock-offset-b-minus-a-s", type=float, default=0.0)
    parser.add_argument("--window-tolerance-s", type=float, default=0.5)
    parser.add_argument("--camera-a", required=True, type=Path)
    parser.add_argument("--camera-b", type=Path)
    parser.add_argument("--camera-a-frames", required=True, type=Path)
    parser.add_argument("--camera-b-frames", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    output, metadata, _ = render_dual_fusion_video(
        args.session_a,
        args.session_b,
        distance_m=args.distance_m,
        clock_offset_b_minus_a_s=args.clock_offset_b_minus_a_s,
        window_tolerance_s=args.window_tolerance_s,
        calibration_a=args.calibration_a,
        calibration_b=args.calibration_b,
        camera_a=args.camera_a,
        camera_b=args.camera_b,
        camera_a_frames=args.camera_a_frames,
        camera_b_frames=args.camera_b_frames,
        output_path=args.output,
        fps=args.fps,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output": str(output), "metadata": str(metadata)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
