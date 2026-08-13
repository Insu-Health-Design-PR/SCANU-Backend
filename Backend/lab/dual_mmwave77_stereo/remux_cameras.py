#!/usr/bin/env python3
"""Remux USB camera mp4s onto their real host timestamps and rebuild the 2x2.

The camera_capture recorder muxed every camera at the requested fps even when
the device delivered frames slower (e.g. 10 fps delivered, 20 fps muxed),
so the mp4 plays faster than wall clock. This script re-times each camera
frame using the host_monotonic_ns recorded in camera_frames.jsonl, then
recreates the 2x2 combined video aligned to the radar-start timeline.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def real_timestamps(camera_frames: Path) -> list[float]:
    ts: list[int] = []
    with camera_frames.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            ts.append(int(json.loads(line)["host_monotonic_ns"]))
    if not ts:
        raise ValueError(f"no camera frames in {camera_frames}")
    base = ts[0]
    return [(t - base) / 1e9 for t in ts]


def extract_frames(mp4: Path, work: Path, prefix: str) -> list[Path]:
    out_dir = work / f"{prefix}_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "f%05d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-q:v", "2", pattern],
        check=True,
    )
    return sorted(out_dir.glob("f*.jpg"))


def remux(frames: list[Path], timestamps: list[float], output: Path) -> None:
    if len(frames) != len(timestamps):
        raise ValueError(
            f"frame count {len(frames)} != timestamp count {len(timestamps)}"
        )
    concat = output.with_suffix(".concat.txt")
    with concat.open("w") as f:
        for i, frame in enumerate(frames):
            f.write(f"file '{frame.resolve()}'\n")
            if i + 1 < len(frames):
                duration = timestamps[i + 1] - timestamps[i]
                if duration > 0:
                    f.write(f"duration {duration:.6f}\n")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ],
        check=True,
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception-a", required=True, type=Path)
    ap.add_argument("--camera-a", required=True, type=Path)
    ap.add_argument("--camera-a-frames", required=True, type=Path)
    ap.add_argument("--trim-a-s", required=True, type=float)
    ap.add_argument("--perception-b", required=True, type=Path)
    ap.add_argument("--camera-b", required=True, type=Path)
    ap.add_argument("--camera-b-frames", required=True, type=Path)
    ap.add_argument("--trim-b-s", required=True, type=float)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--keep-fixed", type=Path, default=None)
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="scanu_remux_") as tmp:
        work = Path(tmp)
        ts_a = real_timestamps(args.camera_a_frames)
        ts_b = real_timestamps(args.camera_b_frames)

        print("extracting camera A frames...", file=sys.stderr)
        frames_a = extract_frames(args.camera_a, work, "a")
        print("remuxing camera A on real timestamps...", file=sys.stderr)
        fixed_a = work / "camera_a_fixed.mp4"
        remux(frames_a, ts_a, fixed_a)

        print("extracting camera B frames...", file=sys.stderr)
        frames_b = extract_frames(args.camera_b, work, "b")
        print("remuxing camera B on real timestamps...", file=sys.stderr)
        fixed_b = work / "camera_b_fixed.mp4"
        remux(frames_b, ts_b, fixed_b)

        if args.keep_fixed is not None:
            import shutil
            out_dir = args.keep_fixed
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixed_a, out_dir / "camera_a_fixed.mp4")
            shutil.copyfile(fixed_b, out_dir / "camera_b_fixed.mp4")
            print(f"saved fixed cameras to {out_dir}", file=sys.stderr)

        filter_complex = (
            "[0:v]scale=960:540:force_original_aspect_ratio=decrease,"
            "pad=960:540:(ow-iw)/2:(oh-ih)/2:color=0x050b14[ra];"
            "[1:v]scale=960:540:force_original_aspect_ratio=decrease,"
            "pad=960:540:(ow-iw)/2:(oh-ih)/2:color=black[ca];"
            "[2:v]scale=960:540:force_original_aspect_ratio=decrease,"
            "pad=960:540:(ow-iw)/2:(oh-ih)/2:color=0x050b14[rb];"
            "[3:v]scale=960:540:force_original_aspect_ratio=decrease,"
            "pad=960:540:(ow-iw)/2:(oh-ih)/2:color=black[cb];"
            "[ra][ca][rb][cb]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0"
        )
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(args.perception_a),
            "-ss", f"{args.trim_a_s:.6f}", "-i", str(fixed_a),
            "-i", str(args.perception_b),
            "-ss", f"{args.trim_b_s:.6f}", "-i", str(fixed_b),
            "-filter_complex", filter_complex,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        if args.duration_s is not None:
            cmd.append("-t")
            cmd.append(f"{args.duration_s:.3f}")
        cmd.append(str(args.output))
        print("building 2x2 combined video...", file=sys.stderr)
        subprocess.run(cmd, check=True)
    print(json.dumps({"output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
