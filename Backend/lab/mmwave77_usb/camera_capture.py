#!/usr/bin/env python3
"""Record an ordinary USB/V4L2 camera beside an experimental AWR1843 run.

This is deliberately a laboratory utility.  Its timestamp log describes host
receipt time only; it does not establish camera-to-radar calibration or make
any material/object classification claim.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _resolve_device(value: str) -> str:
    if value.strip().lower() != "auto":
        return value.strip()
    from layer1_sensor_hub.hardware_registry import discover_rgb_camera_index

    index = discover_rgb_camera_index()
    if index < 0:
        raise RuntimeError("no suitable RGB USB camera was discovered")
    return str(index)


def _open_camera(device: str, width: int, height: int, fps: float):
    from layer8_ui.camera_device import display_device, open_v4l2_capture

    import cv2

    capture = open_v4l2_capture(device)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open camera {display_device(device)}")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    capture.set(cv2.CAP_PROP_FPS, float(fps))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def probe(device: str, width: int, height: int, fps: float) -> dict:
    """Open and read one frame, then release the USB camera immediately."""
    from layer8_ui.camera_device import display_device

    capture = _open_camera(device, width, height, fps)
    try:
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError(f"camera {display_device(device)} opened but returned no frame")
        return {
            "ok": True,
            "device": display_device(device),
            "resolution_px": [int(frame.shape[1]), int(frame.shape[0])],
            "reported_fps": float(capture.get(5)),
        }
    finally:
        capture.release()


def record(
    *, device: str, duration_s: float, output_dir: Path, width: int, height: int, fps: float
) -> dict:
    """Write an MP4 plus a per-frame host-monotonic timestamp log."""
    if duration_s <= 0 or width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("duration, width, height, and fps must be greater than zero")
    from layer8_ui.camera_device import display_device

    import cv2

    output_dir.mkdir(parents=True, exist_ok=False)
    video_path = output_dir / "camera.mp4"
    frames_path = output_dir / "camera_frames.jsonl"
    metadata_path = output_dir / "camera_metadata.json"
    capture = _open_camera(device, width, height, fps)
    writer = None
    started_ns = time.monotonic_ns()
    frames_written = 0
    actual_size: list[int] | None = None
    try:
        deadline = time.monotonic() + duration_s
        with frames_path.open("w") as timestamps:
            while time.monotonic() < deadline:
                ok, frame = capture.read()
                receipt_ns = time.monotonic_ns()
                if not ok or frame is None or frame.size == 0:
                    continue
                if writer is None:
                    actual_size = [int(frame.shape[1]), int(frame.shape[0])]
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(fps),
                        tuple(actual_size),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"could not create camera video: {video_path}")
                writer.write(frame)
                timestamps.write(json.dumps({"index": frames_written, "host_monotonic_ns": receipt_ns}) + "\n")
                frames_written += 1
    finally:
        if writer is not None:
            writer.release()
        capture.release()
    if frames_written == 0 or not video_path.is_file():
        raise RuntimeError("camera recording produced no frames")
    metadata = {
        "schema_version": "scanu_lab_usb_camera_v1",
        "experimental": True,
        "device": display_device(device),
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": time.monotonic_ns(),
        "duration_s_requested": duration_s,
        "frames_written": frames_written,
        "resolution_px": actual_size,
        "fps_requested": fps,
        "limitations": [
            "camera timestamps are host receipt times, not hardware synchronization",
            "camera imagery is visual context only and not a material or weapon classifier",
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"video": str(video_path), "frames": str(frames_path), "metadata": str(metadata_path), **metadata}


def radar_camera_offset_s(camera_frames_path: Path, radar_frames_path: Path) -> float:
    """Return the camera trim needed to start at the first valid radar TLV."""
    def first_timestamp(path: Path, *, radar: bool) -> int:
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not radar or row.get("parse_ok", False):
                    return int(row["host_monotonic_ns"])
        kind = "valid radar" if radar else "camera"
        raise ValueError(f"no {kind} timestamp in {path}")

    camera_ns = first_timestamp(camera_frames_path, radar=False)
    radar_ns = first_timestamp(radar_frames_path, radar=True)
    return max(0.0, (radar_ns - camera_ns) / 1_000_000_000.0)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experimental USB camera recorder for AWR1843 lab comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "record"):
        item = sub.add_parser(name)
        item.add_argument("--device", default="auto", help="auto, V4L2 index, or /dev/v4l/by-id path")
        item.add_argument("--width", type=int, default=1280)
        item.add_argument("--height", type=int, default=720)
        item.add_argument("--fps", type=float, default=20.0)
    record_parser = sub.choices["record"]
    record_parser.add_argument("--duration-s", required=True, type=float)
    record_parser.add_argument("--output-dir", required=True, type=Path)
    offset_parser = sub.add_parser("offset", help="calculate host-time camera trim for a radar session")
    offset_parser.add_argument("--camera-frames", required=True, type=Path)
    offset_parser.add_argument("--radar-frames", required=True, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "offset":
            print(f"{radar_camera_offset_s(args.camera_frames, args.radar_frames):.6f}")
            return 0
        device = _resolve_device(args.device)
        if args.command == "probe":
            result = probe(device, args.width, args.height, args.fps)
        else:
            result = record(device=device, duration_s=args.duration_s, output_dir=args.output_dir,
                            width=args.width, height=args.height, fps=args.fps)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
