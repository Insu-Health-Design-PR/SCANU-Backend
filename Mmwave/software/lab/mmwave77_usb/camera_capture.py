#!/usr/bin/env python3
"""Record an ordinary USB/V4L2 camera beside an experimental AWR1843 run.

This is deliberately a laboratory utility.  Its timestamp log describes host
receipt time only; it does not establish camera-to-radar calibration or make
any material/object classification claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _is_numeric_device(value: str) -> bool:
    normalized = str(value).strip()
    return normalized.isdigit() or (
        normalized.startswith("-") and normalized[1:].isdigit()
    )


def _display_device(value: str) -> str:
    normalized = str(value).strip()
    return f"/dev/video{normalized}" if _is_numeric_device(normalized) else normalized


def _resolve_device(value: str) -> str:
    if value.strip().lower() != "auto":
        return value.strip()
    from layer1_sensor_hub.hardware_registry import discover_rgb_camera_index

    index = discover_rgb_camera_index()
    if index < 0:
        raise RuntimeError("no suitable RGB USB camera was discovered")
    return str(index)


def _open_camera(device: str, width: int, height: int, fps: float):
    import cv2

    normalized = str(device).strip()
    if _is_numeric_device(normalized):
        capture = cv2.VideoCapture(int(normalized), cv2.CAP_V4L2)
    else:
        expanded = str(Path(normalized).expanduser())
        capture = cv2.VideoCapture(expanded, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(expanded)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open camera {_display_device(device)}")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    capture.set(cv2.CAP_PROP_FPS, float(fps))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def probe(device: str, width: int, height: int, fps: float) -> dict:
    """Open and read one frame, then release the USB camera immediately."""
    capture = _open_camera(device, width, height, fps)
    try:
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError(f"camera {_display_device(device)} opened but returned no frame")
        return {
            "ok": True,
            "device": _display_device(device),
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
        "device": _display_device(device),
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


def record_ffmpeg_copy(
    *, device: str, duration_s: float, output_dir: Path, width: int, height: int,
    fps: float, input_format: str,
) -> dict:
    """Capture a compressed V4L2 stream without decoding/re-encoding it."""
    if duration_s <= 0 or width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("duration, width, height, and fps must be greater than zero")
    if not input_format:
        raise ValueError("input_format must be non-empty for ffmpeg-copy")
    output_dir.mkdir(parents=True, exist_ok=False)
    video_path = output_dir / "camera.mp4"
    frames_path = output_dir / "camera_frames.jsonl"
    metadata_path = output_dir / "camera_metadata.json"
    ffmpeg_device = device if device.startswith("/") else f"/dev/video{device}"
    started_ns = time.monotonic_ns()
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "v4l2",
            "-input_format", input_format, "-video_size", f"{width}x{height}",
            "-framerate", str(fps), "-i", ffmpeg_device, "-t", str(duration_s),
            "-c:v", "copy", "-movflags", "+faststart", str(video_path),
        ],
        check=True,
    )
    count = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_packets",
            "-of", "default=nokey=1:noprint_wrappers=1", str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    frames_written = int(count.stdout.strip())
    if frames_written <= 0:
        raise RuntimeError("ffmpeg-copy camera recording produced no packets")
    interval_ns = int(round(1_000_000_000 / fps))
    with frames_path.open("w") as timestamps:
        for index in range(frames_written):
            timestamps.write(json.dumps({
                "index": index,
                "host_monotonic_ns": started_ns + index * interval_ns,
            }) + "\n")
    metadata = {
        "schema_version": "scanu_lab_usb_camera_v1",
        "experimental": True,
        "device": ffmpeg_device,
        "backend": "ffmpeg-copy",
        "input_format": input_format,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": time.monotonic_ns(),
        "duration_s_requested": duration_s,
        "frames_written": frames_written,
        "resolution_px": [width, height],
        "fps_requested": fps,
        "fps_observed_estimate": frames_written / duration_s,
        "limitations": [
            "camera timestamps are host-start plus nominal frame cadence, not hardware synchronization",
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


def radar_window_center_camera_offset_s(
    camera_frames_path: Path,
    radar_frames_path: Path,
    perception_path: Path,
) -> float:
    """Camera seek for the temporal center of the first perception window.

    A perception image combines ten overlapping radar frames.  Associating
    it with the first radar frame makes radar motion appear roughly half a
    second ahead of the camera.  Use the median host timestamp of the exact
    frame range recorded in the first perception row instead.
    """
    camera_ns: int | None = None
    with camera_frames_path.open() as stream:
        for line in stream:
            if line.strip():
                camera_ns = int(json.loads(line)["host_monotonic_ns"])
                break
    if camera_ns is None:
        raise ValueError(f"no camera timestamp in {camera_frames_path}")

    first_window: dict | None = None
    with perception_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if "frame_start" in row and "frame_end" in row:
                first_window = row
                break
    if first_window is None:
        raise ValueError(f"no perception window in {perception_path}")
    frame_start = int(first_window["frame_start"])
    frame_end = int(first_window["frame_end"])

    timestamps: list[int] = []
    with radar_frames_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("parse_ok", False):
                continue
            frame_number = int(row.get("frame_number", -1))
            if frame_start <= frame_number <= frame_end:
                timestamps.append(int(row["host_monotonic_ns"]))
    if not timestamps:
        raise ValueError(
            f"no valid radar timestamps for frames {frame_start}-{frame_end}"
        )
    radar_center_ns = statistics.median(timestamps)
    return max(0.0, (radar_center_ns - camera_ns) / 1_000_000_000.0)


def radar_camera_alignment_s(
    camera_frames_path: Path,
    radar_frames_path: Path,
    perception_path: Path,
) -> tuple[float, float, float]:
    """Return start trim, end trim, and radar span for endpoint alignment."""
    camera_ns: int | None = None
    with camera_frames_path.open() as stream:
        for line in stream:
            if line.strip():
                camera_ns = int(json.loads(line)["host_monotonic_ns"])
                break
    if camera_ns is None:
        raise ValueError(f"no camera timestamp in {camera_frames_path}")
    windows: list[dict] = []
    with perception_path.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if "frame_start" in row and "frame_end" in row:
                    windows.append(row)
    if not windows:
        raise ValueError(f"no perception windows in {perception_path}")
    timestamps: dict[int, int] = {}
    with radar_frames_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("parse_ok", False):
                timestamps[int(row.get("frame_number", -1))] = int(row["host_monotonic_ns"])

    def window_center(row: dict) -> int:
        values = [
            timestamp for frame, timestamp in timestamps.items()
            if int(row["frame_start"]) <= frame <= int(row["frame_end"])
        ]
        if not values:
            raise ValueError(f"no valid radar timestamps for window {row['frame_start']}-{row['frame_end']}")
        return int(statistics.median(values))

    first_ns = window_center(windows[0])
    last_ns = window_center(windows[-1])
    start_s = max(0.0, (first_ns - camera_ns) / 1_000_000_000.0)
    end_s = max(start_s, (last_ns - camera_ns) / 1_000_000_000.0)
    return start_s, end_s, max(0.0, (last_ns - first_ns) / 1_000_000_000.0)


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
    record_parser.add_argument("--backend", choices=("opencv", "ffmpeg-copy"), default="opencv")
    record_parser.add_argument("--input-format", default=None)
    offset_parser = sub.add_parser("offset", help="calculate host-time camera trim for a radar session")
    offset_parser.add_argument("--camera-frames", required=True, type=Path)
    offset_parser.add_argument("--radar-frames", required=True, type=Path)
    offset_parser.add_argument(
        "--perception",
        type=Path,
        help="align to the center of the first perception window",
    )
    alignment_parser = sub.add_parser("alignment", help="calculate camera trim and endpoint drift correction")
    alignment_parser.add_argument("--camera-frames", required=True, type=Path)
    alignment_parser.add_argument("--radar-frames", required=True, type=Path)
    alignment_parser.add_argument("--perception", required=True, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "offset":
            offset_s = (
                radar_window_center_camera_offset_s(
                    args.camera_frames, args.radar_frames, args.perception
                )
                if args.perception is not None
                else radar_camera_offset_s(args.camera_frames, args.radar_frames)
            )
            print(f"{offset_s:.6f}")
            return 0
        if args.command == "alignment":
            start_s, end_s, radar_span_s = radar_camera_alignment_s(
                args.camera_frames, args.radar_frames, args.perception
            )
            print(f"{start_s:.9f}|{end_s:.9f}|{radar_span_s:.9f}")
            return 0
        device = _resolve_device(args.device)
        if args.command == "probe":
            result = probe(device, args.width, args.height, args.fps)
        else:
            if args.backend == "ffmpeg-copy":
                result = record_ffmpeg_copy(
                    device=device, duration_s=args.duration_s, output_dir=args.output_dir,
                    width=args.width, height=args.height, fps=args.fps,
                    input_format=args.input_format or "mjpeg",
                )
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
