#!/usr/bin/env python3
"""
Runtime entrypoint: live webcam + weapon_ai.infer_objects (6-class Sohas YOLO).

Run with cwd = New_Backend/Backend (see runtime.sensor_runner).

  python -m runtime.multi_camera_layer8_runner \\
    --checkpoint trained_models/gun_detection/gun_sohas_6class.pt \\
    --webcam-device 0 \\
    --live-frame /abs/path/live_multi_camera.jpg \\
    --video /abs/path/out.mp4
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


def _infer_main_callable():
    from weapon_ai.infer_objects import main

    return main


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-camera live infer for Layer 8 UI.", allow_abbrev=False)
    p.add_argument("--webcam-device", type=str, default="0")
    p.add_argument("--capture-width", type=int, default=3840)
    p.add_argument("--capture-height", type=int, default=2160)
    p.add_argument("--capture-fps", type=float, default=30.0)
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint (e.g. gun_sohas_6class.pt).",
    )
    p.add_argument(
        "--live-frame",
        type=str,
        default="",
        help="Optional JPEG path updated every frame (legacy MJPEG source for UI).",
    )
    p.add_argument(
        "--video",
        type=str,
        default="",
        help="Optional annotated MP4 output (same as infer --output).",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Max frames (0 = until process stopped / stream ends).",
    )
    p.add_argument(
        "--metrics-json",
        type=str,
        default="",
        help="Optional JSON path for Layer 8 dashboard threat metrics.",
    )
    p.add_argument(
        "--live-ipc-frame",
        type=str,
        default="",
        help="Optional mmap latest-frame path for dashboard low-latency preview.",
    )
    p.add_argument(
        "--live-ipc-bgr-frame",
        type=str,
        default="",
        help="Optional mmap latest raw-BGR frame path for low-CPU WebRTC preview.",
    )
    p.add_argument(
        "--weapon-extra-args",
        type=str,
        default="",
        help="Extra arguments forwarded to infer_objects (quoted shell string).",
    )
    p.add_argument(
        "--gstreamer-capture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use GStreamer v4l2src→jpegdec for webcam capture (off by default; FFmpeg CUVID is preferred).",
    )
    args = p.parse_args()

    forward: list[str] = [
        "infer_objects",
        "--checkpoint",
        args.checkpoint,
        "--source",
        str(args.webcam_device).strip(),
        "--capture_width",
        str(int(args.capture_width)),
        "--capture_height",
        str(int(args.capture_height)),
        "--capture_fps",
        str(float(args.capture_fps)),
        "--no_imshow",
    ]
    if bool(args.gstreamer_capture):
        forward.append("--gstreamer_capture")
    else:
        forward.append("--no-gstreamer_capture")
    if args.live_frame.strip():
        forward.extend(["--live_jpg", args.live_frame.strip()])
    if args.video.strip():
        forward.extend(["--output", args.video.strip()])
    if int(args.frames) > 0:
        forward.extend(["--max_frames", str(int(args.frames))])
    if args.metrics_json.strip():
        forward.extend(["--live_metrics_json", args.metrics_json.strip()])
    if args.live_ipc_frame.strip():
        forward.extend(["--live_ipc_frame", args.live_ipc_frame.strip()])
    if args.live_ipc_bgr_frame.strip():
        forward.extend(["--live_ipc_bgr_frame", args.live_ipc_bgr_frame.strip()])
    extra = args.weapon_extra_args.strip()
    if extra:
        forward.extend(shlex.split(extra))

    sys.argv = forward
    infer_main = _infer_main_callable()
    infer_main()


if __name__ == "__main__":
    main()
