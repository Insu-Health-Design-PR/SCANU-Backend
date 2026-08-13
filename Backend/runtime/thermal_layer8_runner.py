#!/usr/bin/env python3
"""
Runtime entrypoint: thermal weapon overlay via weapon_ai.infer_thermal_objects.

Modes:
  --thermal-device PATH       open PureThermal / Y16 directly (direct pipeline)
  --live-frame-poll PATH      overlay on JPEG; optional --thermal-capture-feed opens camera
                              in a side thread and writes that JPEG (shared pipeline)

  python -m runtime.thermal_layer8_runner ...
"""

from __future__ import annotations

import argparse
import shlex
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def _infer_main_callable():
    from weapon_ai.infer_thermal_objects import main

    return main


def _v4l2_open_target(raw: str) -> str:
    s = str(raw).strip()
    if s.startswith("/dev/video"):
        return s
    if s.isdigit():
        return f"/dev/video{int(s)}"
    return s


def _frame_to_colormap_bgr(frame: np.ndarray, panel_w: int, panel_h: int) -> np.ndarray | None:
    if frame is None or frame.size == 0:
        return None
    if frame.dtype == np.uint16:
        f32 = frame.astype(np.float32)
        mn = float(f32.min())
        mx = float(f32.max())
        if mx - mn > 1e-6:
            gray = ((f32 - mn) / (mx - mn) * 255.0).astype(np.uint8)
        else:
            gray = cv2.convertScaleAbs(frame)
    elif frame.ndim == 2:
        gray = frame if frame.dtype == np.uint8 else cv2.convertScaleAbs(frame)
    elif frame.ndim == 3:
        ch = int(frame.shape[2])
        if ch == 1:
            gray = frame[:, :, 0]
        elif ch == 3:
            return cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        return None
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.resize(heat, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)


def _start_thermal_jpeg_feeder(
    *,
    device: str,
    width: int,
    height: int,
    fps: float,
    panel_w: int,
    panel_h: int,
    out_path: Path,
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    interval = max(0.03, 1.0 / max(1.0, float(fps)))

    def _run() -> None:
        cap: cv2.VideoCapture | None = None
        open_target = _v4l2_open_target(device)
        while not stop.is_set():
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(open_target, cv2.CAP_V4L2)
                if not cap.isOpened():
                    time.sleep(0.5)
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                cap.set(cv2.CAP_PROP_FPS, float(fps))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(
                    f"Thermal JPEG feeder: opened {device} {width}x{height} → {out_path}",
                    flush=True,
                )
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            bgr = _frame_to_colormap_bgr(frame, int(panel_w), int(panel_h))
            if bgr is None:
                time.sleep(0.02)
                continue
            ok_enc, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok_enc:
                tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                tmp.write_bytes(buf.tobytes())
                tmp.replace(out_path)
            time.sleep(interval)
        if cap is not None:
            cap.release()

    thread = threading.Thread(target=_run, name="thermal-jpeg-feeder", daemon=True)
    thread.start()
    return stop, thread


def main() -> None:
    p = argparse.ArgumentParser(description="Thermal live infer for Layer 8 UI.", allow_abbrev=False)
    p.add_argument("--thermal-device", type=str, default="")
    p.add_argument("--live-frame-poll", type=str, default="")
    p.add_argument(
        "--thermal-capture-feed",
        type=str,
        default="",
        help="With --live-frame-poll: capture Y16 from this device into the poll JPEG path.",
    )
    p.add_argument("--thermal-width", type=int, default=160)
    p.add_argument("--thermal-height", type=int, default=120)
    p.add_argument("--thermal-fps", type=float, default=30.0)
    p.add_argument("--panel-w", type=int, default=640)
    p.add_argument("--panel-h", type=int, default=480)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--live-frame", type=str, default="")
    p.add_argument("--video", type=str, default="")
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--metrics-json", type=str, default="")
    p.add_argument("--live-ipc-frame", type=str, default="")
    p.add_argument("--live-ipc-bgr-frame", type=str, default="")
    p.add_argument("--weapon-extra-args", type=str, default="")
    args = p.parse_args()

    poll = args.live_frame_poll.strip()
    device = args.thermal_device.strip()
    feed = args.thermal_capture_feed.strip()
    if device and poll:
        raise SystemExit("Use either --live-frame-poll (shared) or --thermal-device (direct), not both.")
    if not device and not poll:
        raise SystemExit("One of --live-frame-poll or --thermal-device is required.")
    if feed and not poll:
        raise SystemExit("--thermal-capture-feed requires --live-frame-poll.")

    feeder_stop: threading.Event | None = None
    feeder_thread: threading.Thread | None = None
    if poll and feed:
        feeder_stop, feeder_thread = _start_thermal_jpeg_feeder(
            device=str(feed),
            width=int(args.thermal_width),
            height=int(args.thermal_height),
            fps=float(args.thermal_fps),
            panel_w=int(args.panel_w),
            panel_h=int(args.panel_h),
            out_path=Path(poll),
        )
        time.sleep(0.35)

    forward: list[str] = [
        "infer_thermal_objects",
        "--checkpoint",
        args.checkpoint,
        "--capture_fps",
        str(float(args.thermal_fps)),
        "--panel_w",
        str(int(args.panel_w)),
        "--panel_h",
        str(int(args.panel_h)),
        "--no_imshow",
        "--no-show_fps",
        "--thin_overlay",
    ]
    if poll:
        forward.extend(["--live_frame_poll", poll, "--source", poll])
    else:
        forward.extend(
            [
                "--source",
                str(_v4l2_open_target(device)),
                "--capture_width",
                str(int(args.thermal_width)),
                "--capture_height",
                str(int(args.thermal_height)),
                "--thermal_v4l2",
            ]
        )
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
    try:
        infer_main()
    finally:
        if feeder_stop is not None:
            feeder_stop.set()
        if feeder_thread is not None:
            feeder_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
