#!/usr/bin/env python3
"""Replay validation: fuse mmWave perception sessions onto camera MP4 frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weapon_ai.overlay.mmwave_fusion import MmwaveFusionConfig, draw_mmwave_fusion_overlay


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _points_for_window(frames: list[dict], start: int, end: int) -> list[dict]:
    out = []
    for fr in frames:
        n = int(fr.get("frame_number") or -1)
        if n < start or n > end:
            continue
        for p in fr.get("points") or []:
            if isinstance(p, dict):
                out.append(p)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Validate mmWave+camera overlay from capture manifest")
    p.add_argument("--manifest", required=True, help="dual_server manifest.json path")
    p.add_argument("--side", default="front", choices=("front", "back"))
    p.add_argument("--output", default="", help="Optional output MP4 path")
    p.add_argument("--max-frames", type=int, default=300)
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).expanduser().read_text())
    participant = manifest.get("participant") or {}
    cam_key = "camera_a" if args.side == "front" else "camera_b"
    radar_key = "radar_a" if args.side == "front" else "radar_b"
    cam_sess = ((participant.get(cam_key) or {}).get("session") or "")
    rad_sess = ((participant.get(radar_key) or {}).get("session") or "")
    if not cam_sess or not rad_sess:
        print("manifest missing camera/radar session paths", file=sys.stderr)
        return 1

    cam_dir = Path(cam_sess)
    rad_dir = Path(rad_sess)
    video = cam_dir / "camera.mp4"
    perception = rad_dir / "perception.jsonl"
    frames_jsonl = rad_dir / "frames.jsonl"
    if not video.is_file() or not perception.is_file() or not frames_jsonl.is_file():
        print("missing video or radar artifacts", file=sys.stderr)
        return 1

    rows = _load_jsonl(perception)
    frames = _load_jsonl(frames_jsonl)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"cannot open {video}", file=sys.stderr)
        return 1

    cfg = MmwaveFusionConfig(side="front" if args.side == "front" else "back")
    writer = None
    idx = 0
    frame_i = 0
    while frame_i < args.max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        row = rows[idx % len(rows)] if rows else {}
        pts = _points_for_window(
            frames,
            int(row.get("frame_start", 0)),
            int(row.get("frame_end", 0)),
        )
        metrics = {
            "schema_version": "scanu_mmwave_live_v1",
            "front" if cfg.side == "front" else "back": {
                "screening_state": row.get("screening_state", "background"),
                "track": row.get("track"),
                "points": pts,
                "anomalies": row.get("anomalies") or [],
            },
        }
        vis = draw_mmwave_fusion_overlay(bgr, [], metrics, cfg=cfg)
        if args.output:
            if writer is None:
                h, w = vis.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                writer = cv2.VideoWriter(
                    args.output,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )
            writer.write(vis)
        idx += 1
        frame_i += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Wrote {args.output}")
    else:
        print(f"Processed {frame_i} frames (dry run — pass --output to save MP4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
