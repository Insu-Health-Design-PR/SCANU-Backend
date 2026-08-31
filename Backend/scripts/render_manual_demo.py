#!/usr/bin/env python3
"""Render demo-perfect video from manual annotation JSON (no YOLO).

  python scripts/render_manual_demo.py --clip 152636
  python scripts/render_manual_demo.py --clip 152636 --manual sync_out/152636/manual_annotations.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from manual_annotations import (  # noqa: E402
    ManualAnnotationStore,
    draw_manual_frame,
    empty_document,
    save_manual_annotations,
)


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    return cap


def _writer_for(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(max(fps, 1.0)), size)
    if not writer.isOpened():
        raise SystemExit(f"Cannot open VideoWriter for {path}")
    return writer


def default_out_path(src: Path) -> Path:
    return src.with_name(f"{src.stem}_inferred{src.suffix or '.mp4'}")

_PAIR_RE = __import__("re").compile(r"dual_awr1843_\d+_(\d+)_camera_A_4k30\.mp4$")


def discover_pair(demo_root: Path, clip_id: str) -> tuple[Path, Path]:
    for a_path in demo_root.rglob(f"*_camera_A_4k30.mp4"):
        m = _PAIR_RE.match(a_path.name)
        if not m or m.group(1) != clip_id:
            continue
        b_path = a_path.with_name(a_path.name.replace("camera_A", "camera_B"))
        if b_path.is_file():
            return a_path.resolve(), b_path.resolve()
    raise SystemExit(f"No camera pair found for clip {clip_id} under {demo_root}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True, help="Clip id e.g. 152636")
    p.add_argument("--demo-root", type=Path, default=Path.home() / "Desktop" / "demo-video")
    p.add_argument("--sync-out", type=Path, default=None)
    p.add_argument("--manual", type=Path, default=None, help="manual_annotations.json path")
    p.add_argument("--camera-front", default="camera_1")
    p.add_argument("--camera-back", default="camera_2")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clip_id = str(args.clip)
    demo_root = args.demo_root.expanduser().resolve()
    sync_out = (args.sync_out or demo_root / "sync_out").expanduser().resolve()
    out_dir = sync_out / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manual_path = (args.manual or out_dir / "manual_annotations.json").expanduser().resolve()
    if not manual_path.is_file():
        doc = empty_document(
            clip_id=clip_id,
            camera_front=str(args.camera_front),
            camera_back=str(args.camera_back),
        )
        save_manual_annotations(manual_path, doc)
        print(f"Created empty template: {manual_path}")
        print("Add keyframes in Sync Review (Manual box mode) then re-run.")
        return 0

    front_path, back_path = discover_pair(demo_root, clip_id)
    front_out = out_dir / "camera_A_inferred.mp4"
    back_out = out_dir / "camera_B_inferred.mp4"

    store = ManualAnnotationStore.from_path(manual_path)
    if not store.has_any():
        raise SystemExit(f"No keyframes in {manual_path}")

    cap_f = _open_video(front_path)
    cap_b = _open_video(back_path)
    fps = float(cap_f.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0

    # Update ref sizes from source if missing
    fw, fh = int(cap_f.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap_f.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    bw, bh = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    store.data.setdefault("ref", {})
    ref = store.data["ref"]
    cf, cb = str(args.camera_front), str(args.camera_back)
    if not int((ref.get(cf) or {}).get("width") or 0):
        ref[cf] = {"width": fw, "height": fh}
    if not int((ref.get(cb) or {}).get("width") or 0):
        ref[cb] = {"width": bw, "height": bh}
    save_manual_annotations(manual_path, store.data)

    writer_f = writer_b = None
    frame_i = 0
    written = 0
    t0 = time.time()
    max_frames = int(args.max_frames) if args.max_frames > 0 else 0

    try:
        while True:
            ok_f, frame_f = cap_f.read()
            ok_b, frame_b = cap_b.read()
            if not ok_f or not ok_b or frame_f is None or frame_b is None:
                break
            if max_frames and written >= max_frames:
                break

            rw_f, rh_f = store.ref_size(cf)
            rw_b, rh_b = store.ref_size(cb)
            vis_f = draw_manual_frame(
                frame_f, store.frame(cf, frame_i), ref_w=rw_f, ref_h=rh_f
            )
            vis_b = draw_manual_frame(
                frame_b, store.frame(cb, frame_i), ref_w=rw_b, ref_h=rh_b
            )

            if writer_f is None:
                hf, wf = vis_f.shape[:2]
                hb, wb = vis_b.shape[:2]
                writer_f = _writer_for(front_out, fps, (wf, hf))
                writer_b = _writer_for(back_out, fps, (wb, hb))

            writer_f.write(vis_f)
            writer_b.write(vis_b)
            written += 1
            frame_i += 1
            if written % 100 == 0 or written == 1:
                print(f"  frame {written}", flush=True)
    finally:
        cap_f.release()
        cap_b.release()
        if writer_f is not None:
            writer_f.release()
        if writer_b is not None:
            writer_b.release()

    elapsed = time.time() - t0
    print(f"Wrote {front_out}")
    print(f"Wrote {back_out}")
    print(f"Frames={written} keyframes={store.keyframe_count()} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
