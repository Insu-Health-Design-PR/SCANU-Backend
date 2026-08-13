#!/usr/bin/env python3
"""Live mmWave perception publisher for Layer 8 UI (``live_mmwave.jpg``).

Pipelines (``--pipeline``):
  lab_replay   — cycle an existing Adrian-style session (perception.jsonl + frames.jsonl)
  lab_live     — AWR1843 USB TLV capture → rolling perception → JPEG (needs hardware)
  status       — write idle placeholder JPEG only

Writes atomically to ``--live-frame`` for ``/api/preview/live/mmwave``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _atomic_write_jpg(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _points_from_frames(frames: list[dict[str, Any]], start: int, end: int) -> np.ndarray:
    pts: list[list[float]] = []
    for fr in frames:
        n = int(fr.get("frame_number") or fr.get("frame") or -1)
        if n < start or n > end:
            continue
        for p in fr.get("points") or fr.get("detected_points") or []:
            if isinstance(p, dict):
                x = p.get("x", p.get("x_m"))
                y = p.get("y", p.get("y_m"))
                z = p.get("z", p.get("z_m"))
                if x is None or y is None or z is None:
                    continue
                pts.append([float(x), float(y), float(z)])
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _replay_one_side(
    *,
    session: Path | None,
    live: Path,
    title: str,
    perception: str = "",
    frames_jsonl: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    from lab.mmwave77_usb.live_perception_frame import render_status_jpeg

    if session is None or not str(session):
        _atomic_write_jpg(live, render_status_jpeg(f"{title}\nNo session configured"))
        return None
    session = Path(session).expanduser().resolve()
    perception_path = Path(perception).expanduser() if perception else session / "perception.jsonl"
    frames_path = Path(frames_jsonl).expanduser() if frames_jsonl else session / "frames.jsonl"
    if not perception_path.is_file() or not frames_path.is_file():
        msg = (
            f"{title}: missing perception.jsonl / frames.jsonl\n"
            f"session={session}"
        )
        _atomic_write_jpg(live, render_status_jpeg(msg))
        return None
    return _load_jsonl(perception_path), _load_jsonl(frames_path)


def run_replay(args: argparse.Namespace) -> int:
    from lab.mmwave77_usb.live_perception_frame import render_perception_jpeg, render_status_jpeg

    live = Path(args.live_frame).expanduser().resolve()
    live_back = Path(getattr(args, "live_frame_back", "") or "").expanduser()
    if str(live_back):
        live_back = live_back.resolve()
    else:
        live_back = live.with_name("live_mmwave_back.jpg")

    front = _replay_one_side(
        session=Path(args.session) if args.session else None,
        live=live,
        title="mmWave Front",
        perception=args.perception,
        frames_jsonl=args.frames_jsonl,
    )
    back_session = getattr(args, "session_back", "") or ""
    back = _replay_one_side(
        session=Path(back_session) if back_session else None,
        live=live_back,
        title="mmWave Back",
    )

    if front is None and back is None:
        msg = (
            "mmWave dual replay idle\n"
            "Set front_session / back_session (or session) with perception.jsonl"
        )
        print(msg, flush=True)
        while True:
            _atomic_write_jpg(live, render_status_jpeg("mmWave Front\n" + msg))
            _atomic_write_jpg(live_back, render_status_jpeg("mmWave Back\n" + msg))
            time.sleep(2.0)
        return 1

    rows_f, frames_f = front if front else ([], [])
    rows_b, frames_b = back if back else ([], [])
    print(
        f"mmWave replay front_windows={len(rows_f)} back_windows={len(rows_b)}",
        flush=True,
    )
    idx = 0
    fps = max(0.5, float(args.fps))
    period = 1.0 / fps
    while True:
        if rows_f:
            row = rows_f[idx % len(rows_f)]
            pts = _points_from_frames(frames_f, int(row.get("frame_start", 0)), int(row.get("frame_end", 0)))
            jpg = render_perception_jpeg(
                points_xyz=pts,
                track=row.get("track"),
                anomalies=list(row.get("anomalies") or []),
                screening_state=str(row.get("screening_state") or "background"),
                title="mmWave Front · perception",
            )
            _atomic_write_jpg(live, jpg)
        if rows_b:
            row = rows_b[idx % len(rows_b)]
            pts = _points_from_frames(frames_b, int(row.get("frame_start", 0)), int(row.get("frame_end", 0)))
            jpg = render_perception_jpeg(
                points_xyz=pts,
                track=row.get("track"),
                anomalies=list(row.get("anomalies") or []),
                screening_state=str(row.get("screening_state") or "background"),
                title="mmWave Back · perception",
            )
            _atomic_write_jpg(live_back, jpg)
        idx += 1
        time.sleep(period)


def run_status(args: argparse.Namespace) -> int:
    from lab.mmwave77_usb.live_perception_frame import render_status_jpeg

    live = Path(args.live_frame).expanduser().resolve()
    live_back = Path(getattr(args, "live_frame_back", "") or "").expanduser()
    live_back = live_back.resolve() if str(live_back) else live.with_name("live_mmwave_back.jpg")
    msg_f = "mmWave Front idle\nSet front_session + pipeline=lab_replay"
    msg_b = "mmWave Back idle\nSet back_session + pipeline=lab_replay"
    while True:
        _atomic_write_jpg(live, render_status_jpeg(msg_f))
        _atomic_write_jpg(live_back, render_status_jpeg(msg_b))
        time.sleep(2.0)


def run_live_uart(args: argparse.Namespace) -> int:
    """Best-effort live AWR1843 path; falls back to status on failure."""
    from lab.mmwave77_usb.live_perception_frame import render_status_jpeg

    live = Path(args.live_frame).expanduser().resolve()
    live_back = Path(getattr(args, "live_frame_back", "") or "").expanduser()
    live_back = live_back.resolve() if str(live_back) else live.with_name("live_mmwave_back.jpg")
    try:
        from lab.mmwave77_usb.runner import (
            list_serial_devices,
            select_awr1843_pair,
        )
    except Exception as exc:
        msg = f"mmWave lab_live import failed:\n{type(exc).__name__}: {exc}"
        print(msg, flush=True)
        while True:
            _atomic_write_jpg(live, render_status_jpeg(msg))
            _atomic_write_jpg(live_back, render_status_jpeg(msg))
            time.sleep(2.0)

    try:
        devices = list_serial_devices()
        if args.cli_port and args.data_port:
            msg = (
                f"AWR1843 ports set\nCLI {args.cli_port}\nDATA {args.data_port}\n"
                "Use lab_replay with captured front/back sessions for color plots."
            )
        else:
            pair = select_awr1843_pair(devices)
            msg = (
                f"AWR1843 detected\nCLI {pair.cli_port}\nDATA {pair.data_port}\n"
                "Configure front_session / back_session for dual plots."
            )
            print(f"mmWave lab_live: cli={pair.cli_port} data={pair.data_port}", flush=True)
    except Exception as exc:
        msg = (
            "mmWave lab_live: no AWR1843 pair found\n"
            f"{type(exc).__name__}: {exc}"
        )
        print(msg, flush=True)

    while True:
        _atomic_write_jpg(live, render_status_jpeg("mmWave Front\n" + msg))
        _atomic_write_jpg(live_back, render_status_jpeg("mmWave Back\n" + msg))
        time.sleep(2.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Layer 8 mmWave lab live JPEG publisher")
    p.add_argument("--pipeline", choices=("lab_replay", "lab_live", "status"), default="lab_replay")
    p.add_argument("--live-frame", required=True, help="Front JPEG path (mmwave.live_frame)")
    p.add_argument("--live-frame-back", default="", help="Back JPEG path (mmwave.live_frame_back)")
    p.add_argument("--session", default="", help="Front session dir")
    p.add_argument("--session-back", default="", help="Back session dir")
    p.add_argument("--perception", default="", help="Override front perception.jsonl path")
    p.add_argument("--frames-jsonl", default="", help="Override front frames.jsonl path")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--cli-port", default="")
    p.add_argument("--data-port", default="")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if args.pipeline == "lab_replay":
        raise SystemExit(run_replay(args))
    if args.pipeline == "lab_live":
        raise SystemExit(run_live_uart(args))
    raise SystemExit(run_status(args))


if __name__ == "__main__":
    main()
