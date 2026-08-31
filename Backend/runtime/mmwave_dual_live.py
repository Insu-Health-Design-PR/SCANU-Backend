#!/usr/bin/env python3
"""Dual mmWave live/replay publisher for Layer 8 (JPEG + live_mmwave_metrics.json)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from services.mmwave_root import ensure_mmwave_imports, preflight, resolve_mmwave_root

SCHEMA_VERSION = "scanu_mmwave_live_v1"
WINDOW_FRAMES = 10
DEFAULT_BAUD = 921_600


def _ensure_backend_lab_imports() -> None:
    """JPEG render helpers live under Backend/lab (not always in MMWAVE_ROOT)."""
    backend_root = Path(__file__).resolve().parents[1]
    root_s = str(backend_root)
    # Prefer Backend/lab over MMWAVE_ROOT/software/lab (insert at front).
    while root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
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
                x, y, z = p.get("x", p.get("x_m")), p.get("y", p.get("y_m")), p.get("z", p.get("z_m"))
                if x is None or y is None or z is None:
                    continue
                snr = float(p.get("snr") or 0.0)
                pts.append([float(x), float(y), float(z), snr])
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                snr = float(p[3]) if len(p) > 3 else 0.0
                pts.append([float(p[0]), float(p[1]), float(p[2]), snr])
    if not pts:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _cluster_points(points: np.ndarray, radius_m: float, min_pts: int) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    remaining = points.copy()
    clusters: list[np.ndarray] = []
    while len(remaining) >= min_pts:
        seed = remaining[0]
        dists = np.linalg.norm(remaining[:, :3] - seed[:3], axis=1)
        mask = dists <= radius_m
        if int(mask.sum()) < min_pts:
            remaining = remaining[1:]
            continue
        cluster = remaining[mask]
        clusters.append(cluster)
        remaining = remaining[~mask]
    return clusters


def _live_perceive_window(points: np.ndarray) -> dict[str, Any]:
    """Lightweight rolling-window perception (experimental, not weapon classification)."""
    if len(points) == 0:
        return {
            "screening_state": "insufficient_signal",
            "track": None,
            "points": [],
            "anomalies": [],
        }
    near, far = 0.5, 5.0
    valid = points[(points[:, 1] >= near) & (points[:, 1] <= far)]
    if len(valid) == 0:
        return {
            "screening_state": "background",
            "track": None,
            "points": [],
            "anomalies": [],
        }
    clusters = _cluster_points(valid, radius_m=0.42, min_pts=8)
    if not clusters:
        return {
            "screening_state": "background",
            "track": None,
            "points": _serialize_points(valid[:120]),
            "anomalies": [],
        }
    person = max(clusters, key=len)
    center = person[:, :3].mean(axis=0)
    extent = person[:, :3].max(axis=0) - person[:, :3].min(axis=0)
    half = np.maximum(0.5 * extent + np.array([0.25, 0.25, 0.3]), [0.3, 0.3, 0.35])
    body_mask = np.all(np.abs(valid[:, :3] - center) <= half, axis=1)
    anomalies: list[dict[str, Any]] = []
    off_body = valid[~body_mask]
    if len(off_body):
        for cl in _cluster_points(off_body, radius_m=0.18, min_pts=2):
            rel = cl[:, :3].mean(axis=0) - center
            snr_med = float(np.median(cl[:, 3])) if cl.shape[1] > 3 else 0.0
            score = min(1.0, 0.45 + 0.02 * snr_med)
            if score >= 0.68:
                anomalies.append(
                    {
                        "centroid_m": [float(x) for x in cl[:, :3].mean(axis=0)],
                        "position_m": [float(x) for x in center + rel],
                        "score": round(score, 3),
                        "kind": "reflective",
                        "persistent": False,
                    }
                )
    screening = "person"
    if anomalies:
        screening = "suspicious_metal"
    track = {
        "centroid_m": [float(x) for x in center],
        "position_m": [float(x) for x in center],
        "velocity_mps": [0.0, 0.0, 0.0],
        "observed_extent_m": [float(x) for x in extent],
        "confidence": round(min(1.0, len(person) / 40.0), 3),
    }
    return {
        "screening_state": screening,
        "track": track,
        "points": _serialize_points(valid[:160]),
        "anomalies": anomalies,
    }


def _serialize_points(arr: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in arr:
        x, y, z = float(row[0]), float(row[1]), float(row[2])
        snr = float(row[3]) if len(row) > 3 else 0.0
        rng = math.sqrt(x * x + y * y + z * z)
        out.append({"x": x, "y": y, "z": z, "snr": snr, "range": round(rng, 3)})
    return out


@dataclass
class RadarSideReader:
    title: str
    cli_port: str
    data_port: str
    config_path: str
    frames: deque = field(default_factory=lambda: deque(maxlen=WINDOW_FRAMES * 3))
    lock: threading.Lock = field(default_factory=threading.Lock)
    error: str = ""
    running: bool = False

    def start(self) -> None:
        t = threading.Thread(target=self._run, name=f"mmwave-{self.title}", daemon=True)
        t.start()

    def _run(self) -> None:
        try:
            import serial
        except ImportError:
            self.error = "pyserial not installed"
            return
        root = ensure_mmwave_imports()
        from lab.mmwave77_usb.runner import TiTlvFramer, _configure_ti_sensor, _decode_frame
        from layer1_sensor_hub.radar.tlv_parser import TLVParser

        cfg = Path(self.config_path).expanduser() if self.config_path else None
        if cfg and not cfg.is_absolute():
            cfg = (root / cfg).resolve()
        try:
            if cfg and cfg.is_file() and self.cli_port:
                _configure_ti_sensor(
                    self.cli_port,
                    cfg,
                    validate_awr1843=True,
                    audit_path=None,
                )
        except Exception as exc:
            self.error = f"config failed: {exc}"
            return

        framer = TiTlvFramer()
        parser = TLVParser()
        self.running = True
        try:
            with serial.Serial(self.data_port, baudrate=DEFAULT_BAUD, timeout=0.05) as stream:
                stream.reset_input_buffer()
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        continue
                    for frame in framer.feed(chunk):
                        row, _ = _decode_frame(frame, parser)
                        if row.get("parse_ok"):
                            with self.lock:
                                self.frames.append(row)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.running = False

    def window_points(self) -> np.ndarray:
        with self.lock:
            rows = list(self.frames)[-WINDOW_FRAMES:]
        return _points_from_frames(rows, 0, 10_000_000)


def _side_payload(reader: RadarSideReader | None, replay: dict[str, Any] | None) -> dict[str, Any]:
    if replay is not None:
        pts = np.asarray(replay.get("_points_xyz") or np.zeros((0, 4)), dtype=np.float32)
        perceived = {
            "screening_state": str(replay.get("screening_state") or "background"),
            "track": replay.get("track"),
            "points": _serialize_points(pts) if len(pts) else [],
            "anomalies": list(replay.get("anomalies") or []),
        }
        return perceived
    if reader is None:
        return {"screening_state": "insufficient_signal", "track": None, "points": [], "anomalies": []}
    pts = reader.window_points()
    return _live_perceive_window(pts)


def _render_side_jpeg(side: dict[str, Any], title: str) -> bytes:
    from lab.mmwave77_usb.live_perception_frame import render_perception_jpeg, render_status_jpeg

    pts_list = side.get("points") or []
    if not pts_list:
        return render_status_jpeg(f"{title}\nNo points in window")
    xyz = np.array([[p["x"], p["y"], p["z"]] for p in pts_list], dtype=np.float32)
    return render_perception_jpeg(
        points_xyz=xyz,
        track=side.get("track"),
        anomalies=list(side.get("anomalies") or []),
        screening_state=str(side.get("screening_state") or "background"),
        title=title,
    )


def _maybe_fuse(front: dict[str, Any], back: dict[str, Any], sensor_distance_m: float) -> dict[str, Any]:
    fused: dict[str, Any] = {"global_person_count": 0, "fused_centroid_m": None, "active_tracks": []}
    tf = front.get("track") if isinstance(front.get("track"), dict) else None
    tb = back.get("track") if isinstance(back.get("track"), dict) else None
    if not tf or not tb:
        return fused
    try:
        cf = np.asarray(tf.get("centroid_m") or tf.get("position_m"), dtype=np.float32)
        cb = np.asarray(tb.get("centroid_m") or tb.get("position_m"), dtype=np.float32)
        if cf.shape != (3,) or cb.shape != (3,):
            return fused
        cb_a = np.array([-cb[0], sensor_distance_m - cb[1], cb[2]], dtype=np.float32)
        centroid = (0.5 * (cf + cb_a)).tolist()
        fused["global_person_count"] = 1
        fused["fused_centroid_m"] = [float(x) for x in centroid]
        fused["active_tracks"] = [{"centroid_m": fused["fused_centroid_m"], "global_id": "G001"}]
    except Exception:
        pass
    return fused


def publish_cycle(
    *,
    live_front: Path,
    live_back: Path,
    metrics_path: Path,
    metrics_shm: Path | None,
    front_side: dict[str, Any],
    back_side: dict[str, Any],
    sensor_distance_m: float,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experimental": True,
        "ts_monotonic_ns": time.monotonic_ns(),
        "sensor_distance_m": float(sensor_distance_m),
        "front": front_side,
        "back": back_side,
        "fused": _maybe_fuse(front_side, back_side, sensor_distance_m),
    }
    _atomic_write_jpg = lambda p, b: _atomic_write_bytes(p, b)
    _atomic_write_jpg(live_front, _render_side_jpeg(front_side, "mmWave Front · live"))
    _atomic_write_jpg(live_back, _render_side_jpeg(back_side, "mmWave Back · live"))
    _atomic_write_json(metrics_path, payload)
    if metrics_shm is not None:
        _atomic_write_json(metrics_shm, payload)


def run_dual_replay(args: argparse.Namespace) -> int:
    from lab.mmwave77_usb.live_perception_frame import render_status_jpeg

    live = Path(args.live_frame).expanduser().resolve()
    live_back = Path(args.live_frame_back or "").expanduser().resolve() if args.live_frame_back else live.with_name("live_mmwave_back.jpg")
    metrics_path = Path(args.metrics_json).expanduser().resolve()
    metrics_shm = Path(args.metrics_shm).expanduser().resolve() if args.metrics_shm else None
    sensor_distance_m = float(args.sensor_distance_m or 3.6576)

    def _replay_side(session: str, perception: str, frames: str) -> dict[str, Any] | None:
        if not session:
            return None
        sp = Path(session).expanduser().resolve()
        pp = Path(perception).expanduser().resolve() if perception else sp / "perception.jsonl"
        fp = Path(frames).expanduser().resolve() if frames else sp / "frames.jsonl"
        if not pp.is_file() or not fp.is_file():
            return None
        rows = _load_jsonl(pp)
        frs = _load_jsonl(fp)
        if not rows:
            return None
        return {"rows": rows, "frames": frs, "idx": 0}

    front = _replay_side(args.session or "", args.perception or "", args.frames_jsonl or "")
    back = _replay_side(args.session_back or "", "", "")

    if front is None and back is None:
        msg = "dual_replay idle — set front_session / back_session"
        while True:
            _atomic_write_bytes(live, render_status_jpeg(msg))
            _atomic_write_bytes(live_back, render_status_jpeg(msg))
            time.sleep(2.0)

    idx = 0
    period = 1.0 / max(0.5, float(args.fps))
    while True:
        front_side = {"screening_state": "insufficient_signal", "track": None, "points": [], "anomalies": []}
        back_side = dict(front_side)
        if front is not None:
            row = front["rows"][idx % len(front["rows"])]
            pts = _points_from_frames(front["frames"], int(row.get("frame_start", 0)), int(row.get("frame_end", 0)))
            front_side = {
                "screening_state": str(row.get("screening_state") or "background"),
                "track": row.get("track"),
                "points": _serialize_points(pts),
                "anomalies": list(row.get("anomalies") or []),
                "_points_xyz": pts,
            }
        if back is not None:
            row = back["rows"][idx % len(back["rows"])]
            pts = _points_from_frames(back["frames"], int(row.get("frame_start", 0)), int(row.get("frame_end", 0)))
            back_side = {
                "screening_state": str(row.get("screening_state") or "background"),
                "track": row.get("track"),
                "points": _serialize_points(pts),
                "anomalies": list(row.get("anomalies") or []),
                "_points_xyz": pts,
            }
        publish_cycle(
            live_front=live,
            live_back=live_back,
            metrics_path=metrics_path,
            metrics_shm=metrics_shm,
            front_side={k: v for k, v in front_side.items() if not k.startswith("_")},
            back_side={k: v for k, v in back_side.items() if not k.startswith("_")},
            sensor_distance_m=sensor_distance_m,
        )
        idx += 1
        time.sleep(period)


def run_dual_live(args: argparse.Namespace) -> int:
    from lab.mmwave77_usb.live_perception_frame import render_status_jpeg

    live = Path(args.live_frame).expanduser().resolve()
    live_back = Path(args.live_frame_back or "").expanduser().resolve() if args.live_frame_back else live.with_name("live_mmwave_back.jpg")
    metrics_path = Path(args.metrics_json).expanduser().resolve()
    metrics_shm = Path(args.metrics_shm).expanduser().resolve() if args.metrics_shm else Path("/dev/shm/scanu_mmwave_live_metrics.json")

    pf = preflight(settings={
        "mmwave_root": {"path": args.mmwave_root or ""},
        "mmwave": {
            "radar_a_usb_location": args.radar_a_usb_location or "",
            "radar_b_usb_location": args.radar_b_usb_location or "",
        },
    })
    if not pf.get("ok"):
        msg = f"MMWAVE_ROOT invalid\n{pf.get('missing')}"
        while True:
            _atomic_write_bytes(live, render_status_jpeg(msg))
            _atomic_write_bytes(live_back, render_status_jpeg(msg))
            time.sleep(2.0)

    pair_a = pf.get("pair_a") or {}
    pair_b = pf.get("pair_b") or {}
    front_cli = args.front_cli_port or args.cli_port or pair_a.get("cli_port") or ""
    front_data = args.front_data_port or args.data_port or pair_a.get("data_port") or ""
    back_cli = args.back_cli_port or pair_b.get("cli_port") or ""
    back_data = args.back_data_port or pair_b.get("data_port") or ""
    config_path = args.radar_config or "software/lab/mmwave77_usb/configs/awr1843boost_sdk_3_4_profile_3d.cfg"

    readers: list[RadarSideReader] = []
    if front_data:
        readers.append(RadarSideReader("front", front_cli, front_data, config_path))
        readers[-1].start()
    if back_data:
        readers.append(RadarSideReader("back", back_cli, back_data, config_path))
        readers[-1].start()

    if not readers:
        msg = "dual_live: no radar data ports resolved — run preflight"
        while True:
            _atomic_write_bytes(live, render_status_jpeg(msg))
            _atomic_write_bytes(live_back, render_status_jpeg(msg))
            time.sleep(2.0)

    front_reader = readers[0] if readers and readers[0].title == "front" else None
    back_reader = next((r for r in readers if r.title == "back"), None)
    if front_reader is None and len(readers) == 1:
        front_reader = readers[0]

    period = 1.0 / max(0.5, float(args.fps))
    sensor_distance_m = float(args.sensor_distance_m or pf.get("sensor_distance_m") or 3.6576)
    while True:
        front_side = _side_payload(front_reader, None)
        back_side = _side_payload(back_reader, None)
        publish_cycle(
            live_front=live,
            live_back=live_back,
            metrics_path=metrics_path,
            metrics_shm=metrics_shm,
            front_side=front_side,
            back_side=back_side,
            sensor_distance_m=sensor_distance_m,
        )
        time.sleep(period)


def run_status(args: argparse.Namespace) -> int:
    from lab.mmwave77_usb.live_perception_frame import render_status_jpeg

    live = Path(args.live_frame).expanduser().resolve()
    live_back = Path(args.live_frame_back or "").expanduser().resolve() if args.live_frame_back else live.with_name("live_mmwave_back.jpg")
    msg = "mmWave dual idle\nSet pipeline=dual_live or dual_replay"
    while True:
        _atomic_write_bytes(live, render_status_jpeg("Front\n" + msg))
        _atomic_write_bytes(live_back, render_status_jpeg("Back\n" + msg))
        time.sleep(2.0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dual mmWave publisher for Layer 8")
    p.add_argument("--pipeline", default="dual_replay", choices=("dual_live", "dual_replay", "status", "lab_replay", "lab_live"))
    p.add_argument("--live-frame", required=True)
    p.add_argument("--live-frame-back", default="")
    p.add_argument("--metrics-json", default="layer8_ui/artifacts/live_mmwave_metrics.json")
    p.add_argument("--metrics-shm", default="/dev/shm/scanu_mmwave_live_metrics.json")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--session", default="")
    p.add_argument("--session-back", default="")
    p.add_argument("--perception", default="")
    p.add_argument("--frames-jsonl", default="")
    p.add_argument("--cli-port", default="")
    p.add_argument("--data-port", default="")
    p.add_argument("--front-cli-port", default="")
    p.add_argument("--front-data-port", default="")
    p.add_argument("--back-cli-port", default="")
    p.add_argument("--back-data-port", default="")
    p.add_argument("--mmwave-root", default="")
    p.add_argument("--server-config", default="")
    p.add_argument("--radar-config", default="")
    p.add_argument("--radar-a-usb-location", default="")
    p.add_argument("--radar-b-usb-location", default="")
    p.add_argument("--sensor-distance-m", type=float, default=0.0)
    args = p.parse_args(argv)

    root = resolve_mmwave_root(settings_root=args.mmwave_root or "")
    ensure_mmwave_imports(root)
    if args.pipeline in ("dual_live", "lab_live"):
        # Mmwave is the single owner of calibration, acquisition, fusion and
        # rendering.  The Backend process remains only a lifecycle adapter.
        server_config = Path(args.server_config).expanduser() if args.server_config else root / "configs/server_local.json"
        if not server_config.is_absolute():
            server_config = (root / server_config).resolve()
        live_frame = Path(args.live_frame).expanduser().resolve()
        metrics_json = Path(args.metrics_json).expanduser().resolve()
        command = [
            sys.executable,
            "-m",
            "lab.dual_server_77ghz.live_runtime",
            "--config",
            str(server_config),
            "--publish-dir",
            str(live_frame.parent),
            "--live-frame",
            str(live_frame),
            "--metrics-json",
            str(metrics_json),
            "--fps",
            str(float(args.fps)),
        ]
        env = os.environ.copy()
        mmwave_software = str(root / "software")
        env["PYTHONPATH"] = mmwave_software + os.pathsep + env.get("PYTHONPATH", "")
        os.chdir(root)
        os.execvpe(command[0], command, env)
        raise AssertionError("execvpe returned unexpectedly")
    # Backend/lab has live_perception_frame; MMWAVE_ROOT/software/lab may not.
    _ensure_backend_lab_imports()
    # Legacy aliases
    if args.pipeline in ("lab_replay",):
        args.pipeline = "dual_replay"
    if args.pipeline in ("lab_live",):
        args.pipeline = "dual_live"

    if args.pipeline == "dual_replay":
        run_dual_replay(args)
        return 0
    if args.pipeline == "dual_live":
        run_dual_live(args)
        return 0
    run_status(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
