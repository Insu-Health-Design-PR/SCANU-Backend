#!/usr/bin/env python3
"""Calibrate two facing AWR1843 radars, then publish fused evidence until stopped."""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab.dual_mmwave77_stereo.live_dashboard import LiveDashboardRenderer
from lab.dual_mmwave77_stereo.live_fusion import LiveFusionEngine, OnlineClutterCalibration
from lab.dual_server_77ghz.live_acquisition import RadarStream, points_from_rows
from lab.dual_server_77ghz.live_contracts import SCHEMA_VERSION, status_payload
from lab.dual_server_77ghz.orchestrate import load_config, select_radar_pairs
from lab.mmwave77_usb.runner import discover_awr1843_pairs, list_serial_devices


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _resolve(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _hardware(config: dict[str, Any]) -> tuple[Any, Any]:
    pairs = discover_awr1843_pairs(list_serial_devices())
    return select_radar_pairs(
        pairs,
        str(config.get("radar_a_usb_location") or ""),
        str(config.get("radar_b_usb_location") or ""),
    )


def preflight(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    pair_a, pair_b = _hardware(config)
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "experimental": True,
        "pair_a": {
            "usb_location": pair_a.usb_location,
            "cli_port": pair_a.cli_port,
            "data_port": pair_a.data_port,
        },
        "pair_b": {
            "usb_location": pair_b.usb_location,
            "cli_port": pair_b.cli_port,
            "data_port": pair_b.data_port,
        },
        "sensor_distance_m": float(config["sensor_distance_m"]),
    }


def run_live(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[3]
    config_path = _resolve(root, str(args.config))
    config = load_config(config_path)
    live_config = config.get("live") if isinstance(config.get("live"), dict) else {}
    publish_dir = Path(args.publish_dir).expanduser().resolve()
    fused_path = Path(args.live_frame).expanduser().resolve() if args.live_frame else publish_dir / "live_mmwave_fused_dashboard.jpg"
    metrics_path = Path(args.metrics_json).expanduser().resolve() if args.metrics_json else publish_dir / "live_metrics.json"
    status_path = publish_dir / "live_status.json"
    manifest_path = publish_dir / "live_manifest.json"
    calibration_seconds = float(args.calibration_seconds or live_config.get("calibration_seconds", 20.0))
    entry_delay_seconds = float(args.entry_delay_seconds if args.entry_delay_seconds is not None else live_config.get("entry_delay_seconds", 10.0))
    render_fps = float(args.fps or live_config.get("render_fps", config.get("render_fps", 2)))
    window_frames = int(args.window_frames or live_config.get("window_frames", 5))
    minimum_calibration_frames = int(live_config.get("minimum_calibration_frames", 40))
    started_utc = datetime.now(timezone.utc).isoformat()
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    renderer = LiveDashboardRenderer(
        sensor_distance_m=float(config["sensor_distance_m"]),
        dpi=int(config.get("render_dpi", 100)),
    )

    def publish_status(state: str, detail: str = "", progress: float = 0.0, calibration_id: str = "") -> None:
        payload = status_payload(
            state,
            detail=detail,
            calibration_progress=progress,
            calibration_id=calibration_id,
            started_utc=started_utc,
        )
        _atomic_json(status_path, payload)
        _atomic_bytes(fused_path, renderer.render_status(state, detail))

    publish_status("PREFLIGHT", "Discovering and validating both AWR1843BOOST radars")
    pair_a, pair_b = _hardware(config)
    if args.preflight_only:
        result = preflight(config_path)
        _atomic_json(status_path, {**result, "state": "PREFLIGHT_OK"})
        print(json.dumps(result, indent=2), flush=True)
        return 0

    profile = _resolve(root, str(config["radar_profile"]))
    stream_a = RadarStream("A", pair_a.cli_port, pair_a.data_port, profile)
    stream_b = RadarStream("B", pair_b.cli_port, pair_b.data_port, profile)
    calibration_a = OnlineClutterCalibration(minimum_frames=minimum_calibration_frames)
    calibration_b = OnlineClutterCalibration(minimum_frames=minimum_calibration_frames)
    last_calibration_frame_a: int | None = None
    last_calibration_frame_b: int | None = None
    calibration_id = "dual-empty-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        stream_a.start()
        if not stream_a.wait_ready():
            raise RuntimeError(f"radar A did not start: {stream_a.error or 'no TLV frames'}")
        stream_b.start()
        if not stream_b.wait_ready():
            raise RuntimeError(f"radar B did not start: {stream_b.error or 'no TLV frames'}")

        calibration_start = time.monotonic()
        last_status_second = -1
        while not stop_requested and time.monotonic() - calibration_start < calibration_seconds:
            elapsed = time.monotonic() - calibration_start
            status_second = int(elapsed)
            if status_second != last_status_second:
                publish_status(
                    "CALIBRATING",
                    "Keep the complete measurement area empty and do not move either radar",
                    elapsed / max(calibration_seconds, 0.1),
                    calibration_id,
                )
                last_status_second = status_second
            row_a = stream_a.snapshot(1)
            row_b = stream_b.snapshot(1)
            if row_a:
                number = int(row_a[0].get("frame_number") or row_a[0].get("frame") or 0)
                if number != last_calibration_frame_a:
                    calibration_a.update(points_from_rows(row_a))
                    last_calibration_frame_a = number
            if row_b:
                number = int(row_b[0].get("frame_number") or row_b[0].get("frame") or 0)
                if number != last_calibration_frame_b:
                    calibration_b.update(points_from_rows(row_b))
                    last_calibration_frame_b = number
            time.sleep(0.05)
        if stop_requested:
            return 0
        calibration_a.finalize()
        calibration_b.finalize()
        if not calibration_a.valid or not calibration_b.valid:
            raise RuntimeError(
                "calibration quality failed: "
                f"A={calibration_a.frame_count} frames, B={calibration_b.frame_count} frames, "
                f"minimum={minimum_calibration_frames}"
            )

        countdown_end = time.monotonic() + max(0.0, entry_delay_seconds)
        while not stop_requested and time.monotonic() < countdown_end:
            remaining = max(0, int(round(countdown_end - time.monotonic())))
            publish_status(
                "ENTRY_COUNTDOWN",
                f"Calibration valid. Enter the measurement area now · live starts in {remaining} s",
                1.0,
                calibration_id,
            )
            time.sleep(min(1.0, max(0.05, countdown_end - time.monotonic())))
        if stop_requested:
            return 0

        engine = LiveFusionEngine(
            sensor_distance_m=float(config["sensor_distance_m"]),
            calibration_a=calibration_a,
            calibration_b=calibration_b,
            calibration_id=calibration_id,
        )
        period = 1.0 / max(0.5, render_fps)
        while not stop_requested:
            cycle_start = time.monotonic()
            if stream_a.error or stream_b.error:
                raise RuntimeError(f"radar stream failure: A={stream_a.error!r}, B={stream_b.error!r}")
            frame = engine.update(
                stream_a.points(window_frames),
                stream_b.points(window_frames),
                timestamp_a_ns=stream_a.latest_timestamp_ns(),
                timestamp_b_ns=stream_b.latest_timestamp_ns(),
                frames_a=stream_a.frames_received,
                frames_b=stream_b.frames_received,
                dropped_frames_a=stream_a.dropped_frames,
                dropped_frames_b=stream_b.dropped_frames,
            )
            payload = frame.to_dict()
            payload["state"] = "LIVE"
            _atomic_json(metrics_path, payload)
            _atomic_json(status_path, status_payload("LIVE", calibration_id=calibration_id, started_utc=started_utc))
            _atomic_bytes(fused_path, renderer.render(frame))
            delay = period - (time.monotonic() - cycle_start)
            if delay > 0:
                time.sleep(delay)
        return 0
    except Exception as exc:
        publish_status("FAULT", str(exc), calibration_id=calibration_id)
        _atomic_json(manifest_path, {
            "schema_version": SCHEMA_VERSION,
            "experimental": True,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise
    finally:
        stream_a.stop()
        stream_b.stop()
        final_state = "STOPPED" if not (stream_a.error or stream_b.error) else "FAULT"
        _atomic_json(manifest_path, {
            "schema_version": SCHEMA_VERSION,
            "experimental": True,
            "ok": final_state == "STOPPED",
            "state": final_state,
            "calibration_id": calibration_id,
            "calibration_frames": {"radar_a": calibration_a.frame_count, "radar_b": calibration_b.frame_count},
            "frames": {"radar_a": stream_a.frames_received, "radar_b": stream_b.frames_received},
            "dropped_frames": {"radar_a": stream_a.dropped_frames, "radar_b": stream_b.dropped_frames},
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "limitations": [
                "USB contains processed TI TLVs, not raw ADC samples",
                "host receipt timestamps are not hardware acquisition timestamps",
                "reflective candidates do not confirm material or classify a weapon",
            ],
        })
        if final_state == "STOPPED":
            publish_status("STOPPED", "Both radar ports were released", 1.0, calibration_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrated dual-AWR1843 fused live runtime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--publish-dir", default="/dev/shm/scanu_mmwave")
    parser.add_argument("--live-frame", default="")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--calibration-seconds", type=float, default=0.0)
    parser.add_argument("--entry-delay-seconds", type=float, default=None)
    parser.add_argument("--window-frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_live(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
