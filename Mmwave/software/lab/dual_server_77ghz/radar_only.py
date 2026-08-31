#!/usr/bin/env python3
"""Interactive two-AWR1843 capture and fusion without opening cameras."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from lab.dual_server_77ghz.orchestrate import (
    _capture_radars,
    _process_calibration,
    _process_participant,
    _profile_path,
    _probe_radar_cli,
    _run_module,
    load_config,
    select_radar_pairs,
)
from lab.mmwave77_usb.runner import discover_awr1843_pairs, list_serial_devices


SCHEMA_VERSION = "scanu_mmwave_radar_only_interactive_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate and capture two facing AWR1843BOOST radars, then create "
            "one fused classic dashboard. No camera is discovered or opened."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--calibration-s", type=float, default=20.0)
    return parser


def _positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def run_capture(config_path: Path, duration_s: float, calibration_s: float) -> Path:
    duration_s = _positive("duration_s", duration_s)
    calibration_s = _positive("calibration_s", calibration_s)
    config = load_config(config_path)
    pairs = discover_awr1843_pairs(list_serial_devices())
    pair_a, pair_b = select_radar_pairs(
        pairs,
        str(config.get("radar_a_usb_location", "")),
        str(config.get("radar_b_usb_location", "")),
    )

    print("Radar-only preflight: probing A and B. Cameras will not be opened.", flush=True)
    _probe_radar_cli(pair_a)
    _probe_radar_cli(pair_b)
    report = {
        "radar_a": asdict(pair_a),
        "radar_b": asdict(pair_b),
        "camera_a": None,
        "camera_b": None,
        "hardware_synchronized": False,
    }

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config["output_root"]).expanduser().resolve()
    run_root = output_root / f"dual_radar_only_{run_tag}"
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=False)
    (run_root / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    profile = _profile_path(config)
    stagger_ms = float(config.get("radar_start_stagger_ms", 50.0))

    print(
        f"STEP 1/3 - Keep the area empty: calibrating for {calibration_s:.0f} seconds.",
        flush=True,
    )
    calibration_a, calibration_b = _capture_radars(
        report,
        profile,
        calibration_s,
        run_root / "calibration",
        stagger_ms,
        logs,
    )
    _process_calibration(calibration_a)
    _process_calibration(calibration_b)
    print("CALIBRATION_READY", flush=True)
    input("Enter the measurement area, then press ENTER to begin capture... ")

    print(
        f"STEP 2/3 - Capturing radar A and radar B for {duration_s:.0f} seconds.",
        flush=True,
    )
    radar_a, radar_b = _capture_radars(
        report,
        profile,
        duration_s,
        run_root / "participant",
        stagger_ms,
        logs,
    )
    _process_participant(radar_a, calibration_a)
    _process_participant(radar_b, calibration_b)

    print("STEP 3/3 - Fusing both radars and rendering videos.", flush=True)
    common = [
        "--session-a", radar_a["session"],
        "--session-b", radar_b["session"],
        "--calibration-a", calibration_a["session"],
        "--calibration-b", calibration_b["session"],
        "--distance-m", str(float(config["sensor_distance_m"])),
        "--clock-offset-b-minus-a-s", "0.0",
        "--window-tolerance-s", str(float(config.get("window_tolerance_s", 0.25))),
    ]
    fusion_path = run_root / "fusion_report.json"
    tracks_path = run_root / "global_tracks.json"
    _run_module(
        "lab.dual_mmwave77_stereo.point_cloud_fusion",
        [*common, "--output", str(fusion_path)],
        "fusion",
        logs,
    )
    _run_module(
        "lab.dual_mmwave77_stereo.global_tracks",
        [*common, "--output", str(tracks_path)],
        "global_tracks",
        logs,
    )

    separate_videos: list[str] = []
    for label, radar, calibration in (
        ("radar_a", radar_a, calibration_a),
        ("radar_b", radar_b, calibration_b),
    ):
        session = Path(radar["session"])
        output = run_root / f"{label}_classic.mp4"
        _run_module(
            "lab.mmwave77_usb.perception_video",
            [
                "--perception", str(session / "perception.jsonl"),
                "--frames", str(session / "frames.jsonl"),
                "--calibration-session", calibration["session"],
                "--output", str(output),
                "--fps", str(int(config.get("render_fps", 2))),
                "--dpi", str(int(config.get("render_dpi", 100))),
            ],
            f"{label}_video",
            logs,
        )
        separate_videos.append(str(output))

    output_prefix = run_root / "dual_radar_only"
    _run_module(
        "lab.dual_mmwave77_stereo.unified_fusion_video",
        [
            *common,
            "--output-prefix", str(output_prefix),
            "--fps", str(int(config.get("render_fps", 2))),
            "--dpi", str(int(config.get("render_dpi", 100))),
            "--classic-human-centric",
            "--only-classic",
        ],
        "fused_classic_video",
        logs,
    )
    fused_video = output_prefix.with_name(
        output_prefix.name + "_classic_fused_dashboard.mp4"
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experimental": True,
        "cameras_opened": False,
        "material_confirmed": False,
        "weapon_classification": False,
        "run_root": str(run_root),
        "calibration": {"radar_a": calibration_a, "radar_b": calibration_b},
        "participant": {"radar_a": radar_a, "radar_b": radar_b},
        "fusion_report": str(fusion_path),
        "global_tracks": str(tracks_path),
        "fused_video": str(fused_video),
        "separate_radar_videos": separate_videos,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "ok": True,
                "run_root": str(run_root),
                "manifest": str(manifest_path),
                "fused_video": str(fused_video),
                "separate_radar_videos": separate_videos,
            },
            indent=2,
        ),
        flush=True,
    )
    return run_root


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    run_capture(args.config, args.duration_s, args.calibration_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
