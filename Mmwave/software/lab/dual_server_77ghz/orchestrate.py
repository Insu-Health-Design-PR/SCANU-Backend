#!/usr/bin/env python3
"""One-host two-AWR1843/two-camera experimental capture orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from lab.dual_server_77ghz import SCHEMA_VERSION
from lab.mmwave77_usb.runner import (
    Awr1843PortPair,
    discover_awr1843_pairs,
    list_serial_devices,
)


CONFIG_SCHEMA = "scanu_lab_dual_server_77ghz_config_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"unexpected configuration schema: {config.get('schema_version')!r}")
    for key in (
        "output_root", "sensor_distance_m", "radar_profile", "camera_a",
        "camera_b", "calibration_seconds", "entry_delay_seconds",
        "capture_seconds",
    ):
        if key not in config:
            raise ValueError(f"configuration is missing {key!r}")
    if float(config["sensor_distance_m"]) <= 0:
        raise ValueError("sensor_distance_m must be greater than zero")
    for key in ("calibration_seconds", "capture_seconds"):
        if float(config[key]) <= 0:
            raise ValueError(f"{key} must be greater than zero")
    if float(config["entry_delay_seconds"]) < 0:
        raise ValueError("entry_delay_seconds cannot be negative")
    config["_config_path"] = str(path)
    return config


def _profile_path(config: dict[str, Any]) -> Path:
    value = Path(str(config["radar_profile"])).expanduser()
    path = value if value.is_absolute() else REPO_ROOT / value
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"radar profile does not exist: {path}")
    return path


def select_radar_pairs(
    pairs: Sequence[Awr1843PortPair],
    location_a: str = "",
    location_b: str = "",
) -> tuple[Awr1843PortPair, Awr1843PortPair]:
    if len(pairs) < 2:
        raise RuntimeError(f"two complete AWR1843 XDS110 pairs are required; found {len(pairs)}")
    by_location = {pair.usb_location: pair for pair in pairs}
    if location_a:
        if location_a not in by_location:
            raise RuntimeError(f"radar A location {location_a!r} is absent; available: {sorted(by_location)}")
        pair_a = by_location[location_a]
    else:
        pair_a = sorted(pairs, key=lambda item: item.usb_location)[0]
    if location_b:
        if location_b not in by_location:
            raise RuntimeError(f"radar B location {location_b!r} is absent; available: {sorted(by_location)}")
        pair_b = by_location[location_b]
    else:
        pair_b = next(
            item for item in sorted(pairs, key=lambda value: value.usb_location)
            if item.usb_location != pair_a.usb_location
        )
    if pair_a.usb_location == pair_b.usb_location:
        raise RuntimeError("radar A and B resolve to the same USB device")
    return pair_a, pair_b


def _deduplicate_devices(paths: Sequence[Path]) -> list[str]:
    selected: list[str] = []
    real_devices: set[str] = set()
    for path in paths:
        try:
            real = str(path.resolve(strict=True))
        except OSError:
            continue
        if real in real_devices:
            continue
        real_devices.add(real)
        selected.append(str(path))
    return selected


def discover_camera_devices(dev_root: Path = Path("/dev")) -> list[str]:
    by_id = sorted((dev_root / "v4l" / "by-id").glob("*-video-index0"))
    devices = _deduplicate_devices(by_id)
    # A physical UVC camera commonly exposes both video-index0 (capture) and
    # video-index1 (metadata/auxiliary).  Once stable index-zero links exist,
    # never append raw /dev/video* nodes: doing so can count one camera twice.
    if devices:
        return devices
    by_path = sorted((dev_root / "v4l" / "by-path").glob("*-video-index0"))
    devices = _deduplicate_devices(by_path)
    if devices:
        return devices
    fallback = sorted(
        dev_root.glob("video*"),
        key=lambda path: int(path.name.removeprefix("video") or 0),
    )
    return _deduplicate_devices(fallback)


def select_camera_devices(
    candidates: Sequence[str], device_a: str, device_b: str, *, camera_b_enabled: bool = True,
) -> tuple[str, str | None]:
    available = list(candidates)
    if device_a != "auto" and device_a not in available:
        available.insert(0, device_a)
    if camera_b_enabled and device_b != "auto" and device_b not in available:
        available.append(device_b)
    camera_a = device_a if device_a != "auto" else (available[0] if available else "")
    camera_b = (
        device_b if device_b != "auto" else next(
            (item for item in available if os.path.realpath(item) != os.path.realpath(camera_a)),
            "",
        )
        if camera_b_enabled else None
    )
    if not camera_a or (camera_b_enabled and not camera_b):
        required = "two distinct camera capture devices" if camera_b_enabled else "one camera capture device"
        raise RuntimeError(f"{required} is required; found {available}")
    if camera_b is not None and os.path.realpath(camera_a) == os.path.realpath(camera_b):
        raise RuntimeError("camera A and B resolve to the same video device")
    return camera_a, camera_b


def _camera_formats(device: str) -> str:
    command = ["v4l2-ctl", "--list-formats-ext", "--device", device]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=8)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    return result.stdout


def camera_settings(spec: dict[str, Any], device: str) -> dict[str, Any]:
    result = dict(spec)
    result["device"] = device
    backend = str(result.get("backend", "auto"))
    input_format = str(result.get("input_format", "auto"))
    formats = _camera_formats(device) if backend == "auto" or input_format == "auto" else ""
    upper = formats.upper()
    if input_format == "auto":
        if "H264" in upper or "H.264" in upper:
            input_format = "h264"
        elif "MJPG" in upper or "MOTION-JPEG" in upper:
            input_format = "mjpeg"
        else:
            input_format = "mjpeg"
    if backend == "auto":
        backend = "ffmpeg-copy" if formats and ("H264" in upper or "MJPG" in upper or "MOTION-JPEG" in upper) else "opencv"
    if backend not in ("opencv", "ffmpeg-copy"):
        raise ValueError(f"unsupported camera backend: {backend}")
    result["backend"] = backend
    result["input_format"] = input_format
    return result


def _probe_radar_cli(pair: Awr1843PortPair) -> str:
    from layer1_sensor_hub.radar.radar_cli import RadarCliConfig, send_cli_commands

    responses = send_cli_commands(
        RadarCliConfig(port=pair.cli_port, baud=115_200, timeout_s=1.0, command_delay_s=0.1),
        ["version"],
    )
    response = responses[0] if responses else ""
    if "xwr18" not in response.lower():
        raise RuntimeError(f"AWR1843 CLI probe failed on {pair.cli_port}: {response[-200:] or 'empty response'}")
    return response


def _probe_camera(settings: dict[str, Any]) -> None:
    if settings["backend"] == "ffmpeg-copy":
        device = str(settings["device"])
        ffmpeg_device = device if device.startswith("/") else f"/dev/video{device}"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "v4l2",
                "-input_format", str(settings["input_format"]), "-video_size",
                f"{int(settings['width'])}x{int(settings['height'])}", "-framerate",
                str(float(settings["fps"])), "-i", ffmpeg_device, "-frames:v", "1",
                "-f", "null", "-",
            ],
            check=True,
            timeout=15,
        )
    else:
        from lab.mmwave77_usb.camera_capture import probe

        probe(
            str(settings["device"]), int(settings["width"]),
            int(settings["height"]), float(settings["fps"]),
        )


def preflight(config: dict[str, Any], *, open_devices: bool = True) -> dict[str, Any]:
    _profile_path(config)
    all_pairs = discover_awr1843_pairs(list_serial_devices())
    pair_a, pair_b = select_radar_pairs(
        all_pairs,
        str(config.get("radar_a_usb_location", "")),
        str(config.get("radar_b_usb_location", "")),
    )
    camera_b_enabled = bool(config["camera_b"].get("enabled", True))
    camera_a_path, camera_b_path = select_camera_devices(
        discover_camera_devices(),
        str(config["camera_a"].get("device", "auto")),
        str(config["camera_b"].get("device", "auto")),
        camera_b_enabled=camera_b_enabled,
    )
    camera_a = camera_settings(config["camera_a"], camera_a_path)
    camera_b = camera_settings(config["camera_b"], camera_b_path) if camera_b_path else None
    if open_devices:
        _probe_radar_cli(pair_a)
        _probe_radar_cli(pair_b)
        _probe_camera(camera_a)
        if camera_b is not None:
            _probe_camera(camera_b)
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "host": os.uname().nodename,
        "radar_a": asdict(pair_a),
        "radar_b": asdict(pair_b),
        "camera_a": camera_a,
        "camera_b": camera_b,
        "single_host_clock": True,
        "hardware_synchronized": False,
    }


def _radar_command(pair: dict[str, Any], profile: Path, duration_s: float, output_root: Path) -> list[str]:
    return [
        sys.executable, "-m", "lab.mmwave77_usb.runner", "capture",
        "--data-port", str(pair["data_port"]), "--cli-port", str(pair["cli_port"]),
        "--config", str(profile), "--protocol", "ti-tlv", "--duration-s",
        str(duration_s), "--sensor-model", "Texas Instruments AWR1843BOOST",
        "--output-root", str(output_root),
    ]


def _camera_command(settings: dict[str, Any], duration_s: float, output_dir: Path) -> list[str]:
    command = [
        sys.executable, "-m", "lab.mmwave77_usb.camera_capture", "record",
        "--device", str(settings["device"]), "--width", str(int(settings["width"])),
        "--height", str(int(settings["height"])), "--fps", str(float(settings["fps"])),
        "--duration-s", str(duration_s), "--output-dir", str(output_dir),
        "--backend", str(settings["backend"]),
    ]
    if settings["backend"] == "ffmpeg-copy":
        command.extend(["--input-format", str(settings["input_format"])])
    return command


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT), str(REPO_ROOT / "software"), env.get("PYTHONPATH", ""))
    )
    return env


def _start(command: Sequence[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        list(command), cwd=REPO_ROOT, env=_child_env(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _finish(
    process: subprocess.Popen[str], label: str, log_dir: Path, *, require_ok: bool = True
) -> dict[str, Any]:
    stdout, stderr = process.communicate()
    (log_dir / f"{label}.stdout.txt").write_text(stdout)
    (log_dir / f"{label}.stderr.txt").write_text(stderr)
    if process.returncode:
        raise RuntimeError(f"{label} failed ({process.returncode}): {stderr[-500:] or stdout[-500:]}")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} returned invalid JSON: {stdout[-500:]}") from exc
    if require_ok and not result.get("ok"):
        raise RuntimeError(f"{label} reported failure: {result}")
    return result


def _capture_radars(
    preflight_report: dict[str, Any], profile: Path, duration_s: float,
    output_root: Path, stagger_ms: float, log_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    process_a = _start(_radar_command(preflight_report["radar_a"], profile, duration_s, output_root / "radar_a"))
    time.sleep(max(0.0, stagger_ms) / 1000.0)
    process_b = _start(_radar_command(preflight_report["radar_b"], profile, duration_s, output_root / "radar_b"))
    result_a = _finish(process_a, "radar_a", log_dir)
    result_b = _finish(process_b, "radar_b", log_dir)
    return result_a, result_b


def _process_calibration(result: dict[str, Any]) -> None:
    from lab.mmwave77_usb.background import build_empty_room_baseline
    from lab.mmwave77_usb.cube import CubeSpec, build_session_cube

    session = Path(result["session"])
    build_session_cube(session, None, CubeSpec())
    build_empty_room_baseline(session, condition="empty_room")


def _process_participant(result: dict[str, Any], calibration: dict[str, Any]) -> None:
    from lab.mmwave77_usb.cube import CubeSpec, build_session_cube
    from lab.mmwave77_usb.perception import run_perception

    session = Path(result["session"])
    build_session_cube(session, None, CubeSpec())
    run_perception(session, calibration_session=Path(calibration["session"]))


def _run_module(module: str, arguments: Sequence[str], label: str, log_dir: Path) -> None:
    process = _start([sys.executable, "-m", module, *arguments])
    _finish(process, label, log_dir, require_ok=False)


def _countdown(seconds: float) -> None:
    remaining = int(round(seconds))
    while remaining > 0:
        print(f"Participant entry begins in {remaining:02d} s", flush=True)
        time.sleep(1.0)
        remaining -= 1


def capture(
    config: dict[str, Any], *, calibration_s: float, entry_delay_s: float,
    duration_s: float, skip_render: bool,
) -> dict[str, Any]:
    report = preflight(config, open_devices=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(config["output_root"]).expanduser().resolve() / f"dual_server_{run_tag}"
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=False)
    (run_root / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    profile = _profile_path(config)
    stagger_ms = float(config.get("radar_start_stagger_ms", 50.0))

    print(f"STEP 1/3 — empty-room calibration for {calibration_s:.1f} seconds", flush=True)
    calibration_a, calibration_b = _capture_radars(
        report, profile, calibration_s, run_root / "calibration", stagger_ms, logs,
    )
    _process_calibration(calibration_a)
    _process_calibration(calibration_b)

    print("Calibration complete. Do not move the radars or large room objects.", flush=True)
    _countdown(entry_delay_s)
    print(f"STEP 2/3 — recording two radars and two cameras for {duration_s:.1f} seconds", flush=True)
    camera_lead_s = float(config.get("camera_lead_s", 1.5))
    camera_duration_s = duration_s + 2.0 * camera_lead_s
    camera_a_process = _start(_camera_command(report["camera_a"], camera_duration_s, run_root / "participant" / "camera_a"))
    camera_b_process = (
        _start(_camera_command(report["camera_b"], camera_duration_s, run_root / "participant" / "camera_b"))
        if report["camera_b"] is not None else None
    )
    time.sleep(camera_lead_s)
    radar_a, radar_b = _capture_radars(
        report, profile, duration_s, run_root / "participant", stagger_ms, logs,
    )
    # The reusable camera capture contract predates the radar runner's
    # ``{"ok": true}`` convention. A successful camera process returns the
    # artifact paths directly, so do not require the radar-specific flag.
    camera_a = _finish(camera_a_process, "camera_a", logs, require_ok=False)
    camera_b = (
        _finish(camera_b_process, "camera_b", logs, require_ok=False)
        if camera_b_process is not None else None
    )
    for label, camera in (("camera_a", camera_a), ("camera_b", camera_b)):
        if camera is None:
            continue
        missing = [key for key in ("video", "frames", "metadata") if not camera.get(key)]
        if missing:
            raise RuntimeError(f"{label} omitted required artifacts: {', '.join(missing)}")
    _process_participant(radar_a, calibration_a)
    _process_participant(radar_b, calibration_b)

    print("STEP 3/3 — fusing point clouds and building global tracks", flush=True)
    fusion_path = run_root / "fusion_report.json"
    tracks_path = run_root / "global_tracks.json"
    common = [
        "--session-a", radar_a["session"], "--session-b", radar_b["session"],
        "--calibration-a", calibration_a["session"], "--calibration-b", calibration_b["session"],
        "--distance-m", str(float(config["sensor_distance_m"])),
        "--clock-offset-b-minus-a-s", "0.0", "--window-tolerance-s",
        str(float(config.get("window_tolerance_s", 0.25))),
    ]
    _run_module(
        "lab.dual_mmwave77_stereo.point_cloud_fusion",
        [*common, "--output", str(fusion_path)], "fusion", logs,
    )
    _run_module(
        "lab.dual_mmwave77_stereo.global_tracks",
        [*common, "--output", str(tracks_path)], "global_tracks", logs,
    )
    triptych = None
    if not skip_render:
        prefix = run_root / "dual_server"
        _run_module(
            "lab.dual_mmwave77_stereo.unified_fusion_video",
            [
                *common,
                "--camera-a", camera_a["video"],
                "--camera-a-frames", camera_a["frames"],
                "--output-prefix", str(prefix), "--fps", str(int(config.get("render_fps", 2))),
                "--dpi", str(int(config.get("render_dpi", 100))), "--only-triptych",
                "--human-focus",
                *(
                    ["--classic-human-centric"]
                    if bool(config.get("classic_human_centric_dashboard", True))
                    else []
                ),
                "--temporal-history-windows",
                str(int(config.get("human_focus_temporal_history_windows", 8))),
                "--temporal-voxel-m",
                str(float(config.get("human_focus_temporal_voxel_m", 0.08))),
                *( ["--camera-b", camera_b["video"], "--camera-b-frames", camera_b["frames"]]
                   if camera_b is not None else [] ),
            ],
            "triptych", logs,
        )
        candidates = sorted(run_root.glob("*triptych*.mp4"))
        triptych = str(candidates[-1]) if candidates else None
        classic_candidates = sorted(run_root.glob("*classic_fused_dashboard*.mp4"))
        classic_dashboard = (
            str(classic_candidates[-1]) if classic_candidates else None
        )
    else:
        classic_dashboard = None

    artifacts = [fusion_path, tracks_path, Path(camera_a["video"])]
    if camera_b is not None:
        artifacts.append(Path(camera_b["video"]))
    if triptych:
        artifacts.append(Path(triptych))
    if classic_dashboard:
        artifacts.append(Path(classic_dashboard))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experimental": True,
        "canonical_training_compatible": False,
        "material_confirmed": False,
        "weapon_classification": False,
        "run_root": str(run_root),
        "preflight": report,
        "calibration": {"radar_a": calibration_a, "radar_b": calibration_b},
        "participant": {"radar_a": radar_a, "radar_b": radar_b, "camera_a": camera_a, "camera_b": camera_b},
        "fusion_report": str(fusion_path),
        "global_tracks": str(tracks_path),
        "triptych_video": triptych,
        "classic_fused_dashboard_video": classic_dashboard,
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path)} for path in artifacts if path.is_file()
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "single-host USB timestamps are not hardware chirp/exposure synchronization",
            "fusion is geometric post-CFAR fusion, not coherent ADC/IQ fusion",
            "reflectivity does not confirm material and does not classify a weapon",
        ],
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "ok": True,
        "run_root": str(run_root),
        "manifest": str(manifest_path),
        "triptych_video": triptych,
        "classic_fused_dashboard_video": classic_dashboard,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Server-local two-AWR1843/two-camera experimental lab")
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--enumerate-only", action="store_true")
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--calibration-s", type=float)
    capture_parser.add_argument("--entry-delay-s", type=float)
    capture_parser.add_argument("--duration-s", type=float)
    capture_parser.add_argument("--skip-render", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        config = load_config(args.config)
        if args.command == "preflight":
            result = preflight(config, open_devices=not args.enumerate_only)
        else:
            result = capture(
                config,
                calibration_s=float(args.calibration_s if args.calibration_s is not None else config["calibration_seconds"]),
                entry_delay_s=float(args.entry_delay_s if args.entry_delay_s is not None else config["entry_delay_seconds"]),
                duration_s=float(args.duration_s if args.duration_s is not None else config["capture_seconds"]),
                skip_render=bool(args.skip_render),
            )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
