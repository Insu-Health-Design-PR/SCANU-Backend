#!/usr/bin/env python3
"""Capture two AWR1843BOOST USB sensors and build a provisional dual overlay."""

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
from typing import Iterable

import numpy as np

from .cube import CubeSpec, build_session_cube
from .dual_video import render_dual_video
from .runner import discover_awr1843_pairs, list_serial_devices


DUAL_SCHEMA_VERSION = "scanu_lab_awr1843_dual_usb_v1"
DEFAULT_PROFILE = (
    Path(__file__).resolve().parent
    / "configs"
    / "awr1843boost_sdk_3_4_profile_3d.cfg"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_pairs(sensor_a_location: str, sensor_b_location: str):
    pairs = discover_awr1843_pairs(list_serial_devices())
    if len(pairs) != 2:
        identities = [
            f"{pair.serial_number}@{pair.usb_location}" for pair in pairs
        ]
        raise RuntimeError(
            f"dual capture requires exactly two complete XDS110 pairs; found {identities}"
        )
    by_location = {pair.usb_location: pair for pair in pairs}
    if sensor_a_location or sensor_b_location:
        requested_a = sensor_a_location or pairs[0].usb_location
        requested_b = sensor_b_location or next(
            pair.usb_location for pair in pairs if pair.usb_location != requested_a
        )
        try:
            selected = (by_location[requested_a], by_location[requested_b])
        except KeyError as exc:
            raise RuntimeError(
                f"requested USB location is absent; available: {sorted(by_location)}"
            ) from exc
        if selected[0].usb_location == selected[1].usb_location:
            raise RuntimeError("sensor A and B must use different USB locations")
        return selected
    return pairs[0], pairs[1]


def _capture_command(
    pair,
    *,
    profile: Path,
    duration_s: float,
    output_root: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "lab.mmwave77_usb.runner",
        "capture",
        "--data-port",
        pair.data_port,
        "--cli-port",
        pair.cli_port,
        "--config",
        str(profile),
        "--protocol",
        "ti-tlv",
        "--duration-s",
        str(duration_s),
        "--sensor-model",
        "Texas Instruments AWR1843BOOST",
        "--output-root",
        str(output_root),
    ]


def _parse_child_result(stdout: str, label: str) -> dict:
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} returned invalid capture output: {stdout[-500:]}") from exc
    if not result.get("ok"):
        raise RuntimeError(f"{label} capture reported failure: {result}")
    return result


def _build_overlay(
    cube_a: Path,
    cube_b: Path,
    output: Path,
) -> tuple[Path, Path, dict]:
    with np.load(cube_a) as a, np.load(cube_b) as b:
        axes = (
            "range_edges_m",
            "range_centers_m",
            "azimuth_edges_deg",
            "azimuth_centers_deg",
            "elevation_edges_deg",
            "elevation_centers_deg",
        )
        for name in axes:
            if not np.allclose(a[name], b[name]):
                raise RuntimeError(f"cube axes differ: {name}")
        windows = min(len(a["hit_count"]), len(b["hit_count"]))
        hit_a = np.asarray(a["hit_count"][:windows], dtype=np.uint16)
        hit_b = np.asarray(b["hit_count"][:windows], dtype=np.uint16)
        summed = np.minimum(
            hit_a.astype(np.uint32) + hit_b.astype(np.uint32),
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        arrays = {
            "hit_count_a": hit_a,
            "hit_count_b": hit_b,
            "hit_count_overlay": summed,
            **{name: np.asarray(a[name]) for name in axes},
            "frame_start_a": np.asarray(a["frame_start"][:windows]),
            "frame_end_a": np.asarray(a["frame_end"][:windows]),
            "frame_start_b": np.asarray(b["frame_start"][:windows]),
            "frame_end_b": np.asarray(b["frame_end"][:windows]),
        }
    np.savez_compressed(output, **arrays)
    metadata_path = output.with_suffix(".metadata.json")
    metadata = {
        "schema_version": DUAL_SCHEMA_VERSION,
        "experimental": True,
        "canonical_training_compatible": False,
        "extrinsic_calibrated": False,
        "fusion_mode": "identity_overlay_comparison_only",
        "sensor_a_cube": str(cube_a),
        "sensor_a_cube_sha256": _sha256(cube_a),
        "sensor_b_cube": str(cube_b),
        "sensor_b_cube_sha256": _sha256(cube_b),
        "overlay": str(output),
        "overlay_sha256": _sha256(output),
        "windows": windows,
        "sensor_a_observations": int(hit_a.sum()),
        "sensor_b_observations": int(hit_b.sum()),
        "overlay_observations": int(summed.sum()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "identity overlay only; sensor baseline, yaw, pitch and roll are not calibrated",
            "overlap does not deduplicate detections",
            "not coherent multi-radar processing and not canonical SCAN-U training data",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output, metadata_path, metadata


def dual_capture(args: argparse.Namespace) -> int:
    if args.duration_s <= 0:
        raise RuntimeError("--duration-s must be greater than zero")
    if args.stagger_ms < 0:
        raise RuntimeError("--stagger-ms cannot be negative")
    profile = args.config.expanduser().resolve()
    if not profile.is_file():
        raise RuntimeError(f"configuration does not exist: {profile}")
    pair_a, pair_b = _select_pairs(
        args.sensor_a_location, args.sensor_b_location
    )
    output_root = args.output_root.expanduser().resolve()
    session_dir = output_root / datetime.now().strftime("dual_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = os.pathsep.join(
        (str(repo_root), str(repo_root / "software"), env.get("PYTHONPATH", ""))
    )
    launch_a_ns = time.monotonic_ns()
    process_a = subprocess.Popen(
        _capture_command(
            pair_a,
            profile=profile,
            duration_s=args.duration_s,
            output_root=session_dir / "sensor_a",
        ),
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(args.stagger_ms / 1000.0)
    launch_b_ns = time.monotonic_ns()
    process_b = subprocess.Popen(
        _capture_command(
            pair_b,
            profile=profile,
            duration_s=args.duration_s,
            output_root=session_dir / "sensor_b",
        ),
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_a, stderr_a = process_a.communicate()
    stdout_b, stderr_b = process_b.communicate()
    (session_dir / "sensor_a.stderr.txt").write_text(stderr_a)
    (session_dir / "sensor_b.stderr.txt").write_text(stderr_b)
    if process_a.returncode or process_b.returncode:
        failure = {
            "sensor_a_returncode": process_a.returncode,
            "sensor_b_returncode": process_b.returncode,
            "sensor_a_stderr": stderr_a[-1000:],
            "sensor_b_stderr": stderr_b[-1000:],
        }
        (session_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise RuntimeError(f"dual capture failed: {failure}")

    result_a = _parse_child_result(stdout_a, "sensor A")
    result_b = _parse_child_result(stdout_b, "sensor B")
    cube_a, cube_a_metadata, stats_a = build_session_cube(
        Path(result_a["session"]), session_dir / "sensor_a_cube.npz", CubeSpec()
    )
    cube_b, cube_b_metadata, stats_b = build_session_cube(
        Path(result_b["session"]), session_dir / "sensor_b_cube.npz", CubeSpec()
    )
    overlay, overlay_metadata, overlay_stats = _build_overlay(
        cube_a, cube_b, session_dir / "dual_overlay.npz"
    )
    video, video_metadata, video_stats = render_dual_video(overlay)
    manifest = {
        "schema_version": DUAL_SCHEMA_VERSION,
        "experimental": True,
        "canonical_training_compatible": False,
        "session": str(session_dir),
        "profile": str(profile),
        "profile_sha256": _sha256(profile),
        "requested_stagger_ms": args.stagger_ms,
        "process_launch_stagger_ms": (launch_b_ns - launch_a_ns) / 1e6,
        "sensor_a": {
            "pair": asdict(pair_a),
            "capture": result_a,
            "cube": str(cube_a),
            "cube_metadata": str(cube_a_metadata),
            "cube_statistics": stats_a["statistics"],
        },
        "sensor_b": {
            "pair": asdict(pair_b),
            "capture": result_b,
            "cube": str(cube_b),
            "cube_metadata": str(cube_b_metadata),
            "cube_statistics": stats_b["statistics"],
        },
        "overlay": str(overlay),
        "overlay_metadata": str(overlay_metadata),
        "overlay_statistics": overlay_stats,
        "video": str(video),
        "video_metadata": str(video_metadata),
        "video_statistics": video_stats,
        "limitations": [
            "USB process staggering is best-effort and is not hardware frame synchronization",
            "extrinsic geometry is not calibrated; overlay uses identity transforms",
            "two AWR1843 devices cannot form a coherent cascaded aperture",
        ],
    }
    manifest_path = session_dir / "dual_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "ok": True,
        "session": str(session_dir),
        "manifest": str(manifest_path),
        "sensor_a_frames": result_a["tlv_frames"],
        "sensor_b_frames": result_b["tlv_frames"],
        "video": str(video),
        "video_sha256": video_stats["output_video_sha256"],
    }, indent=2))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Experimental parallel capture of exactly two AWR1843BOOST sensors"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--stagger-ms", type=float, default=50.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/lab/mmwave77_usb/dual"),
    )
    parser.add_argument("--sensor-a-location", default="")
    parser.add_argument("--sensor-b-location", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return dual_capture(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
