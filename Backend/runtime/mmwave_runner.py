"""mmWave runner — dual live/replay via MMWAVE_ROOT adapter."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any

from layer8_ui.artifact_paths import abs_software_path, software_root_from_settings


def _mmwave_root_block(settings: dict[str, Any]) -> dict[str, Any]:
    block = settings.get("mmwave_root")
    return block if isinstance(block, dict) else {}


def build_mmwave_lab_command(settings: dict[str, Any], layer8_dir: Path) -> list[str]:
    """Publish dual mmWave JPEGs + live_mmwave_metrics.json."""
    del layer8_dir
    m = settings.get("mmwave") or {}
    mr = _mmwave_root_block(settings)
    py = os.environ.get("PYTHON", sys.executable)
    sw = software_root_from_settings(settings)
    live = abs_software_path(
        settings,
        str(
            m.get("live_frame_fused")
            or m.get("live_frame")
            or "layer8_ui/artifacts/live_mmwave_fused_dashboard.jpg"
        ),
    )
    live_back = abs_software_path(
        settings, str(m.get("live_frame_back") or "layer8_ui/artifacts/live_mmwave_back.jpg")
    )
    metrics_json = abs_software_path(
        settings,
        str(m.get("live_metrics_json") or "layer8_ui/artifacts/live_mmwave_metrics.json"),
    )
    pipeline = str(m.get("pipeline") or "dual_replay").strip() or "dual_replay"
    if pipeline in ("lab_replay",):
        pipeline = "dual_replay"
    if pipeline in ("lab_live",):
        pipeline = "dual_live"

    cmd = [
        py,
        "-m",
        "runtime.mmwave_dual_live",
        "--pipeline",
        pipeline,
        "--live-frame",
        str(live),
        "--live-frame-back",
        str(live_back),
        "--metrics-json",
        str(metrics_json),
        "--fps",
        str(float(m.get("plot_fps") or 2.0)),
    ]
    mmwave_root = str(mr.get("path") or mr.get("mmwave_root") or os.environ.get("MMWAVE_ROOT") or "").strip()
    if mmwave_root:
        p = Path(mmwave_root).expanduser()
        if not p.is_absolute():
            backend_root = Path(__file__).resolve().parents[1]
            p = (backend_root.parent / p).resolve() if p.parts and p.parts[0] == ".." else (backend_root / p).resolve()
        cmd.extend(["--mmwave-root", str(p)])
    config_path = str(mr.get("config_path") or "configs/server_local.json").strip()
    if config_path:
        cmd.extend(["--server-config", config_path])

    sensor_d = mr.get("sensor_distance_m", m.get("sensor_distance_m"))
    if sensor_d is not None and str(sensor_d).strip() != "":
        try:
            cmd.extend(["--sensor-distance-m", str(float(sensor_d))])
        except (TypeError, ValueError):
            pass

    loc_a = str(mr.get("radar_a_usb_location") or m.get("radar_a_usb_location") or "").strip()
    loc_b = str(mr.get("radar_b_usb_location") or m.get("radar_b_usb_location") or "").strip()
    if loc_a:
        cmd.extend(["--radar-a-usb-location", loc_a])
    if loc_b:
        cmd.extend(["--radar-b-usb-location", loc_b])

    radar_cfg = str(mr.get("radar_profile") or m.get("config") or "").strip()
    if radar_cfg:
        cmd.extend(["--radar-config", radar_cfg])

    session = str(m.get("front_session") or m.get("session") or "").strip()
    if session:
        cmd.extend(["--session", str(abs_software_path(settings, session))])
    back_session = str(m.get("back_session") or "").strip()
    if back_session:
        cmd.extend(["--session-back", str(abs_software_path(settings, back_session))])
    perception = str(m.get("perception") or "").strip()
    if perception:
        cmd.extend(["--perception", str(abs_software_path(settings, perception))])
    frames = str(m.get("frames_jsonl") or "").strip()
    if frames:
        cmd.extend(["--frames-jsonl", str(abs_software_path(settings, frames))])

    front_cli = str(m.get("front_cli_port") or m.get("cli_port") or "").strip()
    front_data = str(m.get("front_data_port") or m.get("data_port") or "").strip()
    back_cli = str(m.get("back_cli_port") or "").strip()
    back_data = str(m.get("back_data_port") or "").strip()
    if front_cli:
        cmd.extend(["--front-cli-port", front_cli, "--cli-port", front_cli])
    if front_data:
        cmd.extend(["--front-data-port", front_data, "--data-port", front_data])
    if back_cli:
        cmd.extend(["--back-cli-port", back_cli])
    if back_data:
        cmd.extend(["--back-data-port", back_data])

    extra = str(m.get("extra_args") or "").strip()
    if extra:
        cmd.extend(shlex.split(extra))
    return cmd


def mmwave_command_cwd(settings: dict[str, Any]) -> Path:
    return software_root_from_settings(settings)


def main() -> None:
    raise SystemExit("Use runtime.sensor_runner / runtime.mmwave_dual_live")


if __name__ == "__main__":
    main()
