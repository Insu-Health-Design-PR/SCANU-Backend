"""mmWave radar capture / lab perception runner for Layer 8 UI."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any

from layer8_ui.artifact_paths import abs_software_path, software_root_from_settings


def build_mmwave_lab_command(settings: dict[str, Any], layer8_dir: Path) -> list[str]:
    """Publish Adrian-lab perception JPEG to ``mmwave.live_frame`` (+ optional back)."""
    del layer8_dir
    m = settings.get("mmwave") or {}
    py = os.environ.get("PYTHON", sys.executable)
    live = abs_software_path(settings, str(m.get("live_frame") or "layer8_ui/artifacts/live_mmwave.jpg"))
    live_back = abs_software_path(
        settings, str(m.get("live_frame_back") or "layer8_ui/artifacts/live_mmwave_back.jpg")
    )
    pipeline = str(m.get("pipeline") or "lab_replay").strip() or "lab_replay"
    cmd = [
        py,
        "-m",
        "runtime.mmwave_lab_live",
        "--pipeline",
        pipeline,
        "--live-frame",
        str(live),
        "--live-frame-back",
        str(live_back),
        "--fps",
        str(float(m.get("plot_fps") or 2.0)),
    ]
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
    cli = str(m.get("front_cli_port") or m.get("cli_port") or "").strip()
    if cli:
        cmd.extend(["--cli-port", cli])
    data = str(m.get("front_data_port") or m.get("data_port") or "").strip()
    if data:
        cmd.extend(["--data-port", data])
    extra = str(m.get("extra_args") or "").strip()
    if extra:
        cmd.extend(shlex.split(extra))
    return cmd


def mmwave_command_cwd(settings: dict[str, Any]) -> Path:
    """Run from Backend root so ``lab`` / ``layer1_sensor_hub`` import."""
    return software_root_from_settings(settings)


def main() -> None:
    raise SystemExit("Use runtime.sensor_runner / runtime.mmwave_lab_live")


if __name__ == "__main__":
    main()
