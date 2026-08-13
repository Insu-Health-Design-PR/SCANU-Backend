"""
mmWave capture via ``layer1_radar/examples/live_capture.py`` (UART/TLV stack under ``layer1_radar/mmwave``).

Also owns shared PID/log state and ``start`` / ``stop`` / ``status`` for thermal, webcam,
and multi_camera — delegating command lines to ``runtime.*_runner`` modules.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

from layer8_ui.artifact_paths import software_root_from_settings
from runtime import multi_camera_runner, thermal_runner, webcam_runner

SensorId = Literal["thermal", "webcam", "mmwave", "multi_camera"]

_lock = threading.Lock()
_STATE_PATH: Path | None = None


def _state_path(layer8_dir: Path) -> Path:
    global _STATE_PATH
    if _STATE_PATH is None:
        _STATE_PATH = layer8_dir / ".sensor_pids.json"
    return _STATE_PATH


def _logs_dir(layer8_dir: Path) -> Path:
    d = layer8_dir.resolve() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sensor_file_logging() -> bool:
    """Write sensor stdout to layer8_ui/logs/<sensor>.log (default: on). Set LAYER8_SENSOR_LOG=0 to disable."""
    v = (os.environ.get("LAYER8_SENSOR_LOG") or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _truncate_log_file(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    except OSError:
        pass


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def resolved_software_root(settings: dict[str, Any]) -> Path:
    """Absolute ``software/`` directory used for cwd and PYTHONPATH."""
    return software_root_from_settings(settings)


def mmwave_capture_script(software_root: Path) -> Path:
    return thermal_runner.layer1_examples_dir(software_root) / "live_capture.py"


def build_mmwave_command(settings: dict[str, Any], layer8_dir: Path) -> list[str]:
    """mmWave UI plot: Adrian lab perception JPEG publisher (default), or legacy live_capture."""
    m = settings.get("mmwave") or {}
    pipeline = str(m.get("pipeline") or "lab_replay").strip().lower()
    if pipeline in ("legacy", "layer1_live_capture", "live_capture"):
        del layer8_dir
        sw = software_root_from_settings(settings)
        py = os.environ.get("PYTHON", sys.executable)
        script = mmwave_capture_script(sw)
        cmd = [py, str(script), "--frames", str(int(m.get("frames", 100)))]
        if bool(int(m.get("mmwave_only", 1))):
            cmd.append("--mmwave-only")
        cfg = (m.get("config") or "").strip()
        if cfg:
            cmd.extend(["--config", cfg])
        cli = (m.get("cli_port") or "").strip()
        if cli:
            cmd.extend(["--cli-port", cli])
        data = (m.get("data_port") or "").strip()
        if data:
            cmd.extend(["--data-port", data])
        out = (m.get("output") or "").strip()
        if out:
            cmd.extend(["--output", out])
        video = (m.get("video") or "").strip()
        if video:
            cmd.extend(["--video", video])
        live = (m.get("live_frame") or "").strip()
        if live:
            cmd.extend(["--live-frame", live])
        timeout = m.get("no_frame_timeout_s")
        if timeout is not None and float(timeout) > 0:
            cmd.extend(["--no-frame-timeout-s", str(float(timeout))])
        if m.get("verbose"):
            cmd.append("--verbose")
        extra = (m.get("extra_args") or "").strip()
        if extra:
            cmd.extend(shlex.split(extra))
        return cmd
    from runtime import mmwave_runner

    return mmwave_runner.build_mmwave_lab_command(settings, layer8_dir)

def build_command(sensor: SensorId, settings: dict[str, Any], layer8_dir: Path) -> list[str]:
    if sensor == "thermal":
        return thermal_runner.build_thermal_command(settings, layer8_dir)
    if sensor == "webcam":
        return webcam_runner.build_webcam_command(settings, layer8_dir)
    if sensor == "multi_camera":
        return multi_camera_runner.build_multi_camera_command(settings, layer8_dir)
    return build_mmwave_command(settings, layer8_dir)


def command_cwd(sensor: SensorId, settings: dict[str, Any]) -> Path:
    if sensor == "webcam":
        return webcam_runner.webcam_command_cwd(settings)
    if sensor == "multi_camera":
        return multi_camera_runner.multi_camera_command_cwd(settings)
    if sensor == "thermal":
        return thermal_runner.thermal_command_cwd(settings)
    if sensor == "mmwave":
        from runtime import mmwave_runner

        m = (settings.get("mmwave") or {})
        pipeline = str(m.get("pipeline") or "lab_replay").strip().lower()
        if pipeline not in ("legacy", "layer1_live_capture", "live_capture"):
            return mmwave_runner.mmwave_command_cwd(settings)
    return software_root_from_settings(settings)


def status(sensor: SensorId, layer8_dir: Path) -> dict[str, Any]:
    path = _state_path(layer8_dir)
    with _lock:
        st = _read_state(path)
    entry = st.get(sensor) or {}
    pid = int(entry.get("pid") or 0)
    preview_only = bool(entry.get("preview_only"))
    running = preview_only or _pid_alive(pid)
    if not running and pid:
        with _lock:
            st = _read_state(path)
            if st.get(sensor, {}).get("pid") == pid:
                st.pop(sensor, None)
                _write_state(path, st)
        pid = 0
    log_file = entry.get("log_file")
    tail = ""
    if log_file and Path(log_file).is_file():
        try:
            raw = Path(log_file).read_bytes()
            tail = raw[-4000:].decode("utf-8", errors="replace")
        except OSError:
            tail = ""
    return {"running": running, "pid": pid, "preview_only": preview_only, "log_tail": tail, "log_file": log_file or ""}


def stop(sensor: SensorId, layer8_dir: Path) -> dict[str, Any]:
    path = _state_path(layer8_dir)
    with _lock:
        st = _read_state(path)
        entry = st.get(sensor) or {}
        pid = int(entry.get("pid") or 0)
    if pid and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    if sensor == "webcam" and entry.get("preview_only"):
        webcam_runner.end_webcam_preview_only(layer8_dir)
    with _lock:
        st = _read_state(path)
        st.pop(sensor, None)
        _write_state(path, st)
    _truncate_log_file(_logs_dir(layer8_dir) / f"{sensor}.log")
    return {"ok": True, "stopped_pid": pid}


def _tail_log(path: Path, max_chars: int = 2800) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:].strip()
    except OSError:
        return ""


def start(sensor: SensorId, settings: dict[str, Any], layer8_dir: Path) -> dict[str, Any]:
    path = _state_path(layer8_dir)
    cur = status(sensor, layer8_dir)
    if cur["running"]:
        return {"ok": False, "error": f"{sensor} already running (pid {cur['pid']})"}

    if sensor == "thermal" and thermal_runner.thermal_preview_only(settings):
        stop("thermal", layer8_dir)
        sw = software_root_from_settings(settings)
        return {
            "ok": True,
            "pid": 0,
            "preview_only": True,
            "command": [],
            "cwd": str(sw),
            "software_root": str(sw),
            "log_file": "",
            "message": "Thermal preview only (no AI inference). Live view via WebSocket / in-process V4L2.",
        }

    if sensor == "webcam" and webcam_runner.webcam_preview_only(settings):
        stop("webcam", layer8_dir)
        webcam_runner.begin_webcam_preview_only(settings, layer8_dir)
        sw = software_root_from_settings(settings)
        with _lock:
            st = _read_state(path)
            st[sensor] = {"pid": 0, "preview_only": True, "log_file": ""}
            _write_state(path, st)
        return {
            "ok": True,
            "pid": 0,
            "preview_only": True,
            "command": [],
            "cwd": str(sw),
            "software_root": str(sw),
            "log_file": "",
            "message": "Front Camera blank preview (no models). Live view via shared V4L2 reader.",
        }

    sw = software_root_from_settings(settings)
    try:
        cmd = build_command(sensor, settings, layer8_dir)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    cwd = command_cwd(sensor, settings)
    use_file_log = _sensor_file_logging()
    log_path: Path | None = (_logs_dir(layer8_dir) / f"{sensor}.log") if use_file_log else None
    if log_path is not None:
        _truncate_log_file(log_path)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{sw.parent}:{sw}{os.pathsep}{env.get('PYTHONPATH', '')}"

    log_f = open(log_path, "ab", buffering=0) if log_path is not None else open(os.devnull, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    except OSError as e:
        log_f.close()
        return {"ok": False, "error": str(e)}
    log_f.close()

    with _lock:
        st = _read_state(path)
        st[sensor] = {"pid": proc.pid, "log_file": str(log_path) if log_path is not None else ""}
        _write_state(path, st)

    # If the child dies during imports or camera open, surface failure instead of "Run OK" + idle.
    for _ in range(24):
        code = proc.poll()
        if code is not None:
            tail = _tail_log(log_path) if log_path is not None else ""
            with _lock:
                st2 = _read_state(path)
                if st2.get(sensor, {}).get("pid") == proc.pid:
                    st2.pop(sensor, None)
                    _write_state(path, st2)
            if log_path and tail:
                err = f"{sensor} subprocess exited immediately (code {code}). Log ({log_path}):\n{tail}"
            else:
                err = (
                    f"{sensor} subprocess exited immediately (code {code}). "
                    "Check paths, camera index, GPU memory, and checkpoint. "
                    "Sensor logs: layer8_ui/logs/ (set LAYER8_SENSOR_LOG=0 to disable file logging)."
                )
            return {"ok": False, "error": err}
        time.sleep(0.12)

    return {
        "ok": True,
        "pid": proc.pid,
        "command": cmd,
        "cwd": str(cwd),
        "software_root": str(sw),
        "log_file": str(log_path) if log_path is not None else "",
    }


def restart(sensor: SensorId, settings: dict[str, Any], layer8_dir: Path) -> dict[str, Any]:
    stop(sensor, layer8_dir)
    return start(sensor, settings, layer8_dir)
