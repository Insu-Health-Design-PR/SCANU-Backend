"""SSH helper for Back Camera Jetson (MediaMTX + cam-rtsp services)."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ALLOWED_SERVICES = frozenset({"mediamtx", "cam_rtsp"})
_SERVICE_ACTIONS = frozenset({"status", "start", "stop", "restart", "is-active"})


@dataclass(frozen=True)
class JetsonSshConfig:
    host: str
    user: str
    port: int = 22
    identity_file: Path | None = None
    connect_timeout_s: float = 10.0
    mediamtx_unit: str = "mediamtx.service"
    cam_rtsp_unit: str = "cam-rtsp.service"
    cam_rtsp_log: str = "/home/insu/cam_publish.log"

    def service_unit(self, key: str) -> str:
        k = str(key or "").strip().lower()
        if k in ("mediamtx", "mtx"):
            return self.mediamtx_unit
        if k in ("cam_rtsp", "cam-rtsp", "cam"):
            return self.cam_rtsp_unit
        raise ValueError(f"unknown service key: {key!r}")


def _repo_root(layer8_dir: Path) -> Path:
    layer8_dir = Path(layer8_dir).resolve()
    if layer8_dir.name == "layer8_ui":
        return layer8_dir.parent
    return layer8_dir


def resolve_identity_file(layer8_dir: Path, raw: str | None) -> Path | None:
    env = str(os.environ.get("JETSON_SSH_IDENTITY_FILE") or "").strip()
    val = env or str(raw or "").strip()
    if not val:
        return None
    p = Path(val).expanduser()
    if not p.is_absolute():
        p = (_repo_root(layer8_dir) / val).resolve()
    return p if p.is_file() else p



def jetson_config_from_settings(settings: dict[str, Any] | None, layer8_dir: Path) -> JetsonSshConfig:
    """Build config from ``settings['jetson_back']`` with ``multi_camera`` fallbacks."""
    settings = settings if isinstance(settings, dict) else {}
    raw = settings.get("jetson_back") if isinstance(settings.get("jetson_back"), dict) else {}
    mc = settings.get("multi_camera") if isinstance(settings.get("multi_camera"), dict) else {}
    host = str(raw.get("host") or mc.get("jetson_ip") or "").strip()
    user = str(raw.get("user") or "insu").strip() or "insu"
    try:
        port = int(raw.get("port") or 22)
    except (TypeError, ValueError):
        port = 22
    try:
        timeout = float(raw.get("connect_timeout_s") or 10.0)
    except (TypeError, ValueError):
        timeout = 10.0
    identity = resolve_identity_file(layer8_dir, str(raw.get("identity_file") or ""))
    services = raw.get("services") if isinstance(raw.get("services"), dict) else {}
    return JetsonSshConfig(
        host=host,
        user=user,
        port=max(1, min(65535, port)),
        identity_file=identity if identity and identity.is_file() else None,
        connect_timeout_s=max(3.0, min(60.0, timeout)),
        mediamtx_unit=str(services.get("mediamtx") or raw.get("mediamtx_unit") or "mediamtx.service"),
        cam_rtsp_unit=str(services.get("cam_rtsp") or raw.get("cam_rtsp_unit") or "cam-rtsp.service"),
        cam_rtsp_log=str(raw.get("cam_rtsp_log") or "/home/insu/cam_publish.log"),
    )


def config_public_dict(cfg: JetsonSshConfig) -> dict[str, Any]:
    """Safe for API — never includes PEM contents."""
    ident = cfg.identity_file
    return {
        "host": cfg.host,
        "user": cfg.user,
        "port": cfg.port,
        "identity_file": str(ident) if ident else "",
        "identity_file_exists": bool(ident and ident.is_file()),
        "connect_timeout_s": cfg.connect_timeout_s,
        "services": {
            "mediamtx": cfg.mediamtx_unit,
            "cam_rtsp": cfg.cam_rtsp_unit,
        },
        "cam_rtsp_log": cfg.cam_rtsp_log,
    }


def run_ssh(cfg: JetsonSshConfig, remote_command: str, *, timeout_s: float | None = None) -> dict[str, Any]:
    if not cfg.host:
        return {"ok": False, "error": "jetson_back.host is not configured"}
    if not cfg.user:
        return {"ok": False, "error": "jetson_back.user is not configured"}
    timeout = float(timeout_s if timeout_s is not None else cfg.connect_timeout_s)
    timeout = max(3.0, min(120.0, timeout))
    ssh_cmd: list[str] = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={int(timeout)}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-p",
        str(int(cfg.port)),
    ]
    if cfg.identity_file and cfg.identity_file.is_file():
        ssh_cmd.extend(["-i", str(cfg.identity_file)])
    ssh_cmd.append(f"{cfg.user}@{cfg.host}")
    ssh_cmd.append(remote_command)
    try:
        proc = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"SSH timed out after {timeout:.0f}s", "command": remote_command}
    except FileNotFoundError:
        return {"ok": False, "error": "ssh client not found on PATH"}
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    ok = proc.returncode == 0
    return {
        "ok": ok,
        "exit_code": int(proc.returncode),
        "stdout": out,
        "stderr": err,
        "command": remote_command,
        "used_identity_file": str(cfg.identity_file) if cfg.identity_file else None,
    }


def test_connection(cfg: JetsonSshConfig) -> dict[str, Any]:
    res = run_ssh(cfg, "echo ok && hostname && uptime", timeout_s=cfg.connect_timeout_s)
    if not res.get("ok"):
        hint = ""
        ident = cfg.identity_file
        if ident and not ident.is_file():
            hint = f" PEM not found at {ident}. Place the key file or set JETSON_SSH_IDENTITY_FILE."
        elif not ident:
            hint = " No identity_file configured; using default SSH agent keys."
        res["hint"] = hint.strip()
        return res
    lines = [ln.strip() for ln in str(res.get("stdout") or "").splitlines() if ln.strip()]
    return {
        "ok": True,
        "hostname": lines[1] if len(lines) > 1 else "",
        "uptime": lines[2] if len(lines) > 2 else "",
        "stdout": res.get("stdout"),
        "used_identity_file": res.get("used_identity_file"),
    }


def _parse_systemctl_show(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def service_status(cfg: JetsonSshConfig, service_key: str) -> dict[str, Any]:
    unit = cfg.service_unit(service_key)
    remote = (
        f"systemctl is-active {shlex.quote(unit)}; "
        f"systemctl show {shlex.quote(unit)} "
        "-p ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus --no-pager"
    )
    res = run_ssh(cfg, remote, timeout_s=cfg.connect_timeout_s)
    if not res.get("ok") and not res.get("stdout"):
        return {"ok": False, "service": service_key, "unit": unit, **res}
    lines = [ln.strip() for ln in str(res.get("stdout") or "").splitlines() if ln.strip()]
    active_line = lines[0] if lines else "unknown"
    meta = _parse_systemctl_show("\n".join(lines[1:]))
    state = meta.get("ActiveState") or active_line
    return {
        "ok": True,
        "service": service_key,
        "unit": unit,
        "is_active": active_line,
        "active_state": state,
        "sub_state": meta.get("SubState", ""),
        "main_pid": meta.get("MainPID", ""),
        "exec_main_status": meta.get("ExecMainStatus", ""),
        "enabled": meta.get("UnitFileState", ""),
        "running": state == "active" and active_line == "active",
        "stderr": res.get("stderr") or "",
    }


def all_services_status(cfg: JetsonSshConfig) -> dict[str, Any]:
    items = {}
    for key in ("mediamtx", "cam_rtsp"):
        items[key] = service_status(cfg, key)
    ok = all(v.get("running") for v in items.values())
    return {"ok": ok, "services": items, "config": config_public_dict(cfg)}


def service_action(cfg: JetsonSshConfig, service_key: str, action: str) -> dict[str, Any]:
    key = str(service_key or "").strip().lower()
    act = str(action or "").strip().lower()
    if key not in _ALLOWED_SERVICES:
        return {"ok": False, "error": f"service must be one of {sorted(_ALLOWED_SERVICES)}"}
    if act not in _SERVICE_ACTIONS:
        return {"ok": False, "error": f"action must be one of {sorted(_SERVICE_ACTIONS)}"}
    unit = cfg.service_unit(key)
    remote = f"systemctl {act} {shlex.quote(unit)}"
    res = run_ssh(cfg, remote, timeout_s=max(cfg.connect_timeout_s, 30.0))
    stat = service_status(cfg, key)
    return {
        **res,
        "service": key,
        "unit": unit,
        "action": act,
        "status": stat,
    }


def daemon_reload(cfg: JetsonSshConfig) -> dict[str, Any]:
    return run_ssh(cfg, "systemctl daemon-reload", timeout_s=cfg.connect_timeout_s)


def service_journal(cfg: JetsonSshConfig, service_key: str, *, lines: int = 80) -> dict[str, Any]:
    key = str(service_key or "").strip().lower()
    if key not in _ALLOWED_SERVICES:
        return {"ok": False, "error": f"service must be one of {sorted(_ALLOWED_SERVICES)}"}
    n = max(10, min(500, int(lines)))
    unit = cfg.service_unit(key)
    remote = f"journalctl -u {shlex.quote(unit)} -n {n} --no-pager"
    return run_ssh(cfg, remote, timeout_s=max(cfg.connect_timeout_s, 20.0))


def tail_cam_log(cfg: JetsonSshConfig, *, lines: int = 80) -> dict[str, Any]:
    n = max(10, min(500, int(lines)))
    log = str(cfg.cam_rtsp_log or "").strip()
    if not log or not re.match(r"^[/\w.\-_]+$", log):
        return {"ok": False, "error": "invalid cam_rtsp_log path"}
    remote = f"tail -n {n} {shlex.quote(log)} 2>/dev/null || echo '(log missing)'"
    return run_ssh(cfg, remote, timeout_s=cfg.connect_timeout_s)


def list_v4l2_devices(cfg: JetsonSshConfig) -> dict[str, Any]:
    remote = "v4l2-ctl --list-devices 2>/dev/null || echo 'v4l2-ctl not available'"
    res = run_ssh(cfg, remote, timeout_s=cfg.connect_timeout_s)
    return res


def set_cam_rtsp_video_device(cfg: JetsonSshConfig, device: str) -> dict[str, Any]:
    dev = str(device or "").strip()
    if not re.match(r"^/dev/video\d+$", dev):
        return {"ok": False, "error": "device must look like /dev/videoN"}
    # Update systemd drop-in override on Jetson (requires passwordless sudo for insu).
    remote = (
        "sudo mkdir -p /etc/systemd/system/cam-rtsp.service.d && "
        f"printf '%s\\n' '[Service]' 'Environment=VIDEO_DEVICE={dev}' "
        "| sudo tee /etc/systemd/system/cam-rtsp.service.d/override.conf > /dev/null && "
        "sudo systemctl daemon-reload && "
        "sudo systemctl restart cam-rtsp.service"
    )
    res = run_ssh(cfg, remote, timeout_s=45.0)
    stat = service_status(cfg, "cam_rtsp")
    return {**res, "device": dev, "status": stat}
