"""Host CPU / RAM / optional NVIDIA GPU snapshot for the Layer 8 dashboard (Linux)."""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any


def _cpu_jiffies() -> tuple[int, int] | None:
    """Return (idle_jiffies, total_jiffies) for aggregate CPU line, or None."""
    try:
        with open("/proc/stat") as f:
            line = f.readline()
    except OSError:
        return None
    parts = line.split()
    if len(parts) < 5 or parts[0] != "cpu":
        return None
    try:
        nums = [int(x) for x in parts[1:]]
    except ValueError:
        return None
    idle = nums[3] + nums[4]
    total = sum(nums)
    return idle, total


def _cpu_percent(interval: float = 0.12) -> float | None:
    a = _cpu_jiffies()
    if a is None:
        return None
    idle1, total1 = a
    time.sleep(interval)
    b = _cpu_jiffies()
    if b is None:
        return None
    idle2, total2 = b
    di = idle2 - idle1
    dt = total2 - total1
    if dt <= 0:
        return None
    busy = 1.0 - (di / dt)
    return max(0.0, min(100.0, round(busy * 100.0, 1)))


def _mem_linux() -> dict[str, float | int] | None:
    try:
        with open("/proc/meminfo") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    kb: dict[str, int] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        parts = rest.split()
        if parts and parts[0].isdigit():
            kb[key] = int(parts[0])
    total = kb.get("MemTotal", 0)
    if total <= 0:
        return None
    avail = kb.get("MemAvailable")
    if avail is None:
        avail = kb.get("MemFree", 0)
    used_kb = max(0, total - avail)
    total_mb = total // 1024
    used_mb = used_kb // 1024
    pct = round(100.0 * used_kb / total, 1) if total else 0.0
    return {"used_mb": used_mb, "total_mb": total_mb, "percent": pct}


def _nvidia_gpu() -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return None
    line = proc.stdout.strip().splitlines()[0]
    # "45, 1234, 8192" or with spaces
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    try:
        util = float(parts[0])
        mem_used = float(parts[1])
        mem_total = float(parts[2])
    except ValueError:
        return None
    return {
        "util_percent": round(util, 1),
        "mem_used_mb": int(mem_used),
        "mem_total_mb": int(mem_total),
    }


def snapshot() -> dict[str, Any]:
    """JSON-serializable metrics; safe on non-Linux (returns nulls)."""
    out: dict[str, Any] = {
        "cpu_percent": None,
        "load_1m": None,
        "mem": None,
        "gpu": None,
    }
    try:
        if hasattr(os, "getloadavg"):
            la = os.getloadavg()
            out["load_1m"] = round(la[0], 2)
    except OSError:
        pass

    if os.path.isfile("/proc/stat"):
        out["cpu_percent"] = _cpu_percent()
    if os.path.isfile("/proc/meminfo"):
        m = _mem_linux()
        if m:
            out["mem"] = m

    out["gpu"] = _nvidia_gpu()
    return out
