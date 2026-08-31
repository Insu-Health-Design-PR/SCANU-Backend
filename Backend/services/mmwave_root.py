"""Resolve and import the standalone Mmwave lab tree via ``MMWAVE_ROOT``."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REQUIRED = (
    "software/lab/mmwave77_usb/runner.py",
    "software/lab/dual_server_77ghz/orchestrate.py",
)


def backend_repo_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__).resolve()).parent
    for parent in [start, *start.parents]:
        if (parent / "layer8_ui").is_dir() and (parent / "runtime").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


def resolve_mmwave_root(
    *,
    env: str | None = None,
    settings_root: str | None = None,
    backend_root: Path | None = None,
) -> Path:
    """Return absolute Mmwave repo root (env → settings → sibling default)."""
    backend_root = backend_root or backend_repo_root()
    raw = str(env or os.environ.get("MMWAVE_ROOT") or settings_root or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (backend_root / p).resolve()
        else:
            p = p.resolve()
    else:
        p = (backend_root.parent / "Mmwave").resolve()
    return p


def validate_mmwave_root(root: Path) -> list[str]:
    missing = [rel for rel in _REQUIRED if not (root / rel).is_file()]
    return missing


def ensure_mmwave_imports(root: Path | None = None) -> Path:
    """Add ``{MMWAVE_ROOT}/software`` to ``sys.path`` and return root."""
    root = Path(root or resolve_mmwave_root()).resolve()
    missing = validate_mmwave_root(root)
    if missing:
        raise FileNotFoundError(
            f"MMWAVE_ROOT invalid at {root}: missing {missing}. "
            "Set MMWAVE_ROOT to ~/Desktop/New_Backend/Mmwave"
        )
    software = str(root / "software")
    if software not in sys.path:
        sys.path.insert(0, software)
    return root


def mmwave_root_status(
    *,
    settings: dict[str, Any] | None = None,
    backend_root: Path | None = None,
) -> dict[str, Any]:
    settings = settings if isinstance(settings, dict) else {}
    block = settings.get("mmwave_root") if isinstance(settings.get("mmwave_root"), dict) else {}
    root = resolve_mmwave_root(
        settings_root=str(block.get("path") or block.get("mmwave_root") or ""),
        backend_root=backend_root,
    )
    missing = validate_mmwave_root(root)
    out: dict[str, Any] = {
        "ok": not missing,
        "mmwave_root": str(root),
        "missing": missing,
        "env_mmwave_root": str(os.environ.get("MMWAVE_ROOT") or ""),
    }
    cfg_rel = str(block.get("config_path") or "configs/server_local.json").strip()
    cfg_path = Path(cfg_rel)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_rel).resolve()
    out["config_path"] = str(cfg_path)
    out["config_exists"] = cfg_path.is_file()
    return out


def load_mmwave_config(path: Path | str, *, root: Path | None = None) -> dict[str, Any]:
    root = ensure_mmwave_imports(root) if root is None else Path(root).resolve()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"mmWave config not found: {p}")
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError("mmWave config must be a JSON object")
    data["_config_path"] = str(p)
    return data


def discover_radar_pairs(*, root: Path | None = None) -> list[dict[str, Any]]:
    root = ensure_mmwave_imports(root)
    from lab.mmwave77_usb.runner import discover_awr1843_pairs, list_serial_devices

    pairs = discover_awr1843_pairs(list_serial_devices())
    return [asdict(p) for p in pairs]


def select_radar_pair_ab(
    *,
    location_a: str = "",
    location_b: str = "",
    root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return front (A) and back (B) radar port pairs as dicts."""
    root = ensure_mmwave_imports(root)
    from lab.dual_server_77ghz.orchestrate import select_radar_pairs
    from lab.mmwave77_usb.runner import discover_awr1843_pairs, list_serial_devices

    pair_a, pair_b = select_radar_pairs(
        discover_awr1843_pairs(list_serial_devices()),
        location_a or "",
        location_b or "",
    )
    return asdict(pair_a), asdict(pair_b)


def preflight(
    *,
    settings: dict[str, Any] | None = None,
    backend_root: Path | None = None,
) -> dict[str, Any]:
    """Non-destructive discovery: MMWAVE_ROOT, config, radar pairs."""
    settings = settings if isinstance(settings, dict) else {}
    block = settings.get("mmwave_root") if isinstance(settings.get("mmwave_root"), dict) else {}
    mm = settings.get("mmwave") if isinstance(settings.get("mmwave"), dict) else {}
    status = mmwave_root_status(settings=settings, backend_root=backend_root)
    if not status.get("ok"):
        return {**status, "pairs": [], "pair_a": None, "pair_b": None, "error": "MMWAVE_ROOT invalid"}

    root = Path(str(status["mmwave_root"]))
    try:
        ensure_mmwave_imports(root)
    except FileNotFoundError as exc:
        return {**status, "pairs": [], "pair_a": None, "pair_b": None, "error": str(exc)}

    loc_a = str(block.get("radar_a_usb_location") or mm.get("radar_a_usb_location") or "").strip()
    loc_b = str(block.get("radar_b_usb_location") or mm.get("radar_b_usb_location") or "").strip()
    pairs = discover_radar_pairs(root=root)
    pair_a = pair_b = None
    err = ""
    try:
        if len(pairs) >= 2:
            pair_a, pair_b = select_radar_pair_ab(location_a=loc_a, location_b=loc_b, root=root)
        elif len(pairs) == 1:
            pair_a = pairs[0]
    except Exception as exc:
        err = str(exc)

    sensor_distance_m = float(
        block.get("sensor_distance_m")
        or mm.get("sensor_distance_m")
        or 3.6576
    )
    return {
        **status,
        "pairs": pairs,
        "pair_a": pair_a,
        "pair_b": pair_b,
        "sensor_distance_m": sensor_distance_m,
        "radar_a_usb_location": loc_a,
        "radar_b_usb_location": loc_b,
        "error": err,
    }
