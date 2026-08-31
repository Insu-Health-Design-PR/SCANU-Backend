"""Read live mmWave metrics JSON and derive dashboard fields."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from weapon_ai.overlay.mmwave_fusion import compute_mmwave_torso_score

LIVE_METRICS_SCHEMA = "scanu_mmwave_live_v1"
DEFAULT_METRICS_REL = "layer8_ui/artifacts/live_mmwave_metrics.json"
DEFAULT_SHM = Path("/dev/shm/scanu_mmwave_live_metrics.json")


def _side_block(data: dict[str, Any], side: str) -> dict[str, Any]:
    block = data.get(side)
    return block if isinstance(block, dict) else {}


def read_live_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def resolve_metrics_path(
    settings: dict[str, Any],
    *,
    layer8_dir: Path,
    software_root: Path,
) -> Path | None:
    del layer8_dir
    mm = settings.get("mmwave") if isinstance(settings.get("mmwave"), dict) else {}
    fusion = settings.get("mmwave_fusion") if isinstance(settings.get("mmwave_fusion"), dict) else {}
    rel = str(
        fusion.get("metrics_path")
        or mm.get("live_metrics_json")
        or DEFAULT_METRICS_REL
    ).strip()
    if not rel:
        return DEFAULT_SHM if DEFAULT_SHM.is_file() else None
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p
    return (software_root / rel).resolve()


def live_metrics_snapshot(
    settings: dict[str, Any],
    *,
    layer8_dir: Path,
    software_root: Path,
) -> dict[str, Any]:
    from layer8_ui.artifact_paths import software_root_from_settings

    sw = software_root or software_root_from_settings(settings)
    path = resolve_metrics_path(settings, layer8_dir=layer8_dir, software_root=sw)
    shm_path = DEFAULT_SHM
    data = read_live_metrics(path) if path else None
    if data is None and shm_path.is_file():
        data = read_live_metrics(shm_path)
        path = shm_path
    age_s: float | None = None
    if isinstance(data, dict):
        ts = data.get("ts_monotonic_ns")
        try:
            if ts is not None:
                age_s = max(0.0, (time.monotonic_ns() - int(ts)) / 1_000_000_000.0)
        except (TypeError, ValueError):
            age_s = None
    return {
        "ok": isinstance(data, dict),
        "path": str(path) if path else "",
        "schema_version": (data or {}).get("schema_version"),
        "age_s": round(age_s, 3) if age_s is not None else None,
        "mmwave_torso_score": compute_mmwave_torso_score(data),
        "metrics": data,
    }
