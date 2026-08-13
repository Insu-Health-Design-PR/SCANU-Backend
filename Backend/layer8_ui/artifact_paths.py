"""Resolve artifact files under ``software/`` or ``layer8_ui/`` from UI settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def software_root_from_settings(settings: dict[str, Any]) -> Path:
    raw = (settings.get("software_root") or "").strip()
    if raw:
        return Path(raw).resolve()
    return Path(__file__).resolve().parent.parent


def resolved_software_root(settings: dict[str, Any]) -> Path:
    """Absolute ``software/`` directory used for cwd and PYTHONPATH."""
    return software_root_from_settings(settings)


def resolved_artifact_path_from_roots(
    *,
    software_root: str | Path,
    relative_to_software: str,
    layer8_dir: Path,
) -> Path | None:
    rel = relative_to_software.strip()
    if not rel:
        return None
    sw = Path(software_root).resolve()
    layer8 = Path(layer8_dir).resolve()
    p = Path(rel).expanduser()
    if p.is_absolute():
        ar = p.resolve()
        if ar.is_file():
            return ar
        return None
    cand = (sw / p).resolve()
    for base in (sw, layer8):
        try:
            cand.relative_to(base)
            return cand
        except ValueError:
            continue
    return None


def resolved_artifact_path(
    settings: dict[str, Any],
    *,
    relative_to_software: str,
    layer8_dir: Path,
) -> Path | None:
    return resolved_artifact_path_from_roots(
        software_root=software_root_from_settings(settings),
        relative_to_software=relative_to_software,
        layer8_dir=layer8_dir,
    )


def abs_software_path(settings: dict[str, Any], rel: str) -> str:
    rel = rel.strip()
    if not rel:
        return ""
    sw = software_root_from_settings(settings)
    p = Path(rel).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((sw / p).resolve())
