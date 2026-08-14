"""Conventions for latest-frame mmap paths."""

from __future__ import annotations

from pathlib import Path


def derived_full_bgr_ipc_path(preview_path: Path) -> Path:
    """Sidecar mmap for the full-res (4K) frame next to the 1080p preview IPC."""
    path = Path(preview_path)
    name = path.name
    if name.endswith("_bgr_frame.bin"):
        return path.with_name(name.replace("_bgr_frame.bin", "_bgr_full.bin", 1))
    if name.endswith("_frame.bin"):
        return path.with_name(name.replace("_frame.bin", "_full.bin", 1))
    return path.with_name(f"{path.stem}_full{path.suffix}")
