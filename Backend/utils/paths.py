"""software_root and artifact path helpers."""

from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]


def software_root() -> Path:
    return _BACKEND_ROOT


def artifacts_dir() -> Path:
    return _BACKEND_ROOT / "artifacts"


def config_dir() -> Path:
    return _BACKEND_ROOT / "config"
