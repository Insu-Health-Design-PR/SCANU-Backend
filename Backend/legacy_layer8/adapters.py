"""Bridge from the new backend layout to the current Layer 8 implementation."""

from __future__ import annotations

import sys
from pathlib import Path

NEW_BACKEND_ROOT = Path(__file__).resolve().parent.parent
SOFTWARE_ROOT = NEW_BACKEND_ROOT
LAYER8_DIR = NEW_BACKEND_ROOT / "layer8_ui"


def ensure_legacy_imports() -> None:
    """Make the current `software/` package importable from the new backend."""
    root = str(SOFTWARE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_legacy_imports()

from layer8_ui import playground, thermal_runner, webcam_runner  # noqa: E402


def _new_backend_cwd(_settings: dict | Path) -> Path:
    """Run migrated `weapon_ai` subprocesses from the new backend package root."""
    return NEW_BACKEND_ROOT


def _software_env_with_new_backend(sw: Path) -> dict[str, str]:
    """Keep legacy model/artifact roots, but make migrated modules import first."""
    import os

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = (
        f"{NEW_BACKEND_ROOT}{os.pathsep}{sw.parent}{os.pathsep}{sw}"
        f"{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    return env


# Subprocess inference should execute the migrated New Backend package.
webcam_runner.webcam_command_cwd = _new_backend_cwd
thermal_runner.thermal_command_cwd = _new_backend_cwd
playground._infer_cwd = _new_backend_cwd
playground._software_env = _software_env_with_new_backend

