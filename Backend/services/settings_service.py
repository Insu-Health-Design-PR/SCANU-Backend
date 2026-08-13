"""Settings service wrapper for the current Layer 8 settings store."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from layer8_ui.settings_store import (  # noqa: E402
    DEFAULT_SETTINGS,
    load,
    reset_thermal_weapon_defaults,
    reset_multi_camera_weapon_defaults,
    reset_webcam_weapon_defaults,
    save,
)


class SettingsService:
    def __init__(self, layer8_dir: Path) -> None:
        self.layer8_dir = Path(layer8_dir)

    def get(self) -> dict[str, Any]:
        return load(self.layer8_dir)

    def replace(self, settings: dict[str, Any]) -> dict[str, Any]:
        save(self.layer8_dir, settings)
        return self.get()

    def reset_all(self) -> dict[str, Any]:
        save(self.layer8_dir, deepcopy(DEFAULT_SETTINGS))
        return self.get()

    def reset_sensor(self, sensor: str) -> dict[str, Any]:
        current = self.get()
        current[sensor] = deepcopy(DEFAULT_SETTINGS[sensor])
        save(self.layer8_dir, current)
        return self.get()

    def reset_webcam_model(self) -> dict[str, Any]:
        return reset_webcam_weapon_defaults(self.layer8_dir)

    def reset_thermal_model(self) -> dict[str, Any]:
        return reset_thermal_weapon_defaults(self.layer8_dir)

    def reset_multi_camera_model(self) -> dict[str, Any]:
        return reset_multi_camera_weapon_defaults(self.layer8_dir)
