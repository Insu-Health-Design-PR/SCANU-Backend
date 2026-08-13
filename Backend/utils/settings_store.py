"""ui_settings.json load/save."""

import json
from pathlib import Path

from utils.paths import config_dir

SETTINGS_FILE = "settings.default.json"


def settings_path() -> Path:
    return config_dir() / SETTINGS_FILE


def load_settings() -> dict:
    path = settings_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_settings(data: dict) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
