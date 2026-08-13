"""PID registry, logs, and state file for child processes."""

import json
from pathlib import Path

from utils.paths import artifacts_dir

STATE_FILE = "process_registry.json"


class ProcessRegistry:
    def __init__(self, path: Path | None = None):
        self.path = path or (artifacts_dir() / STATE_FILE)

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")
