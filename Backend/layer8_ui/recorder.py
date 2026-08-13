"""Session recording: local disk (default) or S3-compatible upload (stub hooks)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RecordingDestination(str, Enum):
    LOCAL = "local"
    S3 = "s3"


@dataclass
class RecorderConfig:
    destination: RecordingDestination = RecordingDestination.LOCAL
    local_dir: str = ""
    s3_bucket: str = ""
    s3_prefix: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Recorder:
    """Start/stop segmented recordings when the UI enables ``record`` (not wired yet)."""

    def __init__(self, config: RecorderConfig) -> None:
        self._config = config
        self._active: Path | None = None

    @property
    def config(self) -> RecorderConfig:
        return self._config

    def start_segment(self, name: str) -> Path | None:
        if self._config.destination is RecordingDestination.S3:
            return None
        base = Path(self._config.local_dir or ".").expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{name}.mp4"
        self._active = path
        return path

    def stop(self) -> None:
        self._active = None
