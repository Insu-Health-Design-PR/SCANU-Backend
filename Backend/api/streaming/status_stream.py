"""Server-sent status stream helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from runtime import sensor_runner  # noqa: E402


async def status_events(layer8_dir: Path) -> AsyncIterator[str]:
    try:
        while True:
            payload = {
                name: sensor_runner.status(name, layer8_dir)
                for name in ("thermal", "webcam", "multi_camera", "mmwave")
            }
            yield f"event: status\ndata: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        return

