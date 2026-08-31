"""Low-latency structured mmWave metrics stream."""
from __future__ import annotations

import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect


async def websocket_mmwave(websocket: WebSocket, ctx: object) -> None:
    await websocket.accept()
    layer8_dir = getattr(ctx, "layer8_dir")
    path = layer8_dir / "artifacts" / "live_mmwave_metrics.json"
    last_mtime_ns = -1
    try:
        while True:
            if path.is_file():
                try:
                    stat = path.stat()
                    if stat.st_mtime_ns != last_mtime_ns:
                        payload = json.loads(path.read_text())
                        await websocket.send_json(payload)
                        last_mtime_ns = stat.st_mtime_ns
                except (OSError, json.JSONDecodeError):
                    pass
            await asyncio.sleep(0.2)
    except (WebSocketDisconnect, asyncio.CancelledError):
        return
