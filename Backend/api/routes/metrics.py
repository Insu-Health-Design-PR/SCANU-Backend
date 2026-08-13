"""Dashboard and threat metrics routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from api.routes.context import RouterContext


def build_metrics_router(ctx: RouterContext) -> APIRouter:
    router = APIRouter(tags=["metrics"])

    @router.get("/api/system/metrics")
    def get_system_metrics() -> dict[str, Any]:
        """CPU / RAM / optional NVIDIA GPU for the dashboard header."""
        return ctx.metrics.system_snapshot()

    @router.get("/api/threat/metrics")
    def live_threat_metrics() -> dict[str, Any]:
        p = Path(__file__).resolve().parent.parent.parent / "layer8_ui" / "artifacts" / "live_threat_metrics.json"
        if p.exists():
            return json.loads(p.read_text())
        return {"gun_detected": False, "persons_total": 0}

    @router.get("/api/dashboard/metrics")
    def dashboard_metrics() -> dict[str, Any]:
        return ctx.metrics.dashboard_metrics()

    return router
