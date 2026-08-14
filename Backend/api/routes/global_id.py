"""Global Person ID / Re-ID API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from api.routes.context import RouterContext
from services.global_id_service import get_global_id_service, init_global_id_service
from weapon_ai.reid.config import ReIDConfig


def build_global_id_router(ctx: RouterContext) -> APIRouter:
    router = APIRouter(tags=["global_id"])

    def _ensure() -> Any:
        svc = get_global_id_service()
        if svc is not None:
            return svc
        settings = ctx.settings.get() if hasattr(ctx.settings, "get") else {}
        if not isinstance(settings, dict):
            settings = {}
        return init_global_id_service(ctx.layer8_dir, settings, start=True)

    @router.get("/api/global_id/status")
    def global_id_status() -> dict[str, Any]:
        svc = _ensure()
        snap = svc.snapshot()
        snap["enabled"] = bool(svc.config.enable)
        snap["state_path"] = str(svc.state_path)
        return snap

    @router.post("/api/global_id/tick")
    def global_id_tick() -> dict[str, Any]:
        """Force one association cycle (debug / tests)."""
        svc = _ensure()
        return svc.tick()

    @router.post("/api/global_id/reset")
    def global_id_reset() -> dict[str, Any]:
        svc = _ensure()
        svc.manager.reset()
        snap = svc.manager.snapshot()
        return {"ok": True, **snap}

    @router.get("/api/global_id/config")
    def global_id_config() -> dict[str, Any]:
        svc = _ensure()
        return svc.config.to_dict()

    @router.post("/api/global_id/reload_config")
    def global_id_reload_config() -> dict[str, Any]:
        settings = ctx.settings.get() if hasattr(ctx.settings, "get") else {}
        if not isinstance(settings, dict):
            settings = {}
        cfg = ReIDConfig.from_settings(settings)
        svc = _ensure()
        svc.configure(cfg)
        return {"ok": True, "config": cfg.to_dict()}

    return router
