"""Camera corridor alignment API for Front ↔ Back calibration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi import APIRouter, HTTPException

from api.routes.context import RouterContext
from services.global_id_service import get_global_id_service, init_global_id_service
from weapon_ai.reid.alignment import compute_alignment_status
from weapon_ai.reid.config import ReIDConfig

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from runtime import sensor_runner  # noqa: E402


def build_alignment_router(ctx: RouterContext) -> APIRouter:
    router = APIRouter(tags=["alignment"])

    def _settings() -> dict[str, Any]:
        raw = ctx.settings.get() if hasattr(ctx.settings, "get") else {}
        return raw if isinstance(raw, dict) else {}

    def _global_snapshot() -> dict[str, Any] | None:
        svc = get_global_id_service()
        if svc is None:
            try:
                svc = init_global_id_service(ctx.layer8_dir, _settings(), start=False)
            except Exception:
                return None
        if svc is None:
            return None
        try:
            return svc.snapshot()
        except Exception:
            return None

    @router.get("/api/alignment/status")
    def alignment_status() -> dict[str, Any]:
        settings = _settings()
        front = ctx.metrics.threat_metrics_for("webcam")
        back = ctx.metrics.threat_metrics_for("multi_camera")
        front_running = bool(sensor_runner.status("webcam", ctx.layer8_dir).get("running"))
        back_running = bool(sensor_runner.status("multi_camera", ctx.layer8_dir).get("running"))
        return compute_alignment_status(
            front_metrics=front,
            back_metrics=back,
            settings=settings,
            global_snapshot=_global_snapshot(),
            front_running=front_running,
            back_running=back_running,
        )

    @router.get("/api/alignment/config")
    def alignment_config() -> dict[str, Any]:
        settings = _settings()
        cfg = ReIDConfig.from_settings(settings)
        sent = settings.get("sentinel") if isinstance(settings.get("sentinel"), dict) else {}
        return {
            "sentinel": sent,
            "global_id": cfg.to_dict(),
        }

    @router.put("/api/alignment/config")
    def alignment_config_update(body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        settings = deepcopy(_settings())
        sent_patch = body.get("sentinel")
        gid_patch = body.get("global_id")
        if isinstance(sent_patch, dict):
            cur = settings.get("sentinel") if isinstance(settings.get("sentinel"), dict) else {}
            merged = {**cur, **sent_patch}
            settings["sentinel"] = merged
            # Keep global_id baseline in sync when sentinel baseline changes.
            if "baseline_m" in sent_patch:
                gcur = settings.get("global_id") if isinstance(settings.get("global_id"), dict) else {}
                gcur = {**gcur, "baseline_m": sent_patch["baseline_m"]}
                settings["global_id"] = gcur
        if isinstance(gid_patch, dict):
            cur = settings.get("global_id") if isinstance(settings.get("global_id"), dict) else {}
            merged = {**cur, **gid_patch}
            settings["global_id"] = merged
            if "baseline_m" in gid_patch:
                scur = settings.get("sentinel") if isinstance(settings.get("sentinel"), dict) else {}
                scur = {**scur, "baseline_m": gid_patch["baseline_m"]}
                settings["sentinel"] = scur
        saved = ctx.settings.replace(settings)
        cfg = ReIDConfig.from_settings(saved)
        svc = get_global_id_service()
        if svc is not None:
            svc.configure(cfg)
        return {"ok": True, "config": alignment_config()}

    return router
