"""Model profile routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.routes.context import RouterContext
from api.schemas.profiles import (
    ApplyModelProfileBody,
    ApplyModelProfileByNameBody,
    SaveProfileIfNewBody,
    SnapshotModelProfileBody,
)
from services import model_profiles as profiles


def build_profiles_router(ctx: RouterContext) -> APIRouter:
    layer8_dir = ctx.layer8_dir
    router = APIRouter(tags=["profiles"])

    @router.get("/api/model/profiles")
    def get_model_profiles() -> dict[str, Any]:
        return {"profiles": profiles.get_model_profiles_normalized(layer8_dir)}

    @router.get("/api/ai_camera/profiles")
    def get_ai_camera_profiles() -> dict[str, Any]:
        return {"profiles": profiles.ai_camera_profiles_public_list(layer8_dir)}

    @router.put("/api/model/profiles")
    def put_model_profiles(body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        if "profiles" in body and isinstance(body["profiles"], dict):
            norm_in = body["profiles"]
        else:
            norm_in = {k: v for k, v in body.items() if k not in profiles.PROFILE_FILE_META_KEYS}
        norm: dict[str, dict[str, Any]] = {}
        for pid, v in norm_in.items():
            ent = profiles.coerce_profile_entry(pid, v)
            if ent:
                norm[str(pid)] = ent
        profiles.save_model_profiles(layer8_dir, profiles.serialize_profiles_to_disk(norm))
        return {"profiles": profiles.get_model_profiles_normalized(layer8_dir)}

    @router.post("/api/model/profiles/apply")
    def apply_model_profile(body: ApplyModelProfileBody) -> dict[str, Any]:
        pid = body.id.strip()
        if not pid:
            raise HTTPException(400, "id is required")
        norm = profiles.get_model_profiles_normalized(layer8_dir)
        prof = norm.get(pid)
        if prof is None:
            raise HTTPException(404, "profile not found")
        values = prof.get("values") or {}
        if not isinstance(values, dict):
            raise HTTPException(400, "profile.values must be an object")
        current = ctx.settings.get()
        w = profiles.apply_values_to_webcam({**(current.get("webcam") or {})}, values)
        w["active_model_profile_id"] = pid
        current["webcam"] = w
        current["thermal"] = profiles.apply_weapon_profile_to_thermal(current.get("thermal") or {}, values)
        return ctx.settings.replace(current)

    @router.post("/api/ai_camera/profiles/apply_by_name")
    def apply_ai_camera_profile_by_name(body: ApplyModelProfileByNameBody) -> dict[str, Any]:
        raw_name = body.name.strip()
        if not raw_name:
            raise HTTPException(400, "name is required")
        norm = profiles.get_model_profiles_normalized(layer8_dir)
        matches = profiles.profile_ids_matching_name(norm, raw_name)
        if not matches:
            raise HTTPException(404, "no profile with this name")
        if len(matches) > 1:
            raise HTTPException(
                status_code=409,
                detail={"message": "multiple profiles share this name", "matching_ids": matches},
            )
        pid = matches[0]
        prof = norm.get(pid) or {}
        values = prof.get("values") or {}
        if not isinstance(values, dict):
            raise HTTPException(400, "profile.values must be an object")
        current = ctx.settings.get()
        w = profiles.apply_values_to_webcam({**(current.get("webcam") or {})}, values)
        w["active_model_profile_id"] = pid
        current["webcam"] = w
        current["thermal"] = profiles.apply_weapon_profile_to_thermal(current.get("thermal") or {}, values)
        ctx.settings.replace(current)
        applied = ctx.settings.get()
        return {
            "ok": True,
            "applied_profile_id": pid,
            "applied_profile_name": str(prof.get("label") or pid),
            "settings": applied,
        }

    @router.post("/api/model/profiles/snapshot")
    def snapshot_model_profile(body: SnapshotModelProfileBody) -> dict[str, Any]:
        s = ctx.settings.get()
        w_prev = s.get("webcam") or {}
        if body.values is not None and isinstance(body.values, dict):
            w_merged = profiles.apply_values_to_webcam(dict(w_prev), body.values)
            snap = profiles.extract_profile_values(w_merged)
        else:
            snap = profiles.extract_profile_values(w_prev)
        norm = profiles.get_model_profiles_normalized(layer8_dir)
        existing = set(norm.keys())
        pid = (body.id or "").strip()
        if not pid:
            name_in = (body.name or "").strip()
            if not name_in:
                raise HTTPException(400, "name is required")
            pid = profiles.unique_profile_key_from_name(name_in, existing)
        name = (body.name or "").strip() or (norm.get(pid) or {}).get("label") or pid
        desc = (body.description or "").strip()
        prev = norm.get(pid)
        entry: dict[str, Any] = {
            "label": name,
            "description": desc if desc else (prev.get("description", "") if prev else ""),
            "values": snap,
        }
        if prev:
            if not (body.name or "").strip():
                entry["label"] = prev.get("label", pid)
            if not desc:
                entry["description"] = prev.get("description", "")
        norm[pid] = entry
        profiles.save_model_profiles(layer8_dir, profiles.serialize_profiles_to_disk(norm))
        return {"profiles": norm, "saved_as": pid}

    @router.post("/api/model/profiles/save_if_new")
    def save_model_profile_if_new(body: SaveProfileIfNewBody) -> dict[str, Any]:
        """Save profile only when no existing profile shares the same display name."""
        name = (body.name or "").strip()
        if not name:
            raise HTTPException(400, "name is required")
        exists, existing_id = profiles.profile_exists_by_name(layer8_dir, name)
        if exists:
            return {
                "skipped": True,
                "reason": "profile_exists",
                "existing_id": existing_id,
                "profiles": profiles.get_model_profiles_normalized(layer8_dir),
            }
        snap_body = SnapshotModelProfileBody(
            name=name,
            description=body.description,
            values=body.values,
        )
        s = ctx.settings.get()
        w_prev = s.get("webcam") or {}
        if snap_body.values is not None and isinstance(snap_body.values, dict):
            w_merged = profiles.apply_values_to_webcam(dict(w_prev), snap_body.values)
            snap = profiles.extract_profile_values(w_merged)
        else:
            snap = profiles.extract_profile_values(w_prev)
        norm = profiles.get_model_profiles_normalized(layer8_dir)
        pid = profiles.unique_profile_key_from_name(name, set(norm.keys()))
        entry: dict[str, Any] = {
            "label": name,
            "description": (body.description or "").strip(),
            "values": snap,
        }
        norm[pid] = entry
        profiles.save_model_profiles(layer8_dir, profiles.serialize_profiles_to_disk(norm))
        return {
            "skipped": False,
            "saved_as": pid,
            "profiles": norm,
        }

    @router.post("/api/model/profiles/sync_from_config")
    def sync_profile_from_config(body: ApplyModelProfileBody) -> dict[str, Any]:
        """Merge current ui_settings webcam (camera + model keys) into profile.values."""
        pid = body.id.strip()
        if not pid:
            raise HTTPException(400, "id is required")
        s = ctx.settings.get()
        w = s.get("webcam") or {}
        snap = profiles.extract_profile_values(w)
        norm = profiles.get_model_profiles_normalized(layer8_dir)
        prev = norm.get(pid)
        if prev is None:
            norm[pid] = {"label": pid, "description": "", "values": dict(snap)}
        else:
            merged_vals = {**(prev.get("values") or {}), **snap}
            norm[pid] = {**prev, "values": merged_vals}
        profiles.save_model_profiles(layer8_dir, profiles.serialize_profiles_to_disk(norm))
        return {"profiles": norm}

    @router.get("/api/multi_camera/profiles")
    def get_multi_camera_profiles() -> dict[str, Any]:
        return {"profiles": profiles.get_model_profiles_normalized(layer8_dir)}

    @router.post("/api/multi_camera/profiles/apply")
    def apply_multi_camera_profile(body: ApplyModelProfileBody) -> dict[str, Any]:
        pid = body.id.strip()
        if not pid:
            raise HTTPException(400, "id is required")
        norm = profiles.get_model_profiles_normalized(layer8_dir)
        prof = norm.get(pid)
        if prof is None:
            raise HTTPException(404, "profile not found")
        values = prof.get("values") or {}
        if not isinstance(values, dict):
            raise HTTPException(400, "profile.values must be an object")
        current = ctx.settings.get()
        mc = profiles.apply_values_to_multi_camera({**(current.get("multi_camera") or {})}, values)
        mc["active_model_profile_id"] = pid
        current["multi_camera"] = mc
        return ctx.settings.replace(current)

    @router.post("/api/multi_camera/profiles/snapshot")
    def snapshot_multi_camera_profile(body: SnapshotModelProfileBody) -> dict[str, Any]:
        s = ctx.settings.get()
        mc_prev = s.get("multi_camera") or {}
        if body.values is not None and isinstance(body.values, dict):
            mc_merged = profiles.apply_values_to_multi_camera(dict(mc_prev), body.values)
            snap = profiles.extract_profile_values(mc_merged)
        else:
            snap = profiles.extract_profile_values(mc_prev)
        norm = profiles.get_model_profiles_normalized(layer8_dir)
        existing = set(norm.keys())
        pid = (body.id or "").strip()
        if not pid:
            name_in = (body.name or "").strip()
            if not name_in:
                raise HTTPException(400, "name is required")
            pid = profiles.unique_profile_key_from_name(name_in, existing)
        name = (body.name or "").strip() or (norm.get(pid) or {}).get("label") or pid
        desc = (body.description or "").strip()
        prev = norm.get(pid)
        entry: dict[str, Any] = {
            "label": name,
            "description": desc if desc else (prev.get("description", "") if prev else ""),
            "values": snap,
        }
        if prev:
            if not (body.name or "").strip():
                entry["label"] = prev.get("label", pid)
            if not desc:
                entry["description"] = prev.get("description", "")
        norm[pid] = entry
        profiles.save_model_profiles(layer8_dir, profiles.serialize_profiles_to_disk(norm))
        return {"profiles": norm, "saved_as": pid}

    @router.post("/api/multi_camera/profiles/sync_from_config")
    def sync_multi_camera_profile_from_config(body: ApplyModelProfileBody) -> dict[str, Any]:
        """Merge current ui_settings multi_camera (camera + model keys) into profile.values."""
        pid = body.id.strip()
        if not pid:
            raise HTTPException(400, "id is required")
        s = ctx.settings.get()
        mc = s.get("multi_camera") or {}
        snap = profiles.extract_profile_values(mc)
        norm = profiles.get_model_profiles_normalized(layer8_dir)
        prev = norm.get(pid)
        if prev is None:
            norm[pid] = {"label": pid, "description": "", "values": dict(snap)}
        else:
            merged_vals = {**(prev.get("values") or {}), **snap}
            norm[pid] = {**prev, "values": merged_vals}
        profiles.save_model_profiles(layer8_dir, profiles.serialize_profiles_to_disk(norm))
        return {"profiles": norm}

    return router
