"""Operator info and debug command routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.routes.context import RouterContext
from api.schemas.sensors import SensorName


def build_info_router(ctx: RouterContext) -> APIRouter:
    router = APIRouter(tags=["info"])

    @router.get("/api/layer8/module_map")
    def layer8_module_map() -> dict[str, str]:
        """Which backend modules own each tab (for operators / integration)."""
        return {
            "thermal_tab": (
                "Live thermal infer: runtime.thermal_runner → runtime.thermal_layer8_runner "
                "→ weapon_ai.infer_thermal_objects (--thermal_v4l2, inferno colormap + overlay IPC)."
            ),
            "webcam_tab": (
                "Front Cam: runtime.webcam_runner → runtime.webcam_layer8_runner "
                "→ weapon_ai.infer_objects (local USB or Jetson IP RTSP/HTTP). "
                "API: /api/front_camera/* (alias /api/ai_camera/*)."
            ),
            "model_tab": (
                "Front Cam model settings share the webcam sensor block "
                "(infer_objects / multi-class overlays)."
            ),
            "multi_camera_tab": (
                "Back Cam: runtime.multi_camera_runner → runtime.multi_camera_layer8_runner "
                "→ weapon_ai.infer_objects (local USB or Jetson IP RTSP/HTTP source). "
                "API: /api/back_camera/* (alias /api/multi_camera/*)."
            ),
            "mmwave_tab": "runtime.sensor_runner (mmWave CLI)",
        }

    @router.get("/api/command/{sensor}")
    def preview_command(sensor: SensorName) -> dict[str, Any]:
        if sensor not in ("thermal", "webcam", "multi_camera", "mmwave"):
            raise HTTPException(400, "invalid sensor")
        s = ctx.settings.get()
        return ctx.sensors.build_command(sensor, s)

    @router.get("/api/model/options")
    def model_options() -> dict[str, Any]:
        from layer8_ui.artifact_paths import software_root_from_settings

        s = ctx.settings.get()
        sw = software_root_from_settings(s)
        gun_dir = sw / "trained_models" / "gun_detection"
        checkpoints: list[str] = []
        if gun_dir.is_dir():
            checkpoints = sorted(
                {p.name for p in gun_dir.glob("*.pt")} | {p.name for p in gun_dir.glob("*.engine")}
            )
        gun_suggestions = [
            "gun_enhanced_cctv_v2.pt",
            "gun_enhanced_v1_test.pt",
            "gun_sohas_6class.pt",
            "gun_sohas_6class_modern_phone.pt",
            "gun_sohas_6class_full_v2.pt",
            "gun_sohas_7class_user_v2.engine",
        ]
        gun_checkpoints = sorted(set(checkpoints) | set(gun_suggestions))
        suggestions = ["yolov8n.pt", "yolov8n.engine", "yolov8s.pt", "yolov8m.pt"]
        person_dir = sw / "trained_models" / "person_detection"
        person_pts: set[str] = set(suggestions)
        if person_dir.is_dir():
            person_pts |= {p.name for p in person_dir.glob("*.pt")}
            person_pts |= {p.name for p in person_dir.glob("*.engine")}
        person_yolo_options = sorted(person_pts)
        pose_dir = sw / "trained_models" / "pose"
        pose_suggestions = ["yolov8s-pose.pt", "yolov8n-pose.pt", "person_holding_pose_unity_v1.pt"]
        pose_pts: set[str] = set(pose_suggestions)
        if pose_dir.is_dir():
            pose_pts |= {p.name for p in pose_dir.glob("*.pt")}
            pose_pts |= {p.name for p in pose_dir.glob("*.engine")}
        pose_yolo_options = sorted(pose_pts)
        return {
            "gun_checkpoints": gun_checkpoints,
            "gun_checkpoint_suggestions": gun_suggestions,
            "person_yolo_suggestions": suggestions,
            "person_yolo_options": person_yolo_options,
            "pose_yolo_options": pose_yolo_options,
            "pose_yolo_suggestions": pose_suggestions,
        }

    return router
