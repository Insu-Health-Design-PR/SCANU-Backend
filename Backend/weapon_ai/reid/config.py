"""Configuration for Person Re-ID and Global ID association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ReIDConfig:
    """Tunables for embedding cadence, matching, and weapon persistence."""

    enable: bool = False
    # Embedding
    embed_interval_s: float = 0.5
    embed_min_box_px: int = 40
    embed_device: str = "auto"  # cuda | cpu | auto
    # auto | tensorrt | onnx | torchreid | torchvision | mock
    embed_backend: str = "auto"
    embed_model: str = "osnet_x0_25"
    embed_weights: str = "trained_models/reid/osnet_x0_25_msmt17.pth"
    embed_onnx: str = "trained_models/reid/osnet_x0_25_msmt17.onnx"
    embed_engine: str = "trained_models/reid/osnet_x0_25_msmt17.engine"
    embed_input_h: int = 256
    embed_input_w: int = 128
    # Matching
    similarity_threshold: float = 0.72
    soft_similarity_threshold: float = 0.58  # needs extra geometry/temporal support
    max_embedding_history: int = 12
    track_timeout_s: float = 8.0
    # Scoring weights (cosine + corridor depth + temporal freshness)
    weight_reid: float = 0.35
    weight_depth: float = 0.55
    weight_temporal: float = 0.10
    baseline_m: float = 5.0
    depth_tolerance_frac: float = 0.22  # |d_a + d_b - B| / B
    # Facing cameras mirror left/right: cx_front + cx_back ≈ 1
    lateral_tolerance_frac: float = 0.20
    # Corridor depth boost when Re-ID is soft (d_front + d_back ≈ baseline)
    depth_boost_min_depth: float = 0.65
    depth_boost_min_reid: float = 0.48
    depth_boost_strong_depth: float = 0.85
    depth_boost_strong_min_reid: float = 0.45
    # Weapon state on GLOBAL person
    weapon_hold_s: float = 3.0
    weapon_conf_decay: float = 0.15  # per second without refresh
    weapon_latch: bool = False  # once armed globally, never clear until track expires
    # Camera naming
    camera_front: str = "camera_1"
    camera_back: str = "camera_2"

    @classmethod
    def from_settings(cls, settings: dict[str, Any] | None) -> "ReIDConfig":
        """Build from ``settings['global_id']`` / ``settings['sentinel']`` blocks."""
        raw = {}
        if isinstance(settings, dict):
            raw = dict(settings.get("global_id") or {})
            sent = settings.get("sentinel") if isinstance(settings.get("sentinel"), dict) else {}
            if "baseline_m" not in raw and sent.get("baseline_m") is not None:
                raw["baseline_m"] = sent.get("baseline_m")
            if "enable" not in raw and sent.get("reid_enable") is not None:
                raw["enable"] = sent.get("reid_enable")
        cfg = cls()
        for key, val in raw.items():
            if not hasattr(cfg, key) or val is None or str(val).strip() == "":
                continue
            cur = getattr(cfg, key)
            try:
                if isinstance(cur, bool):
                    setattr(
                        cfg,
                        key,
                        bool(int(val))
                        if str(val).strip().lstrip("-").isdigit()
                        else str(val).strip().lower() in {"1", "true", "yes", "on"},
                    )
                elif isinstance(cur, int) and not isinstance(cur, bool):
                    setattr(cfg, key, int(val))
                elif isinstance(cur, float):
                    setattr(cfg, key, float(val))
                elif isinstance(cur, str):
                    setattr(cfg, key, str(val))
            except (TypeError, ValueError):
                continue
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable": bool(self.enable),
            "embed_interval_s": float(self.embed_interval_s),
            "embed_min_box_px": int(self.embed_min_box_px),
            "embed_device": str(self.embed_device),
            "embed_backend": str(self.embed_backend),
            "embed_model": str(self.embed_model),
            "embed_weights": str(self.embed_weights),
            "embed_onnx": str(self.embed_onnx),
            "embed_engine": str(self.embed_engine),
            "embed_input_h": int(self.embed_input_h),
            "embed_input_w": int(self.embed_input_w),
            "similarity_threshold": float(self.similarity_threshold),
            "soft_similarity_threshold": float(self.soft_similarity_threshold),
            "max_embedding_history": int(self.max_embedding_history),
            "track_timeout_s": float(self.track_timeout_s),
            "weight_reid": float(self.weight_reid),
            "weight_depth": float(self.weight_depth),
            "weight_temporal": float(self.weight_temporal),
            "baseline_m": float(self.baseline_m),
            "depth_tolerance_frac": float(self.depth_tolerance_frac),
            "lateral_tolerance_frac": float(self.lateral_tolerance_frac),
            "depth_boost_min_depth": float(self.depth_boost_min_depth),
            "depth_boost_min_reid": float(self.depth_boost_min_reid),
            "depth_boost_strong_depth": float(self.depth_boost_strong_depth),
            "depth_boost_strong_min_reid": float(self.depth_boost_strong_min_reid),
            "weapon_hold_s": float(self.weapon_hold_s),
            "weapon_conf_decay": float(self.weapon_conf_decay),
            "weapon_latch": bool(self.weapon_latch),
            "camera_front": str(self.camera_front),
            "camera_back": str(self.camera_back),
        }


# Concrete defaults for settings_store
GLOBAL_ID_SETTINGS_DEFAULTS: dict[str, Any] = {
    "enable": 0,
    "embed_interval_s": 0.5,
    "embed_min_box_px": 40,
    "embed_device": "auto",
    "embed_backend": "auto",
    "embed_model": "osnet_x0_25",
    "embed_weights": "trained_models/reid/osnet_x0_25_msmt17.pth",
    "embed_onnx": "trained_models/reid/osnet_x0_25_msmt17.onnx",
    "embed_engine": "trained_models/reid/osnet_x0_25_msmt17.engine",
    "embed_input_h": 256,
    "embed_input_w": 128,
    "similarity_threshold": 0.72,
    "soft_similarity_threshold": 0.58,
    "max_embedding_history": 12,
    "track_timeout_s": 8.0,
    "weight_reid": 0.35,
    "weight_depth": 0.55,
    "weight_temporal": 0.10,
    "baseline_m": 5.0,
    "depth_tolerance_frac": 0.22,
    "lateral_tolerance_frac": 0.20,
    "depth_boost_min_depth": 0.65,
    "depth_boost_min_reid": 0.48,
    "depth_boost_strong_depth": 0.85,
    "depth_boost_strong_min_reid": 0.45,
    "weapon_hold_s": 3.0,
    "weapon_conf_decay": 0.15,
    "camera_front": "camera_1",
    "camera_back": "camera_2",
    "state_json": "layer8_ui/configs/global_person_ids.json",
    "note": (
        "Cross-camera Person Re-ID + Global ID. Keep Front/Back infer independent; "
        "association runs in the API Global ID service."
    ),
}
