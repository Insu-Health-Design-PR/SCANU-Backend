"""Global Re-ID should crop from the 4K sidecar, not the 1080p preview."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from media.ipc.paths import derived_full_bgr_ipc_path
from services.global_id_service import GlobalIDService, scale_xyxy
from weapon_ai.reid.config import ReIDConfig
from weapon_ai.reid.embeddings import MockReIDEmbedder


def test_derived_full_bgr_ipc_path():
    preview = Path("/dev/shm/scanu_webcam_live_bgr_frame.bin")
    assert derived_full_bgr_ipc_path(preview) == Path("/dev/shm/scanu_webcam_live_bgr_full.bin")


def test_scale_xyxy_1080p_to_4k():
    assert scale_xyxy((100, 50, 200, 250), 1920, 1080, 3840, 2160) == (200, 100, 400, 500)


def test_scale_xyxy_same_size_is_noop():
    box = (10, 20, 30, 40)
    assert scale_xyxy(box, 1920, 1080, 1920, 1080) == box


class _RecordingEmbedder(MockReIDEmbedder):
    def __init__(self) -> None:
        super().__init__(dim=64)
        self.crops: list[tuple[int, int]] = []

    def embed(self, bgr_crop, *, identity_hint=None):
        if bgr_crop is not None and getattr(bgr_crop, "size", 0):
            self.crops.append((int(bgr_crop.shape[0]), int(bgr_crop.shape[1])))
        return super().embed(bgr_crop, identity_hint=identity_hint)


class _FullFrameProvider:
    def __init__(self, full: np.ndarray, preview: np.ndarray | None = None) -> None:
        self._full = full
        self._preview = preview

    def latest_full_bgr(self) -> np.ndarray | None:
        return self._full

    def latest_bgr(self) -> np.ndarray | None:
        return self._preview


def test_global_id_crops_scaled_4k_box(tmp_path: Path):
    full = np.zeros((2160, 3840, 3), dtype=np.uint8)
    full[100:500, 200:400] = (0, 0, 255)
    preview = np.zeros((1080, 1920, 3), dtype=np.uint8)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        """
        {
          "ts": 1.0,
          "frame_w": 1920,
          "frame_h": 1080,
          "byte_tracks": [
            {"display_id": 3, "bbox": [100, 50, 200, 250], "weapon_gun_conf": 0.0}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    svc = GlobalIDService(
        tmp_path,
        config=ReIDConfig(enable=True, embed_interval_s=0.01, embed_min_box_px=20),
        state_path=tmp_path / "ids.json",
        front_metrics=metrics,
        back_metrics=tmp_path / "missing.json",
        use_mock_embedder=True,
    )
    rec = _RecordingEmbedder()
    svc.cache.embedder = rec
    svc.set_frame_provider(svc.config.camera_front, _FullFrameProvider(full, preview))
    svc.tick(now=1.0)
    assert rec.crops == [(400, 200)]


def test_global_id_falls_back_to_preview_when_no_4k(tmp_path: Path):
    preview = np.zeros((1080, 1920, 3), dtype=np.uint8)
    preview[50:250, 100:200] = (0, 255, 0)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        """
        {
          "ts": 1.0,
          "frame_w": 1920,
          "frame_h": 1080,
          "byte_tracks": [
            {"display_id": 3, "bbox": [100, 50, 200, 250], "weapon_gun_conf": 0.0}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    class _PreviewOnly:
        def latest_bgr(self):
            return preview

    svc = GlobalIDService(
        tmp_path,
        config=ReIDConfig(enable=True, embed_interval_s=0.01, embed_min_box_px=20),
        state_path=tmp_path / "ids.json",
        front_metrics=metrics,
        back_metrics=tmp_path / "missing.json",
        use_mock_embedder=True,
    )
    rec = _RecordingEmbedder()
    svc.cache.embedder = rec
    svc.set_frame_provider(svc.config.camera_front, _PreviewOnly())
    svc.tick(now=1.0)
    assert rec.crops == [(200, 100)]
