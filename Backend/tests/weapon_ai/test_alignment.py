"""Tests for cross-camera alignment scoring."""

from __future__ import annotations

import numpy as np

from weapon_ai.reid.alignment import (
    compute_alignment_status,
    depth_corridor_score,
    find_alignment_pairs,
    score_pair,
    tracks_from_metrics,
)
from weapon_ai.reid.config import ReIDConfig
from weapon_ai.reid.alignment import AlignmentTrack


def test_depth_corridor_score_perfect():
    score, residual = depth_corridor_score(3.0, 2.0, baseline_m=5.0, tolerance_frac=0.22)
    assert score > 0.99
    assert residual is not None and abs(residual) < 0.01


def test_score_pair_aligned_when_geometry_matches():
    cfg = ReIDConfig(baseline_m=5.0, depth_tolerance_frac=0.22, lateral_tolerance_frac=0.20)
    front = AlignmentTrack("camera_1", 1, (100, 100, 200, 400), lateral_norm=0.20, depth_m=2.0)
    back = AlignmentTrack("camera_2", 2, (800, 100, 900, 400), lateral_norm=0.80, depth_m=3.0)
    out = score_pair(front, back, cfg)
    assert out["checks"]["lateral"]["ok"] is True
    assert out["checks"]["depth"]["ok"] is True
    assert out["aligned"] is True


def test_find_pairs_respects_global_id():
    cfg = ReIDConfig(baseline_m=5.0)
    front = AlignmentTrack("camera_1", 1, None, lateral_norm=0.2, depth_m=2.0, global_id=7)
    back = AlignmentTrack("camera_2", 9, None, lateral_norm=0.8, depth_m=3.0, global_id=7)
    other = AlignmentTrack("camera_2", 3, None, lateral_norm=0.5, depth_m=1.0, global_id=2)
    pairs = find_alignment_pairs([front], [back, other], cfg)
    assert pairs
    assert pairs[0]["front_track_id"] == 1
    assert pairs[0]["back_track_id"] == 9


def test_compute_alignment_status_no_person():
    out = compute_alignment_status(
        front_metrics={"byte_tracks": [], "frame_w": 1920, "frame_h": 1080},
        back_metrics={"byte_tracks": [], "frame_w": 1920, "frame_h": 1080},
        settings={"sentinel": {"baseline_m": 5}, "global_id": {"baseline_m": 5}},
        front_running=True,
        back_running=True,
    )
    assert out["phase"] == "no_person"
    assert out["aligned"] is False


def test_tracks_from_metrics_computes_lateral_and_depth():
    metrics = {
        "frame_w": 1000,
        "frame_h": 800,
        "byte_tracks": [{"display_id": 1, "bbox": [200, 100, 400, 500]}],
    }
    tracks, w, h = tracks_from_metrics(metrics, camera_id="camera_1")
    assert w == 1000 and h == 800
    assert len(tracks) == 1
    assert tracks[0].lateral_norm == 0.3
    assert tracks[0].depth_m is not None and tracks[0].depth_m > 0


def test_reid_in_pair_when_embeddings_present():
    cfg = ReIDConfig(baseline_m=5.0, soft_similarity_threshold=0.58)
    e = np.random.randn(512).astype(np.float32)
    e = e / (np.linalg.norm(e) + 1e-8)
    front = AlignmentTrack("camera_1", 1, None, lateral_norm=0.2, depth_m=2.0, embedding=e)
    back = AlignmentTrack("camera_2", 2, None, lateral_norm=0.8, depth_m=3.0, embedding=e)
    out = score_pair(front, back, cfg)
    assert out["checks"]["reid"]["score"] is not None
    assert out["checks"]["reid"]["score"] > 0.99
