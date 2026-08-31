"""Simulation tests for cross-camera Global ID / Re-ID association."""

from __future__ import annotations

import numpy as np

from weapon_ai.reid.config import ReIDConfig
from weapon_ai.reid.embeddings import MockReIDEmbedder, cosine_similarity
from weapon_ai.reid.global_manager import GlobalIDManager, LocalObservation


def _emb(identity: int) -> np.ndarray:
    return MockReIDEmbedder(dim=64).embed(None, identity_hint=identity)


def test_same_person_across_cameras_gets_same_global_id():
    cfg = ReIDConfig(
        similarity_threshold=0.70,
        soft_similarity_threshold=0.55,
        baseline_m=10.0,
        track_timeout_s=8.0,
        weapon_hold_s=2.5,
    )
    mgr = GlobalIDManager(cfg)
    e = _emb(7)
    mgr.update_observations(
        [
            LocalObservation(
                camera_id="camera_1",
                local_track_id=4,
                embedding=e,
                depth_m=3.0,
                weapon_detected=True,
                weapon_confidence=0.91,
                timestamp=1.0,
            )
        ],
        now=1.0,
    )
    snap = mgr.update_observations(
        [
            LocalObservation(
                camera_id="camera_1",
                local_track_id=4,
                embedding=e,
                depth_m=3.0,
                weapon_detected=True,
                weapon_confidence=0.91,
                timestamp=1.2,
            ),
            LocalObservation(
                camera_id="camera_2",
                local_track_id=12,
                embedding=e,  # same identity embedding
                depth_m=7.0,  # 3+7≈10 baseline
                weapon_detected=False,
                weapon_confidence=0.0,
                timestamp=1.2,
            ),
        ],
        now=1.2,
    )
    g1 = mgr.get_global_id("camera_1", 4)
    g2 = mgr.get_global_id("camera_2", 12)
    assert g1 is not None and g1 == g2
    person = mgr.persons[g1]
    assert person.weapon_detected is True
    assert person.weapon_confidence >= 0.9
    assert snap["cameras"]["camera_2"]["12"]["weapon_detected"] is True


def test_different_people_are_not_merged():
    cfg = ReIDConfig(similarity_threshold=0.85, soft_similarity_threshold=0.80)
    mgr = GlobalIDManager(cfg)
    e_a = _emb(1)
    e_b = _emb(99)
    assert cosine_similarity(e_a, e_b) < 0.85
    mgr.update_observations(
        [
            LocalObservation("camera_1", 1, embedding=e_a, timestamp=1.0),
            LocalObservation("camera_2", 2, embedding=e_b, timestamp=1.0),
        ],
        now=1.0,
    )
    assert mgr.get_global_id("camera_1", 1) != mgr.get_global_id("camera_2", 2)


def test_reappearance_preserves_identity_within_timeout():
    cfg = ReIDConfig(similarity_threshold=0.70, track_timeout_s=5.0)
    mgr = GlobalIDManager(cfg)
    e = _emb(3)
    mgr.update_observations(
        [LocalObservation("camera_1", 5, embedding=e, timestamp=1.0)],
        now=1.0,
    )
    gid = mgr.get_global_id("camera_1", 5)
    # Gap shorter than timeout — reappear on other camera
    mgr.update_observations(
        [LocalObservation("camera_2", 8, embedding=e, timestamp=2.5)],
        now=2.5,
    )
    assert mgr.get_global_id("camera_2", 8) == gid


def test_weapon_state_propagates_and_persists():
    cfg = ReIDConfig(similarity_threshold=0.70, weapon_hold_s=2.0, weapon_conf_decay=0.5)
    mgr = GlobalIDManager(cfg)
    e = _emb(11)
    mgr.update_observations(
        [
            LocalObservation(
                "camera_1",
                4,
                embedding=e,
                weapon_detected=True,
                weapon_confidence=0.91,
                timestamp=1.0,
            )
        ],
        now=1.0,
    )
    mgr.update_observations(
        [
            LocalObservation(
                "camera_1",
                4,
                embedding=e,
                weapon_detected=False,
                weapon_confidence=0.0,
                timestamp=1.5,
            ),
            LocalObservation(
                "camera_2",
                12,
                embedding=e,
                weapon_detected=False,
                weapon_confidence=0.0,
                timestamp=1.5,
            ),
        ],
        now=1.5,
    )
    gid = mgr.get_global_id("camera_2", 12)
    assert gid is not None
    assert mgr.persons[gid].weapon_detected is True  # still within hold window
    snapshot = mgr.snapshot()
    assert snapshot["cameras"]["camera_1"]["4"]["weapon_detected"] is True
    assert snapshot["cameras"]["camera_2"]["12"]["weapon_detected"] is True
    assert snapshot["cameras"]["camera_2"]["12"]["weapon_confidence"] == 0.91

    # After hold + decay → clear
    mgr.update_observations(
        [
            LocalObservation("camera_2", 12, embedding=e, timestamp=5.0),
        ],
        now=5.0,
    )
    assert mgr.persons[gid].weapon_detected is False


def test_low_confidence_reid_creates_new_global_id():
    cfg = ReIDConfig(similarity_threshold=0.95, soft_similarity_threshold=0.94)
    mgr = GlobalIDManager(cfg)
    # Nearly orthogonal random embeddings → should not match
    e1 = _emb(1)
    e2 = _emb(2)
    mgr.update_observations(
        [LocalObservation("camera_1", 1, embedding=e1, timestamp=1.0)],
        now=1.0,
    )
    mgr.update_observations(
        [LocalObservation("camera_2", 2, embedding=e2, timestamp=1.1)],
        now=1.1,
    )
    assert mgr.get_global_id("camera_1", 1) != mgr.get_global_id("camera_2", 2)
    assert len(mgr.persons) == 2


def test_depth_corridor_can_link_without_strong_reid():
    """Facing cameras: depth sum ≈ baseline can confirm soft Re-ID."""
    cfg = ReIDConfig(
        similarity_threshold=0.99,  # hard reid almost impossible
        soft_similarity_threshold=0.50,
        baseline_m=10.0,
        depth_tolerance_frac=0.25,
        weight_reid=0.5,
        weight_depth=0.4,
        weight_temporal=0.1,
    )
    mgr = GlobalIDManager(cfg)
    # Same identity → soft reid should pass with depth
    e = _emb(42)
    mgr.update_observations(
        [
            LocalObservation(
                "camera_1", 1, embedding=e, depth_m=3.0, timestamp=1.0
            )
        ],
        now=1.0,
    )
    mgr.update_observations(
        [
            LocalObservation(
                "camera_1", 1, embedding=e, depth_m=3.0, timestamp=1.2
            ),
            LocalObservation(
                "camera_2", 2, embedding=e, depth_m=7.0, timestamp=1.2
            ),
        ],
        now=1.2,
    )
    assert mgr.get_global_id("camera_1", 1) == mgr.get_global_id("camera_2", 2)


def test_depth_boost_links_soft_reid_when_corridor_matches():
    """Re-ID ~0.55 + d_front + d_back ≈ baseline should link via depth_boost."""
    e_strong = _emb(42)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(64).astype(np.float32)
    e_soft = e_strong * 0.72 + noise * 0.28
    e_soft = e_soft / np.linalg.norm(e_soft)
    sim = cosine_similarity(e_strong, e_soft)
    assert 0.48 <= sim < 0.58

    cfg = ReIDConfig(
        similarity_threshold=0.72,
        soft_similarity_threshold=0.58,
        baseline_m=5.0,
        depth_tolerance_frac=0.22,
        depth_boost_min_reid=0.48,
        depth_boost_min_depth=0.65,
    )
    mgr = GlobalIDManager(cfg)
    mgr.update_observations(
        [LocalObservation("camera_1", 1, embedding=e_strong, depth_m=1.0, timestamp=1.0)],
        now=1.0,
    )
    mgr.update_observations(
        [
            LocalObservation("camera_1", 1, embedding=e_strong, depth_m=1.0, timestamp=1.2),
            LocalObservation("camera_2", 2, embedding=e_soft, depth_m=4.0, timestamp=1.2),
        ],
        now=1.2,
    )
    assert mgr.get_global_id("camera_1", 1) == mgr.get_global_id("camera_2", 2)


def test_association_log_mentions_weapon_inheritance():
    cfg = ReIDConfig(similarity_threshold=0.70)
    mgr = GlobalIDManager(cfg)
    e = _emb(5)
    mgr.update_observations(
        [
            LocalObservation(
                "camera_1",
                4,
                embedding=e,
                weapon_detected=True,
                weapon_confidence=0.9,
                timestamp=1.0,
            )
        ],
        now=1.0,
    )
    mgr.update_observations(
        [
            LocalObservation("camera_1", 4, embedding=e, timestamp=1.2),
            LocalObservation("camera_2", 12, embedding=e, timestamp=1.2),
        ],
        now=1.2,
    )
    log = "\n".join(mgr.association_log())
    assert "Global ID" in log
    assert "Weapon state inherited: TRUE" in log


def test_opposite_corridor_side_not_merged_by_depth_alone():
    """Depth sum ≈ baseline must not link people on opposite lateral sides."""
    cfg = ReIDConfig(
        similarity_threshold=0.99,
        soft_similarity_threshold=0.50,
        baseline_m=10.0,
        depth_tolerance_frac=0.25,
        lateral_tolerance_frac=0.20,
    )
    mgr = GlobalIDManager(cfg)
    e_armed = _emb(1)
    e_other = _emb(99)
    mgr.update_observations(
        [
            LocalObservation(
                "camera_1",
                1,
                embedding=e_armed,
                depth_m=3.0,
                lateral_norm=0.20,
                weapon_detected=True,
                weapon_confidence=0.9,
                timestamp=1.0,
            )
        ],
        now=1.0,
    )
    # Same depth corridor as armed person but wrong mirror position (both on left).
    mgr.update_observations(
        [
            LocalObservation(
                "camera_2",
                2,
                embedding=e_other,
                depth_m=7.0,
                lateral_norm=0.22,
                timestamp=1.2,
            )
        ],
        now=1.2,
    )
    assert mgr.get_global_id("camera_1", 1) != mgr.get_global_id("camera_2", 2)
    wrong_gid = mgr.get_global_id("camera_2", 2)
    assert wrong_gid is not None
    assert mgr.persons[wrong_gid].weapon_detected is False


def test_mirrored_lateral_links_same_corridor_person():
    cfg = ReIDConfig(
        similarity_threshold=0.99,
        soft_similarity_threshold=0.50,
        baseline_m=10.0,
        depth_tolerance_frac=0.25,
        lateral_tolerance_frac=0.20,
    )
    mgr = GlobalIDManager(cfg)
    e = _emb(7)
    mgr.update_observations(
        [
            LocalObservation(
                "camera_1", 1, embedding=e, depth_m=3.0, lateral_norm=0.18, timestamp=1.0
            )
        ],
        now=1.0,
    )
    mgr.update_observations(
        [
            LocalObservation(
                "camera_2", 2, embedding=e, depth_m=7.0, lateral_norm=0.82, timestamp=1.2
            )
        ],
        now=1.2,
    )
    assert mgr.get_global_id("camera_1", 1) == mgr.get_global_id("camera_2", 2)
