"""Global person identity manager for cross-camera association."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from weapon_ai.reid.config import ReIDConfig
from weapon_ai.reid.embeddings import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class LocalObservation:
    camera_id: str
    local_track_id: int
    embedding: np.ndarray | None = None
    bbox: tuple[int, int, int, int] | None = None
    depth_m: float | None = None
    weapon_detected: bool = False
    weapon_confidence: float = 0.0
    timestamp: float = 0.0


@dataclass
class GlobalPersonState:
    global_id: int
    camera_tracks: dict[str, int] = field(default_factory=dict)
    embeddings: deque = field(default_factory=lambda: deque(maxlen=12))
    depths: dict[str, float] = field(default_factory=dict)
    weapon_detected: bool = False
    weapon_confidence: float = 0.0
    weapon_last_true_ts: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_match_confidence: float = 0.0
    last_match_camera: str = ""

    def representative_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        stacked = np.stack([np.asarray(e, dtype=np.float32) for e in self.embeddings], axis=0)
        mean = stacked.mean(axis=0)
        n = float(np.linalg.norm(mean))
        return mean / n if n > 1e-8 else mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_id": int(self.global_id),
            "camera_tracks": {str(k): int(v) for k, v in self.camera_tracks.items()},
            "weapon_detected": bool(self.weapon_detected),
            "weapon_confidence": round(float(self.weapon_confidence), 4),
            "first_seen": float(self.first_seen),
            "last_seen": float(self.last_seen),
            "last_match_confidence": round(float(self.last_match_confidence), 4),
            "depths_m": {str(k): round(float(v), 3) for k, v in self.depths.items()},
        }


class GlobalIDManager:
    """Associate local camera tracks into global person IDs.

    Matching uses Re-ID cosine similarity plus optional corridor depth
    consistency (``d_front + d_back ≈ baseline``) and temporal freshness.
    Weapon state is owned by the global person and persists across cameras.
    """

    def __init__(self, config: ReIDConfig | None = None) -> None:
        self.config = config or ReIDConfig()
        self._next_id = 1
        self._persons: dict[int, GlobalPersonState] = {}
        # (camera_id, local_track_id) -> global_id
        self._local_to_global: dict[tuple[str, int], int] = {}
        self._assoc_log: deque[str] = deque(maxlen=100)

    @property
    def persons(self) -> dict[int, GlobalPersonState]:
        return self._persons

    def reset(self) -> None:
        self._next_id = 1
        self._persons.clear()
        self._local_to_global.clear()
        self._assoc_log.clear()

    def get_global_id(self, camera_id: str, local_track_id: int) -> int | None:
        return self._local_to_global.get((str(camera_id), int(local_track_id)))

    def association_log(self) -> list[str]:
        return list(self._assoc_log)

    def update_observations(
        self,
        observations: list[LocalObservation],
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Ingest one frame of local tracks from one or more cameras."""
        ts = float(time.time() if now is None else now)
        active_keys: set[tuple[str, int]] = set()
        for obs in observations:
            cam = str(obs.camera_id)
            lid = int(obs.local_track_id)
            active_keys.add((cam, lid))
            self._upsert_observation(obs, ts=ts)

        self._expire_stale(ts)
        self._decay_weapons(ts)
        # Drop local maps for tracks no longer present on their camera this tick
        # (only if timed out — keep brief gaps for reappearance).
        return self.snapshot()

    def _upsert_observation(self, obs: LocalObservation, *, ts: float) -> None:
        cam = str(obs.camera_id)
        lid = int(obs.local_track_id)
        key = (cam, lid)
        existing_gid = self._local_to_global.get(key)

        if existing_gid is not None and existing_gid in self._persons:
            person = self._persons[existing_gid]
            self._refresh_person(person, obs, ts=ts, match_conf=person.last_match_confidence)
            return

        match_gid, score, reason = self._best_match(obs, ts=ts)
        if match_gid is not None:
            person = self._persons[match_gid]
            prev_cams = [c for c in person.camera_tracks if c != cam]
            person.camera_tracks[cam] = lid
            self._local_to_global[key] = match_gid
            self._refresh_person(person, obs, ts=ts, match_conf=score)
            prev = prev_cams[0] if prev_cams else "none"
            msg = (
                f"{cam} Local ID {lid} → Global ID {match_gid} | "
                f"Re-ID similarity: {score:.2f} | Previous camera: {prev} | "
                f"Weapon state inherited: {str(person.weapon_detected).upper()} | {reason}"
            )
            self._assoc_log.append(msg)
            logger.info(msg)
            return

        # New global identity
        gid = self._next_id
        self._next_id += 1
        person = GlobalPersonState(
            global_id=gid,
            camera_tracks={cam: lid},
            embeddings=deque(maxlen=int(self.config.max_embedding_history)),
            first_seen=ts,
            last_seen=ts,
            last_match_confidence=1.0,
            last_match_camera=cam,
        )
        self._persons[gid] = person
        self._local_to_global[key] = gid
        self._refresh_person(person, obs, ts=ts, match_conf=1.0)
        msg = (
            f"{cam} Local ID {lid} → Global ID {gid} (NEW) | "
            f"Weapon state: {str(person.weapon_detected).upper()}"
        )
        self._assoc_log.append(msg)
        logger.info(msg)

    def _refresh_person(
        self,
        person: GlobalPersonState,
        obs: LocalObservation,
        *,
        ts: float,
        match_conf: float,
    ) -> None:
        person.last_seen = ts
        person.last_match_confidence = float(match_conf)
        person.last_match_camera = str(obs.camera_id)
        person.camera_tracks[str(obs.camera_id)] = int(obs.local_track_id)
        if obs.embedding is not None:
            person.embeddings.append(np.asarray(obs.embedding, dtype=np.float32))
        if obs.depth_m is not None and float(obs.depth_m) > 0:
            person.depths[str(obs.camera_id)] = float(obs.depth_m)
        if obs.weapon_detected and float(obs.weapon_confidence) > 0:
            person.weapon_detected = True
            person.weapon_confidence = max(float(person.weapon_confidence), float(obs.weapon_confidence))
            person.weapon_last_true_ts = ts
        elif obs.weapon_detected:
            person.weapon_detected = True
            person.weapon_last_true_ts = ts
            person.weapon_confidence = max(float(person.weapon_confidence), 0.5)

    def _best_match(
        self, obs: LocalObservation, *, ts: float
    ) -> tuple[int | None, float, str]:
        cfg = self.config
        if obs.embedding is None and obs.depth_m is None:
            return None, 0.0, "no_signals"

        best_gid: int | None = None
        best_score = -1.0
        best_reason = ""
        for gid, person in self._persons.items():
            # Same camera already owns a different local track on this global → skip
            existing_local = person.camera_tracks.get(str(obs.camera_id))
            if existing_local is not None and int(existing_local) != int(obs.local_track_id):
                # Allow reassignment only if that local track is gone (not in map refresh yet)
                # Keep exclusive: one local track per camera per global person.
                continue

            reid_sim = 0.0
            has_reid = False
            rep = person.representative_embedding()
            if obs.embedding is not None and rep is not None:
                reid_sim = cosine_similarity(obs.embedding, rep)
                has_reid = True

            depth_score = 0.0
            has_depth = False
            if obs.depth_m is not None and float(obs.depth_m) > 0 and cfg.baseline_m > 0:
                # Compare against depth from the *other* camera on this global person
                for other_cam, other_d in person.depths.items():
                    if other_cam == obs.camera_id:
                        continue
                    residual = abs(float(obs.depth_m) + float(other_d) - float(cfg.baseline_m))
                    tol = float(cfg.depth_tolerance_frac) * float(cfg.baseline_m)
                    if residual <= tol:
                        depth_score = max(depth_score, 1.0 - residual / max(tol, 1e-6))
                        has_depth = True
                    else:
                        depth_score = max(depth_score, 0.0)

            age = max(0.0, ts - float(person.last_seen))
            temporal = max(0.0, 1.0 - age / max(float(cfg.track_timeout_s), 1e-6))

            # Need at least Re-ID or depth signal
            if not has_reid and not has_depth:
                continue

            score = (
                float(cfg.weight_reid) * reid_sim
                + float(cfg.weight_depth) * depth_score
                + float(cfg.weight_temporal) * temporal
            )
            # Normalize roughly by active weights
            wsum = float(cfg.weight_reid) * (1.0 if has_reid else 0.0)
            wsum += float(cfg.weight_depth) * (1.0 if has_depth else 0.0)
            wsum += float(cfg.weight_temporal)
            if wsum > 1e-6:
                score = score / wsum * (
                    float(cfg.weight_reid) + float(cfg.weight_depth) + float(cfg.weight_temporal)
                ) / (
                    float(cfg.weight_reid)
                    + float(cfg.weight_depth)
                    + float(cfg.weight_temporal)
                )

            hard = float(cfg.similarity_threshold)
            soft = float(cfg.soft_similarity_threshold)
            ok = False
            reason = ""
            if has_reid and reid_sim >= hard:
                ok = True
                reason = f"reid>={hard:.2f}"
            elif has_reid and has_depth and reid_sim >= soft and depth_score >= 0.5:
                ok = True
                reason = f"reid_soft+depth (sim={reid_sim:.2f})"
            elif (not has_reid) and has_depth and depth_score >= 0.75 and temporal >= 0.4:
                ok = True
                reason = "depth_corridor"

            if ok and score > best_score:
                best_score = score if has_reid else depth_score
                best_gid = gid
                best_reason = reason

        if best_gid is None:
            return None, 0.0, "below_threshold"
        return best_gid, float(best_score if best_score >= 0 else 0.0), best_reason

    def _expire_stale(self, ts: float) -> None:
        timeout = float(self.config.track_timeout_s)
        dead_gids: list[int] = []
        for gid, person in self._persons.items():
            if ts - float(person.last_seen) > timeout:
                dead_gids.append(gid)
        for gid in dead_gids:
            person = self._persons.pop(gid, None)
            if person is None:
                continue
            for cam, lid in list(person.camera_tracks.items()):
                self._local_to_global.pop((cam, int(lid)), None)

        # Also prune local map entries whose global is gone
        for key, gid in list(self._local_to_global.items()):
            if gid not in self._persons:
                self._local_to_global.pop(key, None)

    def _decay_weapons(self, ts: float) -> None:
        hold = float(self.config.weapon_hold_s)
        decay = float(self.config.weapon_conf_decay)
        for person in self._persons.values():
            if not person.weapon_detected:
                person.weapon_confidence = max(0.0, float(person.weapon_confidence) - decay * 0.0)
                continue
            age = ts - float(person.weapon_last_true_ts or person.last_seen)
            if age <= hold:
                continue
            # Decay confidence after hold window; clear when near zero
            over = age - hold
            person.weapon_confidence = max(0.0, float(person.weapon_confidence) - decay * over)
            if person.weapon_confidence <= 0.05:
                person.weapon_detected = False
                person.weapon_confidence = 0.0

    def snapshot(self) -> dict[str, Any]:
        """UI / API payload: per-camera local→global map + global persons."""
        cameras: dict[str, dict[str, Any]] = {}
        for (cam, lid), gid in self._local_to_global.items():
            person = self._persons.get(gid)
            if person is None:
                continue
            cameras.setdefault(cam, {})[str(lid)] = {
                "global_id": int(gid),
                "local_track_id": int(lid),
                "match_confidence": round(float(person.last_match_confidence), 4),
                "weapon_detected": bool(person.weapon_detected),
                "weapon_confidence": round(float(person.weapon_confidence), 4),
            }
        return {
            "ts": time.time(),
            "persons": [p.to_dict() for p in sorted(self._persons.values(), key=lambda x: x.global_id)],
            "cameras": cameras,
            "association_log": list(self._assoc_log)[-20:],
            "config": {
                "similarity_threshold": float(self.config.similarity_threshold),
                "track_timeout_s": float(self.config.track_timeout_s),
                "weapon_hold_s": float(self.config.weapon_hold_s),
                "baseline_m": float(self.config.baseline_m),
            },
        }

    def overlay_for(self, camera_id: str, local_display_id: int) -> dict[str, Any] | None:
        """Lookup used by infer overlay / UI for a single local track."""
        gid = self.get_global_id(camera_id, local_display_id)
        if gid is None:
            return None
        person = self._persons.get(gid)
        if person is None:
            return None
        return {
            "global_id": int(gid),
            "local_track_id": int(local_display_id),
            "match_confidence": round(float(person.last_match_confidence), 4),
            "weapon_detected": bool(person.weapon_detected),
            "weapon_confidence": round(float(person.weapon_confidence), 4),
        }
