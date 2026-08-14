"""Multi-camera Person Re-ID and Global ID association.

Keeps Camera A / Camera B inference independent. Each process may publish
per-track embeddings + weapon state into metrics JSON; the GlobalIDManager
associates them into shared identities and weapon state.
"""

from __future__ import annotations

from weapon_ai.reid.config import ReIDConfig
from weapon_ai.reid.embeddings import MockReIDEmbedder, PersonReIDEmbedder
from weapon_ai.reid.global_manager import GlobalIDManager, GlobalPersonState

__all__ = [
    "ReIDConfig",
    "PersonReIDEmbedder",
    "MockReIDEmbedder",
    "GlobalIDManager",
    "GlobalPersonState",
]
