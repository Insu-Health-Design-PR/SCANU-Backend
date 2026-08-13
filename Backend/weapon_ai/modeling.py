"""Compatibility import for the legacy classifier builder module name."""

from weapon_ai.models.classifier import build_gun_prob_model, build_model

__all__ = ["build_gun_prob_model", "build_model"]

