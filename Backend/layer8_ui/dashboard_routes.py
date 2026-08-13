"""Backward-compatible re-export of migrated API routes."""

from api.routes import build_router, index_handler

__all__ = ["build_router", "index_handler"]
