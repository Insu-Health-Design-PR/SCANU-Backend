"""FastAPI extension hooks.

Slice 1 does not add middleware here. Keep this module as the place for future
CORS/rate-limit setup instead of scattering it through `api.app`.
"""

from __future__ import annotations

from fastapi import FastAPI


def init_extensions(app: FastAPI) -> None:
    return None
