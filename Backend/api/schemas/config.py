"""Configuration request schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SettingsBody(BaseModel):
    settings: dict[str, Any]

