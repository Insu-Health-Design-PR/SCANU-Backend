"""Model profile request schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ApplyModelProfileBody(BaseModel):
    id: str


class ApplyModelProfileByNameBody(BaseModel):
    name: str


class SnapshotModelProfileBody(BaseModel):
    """Save under a file key derived from `name`.

    `id` is optional for legacy clients. `values` can include the current
    Model + Webcam form state so operators do not need to save settings first.
    """

    id: str = ""
    name: str = ""
    description: str = ""
    values: dict[str, Any] | None = None


class SaveProfileIfNewBody(BaseModel):
    name: str
    description: str = ""
    values: dict[str, Any] | None = None
