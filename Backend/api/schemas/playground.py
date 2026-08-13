"""Model playground request schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PlaygroundInferBody(BaseModel):
    """Run single-image weapon inference with optional profile-style values."""

    values: dict[str, Any] | None = None
    use_sample: bool = True
    image_b64: str = ""

