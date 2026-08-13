"""Shared type aliases."""

from typing import Any, TypedDict


class BoundingBox(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


Detection = dict[str, Any]
