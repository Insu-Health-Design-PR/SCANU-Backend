"""Metrics request/response schemas."""

from pydantic import BaseModel


class ThreatMetricsResponse(BaseModel):
    threat_level: str = "unknown"
    counts: dict[str, int] = {}
