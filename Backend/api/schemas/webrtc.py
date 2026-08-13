"""WebRTC request schemas."""

from __future__ import annotations

from pydantic import BaseModel


class WebRTCOfferBody(BaseModel):
    sdp: str
    type: str

