"""Minimal physical-radar helpers vendored by the standalone Mmwave app.

Only ``hardware_registry`` and ``radar`` are included.  Keeping this package
initializer intentionally empty prevents imports from the canonical SCAN-U
Sensor Hub, which is not part of this independent application.
"""

from __future__ import annotations

__all__: list[str] = []
