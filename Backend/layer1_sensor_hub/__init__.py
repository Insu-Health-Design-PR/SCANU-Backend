"""Layer 1 — Sensor Hub (vendored subset for ``lab.mmwave77_usb``).

Full Adrian hub includes DCA1000 / multi-radar capture. This Backend tree only
vendors UART/TLV radar helpers under ``layer1_sensor_hub.radar`` plus
``hardware_registry`` discovery constants used by the AWR1843 lab path.
"""

from __future__ import annotations

__all__: list[str] = []
