"""Compatibility shim — implementation lives in ``runtime.multi_camera_layer8_runner``."""

from __future__ import annotations

from runtime.multi_camera_layer8_runner import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
