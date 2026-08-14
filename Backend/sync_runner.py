#!/usr/bin/env python3
"""Shim: ``python sync_runner.py ...`` → ``scripts/sync_runner.py``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "sync_runner.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
