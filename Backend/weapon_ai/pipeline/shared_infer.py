"""Experimental shared dual-camera inference (opt-in, default off).

Front and Back already run as two processes on the same GPU (CUDA_VISIBLE_DEVICES=0).
Cross-camera batching is not enabled unless ``SCANU_SHARED_INFER=1`` or
``--shared_infer_service``. See ``docs/shared_dual_camera_inference.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SharedInferConfig:
    enabled: bool = False
    max_wait_ms: float = 2.0
    socket_path: str = "/tmp/scanu_shared_infer.sock"


def shared_infer_enabled(*, cli_flag: bool | None = None) -> bool:
    if cli_flag is True:
        return True
    env = str(os.environ.get("SCANU_SHARED_INFER") or "").strip().lower()
    return env in {"1", "true", "yes", "on"}


def load_shared_infer_config(*, cli_flag: bool | None = None) -> SharedInferConfig:
    enabled = shared_infer_enabled(cli_flag=cli_flag)
    wait = float(os.environ.get("SCANU_SHARED_INFER_MAX_WAIT_MS") or 2.0)
    wait = min(2.0, max(0.0, wait))
    return SharedInferConfig(enabled=enabled, max_wait_ms=wait)
