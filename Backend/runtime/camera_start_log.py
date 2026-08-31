"""Structured API logging when Front/Back camera pipelines start."""

from __future__ import annotations

import logging
from typing import Any

from runtime.multi_camera_runner import resolve_multi_camera_source

logger = logging.getLogger("scanu.sensors")

_SENSOR_LABEL = {
    "webcam": "Front Cam",
    "multi_camera": "Back Cam",
}


def _sensor_block(sensor: str, settings: dict[str, Any]) -> dict[str, Any]:
    if sensor == "webcam":
        return dict(settings.get("webcam") or {})
    if sensor == "multi_camera":
        return dict(settings.get("multi_camera") or {})
    return {}


def capture_source_summary(sensor: str, settings: dict[str, Any]) -> dict[str, Any]:
    w = _sensor_block(sensor, settings)
    label = _SENSOR_LABEL.get(sensor, sensor)
    mode = str(w.get("source_mode") or "local").strip().lower()
    try:
        network_source, is_network = resolve_multi_camera_source(w)
    except ValueError as exc:
        return {
            "label": label,
            "source_mode": mode,
            "source": f"(config error: {exc})",
            "is_network": mode in ("jetson", "ip", "network", "rtsp"),
            "width": w.get("webcam_width"),
            "height": w.get("webcam_height"),
            "fps": w.get("fps"),
            "rotate": w.get("capture_rotate"),
            "pipeline": w.get("webcam_pipeline", "infer"),
        }
    if is_network:
        source = str(network_source)
    else:
        dev = int(w.get("webcam_device", 0))
        source = f"/dev/video{dev}"
    return {
        "label": label,
        "source_mode": mode,
        "source": source,
        "is_network": is_network,
        "width": w.get("webcam_width", 3840),
        "height": w.get("webcam_height", 2160),
        "fps": w.get("fps", 30),
        "rotate": w.get("capture_rotate"),
        "pipeline": w.get("webcam_pipeline", "infer"),
    }


def log_start_requested(sensor: str, settings: dict[str, Any], *, action: str = "run") -> None:
    if sensor not in _SENSOR_LABEL:
        return
    info = capture_source_summary(sensor, settings)
    logger.info(
        "%s %s requested — source=%s mode=%s %sx%s@%sfps rotate=%s pipeline=%s",
        info["label"],
        action,
        info["source"],
        info["source_mode"],
        info["width"],
        info["height"],
        info["fps"],
        info.get("rotate"),
        info.get("pipeline"),
    )


def log_start_result(sensor: str, settings: dict[str, Any], result: dict[str, Any], *, action: str = "run") -> None:
    if sensor not in _SENSOR_LABEL:
        return
    info = capture_source_summary(sensor, settings)
    label = info["label"]
    if result.get("ok"):
        if result.get("preview_only"):
            logger.info(
                "%s %s OK (preview-only): shared V4L2 reader — infer subprocess not started",
                label,
                action,
            )
            return
        logger.info(
            "%s %s OK — pid=%s source=%s log=%s",
            label,
            action,
            result.get("pid"),
            info["source"],
            result.get("log_file") or "layer8_ui/logs/%s.log" % sensor,
        )
        return
    logger.warning(
        "%s %s FAILED — source=%s error=%s",
        label,
        action,
        info["source"],
        result.get("error") or "unknown",
    )


def start_message(sensor: str, settings: dict[str, Any], result: dict[str, Any]) -> str:
    info = capture_source_summary(sensor, settings)
    label = info["label"]
    if not result.get("ok"):
        return f"{label} failed: {result.get('error') or 'unknown'}"
    if result.get("preview_only"):
        return f"{label} preview-only (no infer subprocess)"
    log_hint = result.get("log_file") or f"layer8_ui/logs/{sensor}.log"
    return f"{label} started on {info['source']} — pid {result.get('pid')} ({log_hint})"
