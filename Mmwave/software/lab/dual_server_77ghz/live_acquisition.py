"""Continuous, stoppable UART/TLV acquisition for two AWR1843BOOST radars."""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA_BAUD = 921_600


def points_from_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    """Return measured post-CFAR points as x,y,z,snr rows."""
    points: list[list[float]] = []
    for frame in rows:
        for point in frame.get("points") or frame.get("detected_points") or []:
            if isinstance(point, dict):
                xyz = (
                    point.get("x", point.get("x_m")),
                    point.get("y", point.get("y_m")),
                    point.get("z", point.get("z_m")),
                )
                if any(value is None for value in xyz):
                    continue
                points.append(
                    [float(xyz[0]), float(xyz[1]), float(xyz[2]), float(point.get("snr") or 0.0)]
                )
            elif isinstance(point, (list, tuple)) and len(point) >= 3:
                points.append(
                    [float(point[0]), float(point[1]), float(point[2]), float(point[3]) if len(point) > 3 else 0.0]
                )
    return np.asarray(points, dtype=np.float32).reshape((-1, 4)) if points else np.zeros((0, 4), dtype=np.float32)


class RadarStream:
    """Own one CLI/data pair for the complete calibration + live lifecycle."""

    def __init__(
        self,
        name: str,
        cli_port: str,
        data_port: str,
        profile: Path,
        *,
        max_frames: int = 1200,
    ) -> None:
        self.name = name
        self.cli_port = cli_port
        self.data_port = data_port
        self.profile = Path(profile)
        self._frames: deque[dict[str, Any]] = deque(maxlen=max_frames)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.running = False
        self.error = ""
        self.frames_received = 0
        self.dropped_frames = 0
        self.last_frame_number: int | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"awr1843-{self.name}", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=max(0.1, timeout_s))

    def wait_ready(self, timeout_s: float = 12.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.error:
                return False
            if self.running and self.frames_received > 0:
                return True
            time.sleep(0.05)
        return False

    def snapshot(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._frames)
        return rows[-max(1, int(limit)):]

    def points(self, limit: int = 10) -> np.ndarray:
        return points_from_rows(self.snapshot(limit))

    def latest_timestamp_ns(self) -> int | None:
        rows = self.snapshot(1)
        if not rows:
            return None
        return int(rows[-1].get("live_host_monotonic_ns") or 0) or None

    def _run(self) -> None:
        try:
            import serial

            from lab.mmwave77_usb.runner import TiTlvFramer, _configure_ti_sensor, _decode_frame
            from layer1_sensor_hub.radar.tlv_parser import TLVParser

            _configure_ti_sensor(
                self.cli_port,
                self.profile,
                validate_awr1843=True,
                audit_path=None,
            )
            framer = TiTlvFramer()
            parser = TLVParser()
            with serial.Serial(self.data_port, baudrate=DEFAULT_DATA_BAUD, timeout=0.05) as port:
                port.reset_input_buffer()
                self.running = True
                while not self._stop.is_set():
                    chunk = port.read(4096)
                    if not chunk:
                        continue
                    for packet in framer.feed(chunk):
                        row, _ = _decode_frame(packet, parser)
                        if not row.get("parse_ok"):
                            continue
                        now_ns = time.monotonic_ns()
                        row["live_host_monotonic_ns"] = now_ns
                        number = int(row.get("frame_number") or row.get("frame") or 0)
                        if self.last_frame_number is not None and number > self.last_frame_number + 1:
                            self.dropped_frames += number - self.last_frame_number - 1
                        self.last_frame_number = number
                        with self._lock:
                            self._frames.append(row)
                        self.frames_received += 1
        except Exception as exc:  # hardware errors must become visible runtime state
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.running = False
