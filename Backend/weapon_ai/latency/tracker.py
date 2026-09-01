"""Stage timestamps and rolling p50/p95/p99 for the live infer loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from weapon_ai.latency.rolling import RollingPercentiles

_WINDOW = 256


def _ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    if end_ns < start_ns:
        return 0.0
    return (end_ns - start_ns) / 1_000_000.0


@dataclass
class FrameTiming:
    """Monotonic_ns marks carried with one capture → publish cycle."""

    capture_ns: int | None = None
    accepted_ns: int | None = None
    infer_submit_ns: int | None = None
    person_done_ns: int | None = None
    gun_crop_done_ns: int | None = None
    gun_done_ns: int | None = None
    tracking_done_ns: int | None = None
    alert_published_ns: int | None = None
    ipc_published_ns: int | None = None
    person_ran: bool = False
    gun_ran: bool = False


@dataclass
class LatencyTracker:
    capture_to_infer: RollingPercentiles = field(default_factory=lambda: RollingPercentiles(_WINDOW))
    queue_wait: RollingPercentiles = field(default_factory=lambda: RollingPercentiles(_WINDOW))
    person_infer: RollingPercentiles = field(default_factory=lambda: RollingPercentiles(_WINDOW))
    gun_crop: RollingPercentiles = field(default_factory=lambda: RollingPercentiles(_WINDOW))
    gun_infer: RollingPercentiles = field(default_factory=lambda: RollingPercentiles(_WINDOW))
    tracking: RollingPercentiles = field(default_factory=lambda: RollingPercentiles(_WINDOW))
    publish: RollingPercentiles = field(default_factory=lambda: RollingPercentiles(_WINDOW))
    vision_alert: RollingPercentiles = field(default_factory=lambda: RollingPercentiles(_WINDOW))
    frame_age_at_publish: RollingPercentiles = field(default_factory=lambda: RollingPercentiles(_WINDOW))
    detection_age: RollingPercentiles = field(default_factory=lambda: RollingPercentiles(_WINDOW))
    mmwave_age: RollingPercentiles = field(default_factory=lambda: RollingPercentiles(_WINDOW))
    loop_fps_ema: float | None = None
    detection_fps_ema: float | None = None
    last_loop_ns: int | None = None
    last_detection_ns: int | None = None
    infer_frames_replaced: int = 0
    display_frames_replaced: int = 0
    last_person_done_ns: int | None = None
    last_gun_done_ns: int | None = None

    def note_loop(self, now_ns: int | None = None) -> float | None:
        now = int(now_ns if now_ns is not None else time.monotonic_ns())
        inst = None
        if self.last_loop_ns is not None and now > self.last_loop_ns:
            inst = 1e9 / float(now - self.last_loop_ns)
            if self.loop_fps_ema is None:
                self.loop_fps_ema = inst
            else:
                self.loop_fps_ema = (0.85 * self.loop_fps_ema) + (0.15 * inst)
        self.last_loop_ns = now
        return inst

    def note_detection(self, now_ns: int | None = None) -> None:
        now = int(now_ns if now_ns is not None else time.monotonic_ns())
        if self.last_detection_ns is not None and now > self.last_detection_ns:
            inst = 1e9 / float(now - self.last_detection_ns)
            if self.detection_fps_ema is None:
                self.detection_fps_ema = inst
            else:
                self.detection_fps_ema = (0.85 * self.detection_fps_ema) + (0.15 * inst)
        self.last_detection_ns = now

    def record_stages(self, t: FrameTiming) -> None:
        v = _ms(t.capture_ns, t.accepted_ns)
        if v is not None:
            self.capture_to_infer.add(v)
        v = _ms(t.accepted_ns, t.infer_submit_ns)
        if v is not None:
            self.queue_wait.add(v)
        if t.person_ran:
            v = _ms(t.infer_submit_ns, t.person_done_ns)
            if v is not None:
                self.person_infer.add(v)
        v = _ms(t.person_done_ns, t.gun_crop_done_ns)
        if v is not None:
            self.gun_crop.add(v)
        if t.gun_ran:
            v = _ms(t.gun_crop_done_ns, t.gun_done_ns)
            if v is not None:
                self.gun_infer.add(v)
        v = _ms(t.gun_done_ns or t.person_done_ns, t.tracking_done_ns)
        if v is not None:
            self.tracking.add(v)
        v = _ms(t.tracking_done_ns, t.alert_published_ns)
        if v is not None:
            self.publish.add(v)
        v = _ms(t.capture_ns, t.alert_published_ns)
        if v is not None:
            self.vision_alert.add(v)
        v = _ms(t.capture_ns, t.ipc_published_ns)
        if v is not None:
            self.frame_age_at_publish.add(v)
        if t.gun_done_ns is not None:
            self.last_gun_done_ns = t.gun_done_ns
        if t.person_done_ns is not None:
            self.last_person_done_ns = t.person_done_ns

    def note_ipc_publish(self, capture_ns: int, now_ns: int | None = None) -> None:
        now = int(now_ns if now_ns is not None else time.monotonic_ns())
        v = _ms(int(capture_ns), now)
        if v is not None:
            self.frame_age_at_publish.add(v)

    def detection_age_ms(self, now_ns: int | None = None) -> float | None:
        done = self.last_gun_done_ns or self.last_person_done_ns
        if done is None:
            return None
        now = int(now_ns if now_ns is not None else time.monotonic_ns())
        age = _ms(done, now)
        if age is not None:
            self.detection_age.add(age)
        return age

    def note_mmwave_age_ms(self, age_ms: float | None) -> None:
        if age_ms is None:
            return
        self.mmwave_age.add(float(age_ms))

    def metrics_payload(self, *, detection_age_ms: float | None = None) -> dict[str, object]:
        """Nested latency block for live threat JSON. ``infer_fps`` stays top-level (loop)."""
        loop = self.loop_fps_ema
        det = self.detection_fps_ema
        return {
            "capture_to_infer_ms": self.capture_to_infer.snapshot(),
            "queue_wait_ms": self.queue_wait.snapshot(),
            "person_infer_ms": self.person_infer.snapshot(),
            "gun_crop_ms": self.gun_crop.snapshot(),
            "gun_infer_ms": self.gun_infer.snapshot(),
            "tracking_ms": self.tracking.snapshot(),
            "publish_ms": self.publish.snapshot(),
            "vision_alert_latency_ms": self.vision_alert.snapshot(),
            "frame_age_at_publish_ms": self.frame_age_at_publish.snapshot(),
            "detection_age_ms": detection_age_ms,
            "detection_age_ms_roll": self.detection_age.snapshot(),
            "mmwave_age_ms": self.mmwave_age.snapshot(),
            "loop_fps": round(float(loop), 2) if loop is not None else None,
            "detection_fps": round(float(det), 2) if det is not None else None,
            "infer_frames_replaced": int(self.infer_frames_replaced),
            "display_frames_replaced": int(self.display_frames_replaced),
            "infer_fps_note": (
                "deprecated: infer_fps is main-loop throughput (loop_fps), not model latency"
            ),
        }
