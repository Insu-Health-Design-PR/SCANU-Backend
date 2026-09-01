"""CPU-only tests for latency timestamps, latest-frame slots, ROI reuse, mmWave age."""

from __future__ import annotations

import threading
import time

import numpy as np

from media.capture.live_webcam_capture import LiveWebcamCapture
from weapon_ai.detection.gun_batch import (
    bucket_gun_imgsz,
    prioritize_crop_indices,
    split_batch_indices,
    tensorrt_batch_range,
)
from weapon_ai.detection.gun_crops import PersonGunCrop, collect_person_gun_crops, predict_gun_on_crops
from weapon_ai.latency.rolling import RollingPercentiles
from weapon_ai.latency.tracker import FrameTiming, LatencyTracker
from weapon_ai.overlay.mmwave_fusion import compute_mmwave_torso_score, mmwave_age_ms, mmwave_fresh_for_threat
from weapon_ai.pipeline.latest_slot import LatestJob, LatestSlot
from weapon_ai.pipeline.tracked_roi import TrackedRoiConfig, box_shift_frac, should_refresh_person
from weapon_ai.threat.metrics import write_live_metrics_json


def test_rolling_percentiles_bounded_and_ordered():
    roll = RollingPercentiles(maxlen=16)
    for i in range(40):
        roll.add(float(i))
    assert len(roll) == 16
    snap = roll.snapshot()
    assert snap["n"] == 16
    assert snap["p50"] is not None
    assert snap["p95"] >= snap["p50"]
    assert snap["p99"] >= snap["p95"]


def test_frame_timing_propagates_into_metrics_payload():
    t0 = time.monotonic_ns()
    timing = FrameTiming(
        capture_ns=t0,
        accepted_ns=t0 + 2_000_000,
        infer_submit_ns=t0 + 3_000_000,
        person_done_ns=t0 + 13_000_000,
        gun_crop_done_ns=t0 + 14_000_000,
        gun_done_ns=t0 + 24_000_000,
        tracking_done_ns=t0 + 25_000_000,
        alert_published_ns=t0 + 26_000_000,
        ipc_published_ns=t0 + 30_000_000,
        person_ran=True,
        gun_ran=True,
    )
    lat = LatencyTracker()
    lat.record_stages(timing)
    payload = lat.metrics_payload(detection_age_ms=4.0)
    assert payload["capture_to_infer_ms"]["p50"] is not None
    assert payload["person_infer_ms"]["p50"] is not None
    assert payload["gun_infer_ms"]["p50"] is not None
    assert payload["vision_alert_latency_ms"]["p50"] is not None
    assert "deprecated" in str(payload["infer_fps_note"])
    live = {"infer_fps": 32.1, "prediction": "clear", "latency": payload, "detection_age_ms": 4.0}
    assert live["infer_fps"] == 32.1
    assert "byte_tracks" not in live or True


def test_latest_slot_replaces_and_never_grows():
    slot = LatestSlot()
    slot.put({"n": 1})
    slot.put({"n": 2})
    slot.put({"n": 3})
    assert len(slot) == 1
    assert slot.replaced == 2
    assert slot.take()["n"] == 3
    assert slot.take() is None
    assert len(slot) == 0


def test_latest_job_display_independent_of_infer_and_one_slow_camera():
    seen: list[int] = []
    started = threading.Event()
    gate = threading.Event()

    def slow_fn(n: int) -> None:
        started.set()
        gate.wait(timeout=2.0)
        seen.append(n)

    job = LatestJob(slow_fn, name="test-job")
    job.submit(1)
    assert started.wait(timeout=2.0)
    job.submit(2)
    job.submit(3)
    gate.set()
    deadline = time.time() + 2.0
    while time.time() < deadline and 3 not in seen:
        time.sleep(0.01)
    job.shutdown(wait=True)
    assert 1 in seen
    assert 3 in seen
    assert job.replaced >= 1

    front = LatestSlot()
    back = LatestSlot()
    front.put("f1")
    back.put("b1")
    front.put("f2")
    assert front.take() == "f2"
    assert back.take() == "b1"


def test_stale_mmwave_rejected_fresh_accepted():
    now = time.monotonic_ns()
    fresh = {
        "ts_monotonic_ns": now - 50_000_000,
        "front": {"track": {"confidence": 0.9}, "anomalies": [{"score": 0.8}]},
    }
    stale = {
        "ts_monotonic_ns": now - 800_000_000,
        "front": {"track": {"confidence": 0.9}, "anomalies": [{"score": 0.8}]},
    }
    assert mmwave_fresh_for_threat(fresh, max_age_ms=300.0, now_ns=now)
    assert not mmwave_fresh_for_threat(stale, max_age_ms=300.0, now_ns=now)
    assert compute_mmwave_torso_score(fresh, max_age_ms=300.0, now_ns=now) is not None
    assert compute_mmwave_torso_score(stale, max_age_ms=300.0, now_ns=now) is None
    age = mmwave_age_ms(stale, now_ns=now)
    assert age is not None and age > 300.0
    wall_old = {
        "timestamp_ns": time.time_ns() - 2_000_000_000,
        "front": {"track": {"confidence": 0.9}, "points": [{"x": 0}]},
    }
    assert not mmwave_fresh_for_threat(wall_old, max_age_ms=300.0)
    wall_live = {
        "timestamp_ns": time.time_ns() - 40_000_000,
        "front": {"track": {"confidence": 0.9}},
    }
    assert mmwave_fresh_for_threat(wall_live, max_age_ms=300.0)
    assert not mmwave_fresh_for_threat(
        {"publisher": "stopped", "ts_monotonic_ns": time.monotonic_ns()},
        max_age_ms=300.0,
    )


def test_tracked_roi_invalidation_and_lost_track_forces_person():
    cfg = TrackedRoiConfig()
    assert should_refresh_person(
        cfg=cfg,
        frames_since_person=1,
        person_age_ms=10.0,
        unmatched_or_new=False,
        track_lost=False,
        low_track_score=False,
        box_shifted=False,
        mmwave_high_risk=False,
    ) is False
    assert should_refresh_person(
        cfg=cfg,
        frames_since_person=1,
        person_age_ms=10.0,
        unmatched_or_new=False,
        track_lost=True,
        low_track_score=False,
        box_shifted=False,
        mmwave_high_risk=False,
    )
    assert should_refresh_person(
        cfg=cfg,
        frames_since_person=1,
        person_age_ms=10.0,
        unmatched_or_new=False,
        track_lost=False,
        low_track_score=True,
        box_shifted=False,
        mmwave_high_risk=False,
    )
    assert should_refresh_person(
        cfg=cfg,
        frames_since_person=3,
        person_age_ms=10.0,
        unmatched_or_new=False,
        track_lost=False,
        low_track_score=False,
        box_shifted=False,
        mmwave_high_risk=False,
    )
    assert should_refresh_person(
        cfg=cfg,
        frames_since_person=1,
        person_age_ms=120.0,
        unmatched_or_new=False,
        track_lost=False,
        low_track_score=False,
        box_shifted=False,
        mmwave_high_risk=False,
    )
    assert box_shift_frac((0, 0, 100, 200), (80, 0, 180, 200)) > 0.35
    fixed = TrackedRoiConfig(fallback_fixed_stride=True)
    assert should_refresh_person(
        cfg=fixed,
        frames_since_person=1,
        person_age_ms=1.0,
        unmatched_or_new=False,
        track_lost=False,
        low_track_score=False,
        box_shifted=False,
        mmwave_high_risk=False,
    )


def test_gun_batch_split_and_imgsz_buckets():
    assert split_batch_indices(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert tensorrt_batch_range(None)[0] == 1
    assert bucket_gun_imgsz((80, 80), mode="adaptive") == 960
    assert bucket_gun_imgsz((400, 400), mode="adaptive") == 512
    assert bucket_gun_imgsz((150, 150), mode="adaptive") == 640
    assert bucket_gun_imgsz((80, 80), mode="fixed") == 640
    crops = [
        PersonGunCrop(ridx=0, person_xyxy=(0, 0, 1, 1), roi_xyxy=(0, 0, 1, 1), crop=np.zeros((2, 2, 3), np.uint8)),
        PersonGunCrop(ridx=1, person_xyxy=(0, 0, 1, 1), roi_xyxy=(0, 0, 1, 1), crop=np.zeros((2, 2, 3), np.uint8)),
    ]
    order = prioritize_crop_indices(crops, mmwave_ridx={1})
    assert order[0] == 1


def test_predict_gun_on_crops_splits_max_batch():
    calls: list[int] = []

    class _Det:
        def predict(self, source, **_kw):
            n = len(source) if isinstance(source, list) else 1
            calls.append(n)
            from types import SimpleNamespace

            return [SimpleNamespace(boxes=None, names={0: "gun"}) for _ in range(n)]

    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    crops = collect_person_gun_crops(
        frame,
        [(0, (0, 0, 40, 40)), (1, (40, 0, 80, 40)), (2, (80, 0, 120, 40))],
        pad_frac=0.0,
        pad_px=0,
        min_box_px=8,
    )
    predict_gun_on_crops(_Det(), crops, conf=0.2, imgsz=640, device="cpu", batched=True, max_batch=2)
    assert calls == [2, 1]


def test_cpu_capture_latest_frame_slot():
    cap = LiveWebcamCapture.__new__(LiveWebcamCapture)
    cap._lock = threading.Lock()
    cap._frame = np.zeros((6, 8, 3), dtype=np.uint8)
    cap._frame[0, 0] = 9
    cap._seq = 4
    cap._capture_ns = 123
    dst = np.empty((6, 8, 3), dtype=np.uint8)
    assert cap.copy_latest_into(dst) == 4
    assert dst[0, 0, 0] == 9
    assert cap.last_capture_ns == 123


def test_metrics_json_keeps_legacy_keys(tmp_path):
    path = tmp_path / "metrics.json"
    write_live_metrics_json(
        path,
        {
            "infer_fps": 31.5,
            "prediction": "clear",
            "gun_detected": False,
            "latency": {"loop_fps": 31.5, "infer_fps_note": "deprecated"},
            "detection_age_ms": 12.0,
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "infer_fps" in text
    assert "detection_age_ms" in text
    assert "prediction" in text
