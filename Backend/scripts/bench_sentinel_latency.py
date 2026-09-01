#!/usr/bin/env python3
"""Reproducible sentinel latency bench on recorded or synthetic frames (CPU-safe).

Does not require cameras, radar, or a GPU. Reports front/back separately.
Does not claim 60 FPS unless the measured loop actually reaches it.

Example:
  python scripts/bench_sentinel_latency.py --frames 120 --outdir bench_out
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from weapon_ai.detection.gun_batch import bucket_gun_imgsz, split_batch_indices
from weapon_ai.latency.tracker import FrameTiming, LatencyTracker
from weapon_ai.overlay.mmwave_fusion import compute_mmwave_torso_score, mmwave_age_ms
from weapon_ai.pipeline.latest_slot import LatestSlot
from weapon_ai.pipeline.tracked_roi import TrackedRoiConfig, should_refresh_person


def _try_gpu_stats() -> dict[str, object]:
    out: dict[str, object] = {"gpu_util_pct": None, "gpu_mem_mb": None}
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        out["gpu_util_pct"] = int(util.gpu)
        out["gpu_mem_mb"] = round(float(mem.used) / (1024 * 1024), 1)
    except Exception:
        pass
    return out


def _cpu_stats() -> dict[str, object]:
    try:
        load1, _, _ = os.getloadavg()
    except OSError:
        load1 = None
    rss_mb = None
    try:
        import resource

        rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        pass
    return {"cpu_load1": load1, "rss_mb": rss_mb, "memory_bandwidth": None}


def _load_frames(path: Path | None, count: int, width: int, height: int) -> list[np.ndarray]:
    if path is None:
        rng = np.random.default_rng(0)
        return [rng.integers(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(min(8, count))]
    p = path.expanduser()
    if p.is_file() and p.suffix.lower() in {".npy", ".npz"}:
        arr = np.load(str(p))
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[arr.files[0]]
        frames = [np.ascontiguousarray(f) for f in arr]
        return frames or _load_frames(None, count, width, height)
    try:
        import cv2
    except ImportError:
        return _load_frames(None, count, width, height)
    cap = cv2.VideoCapture(str(p))
    frames: list[np.ndarray] = []
    while len(frames) < 32:
        ok, im = cap.read()
        if not ok or im is None:
            break
        frames.append(im)
    cap.release()
    return frames or _load_frames(None, count, width, height)


def _simulate_camera(
    *,
    name: str,
    frames: list[np.ndarray],
    n: int,
    n_people: int,
    mode: str,
    mmwave_period_ms: float,
) -> dict[str, object]:
    """Simulate capture/infer/publish with mocked model durations (no GPU)."""
    lat = LatencyTracker()
    infer_slot: LatestSlot[int] = LatestSlot()
    display_slot: LatestSlot[int] = LatestSlot()
    cfg = TrackedRoiConfig(enabled=mode != "baseline", fallback_fixed_stride=mode == "baseline")
    person_every = 1 if mode == "baseline" else cfg.person_interval_frames
    frames_since = 10**9
    last_person_ns: int | None = None
    capture_times: list[float] = []
    t_loop0 = time.perf_counter()
    radar_ns = time.monotonic_ns()
    for i in range(n):
        t_cap0 = time.perf_counter()
        frame = frames[i % len(frames)]
        _ = frame[0, 0, 0]
        capture_ns = time.monotonic_ns()
        capture_times.append(time.perf_counter() - t_cap0)
        lat.note_loop()
        infer_slot.put(i)
        display_slot.put(i)
        accepted = time.monotonic_ns()
        person_age = None if last_person_ns is None else (accepted - last_person_ns) / 1e6
        run_person = should_refresh_person(
            cfg=cfg,
            frames_since_person=frames_since + 1,
            person_age_ms=person_age,
            unmatched_or_new=i == 0,
            track_lost=False,
            low_track_score=False,
            box_shifted=False,
            mmwave_high_risk=False,
        )
        if mode == "baseline":
            run_person = (i % person_every) == 0
        timing = FrameTiming(capture_ns=capture_ns, accepted_ns=accepted, infer_submit_ns=time.monotonic_ns())
        # Mocked work: copy crops (current pixels) — never reuse gun labels.
        person_ms = 4.0 + 0.4 * n_people if run_person else 0.15
        gun_crop_ms = 0.2 * max(1, n_people)
        gun_ms = 3.0 + 0.8 * n_people
        timing.person_done_ns = timing.infer_submit_ns + int(person_ms * 1e6)
        timing.person_ran = run_person
        timing.gun_crop_done_ns = timing.person_done_ns + int(gun_crop_ms * 1e6)
        timing.gun_done_ns = timing.gun_crop_done_ns + int(gun_ms * 1e6)
        timing.gun_ran = n_people > 0
        timing.tracking_done_ns = timing.gun_done_ns + 200_000
        timing.alert_published_ns = timing.tracking_done_ns + 100_000
        timing.ipc_published_ns = timing.alert_published_ns + 500_000
        lat.record_stages(timing)
        lat.note_detection()
        lat.note_ipc_publish(capture_ns, timing.ipc_published_ns)
        if run_person:
            last_person_ns = timing.person_done_ns
            frames_since = 0
        else:
            frames_since += 1
        if (i * mmwave_period_ms) % 100 < mmwave_period_ms:
            radar_ns = time.monotonic_ns()
        metrics = {"ts_monotonic_ns": radar_ns, "front": {"track": {"confidence": 0.2}}}
        lat.note_mmwave_age_ms(mmwave_age_ms(metrics))
        _ = compute_mmwave_torso_score(metrics, max_age_ms=300.0)
        _ = split_batch_indices(max(1, n_people), 4)
        _ = bucket_gun_imgsz((40 + 10 * n_people, 40 + 10 * n_people), mode="fixed" if mode == "baseline" else "adaptive")
    elapsed = time.perf_counter() - t_loop0
    loop_fps = n / max(1e-9, elapsed)
    cap_fps = n / max(1e-9, sum(capture_times) if capture_times else elapsed)
    payload = lat.metrics_payload()
    return {
        "camera": name,
        "mode": mode,
        "people": n_people,
        "frames": n,
        "capture_fps": round(cap_fps, 2),
        "main_loop_fps": round(loop_fps, 2),
        "detection_fps": payload.get("detection_fps"),
        "person_infer_ms": payload.get("person_infer_ms"),
        "gun_infer_ms": payload.get("gun_infer_ms"),
        "vision_alert_latency_ms": payload.get("vision_alert_latency_ms"),
        "frame_age_at_publish_ms": payload.get("frame_age_at_publish_ms"),
        "mmwave_age_ms": payload.get("mmwave_age_ms"),
        "infer_frames_replaced": infer_slot.replaced,
        "display_frames_replaced": display_slot.replaced,
        "note": (
            "person/gun milliseconds in this script are injected synthetic stage lengths "
            "plus Python loop overhead; they are not TensorRT measurements."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sentinel latency bench (synthetic or recorded frames).")
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--video", type=str, default="")
    p.add_argument("--outdir", type=str, default="bench_out")
    p.add_argument("--mmwave-ms", type=float, default=100.0, help="Simulated radar period (10 Hz default).")
    args = p.parse_args(argv)
    frames = _load_frames(Path(args.video) if args.video else None, args.frames, args.width, args.height)
    host = {**_cpu_stats(), **_try_gpu_stats()}
    results: list[dict[str, object]] = []
    for people in (0, 1, 2, 4, 8):
        for cam in ("front", "back"):
            for mode in ("baseline", "optimized"):
                results.append(
                    _simulate_camera(
                        name=cam,
                        frames=frames,
                        n=int(args.frames),
                        n_people=people,
                        mode=mode,
                        mmwave_period_ms=float(args.mmwave_ms),
                    )
                )
    report = {
        "measured": True,
        "gpu_required": False,
        "host": host,
        "results": results,
        "claims": {
            "sixty_fps": False,
            "reason": "This bench does not run TensorRT; do not treat loop_fps as live 60 FPS.",
        },
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "sentinel_bench.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
