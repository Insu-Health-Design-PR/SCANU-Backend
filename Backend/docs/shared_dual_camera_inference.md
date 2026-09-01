# Shared dual-camera inference (experimental)

Front and Back **already share one GPU** by default (`cuda_visible_devices: "0"` in
each sensor block). Capture and overlay stay independent processes.

## Goal

One bounded inference service that:

* Accepts latest-frame person images from each camera (replace, never queue).
* May batch front+back person frames and gun crops.
* Waits at most **2 ms** to form a batch.
* Preserves `camera_id`, capture timestamps, and per-camera order.
* Never lets a stalled camera block the other.

## Why this is feature-flagged

Independent processes already isolate crashes and TensorRT contexts. Cross-camera
batching can *add* latency if one camera is slow. Default remains **independent
infer subprocesses** until a benchmark shows a net win.

Enable later with `SCANU_SHARED_INFER=1` or `--shared_infer_service`. The runtime
currently logs the flag and continues with the existing per-camera pool.

## Skeleton

`weapon_ai/pipeline/shared_infer.py` holds `SharedInferConfig` only. A socket
protocol is intentionally not wired into `infer_objects.py` yet.
