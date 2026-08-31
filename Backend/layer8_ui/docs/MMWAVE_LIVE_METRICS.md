# Live mmWave metrics (`scanu_mmwave_live_v1`)

Published by the standalone `Mmwave` calibrated live runtime, launched through
`runtime.mmwave_dual_live`, to:

- `layer8_ui/artifacts/live_mmwave_metrics.json`
- `/dev/shm/scanu_mmwave_live_metrics.json` (low-latency read for infer overlay)

The default visual endpoint is one fused classic dashboard:

```text
GET /api/mmwave/preview/live?side=fused
```

Lifecycle endpoints:

```text
POST /api/mmwave/live/start
POST /api/mmwave/live/stop
POST /api/mmwave/live/recalibrate
GET  /api/mmwave/live/status
GET  /api/mmwave/live_metrics
WS   /ws/mmwave
```

## Schema

```json
{
  "schema_version": "scanu_mmwave_live_v1",
  "experimental": true,
  "ts_monotonic_ns": 0,
  "sensor_distance_m": 3.6576,
  "front": {
    "screening_state": "person",
    "track": {
      "centroid_m": [0.1, 2.5, 1.0],
      "position_m": [0.1, 2.5, 1.0],
      "velocity_mps": [0, 0, 0],
      "observed_extent_m": [0.5, 0.5, 1.6],
      "confidence": 0.72
    },
    "points": [{"x": 0, "y": 0, "z": 0, "snr": 12, "range": 1.2}],
    "anomalies": [{"centroid_m": [0.2, 2.4, 1.1], "score": 0.75, "kind": "reflective"}]
  },
  "back": {},
  "fused": {
    "global_person_count": 1,
    "fused_centroid_m": [0.1, 2.5, 1.0],
    "active_tracks": []
  }
}
```

## Safety

- `screening_state` and anomalies are **experimental radar evidence**, not weapon classification.
- `mmwave_torso_score` in camera metrics is an engineering correlation score only.

## Consumers

| Consumer | Usage |
|----------|--------|
| `weapon_ai/infer_objects.py` | `--mmwave_overlay` draws dots on camera frame |
| `services/mmwave_metrics_service.py` | `/api/mmwave/live_metrics`, dashboard `#metric-mmwave-score` |
| Layer 8 mmWave tab | Preflight + service status |
