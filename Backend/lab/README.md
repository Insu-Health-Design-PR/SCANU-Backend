# Vendored Adrian lab (mmWave)

Copied from `Adrian_code/software/lab` into this Backend for Layer 8 UI integration.

| Package | Role |
|---------|------|
| `mmwave77_usb/` | AWR1843 USB TLV capture → cube → perception → (offline) perception_video |
| `dual_mmwave77_stereo/` | Facing dual-radar fusion / remux |
| `mmwave77_usb/live_perception_frame.py` | JPEG renderer for UI `live_mmwave.jpg` (body vs anomaly colors) |

## Runtime entry

```bash
python -m runtime.mmwave_lab_live \
  --pipeline lab_replay \
  --session /path/to/PARTICIPANT_CAPTURE \
  --live-frame layer8_ui/artifacts/live_mmwave.jpg
```

Layer 8 **mmWave → Run** uses this publisher when `mmwave.pipeline` is `lab_replay` / `lab_live` / `status` (default `lab_replay`).

Set `mmwave.pipeline=legacy` to use the older `layer1_radar` `live_capture.py` path via `software_root`.

## Scripts

- `scripts/run_dual_awr1843_stereo_test.sh`
- `scripts/measure_clock_offset.sh`

## Dependencies

Minimal UART stack: `layer1_sensor_hub/radar/` + `hardware_registry.py`.
Camera helper: `layer8_ui/camera_device.py`.
