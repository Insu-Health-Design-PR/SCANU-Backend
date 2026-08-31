# Mmwave — standalone dual 77 GHz radar module

This directory contains a self-contained application for capturing and fusing
two **TI AWR1843BOOST** radars connected directly to the same Linux server. It
does not open cameras and does not depend on the main
`Backend` process and does not ask the Backend to open any sensor.

The only operator entrypoint is `./run`.

## Quick start

```bash
cd /home/insu/Desktop/New_Backend/Mmwave
./run preflight
./run live 20
```

The `live` sequence is automatic:

1. verify both configured radars;
2. calibrate the empty room;
3. show a countdown before participant entry;
4. continue reading both radars without reopening their UART ports;
5. suppress calibrated clutter, fuse points, and create global tracks;
6. publish one classic fused dashboard and structured metrics until stopped.

No Enter key is required. Stop the standalone workflow with Ctrl+C; when the
Backend owns it, use the mmWave Stop endpoint.

## Commands

```bash
./run preflight                     # resolve both configured radar pairs
./run live 20                       # calibrate 20 s, countdown, live until stopped
./run capture 120 20                # established finite offline experiment
./run 120 20                        # legacy spelling of the same finite capture
```

The maintained `./run` workflow never opens cameras. Camera integration is
owned by the sibling Backend and consumes the published mmWave contract.

## Live outputs

By default the calibrated runtime atomically publishes under
`/dev/shm/scanu_mmwave`:

- `live_mmwave_fused_dashboard.jpg` — one A+B classic dashboard;
- `live_metrics.json` — fused points, global IDs, quality and reflective evidence;
- `live_status.json` — state and calibration progress;
- `live_manifest.json` — final quality and shutdown record.

## Outputs

Each execution creates a unique directory under `data/captures` containing:

- empty-room calibration sessions for radar A and radar B;
- participant sessions for both radars;
- `fusion_report.json`, containing geometric post-CFAR fusion;
- `global_tracks.json`, containing global identities deduplicated across sensors;
- a classic six-panel fused radar dashboard with the observed A+B point cloud,
  top/side views, per-radar range profiles, world occupancy, and evidence timeline;
- subprocess logs, hashes, and `manifest.json`.

The classic dashboard preserves the established human-centric style: gray
scene returns, small blue track-associated points, a fused observed-body
wireframe, yellow transient reflective anomalies, and red persistent
reflective anomalies. It displays measured post-CFAR coordinates without
voxel interpolation or invented points.

## What the system actually measures

The AWR1843 USB connection carries TLVs processed by TI firmware: post-CFAR
points, range, angles, Doppler velocity, power/SNR, and available profiles. It
does not carry raw ADC/IQ samples. Dual-radar fusion is geometric and temporal;
it does not create a coherent combined antenna aperture.

Anomaly colors represent experimental reflective-return evidence. Reflectivity
alone **does not identify a material**, confirm metal, or classify a firearm.

## Main code

- `software/lab/dual_server_77ghz/orchestrate.py`: complete experiment lifecycle.
- `software/lab/mmwave77_usb/runner.py`: port discovery and TI TLV capture.
- `software/lab/mmwave77_usb/cube.py`: range-azimuth-elevation volume.
- `software/lab/mmwave77_usb/background.py`: empty-room calibration and clutter.
- `software/lab/mmwave77_usb/perception.py`: points, tracks, and reflective evidence.
- `software/lab/dual_mmwave77_stereo/point_cloud_fusion.py`: transformation and fusion.
- `software/lab/dual_mmwave77_stereo/global_tracks.py`: global person association.
- `software/lab/dual_mmwave77_stereo/unified_fusion_video.py`: final video renderer.

The renderer's `--classic-human-centric` option is enabled by both maintained
server configurations. The normal `./run capture` command therefore creates
the triptych and classic dashboard without introducing another operator entrypoint.

See `ARCHITECTURE.md` for contracts, data flow, synchronization, and limitations.
