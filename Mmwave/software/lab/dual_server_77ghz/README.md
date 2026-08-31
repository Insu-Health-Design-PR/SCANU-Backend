# Server-local dual 77 GHz laboratory

This package captures two AWR1843BOOST radars and one or two USB cameras attached
directly to the Linux server. It performs an automatic empty-room calibration,
waits a fixed entry interval, records all four sources, processes each radar,
builds a post-CFAR geometric fusion, creates global person tracks and renders a
three-panel synchronized evidence video.

The module is experimental and independent of canonical Layers 1–8. It uses
processed TI TLVs, not raw ADC. Reflectivity is not material-specific and the
output is not a firearm classification.

## Preflight on the server

```bash
cd /home/insu/Desktop/scanu-stereo-node/SCANU
export PYTHONPATH="$PWD:$PWD/software"
.venv/bin/python3 -m lab.dual_server_77ghz.orchestrate \
  --config software/lab/dual_server_77ghz/configs/server_local.json preflight
```

Preflight requires exactly two selected XDS110 radar pairs and the configured
number of distinct camera devices. Set `camera_b.enabled` to `false` for a
single-camera run. It reports the stable USB locations and camera paths without
starting a radar stream.

## Complete automatic experiment

Keep the area empty when starting. The command calibrates for 20 seconds,
prints a countdown, then records for the configured duration without requiring
ENTER:

```bash
.venv/bin/python3 -m lab.dual_server_77ghz.orchestrate \
  --config software/lab/dual_server_77ghz/configs/server_local.json \
  capture --duration-s 60
```

Use `--calibration-s`, `--entry-delay-s` and `--duration-s` to override the
configuration. `--skip-render` preserves all captures and reports but omits the
costly MP4 composition.

## Outputs

Each unique run contains calibration and participant sessions for both radars,
both native camera videos, per-radar perception, `fusion_report.json`,
`global_tracks.json`, a three-panel MP4, process logs, hashes and `manifest.json`.
