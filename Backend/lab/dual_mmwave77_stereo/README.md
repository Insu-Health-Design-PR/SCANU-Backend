# Dual-node facing AWR1843 stereo lab (experimental)

Two independent AWR1843BOOST + USB-camera pairs, physically facing each other
across a shared space, each connected to a different host:

- Node A: Jetson (`100.92.1.128`), existing single-radar lab path
  (`software/lab/mmwave77_usb`).
- Node B: a second, separately owned host (for example a workstation server),
  running its own independent clone/venv of this repository.

This module adds only what does not already exist for a single radar: cross-
host orchestration and a cross-node comparison of two independently processed
sessions. It does not duplicate or fork the single-radar capture, cube,
background-calibration, or perception pipeline — both nodes run the existing,
unmodified `lab.mmwave77_usb` CLIs (`runner`, `cube`, `background`,
`perception`, `perception_video`, `camera_capture`) locally, exactly as the
single-sensor lab already does. This folder only orchestrates two of those
runs and compares their independent outputs afterward.

## Scope and limitations

- This is a laboratory diagnostic path outside canonical Layers 1-8. Its
  output cannot enter schema-V6 training and is not a Layer 5 fusion
  decision.
- `fusion.py` performs a **descriptive comparison** of two independently
  computed `screening_state` timelines (one per radar). It does not fuse raw
  returns, does not triangulate a shared 3D position between the two radars,
  and does not know whether the two radars actually saw the same physical
  event — it only aligns their windows by wall-clock time.
- There is no controlled ground-truth label in this lab capture (no known
  true class per second). Agreement/union/intersection counts describe how
  often the two independent single-radar pipelines agreed or disagreed; they
  are **not** an accuracy, precision, or recall measurement.
- `screening_state` values (`background`, `person`, `suspicious_metal`,
  `insufficient_signal`) come from the existing experimental
  `lab.mmwave77_usb.perception` pipeline and carry the same limitations
  documented there (post-CFAR points, not raw ADC; no material/weapon
  classification).
- Cross-host time alignment depends on an externally supplied clock-offset
  measurement (`scripts/measure_clock_offset.sh`). Both hosts are expected to
  already be NTP-synchronized; the measured offset is logged in every fusion
  report for transparency, not silently assumed to be zero.

## Interference check (must run once per physical setup)

Two active FMCW radars facing each other can, in principle, raise each
other's noise floor or introduce spurious detections. Before trusting a
"two radars detect better than one" comparison, capture three short empty-room
sessions — radar A alone, radar B alone, and both simultaneously — and compare
`noise_profiles.npz` statistics and average detected-point counts per frame
between the "alone" and "simultaneous" conditions for each radar. Record the
result before deciding whether simultaneous capture is valid for a given
physical layout; do not assume the answer transfers to a different distance,
angle, or room.

## Files

- `fusion.py` — loads `frames.jsonl` + `perception.jsonl` from two independent
  sessions, aligns windows by wall-clock time (with an applied clock-offset
  correction), and writes a comparison report plus per-window match table.
- `testing/test_fusion.py` — unit tests for the alignment and comparison
  logic using synthetic sessions.

## Orchestration

The cross-host orchestrator lives outside this package, in
`scripts/run_dual_awr1843_stereo_test.sh` and
`scripts/measure_clock_offset.sh`, matching the existing convention that
laboratory *host automation* is a `scripts/` concern while laboratory
*processing* is a `software/lab/*` concern.
