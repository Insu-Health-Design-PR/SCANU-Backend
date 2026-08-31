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
- `global_tracks.py` — post-CFAR multi-person association in a common world
  frame. It creates persistent global IDs, reports the simultaneous count, and
  deduplicates A/B observations of the same spatial cluster. A same-window
  cross-view reflective event is reported as `multiview_reflective_anomaly`;
  a one-view event remains `single_view_reflective_anomaly`. Neither is a
  material, weapon, or firearm result.

## Global person counting and deduplication

The new global tracker is the right fusion level for independent boards:

```text
radar A TLV ─┐                        ┌─ calibrated world coordinates
             ├─ local clusters ───────┤
radar B TLV ─┘  B→A rigid transform   ├─ A/B association (one-to-one)
                                        ├─ temporal association → G001, G002…
                                        └─ reflective evidence associated to a global track
```

It does **not** add the raw ADC streams or pretend that the two boards form one
48-channel coherent aperture. That would require a shared trigger/reference
and phase calibration. For the facing geometry, it starts from the existing
measured transform `x'=-x, y'=D-y, z'=z`; the sensor distance, clock offset,
empty-room clutter masks and association gates are recorded in the output.

Example post-capture analysis on the Server:

```bash
export PYTHONPATH="$PWD:$PWD/software"
python3 -m lab.dual_mmwave77_stereo.global_tracks \
  --session-a /path/to/session_A \
  --session-b /path/to/session_B \
  --calibration-a /path/to/empty_A \
  --calibration-b /path/to/empty_B \
  --distance-m 3.6576 \
  --clock-offset-b-minus-a-s 0.0 \
  --output /path/to/global_tracks_report.json
```

`global_person_count` is the current experimental count; `active_tracks` has
the global ID and position for that window. It must be calibrated against a
manually observed count before being treated as a count-performance metric.

## Orchestration

The cross-host orchestrator lives outside this package, in
`scripts/run_dual_awr1843_stereo_test.sh` and
`scripts/measure_clock_offset.sh`, matching the existing convention that
laboratory *host automation* is a `scripts/` concern while laboratory
*processing* is a `software/lab/*` concern.

## Independent raised-hand comparison

`scripts/run_dual_awr1843_raised_hand_laptop_test.sh` is a deliberately
non-fused A/B experiment. It retains the two independent AWR1843 and camera
captures, but renders one point-cloud-only English video per radar and copies
the two camera videos and two radar videos to `demo-video` on the Server and
laptop. The visual colors are: blue for person-associated reported returns,
yellow for an elevated single-radar high-reflectivity cue, and red only after
that elevated cue persists across windows. Red is **not** confirmed metal,
object identity, or a weapon classification. The elevated region is a
display ROI for a controlled hand-raised test, not anatomical localization.

## Classic separate capture

`scripts/run_dual_awr1843_separate_laptop_test.sh` is the short operator path
for the proven July human-centric visualization. One command performs the
two-radar empty-room calibration, waits 15 seconds for participant entry,
captures both radars and both cameras, renders the classic multi-panel radar
dashboard at its measured 1920x1080/2-fps cadence, and downloads the four
videos plus radar/timebase metadata to
`~/Desktop/INSU/77ghz mmwave video`. The cameras remain native 4K/30. The
radar videos are not upscaled or frame-duplicated. The reflectivity/anomaly
panels remain experimental evidence and do not identify material or weapons.

## Synchronized three-panel fused export

The complete dual-node workflow also writes
`*_cameras_fused_radar_triptych_4k.mp4`. It contains exactly three panels in
one shared radar-window timebase: camera A, one geometrically fused A+B point
cloud, and camera B. The center panel transforms B into A's coordinate frame;
it is post-CFAR geometric fusion, not coherent ADC/IQ aperture fusion. Camera
frames are selected from their recorded host-monotonic timestamps for every
matched radar window. The result is software-synchronized laboratory evidence,
not a hardware-triggered acquisition or a material/weapon classification.
