# Experimental 77 GHz USB mmWave runner

This lab package discovers and records a Texas Instruments AWR1843BOOST
connected directly to the Jetson over its onboard XDS110 USB interface. It is
deliberately separate from the canonical SCAN-U collector and does not change
the Layers 1–8 live runtime.

The output is diagnostic evidence only. It is not a schema-V6 capture and
cannot be used directly by `software.train_model`.

## Minimal camera + cube review

The existing laptop runner can keep the capture workflow unchanged while
rendering only one camera and one 3D radar cube.  It deliberately omits the
top view, side view, history/trajectory, and timeline panels:

```bash
CAPTURE_NODE=server USE_PERSON_REFERENCE=0 VISUAL_MODE=handheld_minimal \
CALIBRATION_SECONDS=20 ./scripts/run_awr1843_camera_laptop_test.sh 60
```

The generated radar video uses the same sparse observed cloud and track mask
as the full perception renderer. Blue is a person-associated return and
yellow/red are its reflectivity cues. They are not a material, object, or
firearm result.

The minimal export now aligns the camera at both the first and last radar
window, removes anomaly markers outside the tracked chest envelope, and adds a
range-bin residual graph with measured peaks. It still uses processed TLVs and
does not create raw-ADC resolution or confirm a material.

## 1. Identify the USB device

Connect the new sensor, then run:

```bash
cd ~/Desktop/SCANU-dev_adrian
export PYTHONPATH="$PWD:$PWD/software"
python3 -m lab.mmwave77_usb.runner list
```

The runner marks the three existing SCAN-U CP2105 bridges and excludes them
from automatic selection.

For more detail:

```bash
python3 -m lab.mmwave77_usb.runner list --json
```

Record the model name, `/dev/ttyUSB*` or `/dev/ttyACM*` port, USB VID/PID,
serial number, and the manufacturer's UART baud/protocol.

For the AWR1843BOOST, resolve its two XDS110 ports automatically:

```bash
python3 -m lab.mmwave77_usb.runner detect-awr1843
```

Expected roles:

- XDS110 Application/User UART: CLI/configuration at 115200 baud;
- XDS110 Auxiliary Data Port: processed TLV data at 921600 baud.

The board must also have a suitable external 5 V supply; USB provides the
communications path but is not a substitute for the board's required power.

## 2. Safe AWR1843BOOST capture

If the TI out-of-box demo is already configured and streaming:

```bash
python3 -m lab.mmwave77_usb.runner capture \
  --auto-awr1843 \
  --protocol ti-tlv \
  --duration-s 15
```

The XDS110 port names determine CLI versus data, so volatile
`/dev/ttyACM*` numbers are not hardcoded.

## 3. Safe raw diagnostic capture

Raw mode never sends commands to the sensor:

```bash
python3 -m lab.mmwave77_usb.runner capture \
  --data-port /dev/ttyACM0 \
  --sensor-model MODEL_FROM_LABEL \
  --baud 921600 \
  --duration-s 15 \
  --protocol raw
```

If the sensor is the only non-SCAN-U serial device, `--data-port` may be
omitted. The session is written below `data/lab/mmwave77_usb/` with:

- `raw_uart.bin`: byte-for-byte UART evidence;
- `metadata.json`: USB identity and capture parameters;
- `summary.json`: byte count, SHA-256, rate, and limitations;
- `frames.jsonl`: empty in raw mode.

An `ok: false` result means no bytes arrived. Check power, permissions, the
correct data port, baud rate, and whether the vendor requires a start command.

## 4. Configure the TI out-of-box demo

Use this only if the board runs a TI mmWave demo compatible with the standard
magic word and TLV header:

```bash
python3 -m lab.mmwave77_usb.runner capture \
  --data-port /dev/ttyUSB11 \
  --sensor-model IWR6843ISK \
  --baud 921600 \
  --duration-s 15 \
  --protocol ti-tlv
```

This adds decoded frame numbers, sensor cycle counters, detected points and TLV
types to `frames.jsonl` while preserving every received byte in
`raw_uart.bin`.

The connected laboratory board reports mmWave SDK `03.04.00.03`. A matching
processed-data 2D profile is included at:

```text
software/lab/mmwave77_usb/configs/awr1843boost_sdk_3_4_profile_2d.cfg
```

Do not use SCAN-U's IWR6843 files: those start near 60.75 GHz. This runner
refuses any AWR1843 profile whose `profileCfg` start frequency is outside
76-81 GHz.

Run the SDK 3.4 profile:

```bash
python3 -m lab.mmwave77_usb.runner capture \
  --auto-awr1843 \
  --config software/lab/mmwave77_usb/configs/awr1843boost_sdk_3_4_profile_2d.cfg \
  --protocol ti-tlv \
  --duration-s 15
```

Every CLI command must return a non-empty response without an error before
the next command is sent. Responses are preserved in `configuration.json`.
The SDK 3.4 firmware on the laboratory board does not support the newer
`calibData` command, so it is intentionally absent. The runner sends
`sensorStop` on exit after configuration starts, including partial failure.

## Firmware prerequisite

The board must have an xWR18xx mmWave SDK demo flashed. If the two XDS110 ports
appear but configuration commands do not return the mmWave prompt, flash the
official xWR18xx out-of-box demo with TI UniFlash before collecting data.

The current adapter decodes the common SDK TLV stream. Raw ADC is not available
through this USB stream; TI documents DCA1000/LVDS for raw ADC capture.

## 5. Elevation, azimuth and depth cube

The validated 2D profile activates only the two azimuth transmitters. For an
experimental 3D point capture, use the separate SDK 3.4 profile that also
activates the elevation-offset transmitter:

```bash
python3 -m lab.mmwave77_usb.runner capture \
  --auto-awr1843 \
  --config software/lab/mmwave77_usb/configs/awr1843boost_sdk_3_4_profile_3d.cfg \
  --protocol ti-tlv \
  --duration-s 15 \
  --output-root /mnt/scanu-data/validation/awr1843_usb
```

Then build temporal range/azimuth/elevation voxels from that session:

```bash
python3 -m lab.mmwave77_usb.cube \
  --session /mnt/scanu-data/validation/awr1843_usb/CAPTURE_DIRECTORY
```

The output `rae_cube_tlv.npz` uses axis order
`[window, range, azimuth, elevation]`. It contains:

- `hit_count`: accumulated detected-point observations;
- `snr_mean_db`: mean point SNR per occupied voxel;
- `doppler_mean_mps` and `doppler_abs_max_mps`;
- axis edges/centers and source frame-number bounds for every window.

`rae_cube_tlv.metadata.json` records hashes, coordinate conventions, binning,
statistics and limitations. The defaults use 64 range, 48 azimuth and 24
elevation bins, accumulating 10 frames with a five-frame stride. Bounds and
bin counts are CLI options.

This is a **sparse detection cube**: the demo firmware has already performed
range/Doppler/angle processing and emitted only detected Cartesian points.
An empty voxel means no reported point, not verified empty space. A dense
complex range/azimuth/elevation radar cube requires raw virtual-antenna ADC
data through LVDS/DCA1000 or specialized firmware and is not available over
the XDS110 processed-data USB stream. These laboratory cubes remain outside
schema V6 and cannot be used directly by `software.train_model`.

Render the cube as an MP4 with synchronized 3D, top and side maps:

```bash
python3 -m lab.mmwave77_usb.video \
  --cube /mnt/scanu-data/validation/awr1843_usb/CAPTURE_DIRECTORY/rae_cube_tlv.npz
```

The default `rae_map.mp4` is a 1280×720 H.264 video. Marker size represents
accumulated detections and color represents elevation. Its companion metadata
records the cube and video hashes, frame rate, duration and limitations.

## Verified Jetson result

On 2026-07-28, AWR1843BOOST XDS110 serial `R2091049` reported xWR18xx mmWave
SDK `03.04.00.03`. The included profile produced 150 consecutive frames in
15.12 seconds through the Auxiliary Data UART:

- 373,281 raw UART bytes;
- frame numbers 2 through 151 with no sequence gaps;
- zero parser errors, invalid headers, or discarded bytes;
- detected-point, range-profile, noise-profile, statistics, and side-info TLVs;
- 256-bin range profiles and detected points in every frame.

The auditable session is stored on the Jetson under
`/mnt/scanu-data/validation/awr1843_usb/capture_20260728_143833`.

The three-transmitter profile was validated separately after a board reset.
Session `capture_20260728_151206` produced 150 consecutive frames, 8,735
points and 8,733 nonzero-z points without parser or sequence errors. Its
29-window sparse cube contains 5,118 occupied voxels. The accompanying
1280×720 H.264 map has 29 frames at 5 fps. Observed elevation variation
confirms the 3D data path, but does not constitute an angular accuracy
calibration.

## 6. Two AWR1843BOOST sensors in parallel

Two XDS110 boards may report the same serial number. Discovery therefore
groups each CLI/data pair by its physical USB topology path, such as `1-4.1`
or `1-4.2`. Keep the boards connected to the same Jetson USB sockets so these
laboratory identities remain stable.

Run a bounded dual capture:

```bash
python3 -m lab.mmwave77_usb.dual \
  --duration-s 15 \
  --stagger-ms 50 \
  --output-root /mnt/scanu-data/validation/awr1843_dual
```

The dual runner:

- requires exactly two complete XDS110 CLI/data pairs;
- consumes stale CLI startup responses and verifies `xWR18xx` firmware before
  applying either configuration;
- launches isolated single-sensor captures with a requested USB start stagger;
- stops both sensors through each child runner;
- preserves raw UART, decoded frames, configuration responses and hashes for
  each sensor independently;
- builds one sparse RAE cube per sensor;
- writes `dual_overlay.npz`, `dual_manifest.json` and `dual_rae_map.mp4`.

The initial overlay uses identity transforms because sensor separation,
translation, yaw, pitch and roll have not been measured. It is useful for
side-by-side comparison and interference diagnosis, but it is not calibrated
metric fusion and does not deduplicate targets. USB process staggering is
best-effort, not hardware frame synchronization.

## 7. One-sensor environment with metal-like highlights

The three-TX profile can use one sensor for both 3D geometry and a preliminary
reflectivity ranking. Build the ordinary sparse cube, score it, and render the
full environment:

```bash
python3 -m lab.mmwave77_usb.metal_like \
  --cube CAPTURE_DIRECTORY/rae_cube_tlv.npz
python3 -m lab.mmwave77_usb.metal_video \
  --map CAPTURE_DIRECTORY/metal_like_map.npz
```

The English-language video maps the original Cartesian point occurrences from
each 10-frame window back to their scored voxels. It draws the full raw point
cloud in gray and persistent, high-SNR, same-range-contrast candidates as
colored diamonds. `metal_like_score` combines absolute SNR, temporal hit
persistence and robust contrast against occupied voxels at the same range. It
is a heuristic ranking, not a probability or confirmed material label. Strong
nonmetal reflectors may be highlighted and metal may be missed. No weapon class
is produced.

The one-sensor pipeline was physically exercised at USB location `1-4.2` in
session `capture_20260729_143032`: 150 frames produced 2,290 occupied voxels
and 547 heuristic highlights without UART parser errors. Because the scene had
no controlled material labels, this validates only capture, scoring and
visualization—not metal identification.

For a paper-style view of the same sparse TLV evidence, render interpolated
range–azimuth and azimuth–elevation maps:

```bash
python3 -m lab.mmwave77_usb.heatmap_video \
  --map CAPTURE_DIRECTORY/metal_like_map.npz
```

The output uses Gaussian splatting to make isolated detected points easier to
see and adds white contours around the unverified reflectivity candidates.
It is explicitly labeled as an interpolated sparse-TLV visualization. It does
not reconstruct the dense complex heatmaps obtainable from raw ADC/LVDS data
and does not add physical angular or range resolution.

## 8. Human-centric perception

The human-centric path supersedes the undifferentiated `metal_like` display
for interpretation experiments. It preserves the complete 256-bin range and
noise profiles, models an explicitly empty room, tracks one observed human
volume, and scores only compact reflectors associated with that track.

Every new `ti-tlv` capture writes:

- `range_profiles.npz` and `noise_profiles.npz`, including full values, frame
  identity and UART receipt timing;
- companion profile metadata with hashes;
- `capture_quality.json`, including sequence gaps, profile coverage and
  transport integrity;
- the existing byte-exact `raw_uart.bin` and decoded `frames.jsonl`.

Recover these artifacts from an older preserved UART capture without opening
the sensor:

```bash
python3 -m lab.mmwave77_usb.reprocess \
  --session CAPTURE_DIRECTORY
```

Create an empty-room calibration only while the inspection zone is known to
be empty:

```bash
python3 -m lab.mmwave77_usb.cube \
  --session EMPTY_ROOM_CAPTURE
python3 -m lab.mmwave77_usb.background \
  --session EMPTY_ROOM_CAPTURE \
  --condition empty_room
```

The baseline uses per-range-bin median, MAD and percentiles. The sparse RAE
artifact records occupancy and masks the configured near field plus voxels
that remain occupied in the empty scene. It does not learn a material class.
Move the sensor or major scene geometry and the baseline must be collected
again. Participant and empty-room sessions must also have the same normalized
TI CLI command signature; incompatible profiles are rejected.

Interpret a participant capture against that calibration:

```bash
python3 -m lab.mmwave77_usb.cube \
  --session PARTICIPANT_CAPTURE
python3 -m lab.mmwave77_usb.perception \
  --session PARTICIPANT_CAPTURE \
  --calibration-session EMPTY_ROOM_CAPTURE
python3 -m lab.mmwave77_usb.perception_video \
  --perception PARTICIPANT_CAPTURE/perception.jsonl
```

For a controlled object-reflectivity comparison, add a matched capture of the
same participant without the test object. The participant must use the same
marked position, clothing, pose, and movement sequence in both captures:

```bash
python3 -m lab.mmwave77_usb.perception \
  --session TARGET_CAPTURE \
  --calibration-session EMPTY_ROOM_CAPTURE \
  --person-reference-session PERSON_WITHOUT_OBJECT_CAPTURE
```

This second reference gates candidate returns against the upper range-profile
envelope already produced by the person's body and clothing. It reduces the
previous failure mode in which a strong torso return was labeled solely because
it differed from an empty room. It is still a controlled A/B radar comparison,
not material identification or firearm classification.

When a capture protocol explicitly declares its leading windows empty, they
may also suppress capture-local startup clutter for visualization:

```bash
python3 -m lab.mmwave77_usb.perception \
  --session PARTICIPANT_CAPTURE \
  --initial-empty-windows 18
```

Those windows are labeled background and cannot create a human track. This is
useful for diagnosing an existing capture, but it does not provide the matched
external profile baseline required for anomaly screening. The output therefore
remains `insufficient_signal`.

The 1920×1080 English video displays the measured scene returns, observed
human volume, track history, body-associated reflective anomalies, unsmoothed
range–azimuth point bins, current versus empty-room range profile, and an
evidence timeline. The volume is not an anatomical reconstruction. Track
confidence and reflective-anomaly score are uncalibrated engineering rankings,
not probabilities.

If a compatible empty-room baseline or full profiles are missing, screening
fails closed to `insufficient_signal`. A persistent calibrated body-associated
radar anomaly may be labeled `suspicious_metal`, but the output never confirms
material or produces a firearm class. The controlled capture and evaluation
method is documented in [METHODOLOGY.md](METHODOLOGY.md).

## 9. One Server radar and one Server camera

The existing laptop camera runner has a Server profile; no second operator
entrypoint is required. From the repository on the Mac, run:

```bash
CAPTURE_NODE=server ./scripts/run_awr1843_camera_laptop_test.sh 60
```

By default the command performs a fresh empty-room calibration and then records
the moving participant with the test object, using only the AWR1843 and 4K/30
camera attached to the Server. It downloads these files to
`~/Desktop/Server AWR1843 test videos`:

- the native 4K/30 camera MP4;
- the independent 1920×1080/2-fps radar perception MP4;
- a 3840×1080/30-fps camera/radar comparison MP4;
- the perception summary and timebase sidecars.

Set `USE_PERSON_REFERENCE=1` to enable the optional matched person-only stage;
`PERSON_REFERENCE_SECONDS` controls its duration (default 10 seconds). Without
that stage, temporal persistence and track-relative compactness suppress brief
motion artifacts, but normal strong body reflections cannot be empirically
separated from object reflections. In either mode the target video shows
experimental reflectivity evidence and cannot establish that an object is
metal or a weapon.

The Server dashboard normally owns the camera. The runner traces the camera
holder to the known New_Backend process, pauses only that process immediately
before recording, and restores the API afterward. It refuses to stop an
unrecognized holder. Camera video is aligned to the median host timestamp of
the first ten-frame radar perception window, not to the beginning of that
window. This corrects the deterministic half-window lead but remains software
synchronization; it is not a common hardware trigger or a calibrated
material/weapon classifier.
