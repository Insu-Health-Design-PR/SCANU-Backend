# Dual AWR1843 stereo lab — operations guide

This guide explains the **code**, the **architecture**, and how to run the
dual-radar stereo experiments over **passwordless SSH** across the three
hosts. This is a laboratory diagnostic path **outside** canonical Layers 1–8;
it is experimental evidence only, not a production firearm detector.

## Hosts and users

| Role | Host | User | Repo path | Runs / captures |
|---|---|---|---|---|
| Laptop (local) | macOS dev machine | `adriancordero` | `~/Desktop/INSU/SCANU` | orchestration, downloads, git |
| Node A | Jetson | `insu` | `/home/insu/Desktop/SCANU-dev_adrian` | radar A + camera A capture |
| Node B | Server | `insu` | `/home/insu/Desktop/scanu-stereo-node/SCANU` | radar B + camera B capture + centralized processing |
| Remote git | GitHub | `Insu-Health-Design-PR` | `https://github.com/Insu-Health-Design-PR/SCANU.git` | shared `dev_adrian` branch |

IP addresses: Jetson `100.92.1.128`, Server `100.96.160.1`.

The laptop has SSH keys already deployed so it can reach both nodes and GitHub
**without a password**. Usernames are included in the examples below.

## SSH without a password

Key exchange must already be set up on the laptop:

```bash
ssh-keygen -t ed25519 -C "adrian" -f ~/.ssh/id_ed25519
ssh-copy-id insu@100.92.1.128   # Jetson
ssh-copy-id insu@100.96.160.1   # Server
```

Verify the nodes answer without prompting for a password:

```bash
ssh -o BatchMode=yes insu@100.92.1.128 'echo JETSON_OK'
ssh -o BatchMode=yes insu@100.96.160.1 'echo SERVER_OK'
```

GitHub remotes used from the laptop:

```bash
# fetch/pull over SSH (private key) — user "git" is fixed by GitHub
git remote add origin_ssh git@github.com:Insu-Health-Design-PR/SCANU.git

# or the HTTPS origin currently configured:
git remote -v   # origin https://github.com/Insu-Health-Design-PR/SCANU.git
```

## Architecture overview

Two independent AWR1843BOOST + USB-camera pairs face each other across a shared
space. Each node runs its own capture and per-radar perception locally using
the existing `lab.mmwave77_usb` pipeline. The Server then does the centralized
cross-node comparison and renders the evidence video.

```
            Jetson (A)                       Server (B)
   ┌──────────────────────┐        ┌──────────────────────────┐
   │ AWR1843 + camera A   │        │ AWR1843 + camera B       │
   │ runner  cube         │        │ runner  cube             │
   │ background perception│        │ background perception    │
   └─────────┬────────────┘        └────────────┬─────────────┘
             │  rsync session A                 │
             └──────────────▶  Server ──────────┘
                     ┌──────────────────────────┴──────────┐
                     │ point_cloud_fusion.py (3D fusion)   │
                     │ dual_fusion_video.py (evidence mp4) │
                     │ fusion.py (window comparison)       │
                     └──────────────────────────────────────┘
```

Layers and responsibilities:

- **Capture** (`lab.mmwave77_usb.runner`) — raw TI-TLV frames from each
  AWR1843, plus `camera_capture` recording `camera.mp4` /
  `camera_frames.jsonl` on each node.
- **Per-node processing** (`cube`, `background`, `perception`) — per-radar
  `frames.jsonl` + `perception.jsonl`, including a per-window
  `screening_state` and a `screening_state`/`anomalies[]` timeline that the
  fusion video reads.
- **Clock alignment** (`scripts/measure_clock_offset.sh`) — measures
  (Server − Jetson) offset; never assumed zero.  Fusion pairs the nearest
  unused windows by corrected timestamp, rather than treating local window
  counters as a shared clock.
- **`fusion.py`** — loads both sessions, aligns windows by wall-clock time,
  writes `fusion_report.json` + per-window match table. Descriptive
  comparison only.
- **`point_cloud_fusion.py`** — the current 3D fusion path. Builds a
  combined point cloud in sensor A's frame, applies the A/B offset and
  `transform_b_to_a`, removes clutter, clusters voxels
  (`cluster_points`, 0.18 m voxel), and annotates per-window
  `screening_state` and `anomalies[]` (`anomaly_centers`). Output is
  `fusion_report.json`.
- **`dual_fusion_video.py`** — renders the evidence mp4: camera A + camera B
  panels, fused 3D cloud, top/side/elevation-azimuth panels, per-sensor and
  fused timelines, and yellow/red diamond markers (`_plot_anomalies`) for
  body-associated reflective anomalies. Camera B is optional (device busy
  shows a placeholder panel).
- **`unified_fusion_video.py`** — renders companion exports from the same
  matched windows, including a three-panel 4K view containing only camera A,
  one geometrically fused A+B point cloud, and camera B. Camera lookup uses
  recorded host-monotonic timestamps; this is software synchronization, not a
  hardware trigger.
- **`remux_cas_cameras.py` / `remux_cameras.py`** — rebuild camera MP4s on
  their recorded `host_monotonic_ns` timestamps.  This prevents an OpenCV
  requested-FPS mismatch from accumulating visual drift against the radar.
- **`scripts/run_dual_awr1843_stereo_test.sh`** — end-to-end orchestrator:
  preflight → clock offset → parallel empty-room calibration → parallel
  participant capture → transfer both A sessions to Server → timestamp remux
  of both cameras → centralized fusion/render → download of diagnostic video,
  fused evidence video, separate host-time camera A/B videos, a two-camera
  video, a point-cloud-only video, the synchronized camera/fused-radar
  triptych, and `camera_radar_timebase.json` to the laptop.

## Running end-to-end (passwordless SSH, from the laptop)

The orchestrator uses SSH from the laptop to both nodes, so it runs without
a password once keys are installed:

```bash
cd ~/Desktop/INSU/SCANU
bash scripts/run_dual_awr1843_stereo_test.sh 60     # 60 s capture
```

Manual steps for the same flow, with explicit users:

```bash
# 1. Preflight
ssh insu@100.92.1.128  'cd /home/insu/Desktop/SCANU-dev_adrian && .venv/bin/python3 -m lab.mmwave77_usb.runner detect-awr1843'
ssh insu@100.96.160.1  'cd /home/insu/Desktop/scanu-stereo-node/SCANU && .venv/bin/python3 -m lab.mmwave77_usb.runner detect-awr1843'

# 2. Measure clock offset (Server − Jetson)
bash scripts/measure_clock_offset.sh

# 3. Parallel empty-room calibration (20 s) on both nodes, then participant capture (60 s)
#    — see run_dual_awr1843_stereo_test.sh for the full parallel commands.
```

## Fusion processing on the Server

```bash
ssh insu@100.96.160.1 '
  export PYTHONPATH=/home/insu/Desktop/scanu-stereo-node/SCANU/software:/home/insu/Desktop/scanu-stereo-node/SCANU
  cd /home/insu/Desktop/scanu-stereo-node/SCANU
  .venv/bin/python3 -m lab.dual_mmwave77_stereo.point_cloud_fusion \
    --session-a /path/to/session_A \
    --session-b /path/to/session_B \
    --calibration-a /path/to/cal_A \
    --calibration-b /path/to/cal_B \
    --distance-m 2.0 \
    --clock-offset-b-minus-a-s 0.0 \
    --output /path/to/fusion_report.json

  .venv/bin/python3 -m lab.dual_mmwave77_stereo.dual_fusion_video \
    --session-a /path/to/session_A \
    --session-b /path/to/session_B \
    --distance-m 2.0 \
    --camera-a /path/to/session_A/camera.mp4 \
    --camera-a-frames /path/to/session_A/camera_frames.jsonl \
    --camera-b /path/to/session_B/camera.mp4 \
    --camera-b-frames /path/to/session_B/camera_frames.jsonl \
    --output /path/to/dual_fusion_evidence.mp4 \
    --fps 5 --overwrite
'
```

If camera B is busy, omit `--camera-b` / `--camera-b-frames`; the renderer
shows a "camera B unavailable (device busy)" placeholder panel.

## Downloading results to the laptop

```bash
mkdir -p "$HOME/Desktop/77ghz mmwave video"
scp insu@100.96.160.1:/home/insu/Desktop/scanu-stereo-node/captures/<run>/dual_fusion_evidence.mp4 \
    "$HOME/Desktop/77ghz mmwave video/dual_fusion_evidence_<run>.mp4"
scp insu@100.96.160.1:/home/insu/Desktop/scanu-stereo-node/captures/<run>/fusion_report.json \
    "$HOME/Desktop/77ghz mmwave video/fusion_report_<run>.json"
```

## Testing

Run the module tests on the laptop or on the Server:

```bash
export PYTHONPATH="$PWD:$PWD/software"
python3 -m software.lab.dual_mmwave77_stereo.testing.test_fusion
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/software" \
  python3 -m pytest software/lab/dual_mmwave77_stereo/testing/ -q
```

On the Server (venv): replace `python3` with
`/home/insu/Desktop/scanu-stereo-node/SCANU/.venv/bin/python3`.

## Sync / deploy checklist

After committing on the laptop, propagate the files to all hosts. The laptop
is the source of truth:

```bash
git push origin dev_adrian        # remote

# Server
scp software/lab/dual_mmwave77_stereo/point_cloud_fusion.py \
    insu@100.96.160.1:/home/insu/Desktop/scanu-stereo-node/SCANU/software/lab/dual_mmwave77_stereo/
scp software/lab/dual_mmwave77_stereo/dual_fusion_video.py \
    insu@100.96.160.1:/home/insu/Desktop/scanu-stereo-node/SCANU/software/lab/dual_mmwave77_stereo/

# Jetson
scp software/lab/dual_mmwave77_stereo/point_cloud_fusion.py \
    insu@100.92.1.128:/home/insu/Desktop/SCANU-dev_adrian/software/lab/dual_mmwave77_stereo/
scp software/lab/dual_mmwave77_stereo/dual_fusion_video.py \
    insu@100.92.1.128:/home/insu/Desktop/SCANU-dev_adrian/software/lab/dual_mmwave77_stereo/
```

Verify checksums on each host after copying:

```bash
ssh insu@100.96.160.1 "shasum -a 256 /home/insu/Desktop/scanu-stereo-node/SCANU/software/lab/dual_mmwave77_stereo/point_cloud_fusion.py"
ssh insu@100.92.1.128 "shasum -a 256 /home/insu/Desktop/SCANU-dev_adrian/software/lab/dual_mmwave77_stereo/point_cloud_fusion.py"
```

## Limitations (read before interpreting results)

- Fusion windows are aligned by corrected host timestamps, not by geometric
  triangulation or a hardware trigger; the two radars/cameras can still differ
  by transport and exposure latency.
- No controlled ground-truth labels; agreement/union counts are descriptive,
  not accuracy/precision/recall.
- `screening_state` (`background`, `person`, `suspicious_metal`,
  `insufficient_signal`) and `anomalies[]` come from the experimental
  `lab.mmwave77_usb.perception` pipeline (post-CFAR points, not raw ADC) and
  carry no material/weapon classification.
- Two facing FMCW radars can interfere; run the empty-room interference check
  in `README.md` before trusting a "two radars detect better than one"
  conclusion.
