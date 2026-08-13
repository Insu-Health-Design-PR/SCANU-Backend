#!/usr/bin/env bash
# run_dual_awr1843_stereo_test.sh — two facing AWR1843 + USB camera test.
#
# Nodes: Jetson (radar A + camera A) and Server (radar B + camera B).
# Each node captures locally with the existing, unmodified lab.mmwave77_usb
# pipeline. Processing is centralized on the Server. Clock offset between
# hosts is measured and recorded, never assumed zero.

set -Eeuo pipefail

jetson_target="${JETSON_TARGET:-insu@100.92.1.128}"
server_target="${SERVER_TARGET:-insu@100.96.160.1}"
duration_s="${1:-60}"
camera_device_jetson="${CAMERA_DEVICE_JETSON:-auto}"
camera_device_server="${CAMERA_DEVICE_SERVER:-auto}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
config_rel="software/lab/mmwave77_usb/configs/awr1843boost_sdk_3_4_profile_3d.cfg"
run_tag="$(date +%Y%m%d_%H%M%S)"
run_name="dual_awr1843_${run_tag}"

jetson_repo="${JETSON_REPO:-/home/insu/Desktop/SCANU-dev_adrian}"
server_repo="${SERVER_REPO:-/home/insu/Desktop/scanu-stereo-node/SCANU}"
jetson_run="/mnt/scanu-data/validation/${run_name}_A"
server_run="/home/insu/Desktop/scanu-stereo-node/captures/${run_name}_B"

if ! [[ "$duration_s" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [capture_seconds]" >&2
  exit 2
fi

echo "=== run_tag: $run_tag ==="

# 1. Preflight both hosts.
echo "--- Jetson preflight ---"
ssh "$jetson_target" "
  set -eu
  repo='$jetson_repo'
  python_bin=\"\$repo/.venv/bin/python3\"
  test -x \"\$python_bin\"
  test -d \"\$repo/software/lab/mmwave77_usb\"
  findmnt /mnt/scanu-data >/dev/null
  \"\$python_bin\" -m lab.mmwave77_usb.runner detect-awr1843
  \"\$python_bin\" -m lab.mmwave77_usb.camera_capture probe --device '$camera_device_jetson' 2>/dev/null | tail -n 3
  echo JETSON_OK
"
echo "--- Server preflight ---"
ssh "$server_target" "
  set -eu
  repo='$server_repo'
  python_bin=\"\$repo/.venv/bin/python3\"
  test -x \"\$python_bin\"
  test -d \"\$repo/software/lab/mmwave77_usb\"
  command -v ffmpeg >/dev/null
  \"\$python_bin\" -m lab.mmwave77_usb.runner detect-awr1843
  echo SERVER_OK
"

# 2. Measure clock offset (Server - Jetson).
clock_offset=$("$script_dir/measure_clock_offset.sh" | tail -n 1)
clock_offset=$(echo "$clock_offset" | grep -oE '^-?[0-9.]+' || echo "0")
echo "=== clock offset (Server - Jetson): ${clock_offset} s ==="

# 3. Parallel empty-room calibration (both radars simultaneously).
echo "Step 1/3 — empty-room calibration (20 s, both radars). Keep the space empty."
read -r -p "Press ENTER when the space is empty to start calibration... "
run_a="$jetson_run"
run_b="$server_run"

ssh "$jetson_target" "
  set -eu
  repo='$jetson_repo'
  run_root='$run_a'
  config='$config_rel'
  python_bin=\"\$repo/.venv/bin/python3\"
  mkdir -p \"\$run_root\"
  cd \"\$repo\"
  export PYTHONPATH=\"\$repo/software:\$repo\"
  \"\$python_bin\" -m lab.mmwave77_usb.runner capture --auto-awr1843 --config \"\$config\" --protocol ti-tlv --duration-s 20 --output-root \"\$run_root\"
  empty_session=\$(find \"\$run_root\" -mindepth 1 -maxdepth 1 -type d -name 'capture_*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  printf '%s\n' \"\$empty_session\" > \"\$run_root/empty_session.txt\"
  \"\$python_bin\" -m lab.mmwave77_usb.cube --session \"\$empty_session\"
  \"\$python_bin\" -m lab.mmwave77_usb.background --session \"\$empty_session\" --condition empty_room
  echo JETSON_CALIB_OK=\"\$empty_session\"
" 2>&1 | grep -v -E "WARNING|password" &
pid_a=$!

ssh "$server_target" "
  set -eu
  repo='$server_repo'
  run_root='$run_b'
  config='$config_rel'
  python_bin=\"\$repo/.venv/bin/python3\"
  mkdir -p \"\$run_root\"
  cd \"\$repo\"
  export PYTHONPATH=\"\$repo/software:\$repo\"
  \"\$python_bin\" -m lab.mmwave77_usb.runner capture --auto-awr1843 --config \"\$config\" --protocol ti-tlv --duration-s 20 --output-root \"\$run_root\"
  empty_session=\$(find \"\$run_root\" -mindepth 1 -maxdepth 1 -type d -name 'capture_*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  printf '%s\n' \"\$empty_session\" > \"\$run_root/empty_session.txt\"
  \"\$python_bin\" -m lab.mmwave77_usb.cube --session \"\$empty_session\"
  \"\$python_bin\" -m lab.mmwave77_usb.background --session \"\$empty_session\" --condition empty_room
  echo SERVER_CALIB_OK=\"\$empty_session\"
" 2>&1 | grep -v -E "WARNING|password" &
pid_b=$!

wait "$pid_a" "$pid_b"
echo "Calibration complete. Do not move the radars or large objects."

# 4. Participant placement pause.
read -r -p "Place the participant between the two sensors (1.5-2.5 m from each). Press ENTER to record ${duration_s} s... "

# 5. Parallel capture with camera on both nodes.
ssh "$jetson_target" "
  set -eu
  repo='$jetson_repo'
  run_root='$run_a'
  config='$config_rel'
  camera_device='$camera_device_jetson'
  duration='$duration_s'
  camera_duration=\$((duration + 10))
  python_bin=\"\$repo/.venv/bin/python3\"
  empty_session=\$(cat \"\$run_root/empty_session.txt\")
  cd \"\$repo\"
  export PYTHONPATH=\"\$repo/software:\$repo\"
  camera_dir=\"\$run_root/camera_pending\"
  \"\$python_bin\" -m lab.mmwave77_usb.camera_capture record --device \"\$camera_device\" --duration-s \"\$camera_duration\" --output-dir \"\$camera_dir\" > \"\$run_root/camera.log\" 2>&1 &
  camera_pid=\$!
  \"\$python_bin\" -m lab.mmwave77_usb.runner capture --auto-awr1843 --config \"\$config\" --protocol ti-tlv --duration-s \"\$duration\" --output-root \"\$run_root\"
  wait \"\$camera_pid\"
  person_session=\$(find \"\$run_root\" -mindepth 1 -maxdepth 1 -type d -name 'capture_*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  mv \"\$camera_dir/camera.mp4\" \"\$person_session/camera.mp4\"
  mv \"\$camera_dir/camera_frames.jsonl\" \"\$person_session/camera_frames.jsonl\"
  mv \"\$camera_dir/camera_metadata.json\" \"\$person_session/camera_metadata.json\"
  rmdir \"\$camera_dir\"
  \"\$python_bin\" -m lab.mmwave77_usb.cube --session \"\$person_session\"
  \"\$python_bin\" -m lab.mmwave77_usb.perception --session \"\$person_session\" --calibration-session \"\$empty_session\"
  printf '%s\n' \"\$person_session\" > \"\$run_root/person_session.txt\"
  echo JETSON_PERSON_SESSION=\"\$person_session\"
" 2>&1 | grep -v -E "WARNING|password" &
pid_a=$!

ssh "$server_target" "
  set -eu
  repo='$server_repo'
  run_root='$run_b'
  config='$config_rel'
  camera_device='$camera_device_server'
  duration='$duration_s'
  camera_duration=\$((duration + 10))
  python_bin=\"\$repo/.venv/bin/python3\"
  empty_session=\$(cat \"\$run_root/empty_session.txt\")
  cd \"\$repo\"
  export PYTHONPATH=\"\$repo/software:\$repo\"
  camera_dir=\"\$run_root/camera_pending\"
  \"\$python_bin\" -m lab.mmwave77_usb.camera_capture record --device \"\$camera_device\" --duration-s \"\$camera_duration\" --output-dir \"\$camera_dir\" > \"\$run_root/camera.log\" 2>&1 &
  camera_pid=\$!
  \"\$python_bin\" -m lab.mmwave77_usb.runner capture --auto-awr1843 --config \"\$config\" --protocol ti-tlv --duration-s \"\$duration\" --output-root \"\$run_root\"
  wait \"\$camera_pid\"
  person_session=\$(find \"\$run_root\" -mindepth 1 -maxdepth 1 -type d -name 'capture_*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  mv \"\$camera_dir/camera.mp4\" \"\$person_session/camera.mp4\"
  mv \"\$camera_dir/camera_frames.jsonl\" \"\$person_session/camera_frames.jsonl\"
  mv \"\$camera_dir/camera_metadata.json\" \"\$person_session/camera_metadata.json\"
  rmdir \"\$camera_dir\"
  \"\$python_bin\" -m lab.mmwave77_usb.cube --session \"\$person_session\"
  \"\$python_bin\" -m lab.mmwave77_usb.perception --session \"\$person_session\" --calibration-session \"\$empty_session\"
  printf '%s\n' \"\$person_session\" > \"\$run_root/person_session.txt\"
  echo SERVER_PERSON_SESSION=\"\$person_session\"
" 2>&1 | grep -v -E "WARNING|password" &
pid_b=$!

wait "$pid_a" "$pid_b"
echo "Capture + per-node perception complete on both nodes."

# 6. Transfer Jetson session to Server for centralized processing.
echo "Transferring Jetson session to Server (rsync)..."
jetson_person_session=$(ssh "$jetson_target" "cat '$run_a/person_session.txt'")
server_person_session=$(ssh "$server_target" "cat '$run_b/person_session.txt'")
jetson_session_local="$run_b/jetson_session"
mkdir -p "$jetson_session_local"
rsync -az --info=summary1 "$jetson_target:$jetson_person_session/" "$jetson_session_local/"

# 7. Centralized processing on Server: per-node videos + fusion + combined.
echo "Centralized processing on Server..."
ssh "$server_target" "
  set -eu
  repo='$server_repo'
  python_bin=\"\$repo/.venv/bin/python3\"
  run_root='$run_b'
  jetson_person='$jetson_person_session'
  cd \"\$repo\"
  export PYTHONPATH=\"\$repo/software:\$repo\"
  server_person='$server_person_session'
  jetson_local='$jetson_session_local'
  echo \"Server person session: \$server_person\"

  mmwave_video_a=\"\$run_root/mmwave_perception_A.mp4\"
  mmwave_video_b=\"\$run_root/mmwave_perception_B.mp4\"
  \"\$python_bin\" -m lab.mmwave77_usb.perception_video --perception \"\$jetson_local/perception.jsonl\" --output \"\$mmwave_video_a\" --fps 2
  \"\$python_bin\" -m lab.mmwave77_usb.perception_video --perception \"\$server_person/perception.jsonl\" --output \"\$mmwave_video_b\" --fps 2

  camera_offset_a=\$(\"\$python_bin\" -m lab.mmwave77_usb.camera_capture offset --camera-frames \"\$jetson_local/camera_frames.jsonl\" --radar-frames \"\$jetson_local/frames.jsonl\")
  camera_offset_b=\$(\"\$python_bin\" -m lab.mmwave77_usb.camera_capture offset --camera-frames \"\$server_person/camera_frames.jsonl\" --radar-frames \"\$server_person/frames.jsonl\")
  echo \"camera offsets A=\$camera_offset_a B=\$camera_offset_b\"

  video=\"\$run_root/combined_dual.mp4\"
  ffmpeg -y \
    -i \"\$mmwave_video_a\" -ss \"\$camera_offset_a\" -i \"\$jetson_local/camera.mp4\" \
    -i \"\$mmwave_video_b\" -ss \"\$camera_offset_b\" -i \"\$server_person/camera.mp4\" \
    -filter_complex \
      \"[0:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:color=0x050b14[ra];[1:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:color=black[ca];[2:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:color=0x050b14[rb];[3:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:color=black[cb];[ra][ca][rb][cb]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0\" \
    -map 0:a? -c:v libx264 -pix_fmt yuv420p -movflags +faststart -shortest \"\$video\"

  \"\$python_bin\" -m lab.dual_mmwave77_stereo.fusion \
    --session-a \"\$jetson_local\" \
    --session-b \"\$server_person\" \
    --clock-offset-b-minus-a-s $clock_offset \
    --window-tolerance-s 0.5 \
    --output \"\$run_root/fusion_report.json\"

  echo REMOTE_VIDEO=\"\$video\"
  echo FUSION_REPORT=\"\$run_root/fusion_report.json\"
  ls -la \"\$video\"
"

# 8. Download combined video + report to laptop.
mkdir -p "$HOME/Desktop/77ghz mmwave video"
scp "$server_target:$run_b/combined_dual.mp4" "$HOME/Desktop/77ghz mmwave video/${run_name}.mp4"
scp "$server_target:$run_b/fusion_report.json" "$HOME/Desktop/77ghz mmwave video/${run_name}_fusion_report.json"
echo "Done: $HOME/Desktop/77ghz mmwave video/${run_name}.mp4"
echo "Fusion: $HOME/Desktop/77ghz mmwave video/${run_name}_fusion_report.json"