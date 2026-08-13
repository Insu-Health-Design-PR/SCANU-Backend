#!/usr/bin/env bash
# measure_clock_offset.sh — measure host B clock minus host A clock via parallel SSH
# Usage: measure_clock_offset.sh
# Reads host IPs from environment or defaults:
#   JETSON_TARGET="insu@100.92.1.128"
#   SERVER_TARGET="insu@100.96.160.1"

set -Eeuo pipefail

jetson_target="${JETSON_TARGET:-insu@100.92.1.128}"
server_target="${SERVER_TARGET:-insu@100.96.160.1}"

measure_offset() {
  local target="$1"
  ssh "$target" 'date -u +%s.%N' 2>/dev/null
}

echo "Measuring clock offset (host B - host A) over 10 parallel probes..."
offsets=()
for i in {1..10}; do
  (measure_offset "$jetson_target" > /tmp/jetson_time_$$_$i.txt) &
  (measure_offset "$server_target" > /tmp/server_time_$$_$i.txt) &
  wait
  j=$(cat "/tmp/jetson_time_$$_$i.txt" 2>/dev/null || echo "")
  s=$(cat "/tmp/server_time_$$_$i.txt" 2>/dev/null || echo "")
  if [[ -n "$j" && -n "$s" ]]; then
    offset=$(python3 -c "print(float('$s') - float('$j'))")
    offsets+=("$offset")
    echo "  probe $i: B - A = ${offset} s"
  fi
  rm -f "/tmp/jetson_time_$$_$i.txt" "/tmp/server_time_$$_$i.txt"
done

if [[ ${#offsets[@]} -eq 0 ]]; then
  echo "error: no valid probes" >&2
  exit 1
fi

# median
sorted_offsets=($(printf '%s\n' "${offsets[@]}" | sort -g))
n=${#sorted_offsets[@]}
if (( n % 2 == 0 )); then
  median=$(python3 -c "
import sys
vals = [float(x) for x in sys.argv[1:]]
vals.sort()
mid = len(vals) // 2
print((vals[mid-1] + vals[mid]) / 2)
" "${sorted_offsets[@]}")
else
  median=$(python3 -c "
import sys
vals = [float(x) for x in sys.argv[1:]]
vals.sort()
print(vals[len(vals)//2])
" "${sorted_offsets[@]}")
fi

echo
echo "Clock offset (Server - Jetson): ${median} s"
echo "Use this value as --clock-offset-b-minus-a-s in fusion.py"
echo "$median" > /tmp/clock_offset_median.txt