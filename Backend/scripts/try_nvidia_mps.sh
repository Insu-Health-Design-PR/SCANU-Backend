#!/usr/bin/env bash
# Share one GPU between Front + Back infer via NVIDIA MPS (not two physical GPUs).
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if pgrep -x nvidia-cuda-mps-server >/dev/null 2>&1; then
  echo "MPS server already running"
  exit 0
fi
# Exclusive compute mode needs no CUDA clients. Callers must stop infer first.
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS || true
mkdir -p /tmp/nvidia-mps /tmp/nvidia-log
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
echo "MPS started"
