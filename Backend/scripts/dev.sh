#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  make install
fi

# shellcheck disable=SC1091
source .venv/bin/activate

cp -n .env.example .env 2>/dev/null || true

make api &
API_PID=$!

make infer &
INFER_PID=$!

trap 'kill $API_PID $INFER_PID 2>/dev/null || true' EXIT
wait
