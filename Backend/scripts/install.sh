#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

make install
cp -n .env.example .env 2>/dev/null || true

echo "Install complete. Run: make api"
