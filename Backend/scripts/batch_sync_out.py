#!/usr/bin/env python3
"""Run sync_runner for every Camera A/B pair under demo-video → sync_out/<clip_id>/.

  python scripts/batch_sync_out.py
  python scripts/batch_sync_out.py --only 150731 152636
  python scripts/batch_sync_out.py --force   # re-run even if outputs exist
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNC_RUNNER = ROOT / "sync_runner.py"
_PAIR_RE = re.compile(r"dual_awr1843_\d+_(\d+)_camera_A_4k30\.mp4$")


def discover_pairs(demo_root: Path) -> dict[str, tuple[Path, Path]]:
    found: dict[str, tuple[Path, Path]] = {}
    for a_path in demo_root.rglob("*_camera_A_4k30.mp4"):
        m = _PAIR_RE.match(a_path.name)
        if not m:
            continue
        clip_id = m.group(1)
        b_path = a_path.with_name(a_path.name.replace("camera_A", "camera_B"))
        if b_path.is_file():
            found[clip_id] = (a_path.resolve(), b_path.resolve())
    return dict(sorted(found.items()))


def outputs_ready(out_dir: Path) -> bool:
    return (
        (out_dir / "camera_A_inferred.mp4").is_file()
        and (out_dir / "camera_B_inferred.mp4").is_file()
    )


def run_one(clip_id: str, front: Path, back: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SYNC_RUNNER),
        "--front",
        str(front),
        "--back",
        str(back),
        "--front-out",
        str(out_dir / "camera_A_inferred.mp4"),
        "--back-out",
        str(out_dir / "camera_B_inferred.mp4"),
        "--baseline-m",
        "5",
        "--infer-stride",
        "1",
        "--summary-json",
        str(out_dir / "sync_summary.json"),
    ]
    print(f"\n===== {clip_id} =====", flush=True)
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
    return int(proc.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--demo-root",
        type=Path,
        default=Path.home() / "Desktop" / "demo-video",
    )
    p.add_argument(
        "--sync-out",
        type=Path,
        default=None,
        help="Default: <demo-root>/sync_out",
    )
    p.add_argument("--only", nargs="*", help="Clip ids to run (default: all discovered)")
    p.add_argument("--force", action="store_true", help="Re-run even when outputs exist")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    demo_root = args.demo_root.expanduser().resolve()
    sync_out = (args.sync_out or demo_root / "sync_out").expanduser().resolve()
    pairs = discover_pairs(demo_root)
    if not pairs:
        raise SystemExit(f"No camera pairs under {demo_root}")

    selected = pairs
    if args.only:
        missing = [x for x in args.only if x not in pairs]
        if missing:
            raise SystemExit(f"Unknown clip id(s): {', '.join(missing)}")
        selected = {k: pairs[k] for k in args.only}

    todo: list[tuple[str, Path, Path, Path]] = []
    for clip_id, (front, back) in selected.items():
        out_dir = sync_out / clip_id
        if not args.force and outputs_ready(out_dir):
            print(f"skip {clip_id} (outputs exist)", flush=True)
            continue
        todo.append((clip_id, front, back, out_dir))

    print(f"Demo root : {demo_root}")
    print(f"Sync out  : {sync_out}")
    print(f"Pairs     : {len(selected)} total, {len(todo)} to run", flush=True)

    if args.dry_run:
        for clip_id, front, back, out_dir in todo:
            print(f"would run {clip_id}: {front.name}")
        return 0

    failed: list[str] = []
    for clip_id, front, back, out_dir in todo:
        rc = run_one(clip_id, front, back, out_dir)
        if rc != 0:
            failed.append(clip_id)

    if failed:
        print(f"\nFailed: {', '.join(failed)}", flush=True)
        return 1
    print("\nAll batch runs finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
