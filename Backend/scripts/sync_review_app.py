#!/usr/bin/env python3
"""Local web UI to review sync_runner outputs side-by-side (Camera A / Camera B).

  python scripts/sync_review_app.py
  # → http://127.0.0.1:8099/

Env:
  SYNC_REVIEW_DEMO  default: ~/Desktop/demo-video
  SYNC_REVIEW_OUT   default: ~/Desktop/demo-video/sync_out
  SYNC_REVIEW_PORT  default: 8099
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from manual_annotations import empty_document, load_manual_annotations, save_manual_annotations  # noqa: E402
DEMO_ROOT = Path(os.environ.get("SYNC_REVIEW_DEMO", Path.home() / "Desktop" / "demo-video")).expanduser()
SYNC_OUT = Path(os.environ.get("SYNC_REVIEW_OUT", DEMO_ROOT / "sync_out")).expanduser()
_EXTRA_DEMO_ROOT = Path.home() / "Desktop" / "final_demo-video"
PORT = int(os.environ.get("SYNC_REVIEW_PORT", "8099"))

_FOLDER_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_NUMERIC_RUN_RE = re.compile(r"^\d+$")
_PAIR_RE = re.compile(r"dual_awr1843_\d+_(\d+)_camera_A_4k30\.mp4$")

app = FastAPI(title="Sync Out Review", version="1.0")


def _demo_roots() -> list[Path]:
    roots = [DEMO_ROOT]
    extra = Path(os.environ.get("SYNC_REVIEW_DEMO_EXTRA", _EXTRA_DEMO_ROOT)).expanduser()
    if extra.is_dir() and extra.resolve() not in {r.resolve() for r in roots}:
        roots.append(extra)
    return roots


def _discover_source_pairs() -> list[dict]:
    found: dict[str, dict] = {}
    for demo_root in _demo_roots():
        if not demo_root.is_dir():
            continue
        for a_path in demo_root.rglob("*_camera_A_4k30.mp4"):
            m = _PAIR_RE.match(a_path.name)
            if not m:
                continue
            clip_id = m.group(1)
            b_path = a_path.with_name(a_path.name.replace("camera_A", "camera_B"))
            if not b_path.is_file():
                continue
            found[clip_id] = {
                "id": clip_id,
                "front": str(a_path.resolve()),
                "back": str(b_path.resolve()),
                "source_dir": str(a_path.parent.resolve()),
            }
    return [found[k] for k in sorted(found.keys())]


def _run_folder(run_id: str) -> Path:
    if not _NUMERIC_RUN_RE.match(run_id):
        raise HTTPException(400, "invalid run id (numeric clip folders only)")
    return SYNC_OUT / run_id


def _web_video_path(src: Path) -> Path:
    """Browser-friendly H.264 proxy (OpenCV mp4v/mpeg4 often won't play in Chrome)."""
    cached = src.with_name(f"{src.stem}_web.mp4")
    if cached.is_file():
        try:
            if cached.stat().st_mtime >= src.stat().st_mtime:
                return cached
        except OSError:
            pass
    tmp = cached.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-vf",
        "scale='min(1280,iw)':-2",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(500, f"H.264 transcode failed for {src.name}: {exc}") from exc
    tmp.replace(cached)
    return cached


def _video_path(run_id: str, side: str, *, web: bool = True) -> Path:
    if side not in ("a", "b"):
        raise HTTPException(400, "side must be a or b")
    folder = _run_folder(run_id)
    name = "camera_A_inferred.mp4" if side == "a" else "camera_B_inferred.mp4"
    path = folder / name
    if not path.is_file():
        raise HTTPException(404, f"{name} not found in {run_id}")
    src = path.resolve()
    if web:
        return _web_video_path(src)
    return src


def _list_runs() -> list[dict]:
    runs: list[dict] = []
    if not SYNC_OUT.is_dir():
        return runs
    for entry in sorted(SYNC_OUT.iterdir()):
        if not entry.is_dir():
            continue
        run_id = entry.name
        if not _NUMERIC_RUN_RE.match(run_id):
            continue
        a = entry / "camera_A_inferred.mp4"
        b = entry / "camera_B_inferred.mp4"
        summary_path = entry / "sync_summary.json"
        summary: dict | None = None
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = None
        runs.append(
            {
                "id": run_id,
                "has_a": a.is_file(),
                "has_b": b.is_file(),
                "ready": a.is_file() and b.is_file(),
                "mtime": int(max(a.stat().st_mtime if a.is_file() else 0, b.stat().st_mtime if b.is_file() else 0)),
                "frames": summary.get("frames_written") if summary else None,
                "elapsed_s": summary.get("elapsed_s") if summary else None,
                "global_persons": len(summary.get("persons") or []) if summary else None,
                "armed_count": sum(
                    1 for p in (summary.get("persons") or []) if p.get("weapon_detected")
                )
                if summary
                else None,
            }
        )
    return runs


def _manual_path(run_id: str) -> Path:
    return _run_folder(run_id) / "manual_annotations.json"


def _source_pair(run_id: str) -> tuple[Path, Path]:
    pairs = {p["id"]: (Path(p["front"]), Path(p["back"])) for p in _discover_source_pairs()}
    if run_id not in pairs:
        raise HTTPException(404, f"source pair not found for {run_id}")
    return pairs[run_id]


def _video_size(path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        w, h = out.split(",")
        return int(w), int(h)
    except (subprocess.CalledProcessError, OSError, ValueError):
        return 0, 0


class ManualPayload(BaseModel):
    data: dict


@app.get("/api/runs/{run_id}/manual-annotations")
def get_manual_annotations(run_id: str):
    _run_folder(run_id)
    path = _manual_path(run_id)
    if path.is_file():
        doc = load_manual_annotations(path)
    else:
        doc = empty_document(clip_id=run_id)
    front, back = _source_pair(run_id)
    cf, cb = doc.get("camera_front", "camera_1"), doc.get("camera_back", "camera_2")
    doc.setdefault("ref", {})
    fw, fh = _video_size(front)
    bw, bh = _video_size(back)
    if fw > 0 and fh > 0:
        doc["ref"].setdefault(cf, {})
        if int(doc["ref"].get(cf, {}).get("width") or 0) <= 0:
            doc["ref"][cf] = {"width": fw, "height": fh}
    if bw > 0 and bh > 0:
        doc["ref"].setdefault(cb, {})
        if int(doc["ref"].get(cb, {}).get("width") or 0) <= 0:
            doc["ref"][cb] = {"width": bw, "height": bh}
    return JSONResponse(doc)


@app.put("/api/runs/{run_id}/manual-annotations")
def put_manual_annotations(run_id: str, payload: ManualPayload):
    folder = _run_folder(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    data = payload.data
    data["clip_id"] = run_id
    save_manual_annotations(_manual_path(run_id), data)
    return JSONResponse({"ok": True, "path": str(_manual_path(run_id))})


@app.post("/api/runs/{run_id}/render-manual")
def render_manual(run_id: str):
    folder = _run_folder(run_id)
    manual = _manual_path(run_id)
    if not manual.is_file():
        raise HTTPException(400, "Save manual annotations first")
    front, back = _source_pair(run_id)
    sync_runner = ROOT / "sync_runner.py"
    cmd = [
        sys.executable,
        str(sync_runner),
        "--front",
        str(front),
        "--back",
        str(back),
        "--front-out",
        str(folder / "camera_A_inferred.mp4"),
        "--back-out",
        str(folder / "camera_B_inferred.mp4"),
        "--summary-json",
        str(folder / "sync_summary.json"),
        "--manual-annotations",
        str(manual),
        "--manual-only",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(500, proc.stderr[-4000:] or proc.stdout[-4000:] or "render failed")
    for side in ("a", "b"):
        src = folder / f"camera_{'A' if side == 'a' else 'B'}_inferred.mp4"
        web = src.with_name(f"{src.stem}_web.mp4")
        web.unlink(missing_ok=True)
        try:
            _web_video_path(src)
        except HTTPException:
            pass
    return JSONResponse({"ok": True, "log_tail": (proc.stdout or "")[-2000:]})


@app.get("/api/runs")
def api_runs():
    return JSONResponse({"sync_out": str(SYNC_OUT), "runs": _list_runs()})


@app.get("/api/source-pairs")
def api_source_pairs():
    return JSONResponse({"demo_root": str(DEMO_ROOT), "pairs": _discover_source_pairs()})


@app.get("/api/runs/{run_id}/summary")
def api_summary(run_id: str):
    folder = _run_folder(run_id)
    summary_path = folder / "sync_summary.json"
    if not summary_path.is_file():
        raise HTTPException(404, "sync_summary.json missing")
    return JSONResponse(json.loads(summary_path.read_text(encoding="utf-8")))


@app.get("/media/{run_id}/{side}")
def media_video(run_id: str, side: str):
    path = _video_path(run_id, side, web=True)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
        headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache, must-revalidate"},
    )


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).with_name("sync_review.html")
    if html_path.is_file():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    raise HTTPException(500, "sync_review.html missing")


def main() -> None:
    print(f"Demo root : {DEMO_ROOT}")
    print(f"Sync out  : {SYNC_OUT}")
    print(f"Review UI : http://127.0.0.1:{PORT}/")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
