"""End-to-end mmWave + camera fusion probe for the Layer 8 control center."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from layer8_ui.webcam_device import _probe_device
from runtime import sensor_runner
from services.mmwave_metrics_service import live_metrics_snapshot
from services.mmwave_root import preflight


def _check(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def probe_mmwave_stack(
    *,
    settings: dict[str, Any],
    layer8_dir: Path,
    software_root: Path,
    backend_root: Path | None = None,
) -> dict[str, Any]:
    """Non-destructive checklist: radars, metrics, fusion, cameras, runners."""
    mm = settings.get("mmwave") if isinstance(settings.get("mmwave"), dict) else {}
    fusion = settings.get("mmwave_fusion") if isinstance(settings.get("mmwave_fusion"), dict) else {}
    pipeline = str(mm.get("pipeline") or "dual_replay").strip()

    pf = preflight(settings=settings, backend_root=backend_root)
    pairs_n = len(pf.get("pairs") or [])
    pair_a = pf.get("pair_a")
    pair_b = pf.get("pair_b")

    front_sess = str(mm.get("front_session") or mm.get("session") or "").strip()
    back_sess = str(mm.get("back_session") or "").strip()
    replay_ready = False
    replay_detail = ""
    if front_sess or back_sess:
        parts = []
        for label, sess in (("front", front_sess), ("back", back_sess)):
            if not sess:
                continue
            sp = Path(sess).expanduser()
            pp = sp / "perception.jsonl"
            fp = sp / "frames.jsonl"
            ok = pp.is_file() and fp.is_file()
            parts.append(f"{label}:{'ok' if ok else 'missing perception/frames'} ({sp.name})")
            replay_ready = replay_ready or ok
        replay_detail = "; ".join(parts) if parts else "no sessions configured"
    else:
        replay_detail = "set front_session / back_session for replay without USB radars"

    metrics = live_metrics_snapshot(settings, layer8_dir=layer8_dir, software_root=software_root)
    metrics_ok = bool(metrics.get("ok"))
    age = metrics.get("age_s")
    metrics_detail = (
        f"age={age:.1f}s torso={metrics.get('mmwave_torso_score', '—')} path={metrics.get('path', '')}"
        if metrics_ok and age is not None
        else (f"no file at {metrics.get('path') or '?'}" if not metrics_ok else "metrics present")
    )

    fusion_on = bool(int(fusion.get("enable", 0) or 0))
    fusion_detail = (
        f"webcam→{fusion.get('webcam_side', 'front')} multi→{fusion.get('multi_camera_side', 'back')} "
        f"metrics={fusion.get('metrics_path', '')}"
        if fusion_on
        else "enable fusion in mmWave tab to draw radar dots on camera infer"
    )

    cam_checks: list[dict[str, Any]] = []
    for sensor, block_key, label in (
        ("webcam", "webcam", "Front Cam"),
        ("multi_camera", "multi_camera", "Back Cam"),
    ):
        block = settings.get(block_key) if isinstance(settings.get(block_key), dict) else {}
        mode = str(block.get("source_mode") or "local").strip().lower()
        if mode in ("jetson", "ip", "network", "rtsp"):
            url = str(block.get("jetson_stream_url") or block.get("jetson_ip") or "").strip()
            cam_checks.append(_check(label, bool(url), f"network {url or '(no URL)'}"))
            continue
        dev = int(block.get("webcam_device", 0))
        w = int(block.get("webcam_width", 1280))
        h = int(block.get("webcam_height", 720))
        fps = float(block.get("fps", 30) or 30)
        opened = _probe_device(dev, min(1280, w), min(720, h), min(30.0, fps))
        cam_checks.append(
            _check(label, opened, f"/dev/video{dev} {'open OK' if opened else 'cannot open (busy or no HDMI)'}")
        )

    runner_checks: list[dict[str, Any]] = []
    for sensor, label in (
        ("webcam", "Front Cam infer"),
        ("multi_camera", "Back Cam infer"),
        ("mmwave", "mmWave publisher"),
    ):
        st = sensor_runner.status(sensor, layer8_dir)  # type: ignore[arg-type]
        running = bool(st.get("running"))
        pid = st.get("pid") or 0
        runner_checks.append(
            _check(label, running, f"pid={pid}" if running else "not running — click Run")
        )

    radar_live_ok = pairs_n >= 1 and pipeline in ("dual_live", "lab_live")
    radar_replay_ok = replay_ready and pipeline in ("dual_replay", "lab_replay")
    radar_ok = radar_live_ok or radar_replay_ok

    checks = [
        _check("MMWAVE_ROOT", bool(pf.get("ok")), str(pf.get("mmwave_root") or "")),
        _check(
            "Radar USB pairs",
            pairs_n >= 1,
            f"{pairs_n} pair(s)"
            + (f" — A {pair_a.get('cli_port')}/{pair_a.get('data_port')}" if pair_a else "")
            + (f" B {pair_b.get('cli_port')}/{pair_b.get('data_port')}" if pair_b else "")
            if pairs_n
            else "plug in AWR1843 USB (or use dual_replay sessions)",
        ),
        _check("Replay sessions", replay_ready, replay_detail),
        _check("Live metrics JSON", metrics_ok, metrics_detail),
        _check("Fusion overlay enabled", fusion_on, fusion_detail),
        *cam_checks,
        *runner_checks,
    ]

    critical = [
        c["ok"]
        for c in checks
        if c["name"] in ("MMWAVE_ROOT", "Fusion overlay enabled")
    ]
    operational = metrics_ok and fusion_on and any(c["ok"] for c in runner_checks[:2])
    ok = all(critical) and (metrics_ok or replay_ready or pairs_n >= 1)

    summary_parts = []
    if not pf.get("ok"):
        summary_parts.append("MMWAVE_ROOT invalid")
    if not fusion_on:
        summary_parts.append("fusion off")
    if not metrics_ok:
        summary_parts.append("no live metrics — Run mmWave")
    if not any(c["ok"] for c in runner_checks[:2]):
        summary_parts.append("cameras not running")
    if not summary_parts:
        summary_parts.append("stack ready — stand in corridor to see person boxes + mmWave dots")

    return {
        "ok": ok,
        "operational": operational,
        "summary": "; ".join(summary_parts),
        "pipeline": pipeline,
        "preflight": pf,
        "metrics": metrics,
        "checks": checks,
        "ts": time.time(),
    }
