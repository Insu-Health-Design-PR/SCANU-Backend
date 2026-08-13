"""
Live + final clip analysis for ``infer_thermal_objects``.

Writes three artifacts from a shared base path (e.g. ``runs/foo_analysis``):
  - ``.log``  — append-only live lines while the video runs
  - ``.json`` — structured report (rewritten each flush + final)
  - ``.txt``  — human-readable summary at end
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Any


def _sec(frame: int, fps: float) -> float:
    if fps < 1e-6:
        return 0.0
    return float(frame) / float(fps)


def _ts_range(start_f: int, end_f: int, fps: float) -> str:
    a = _sec(start_f, fps)
    b = _sec(end_f, fps)
    return f"{a:.2f}s–{b:.2f}s (frames {start_f}–{end_f})"


@dataclass
class _Span:
    start_frame: int
    end_frame: int

    def extend(self, frame: int) -> None:
        self.end_frame = frame

    def to_dict(self, fps: float) -> dict[str, Any]:
        return {
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "frame_count": int(self.end_frame - self.start_frame + 1),
            "time_start_sec": round(_sec(self.start_frame, fps), 3),
            "time_end_sec": round(_sec(self.end_frame, fps), 3),
            "time_range": _ts_range(self.start_frame, self.end_frame, fps),
        }


@dataclass
class _EntityTrack:
    key: str
    category: str
    spans: list[_Span] = field(default_factory=list)
    frame_hits: int = 0
    peak_value: float = 0.0
    peak_frame: int = 0
    object_frames: int = 0
    weapon_frames: int = 0
    unsafe_frames: int = 0
    suspicious_frames: int = 0
    safe_frames: int = 0
    majority_kind: str | None = None
    last_frame: int = 0

    def note_frame(
        self,
        frame: int,
        *,
        value: float = 0.0,
        kind: str | None = None,
        bucket: str | None = None,
    ) -> None:
        self.frame_hits += 1
        self.last_frame = frame
        if value > self.peak_value:
            self.peak_value = float(value)
            self.peak_frame = frame
        if kind == "weapon":
            self.weapon_frames += 1
        elif kind == "object":
            self.object_frames += 1
        if bucket == "unsafe":
            self.unsafe_frames += 1
        elif bucket == "suspicious":
            self.safe_frames += 1
        elif bucket == "safe":
            self.safe_frames += 1
        if self.spans and self.spans[-1].end_frame == frame - 1:
            self.spans[-1].extend(frame)
        else:
            self.spans.append(_Span(start_frame=frame, end_frame=frame))

    def to_dict(self, fps: float) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.key,
            "category": self.category,
            "frames_detected": int(self.frame_hits),
            "appearance_spans": [s.to_dict(fps) for s in self.spans],
            "peak_value": round(float(self.peak_value), 4),
            "peak_frame": int(self.peak_frame),
            "peak_time_sec": round(_sec(self.peak_frame, fps), 3) if self.peak_frame > 0 else 0.0,
        }
        if self.category == "person":
            out["gun_conf_peak"] = out["peak_value"]
            out["unsafe_frames"] = int(self.unsafe_frames)
            out["suspicious_frames"] = int(self.suspicious_frames)
            out["safe_frames"] = int(self.safe_frames)
        else:
            out["majority_kind"] = self.majority_kind
            out["object_frames"] = int(self.object_frames)
            out["weapon_frames"] = int(self.weapon_frames)
        return out


class ClipAnalysisReporter:
    """Accumulate per-track stats and flush live logs + JSON during inference."""

    def __init__(
        self,
        base_path: Path,
        *,
        source: str,
        fps: float,
        unsafe_threshold: float,
        flush_every: int = 60,
    ) -> None:
        self.base_path = base_path.expanduser().resolve()
        self.source = str(source)
        self.fps = max(1.0, float(fps))
        self.unsafe_threshold = float(unsafe_threshold)
        self.flush_every = max(1, int(flush_every))
        self._started = time()
        self._persons: dict[str, _EntityTrack] = {}
        self._firearms: dict[str, _EntityTrack] = {}
        self._frames_processed = 0
        self._frames_with_person = 0
        self._frames_with_firearm = 0
        self._frames_weapon_class = 0
        self._clip_peak_gun_conf = 0.0
        self._clip_peak_frame = 0
        self._last_frame_snapshot: dict[str, Any] = {}

        stem = self.base_path
        if stem.suffix.lower() in (".json", ".txt", ".log", ".md"):
            stem = stem.with_suffix("")
        self._json_path = stem.with_suffix(".json")
        self._txt_path = stem.with_suffix(".txt")
        self._log_path = stem.with_suffix(".log")
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text(
            f"# analysis log — {self.source}\n"
            f"# started {datetime.now(timezone.utc).isoformat()}\n"
            f"# fps≈{self.fps:.3f}  unsafe_threshold={self.unsafe_threshold:.2f}\n\n",
            encoding="utf-8",
        )

    def _append_log(self, line: str) -> None:
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

    def on_frame(
        self,
        frame: int,
        *,
        persons: list[dict[str, Any]],
        firearms: list[dict[str, Any]],
        frame_max_gun_conf: float,
    ) -> None:
        self._frames_processed = int(frame)
        if persons:
            self._frames_with_person += 1
        if firearms:
            self._frames_with_firearm += 1
        if any(str(f.get("kind")) == "weapon" for f in firearms):
            self._frames_weapon_class += 1
        if float(frame_max_gun_conf) > self._clip_peak_gun_conf:
            self._clip_peak_gun_conf = float(frame_max_gun_conf)
            self._clip_peak_frame = int(frame)

        active_person: list[str] = []
        for p in persons:
            key = str(p["display"])
            bucket = str(p.get("bucket", "safe"))
            tr = self._persons.get(key)
            if tr is None:
                tr = _EntityTrack(key=key, category="person")
                self._persons[key] = tr
                self._append_log(
                    f"[frame {frame} | {_sec(frame, self.fps):.2f}s] person track {key} started"
                )
            tr.note_frame(frame, value=float(p.get("gun_conf", 0.0)), bucket=bucket)
            active_person.append(key)

        active_gun: list[str] = []
        for g in firearms:
            tag = g.get("display_tag")
            if not tag:
                continue
            key = str(tag)
            kind = str(g.get("kind", "object"))
            mk = str(g.get("majority_kind", kind))
            who = str(g.get("person") or "")
            tr = self._firearms.get(key)
            if tr is None:
                tr = _EntityTrack(key=key, category="firearm")
                tr.majority_kind = mk
                self._firearms[key] = tr
                self._append_log(
                    f"[frame {frame} | {_sec(frame, self.fps):.2f}s] firearm track {key} started ({kind})"
                    + (f" person={who}" if who else "")
                )
            tr.majority_kind = mk
            tr.note_frame(frame, value=float(g.get("conf", 0.0)), kind=kind)
            active_gun.append(key)
            if kind == "weapon":
                self._append_log(
                    f"[frame {frame} | {_sec(frame, self.fps):.2f}s] {key} weapon hit conf={float(g.get('conf', 0)):.2f}"
                )

        self._last_frame_snapshot = {
            "frame": frame,
            "time_sec": round(_sec(frame, self.fps), 3),
            "persons": active_person,
            "firearms": active_gun,
            "frame_max_gun_conf": round(float(frame_max_gun_conf), 4),
        }

        if frame % self.flush_every == 0:
            self._write_json(partial=True)
            self._append_log(
                f"[frame {frame} | {_sec(frame, self.fps):.2f}s] flush — "
                f"persons={active_person} firearms={active_gun} "
                f"clip_peak={self._clip_peak_gun_conf:.2f}"
            )

    def _build_report(self, *, partial: bool) -> dict[str, Any]:
        persons_sorted = sorted(self._persons.values(), key=lambda t: t.key)
        firearms_sorted = sorted(self._firearms.values(), key=lambda t: t.key)
        weapon_tracks = [t for t in firearms_sorted if (t.majority_kind or "") == "weapon"]
        object_tracks = [t for t in firearms_sorted if (t.majority_kind or "") != "weapon"]

        return {
            "partial": partial,
            "source": self.source,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(time() - self._started, 2),
            "video_fps": round(self.fps, 3),
            "unsafe_threshold": self.unsafe_threshold,
            "summary": {
                "frames_processed": self._frames_processed,
                "frames_with_person": self._frames_with_person,
                "frames_with_firearm_box": self._frames_with_firearm,
                "frames_with_weapon_label": self._frames_weapon_class,
                "clip_peak_gun_conf": round(self._clip_peak_gun_conf, 4),
                "clip_peak_frame": int(self._clip_peak_frame),
                "clip_peak_time_sec": round(_sec(self._clip_peak_frame, self.fps), 3),
                "person_tracks": len(self._persons),
                "firearm_tracks_total": len(self._firearms),
                "firearm_tracks_weapon_majority": len(weapon_tracks),
                "firearm_tracks_object_majority": len(object_tracks),
            },
            "last_frame": self._last_frame_snapshot,
            "person_tracks": [t.to_dict(self.fps) for t in persons_sorted],
            "firearm_tracks": [t.to_dict(self.fps) for t in firearms_sorted],
            "paths": {
                "json": str(self._json_path),
                "txt": str(self._txt_path),
                "log": str(self._log_path),
            },
        }

    def _write_json(self, *, partial: bool) -> None:
        payload = self._build_report(partial=partial)
        tmp = self._json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._json_path)

    def _write_txt(self, report: dict[str, Any]) -> None:
        s = report["summary"]
        lines = [
            "=" * 72,
            "CLIP ANALYSIS REPORT",
            "=" * 72,
            f"Source:     {report['source']}",
            f"Generated:  {report['generated_utc']}",
            f"FPS:        {report['video_fps']}",
            f"Threshold:  unsafe gun_conf >= {report['unsafe_threshold']:.2f}",
            "",
            "— SUMMARY —",
            f"  Frames processed:              {s['frames_processed']}",
            f"  Frames with person:            {s['frames_with_person']}",
            f"  Frames with firearm box:       {s['frames_with_firearm_box']}",
            f"  Frames with weapon label:      {s['frames_with_weapon_label']}",
            f"  Clip peak gun_conf:            {s['clip_peak_gun_conf']:.4f}  "
            f"(frame {s['clip_peak_frame']}, {s['clip_peak_time_sec']:.2f}s)",
            f"  Person tracks (T<id>):         {s['person_tracks']}",
            f"  Firearm tracks (total):        {s['firearm_tracks_total']}",
            f"    weapon-majority (weaponN):   {s['firearm_tracks_weapon_majority']}",
            f"    object-majority (objectN):   {s['firearm_tracks_object_majority']}",
            "",
            "— PERSON TRACKS —",
        ]
        if not report["person_tracks"]:
            lines.append("  (none)")
        for p in report["person_tracks"]:
            lines.append(f"  {p['id']}:")
            lines.append(f"    frames seen:     {p['frames_detected']}")
            lines.append(f"    peak gun_conf:   {p['gun_conf_peak']:.4f} @ frame {p['peak_frame']} ({p['peak_time_sec']:.2f}s)")
            lines.append(
                f"    buckets:         unsafe={p['unsafe_frames']} safe={p['safe_frames']}"
            )
            for span in p["appearance_spans"]:
                lines.append(f"    span:            {span['time_range']}")
            lines.append("")
        lines.append("— FIREARM TRACKS —")
        if not report["firearm_tracks"]:
            lines.append("  (none)")
        for g in report["firearm_tracks"]:
            who = g.get("person_key") or g.get("person") or ""
            who_s = f" ({who})" if who else ""
            lines.append(f"  {g['id']}{who_s} (majority {g.get('majority_kind', '?')}):")
            lines.append(f"    frames seen:     {g['frames_detected']}")
            lines.append(f"    peak conf:       {g['peak_value']:.4f} @ frame {g['peak_frame']} ({g['peak_time_sec']:.2f}s)")
            lines.append(f"    object frames:   {g['object_frames']}   weapon frames: {g['weapon_frames']}")
            for span in g["appearance_spans"]:
                lines.append(f"    span:            {span['time_range']}")
            lines.append("")
        lines.append(f"Live log: {self._log_path}")
        lines.append(f"JSON:     {self._json_path}")
        lines.append("=" * 72)
        self._txt_path.write_text("\n".join(lines), encoding="utf-8")

    def finalize(self) -> Path:
        report = self._build_report(partial=False)
        self._write_json(partial=False)
        self._write_txt(report)
        self._append_log(
            f"[DONE] frames={report['summary']['frames_processed']} "
            f"clip_peak={report['summary']['clip_peak_gun_conf']:.4f} "
            f"person_tracks={report['summary']['person_tracks']} "
            f"firearm_tracks={report['summary']['firearm_tracks_total']}"
        )
        return self._txt_path


def default_analysis_base(output_video: Path | None, source: str) -> Path | None:
    """Derive report base path next to ``--output`` or under ``runs/analysis`` from source name."""
    if output_video is not None:
        p = Path(output_video)
        return p.parent / f"{p.stem}_analysis"
    try:
        src_name = Path(source).stem
    except Exception:
        src_name = "clip"
    return Path("runs") / "analysis" / src_name
