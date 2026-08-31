from __future__ import annotations

import json
from pathlib import Path

from lab.dual_mmwave77_stereo.fusion import run_fusion


def _write_session(
    root: Path,
    name: str,
    *,
    start_utc: str,
    fps: float,
    frame_count: int,
    window_states: list[str],
) -> Path:
    session = root / name
    session.mkdir(parents=True)
    from datetime import datetime, timedelta, timezone

    base = datetime.fromisoformat(start_utc).replace(tzinfo=timezone.utc)
    frame_period = 1.0 / fps

    frames_path = session / "frames.jsonl"
    with frames_path.open("w") as handle:
        for frame_number in range(frame_count):
            ts = base + timedelta(seconds=frame_number * frame_period)
            handle.write(
                json.dumps(
                    {
                        "frame_number": frame_number,
                        "host_utc": ts.isoformat(),
                    }
                )
                + "\n"
            )

    window_frames = 10
    stride_frames = 5
    perception_path = session / "perception.jsonl"
    with perception_path.open("w") as handle:
        for window_index, state in enumerate(window_states):
            start = window_index * stride_frames
            end = min(start + window_frames, frame_count)
            handle.write(
                json.dumps(
                    {
                        "window_index": window_index,
                        "frame_start": start,
                        "frame_end": end,
                        "screening_state": state,
                        "scene_state": "person" if state != "background" else "background",
                        "track": {"position_m": [1.0, 1.0, 1.0]}
                        if state != "background"
                        else None,
                    }
                )
                + "\n"
            )
    return session


def test_perfectly_aligned_sessions_agree_fully(tmp_path: Path) -> None:
    session_a = _write_session(
        tmp_path,
        "a",
        start_utc="2026-01-01T00:00:00",
        fps=10.0,
        frame_count=40,
        window_states=["background", "person", "suspicious_metal", "person"],
    )
    session_b = _write_session(
        tmp_path,
        "b",
        start_utc="2026-01-01T00:00:00",
        fps=10.0,
        frame_count=40,
        window_states=["background", "person", "suspicious_metal", "person"],
    )

    report = run_fusion(
        session_a,
        session_b,
        clock_offset_b_minus_a_s=0.0,
        tolerance_s=0.5,
    )

    assert report["matched_window_pairs"] == 4
    assert report["screening_state_agreement_fraction"] == 1.0
    assert report["both_flagged_suspicious_metal"] == 1
    assert report["either_flagged_suspicious_metal"] == 1
    assert report["unmatched_windows_a"] == 0
    assert report["unmatched_windows_b"] == 0


def test_clock_offset_is_corrected_before_matching(tmp_path: Path) -> None:
    session_a = _write_session(
        tmp_path,
        "a",
        start_utc="2026-01-01T00:00:00",
        fps=10.0,
        frame_count=40,
        window_states=["background", "person", "suspicious_metal", "person"],
    )
    # Host B's clock is 3.0 seconds ahead of host A's clock.
    session_b = _write_session(
        tmp_path,
        "b",
        start_utc="2026-01-01T00:00:03",
        fps=10.0,
        frame_count=40,
        window_states=["background", "person", "suspicious_metal", "person"],
    )

    uncorrected = run_fusion(
        session_a,
        session_b,
        clock_offset_b_minus_a_s=0.0,
        tolerance_s=0.5,
    )
    assert uncorrected["matched_window_pairs"] == 0

    corrected = run_fusion(
        session_a,
        session_b,
        clock_offset_b_minus_a_s=3.0,
        tolerance_s=0.5,
    )
    assert corrected["matched_window_pairs"] == 4
    assert corrected["screening_state_agreement_fraction"] == 1.0


def test_disagreement_is_counted_not_hidden(tmp_path: Path) -> None:
    session_a = _write_session(
        tmp_path,
        "a",
        start_utc="2026-01-01T00:00:00",
        fps=10.0,
        frame_count=40,
        window_states=["background", "suspicious_metal", "person", "background"],
    )
    session_b = _write_session(
        tmp_path,
        "b",
        start_utc="2026-01-01T00:00:00",
        fps=10.0,
        frame_count=40,
        window_states=["background", "person", "person", "background"],
    )

    report = run_fusion(
        session_a,
        session_b,
        clock_offset_b_minus_a_s=0.0,
        tolerance_s=0.5,
    )

    assert report["matched_window_pairs"] == 4
    assert report["screening_state_agreement_fraction"] == 0.75
    assert report["both_flagged_suspicious_metal"] == 0
    assert report["either_flagged_suspicious_metal"] == 1
    assert report["both_flagged_person_or_suspicious_metal"] == 2
    assert report["either_flagged_person_or_suspicious_metal"] == 2
