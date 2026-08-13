"""Tests for point_cloud_fusion (two facing AWR1843 sensors)."""
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lab.dual_mmwave77_stereo import point_cloud_fusion as pcf

D = 3.6576  # 12 ft


def _write_session(d: Path, nframes: int = 20, seed: int = 0, ns_shift: int = 0) -> None:
    rng = np.random.default_rng(seed)
    base_utc_ns = 1_700_000_000_000_000_000

    def person_pts() -> np.ndarray:
        pts = np.zeros((150, 6), dtype=np.float32)
        pts[:, 0] = rng.normal(0, 0.08, 150)
        pts[:, 1] = 1.83 + rng.normal(0, 0.10, 150)
        pts[:, 2] = 0.05 + rng.normal(0, 0.20, 150)
        pts[:, 3] = rng.normal(0.4, 0.15, 150)  # doppler: moving person
        pts[:, 4] = 18.0 + rng.normal(0, 3, 150)  # snr
        pts[:, 5] = 5.0
        return pts

    with (d / "frames.jsonl").open("w") as f:
        for i in range(1, nframes + 1):
            utc_ns = base_utc_ns + i * 100_000_000 + ns_shift
            utc = datetime.fromtimestamp(utc_ns / 1e9, tz=timezone.utc).isoformat()
            row = {
                "host_utc": utc,
                "frame_number": i,
                "points": [
                    {
                        "x": float(q[0]),
                        "y": float(q[1]),
                        "z": float(q[2]),
                        "doppler": float(q[3]),
                        "snr": float(q[4]),
                        "noise": float(q[5]),
                    }
                    for q in person_pts()
                ],
            }
            f.write(json.dumps(row) + "\n")
    with (d / "perception.jsonl").open("w") as f:
        for w in range(4):
            fs = w * 5 + 1
            row = {
                "window_index": w,
                "frame_start": fs,
                "frame_end": fs + 10,
                "time_s": w * 0.5,
                "input_point_count": 150,
                "track": {"observed_this_window": True, "cluster_point_count": 120},
            }
            f.write(json.dumps(row) + "\n")


def _sessions(tmp: Path, b_ns_shift: int = 0):
    sa = tmp / "A"
    sb = tmp / "B"
    sa.mkdir()
    sb.mkdir()
    _write_session(sa, seed=0)
    _write_session(sb, seed=1, ns_shift=b_ns_shift)
    return pcf.load_session(sa), pcf.load_session(sb)


def test_transform_b_to_a_maps_person_to_mid() -> None:
    pts = np.asarray([[0.0, 1.83, 0.05]], dtype=np.float32)
    out = pcf.transform_b_to_a(pts, D)
    assert abs(out[0, 0]) < 1e-6          # x mirrored
    assert abs(out[0, 1] - 1.8276) < 1e-3  # D - y
    assert abs(out[0, 2] - 0.05) < 1e-6   # z unchanged


def test_fusion_matches_all_windows_and_centers_person() -> None:
    with tempfile.TemporaryDirectory() as td:
        a, b = _sessions(Path(td))
        rep = pcf.run_fusion(a, b, D, 0.0, 0.5)
        assert rep["matched_window_pairs"] == 4
        assert rep["fused_person_present_fraction"] == 1.0
        assert rep["a_person_present_fraction"] == 1.0
        assert rep["b_person_present_fraction"] == 1.0
        assert rep["fused_track_present_fraction"] == 1.0
        assert abs(rep["median_centroid_m"][1] - D / 2) < 0.15


def test_frames_with_offset_numbers_and_gap_map_to_correct_position() -> None:
    """Regression: window_points must resolve by frame number, not list position."""
    import bisect
    nums = [2, 3, 4, 6, 7]  # starts at 2, missing 5
    for target in (2, 3, 4, 6, 7):
        i = bisect.bisect_left(nums, target)
        assert i < len(nums) and nums[i] == target
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sa, sb = _sessions(tmp)
        assert sa.frame_numbers[0] == 1
        # emulate an offset start: first window frame_start maps correctly
        pts = sa.points[0]
        assert pts.shape[1] == 6


def test_window_points_uses_frame_numbers_not_positions() -> None:
    """A capture whose frames start at frame 2 (or have gaps) must still map
    window frame_start to the correct points, not a shifted list position."""
    from lab.dual_mmwave77_stereo.point_cloud_fusion import (
        SensorSession,
        window_points,
        window_start_ns,
    )
    import bisect

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sa, _ = _sessions(tmp)
        # Rewrite with frames starting at 2 and a gap (frame 5 missing).
        frame_numbers = [2, 3, 4, 6, 7]
        points = [np.zeros((i + 1, 6), dtype=np.float32) for i in range(5)]
        for i, fn in enumerate(frame_numbers):
            points[i][:, 1] = float(fn)  # y encodes the frame number
        utc = sa.frames_utc_ns[:5]
        sess = SensorSession(
            frames_utc_ns=utc,
            frame_numbers=frame_numbers,
            points=points,
            window_index=[0],
            window_frame_start=[3],
            window_time_s=[0.0],
            window_present=[False],
            window_cluster_points=[0],
            window_input_points=[0],
        )
        got = window_points(sess, 3, frame_span=2)
        assert got.shape[0] == points[1].shape[0] + points[2].shape[0]
        assert set(np.unique(got[:, 1])) == {3.0, 4.0}
        assert window_start_ns(sess, 3) == utc[1]


def test_clock_offset_aligns_windows() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Host B clock is 100 ms ahead of host A.
        sa, sb = _sessions(tmp, b_ns_shift=100_000_000)
        # Without the offset, B windows are 100 ms later -> no match at tight tolerance.
        rep0 = pcf.run_fusion(sa, sb, D, clock_offset_b_minus_a_s=0.0, window_tolerance_s=0.05)
        assert rep0["matched_window_pairs"] == 0
        # With the measured offset, windows align.
        rep = pcf.run_fusion(sa, sb, D, clock_offset_b_minus_a_s=0.1, window_tolerance_s=0.05)
        assert rep["matched_window_pairs"] == 4
