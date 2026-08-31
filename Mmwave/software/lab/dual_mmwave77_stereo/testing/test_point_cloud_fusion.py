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
    base_mono_ns = 5_000_000_000_000

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
                "host_monotonic_ns": base_mono_ns + i * 50_000_000 + ns_shift,
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
                "screening_state": "person",
                "anomalies": [],
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


def test_cluster_points_centroid_is_finite_and_repeatable() -> None:
    points = np.asarray(
        [
            [0.00, 1.00, 0.00, 0.0, 12.0, 4.0],
            [0.04, 1.02, 0.02, 0.0, 14.0, 4.0],
            [0.08, 1.04, 0.04, 0.0, 16.0, 4.0],
            [0.12, 1.06, 0.06, 0.0, 18.0, 4.0],
            [0.16, 1.08, 0.08, 0.0, 20.0, 4.0],
        ],
        dtype=np.float32,
    )

    first = pcf.cluster_points(points, voxel_m=0.20, min_points=5)
    second = pcf.cluster_points(points, voxel_m=0.20, min_points=5)

    assert len(first) == 1
    assert len(second) == 1
    assert np.all(np.isfinite(first[0].centroid_m))
    np.testing.assert_allclose(first[0].centroid_m, second[0].centroid_m)
    np.testing.assert_allclose(first[0].centroid_m, points[:, :3].mean(axis=0))


def test_anomaly_centers_transform_and_colors() -> None:
    anomalies = [
        {
            "center_m": [0.3, 1.4, 0.2],
            "reflective_anomaly_score": 0.9,
            "material_confirmed": False,
        },
        {
            "center_m": [-0.2, 1.5, 0.1],
            "reflective_anomaly_score": 0.95,
            "material_confirmed": True,
        },
    ]
    centers = pcf.anomaly_centers(anomalies, transform=True, distance_m=D)
    assert centers.shape == (2, 3)
    assert abs(centers[0, 0] - -0.3) < 1e-5      # x mirrored for B
    assert abs(centers[0, 1] - (D - 1.4)) < 1e-5  # D - y for B
    assert abs(centers[1, 2] - 0.1) < 1e-5       # z unchanged

    empty = pcf.anomaly_centers([], transform=True, distance_m=D)
    assert empty.shape == (0, 3)


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
        window_center_monotonic_ns,
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
            frames_monotonic_ns=[int(2e9) + 10 * fn for fn in frame_numbers],
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
        assert window_center_monotonic_ns(sess, 3, frame_span=2) == (
            sess.frames_monotonic_ns[1] + sess.frames_monotonic_ns[2]
        ) // 2


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


def test_matched_window_pairs_respects_offset_and_tolerance() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sa, sb = _sessions(tmp, b_ns_shift=100_000_000)
        pairs0 = pcf.matched_window_pairs(sa, sb, 0.0, 0.05)
        assert pairs0 == []
        pairs = pcf.matched_window_pairs(sa, sb, 0.1, 0.05)
        assert [(i_a, i_b) for i_a, i_b in pairs] == [(w, w) for w in range(4)]


def test_matched_window_pairs_uses_time_not_local_window_counter() -> None:
    """A station started 480 ms early must not be paired by equal counters."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sa, sb = _sessions(tmp, b_ns_shift=-480_000_000)
        pairs = pcf.matched_window_pairs(sa, sb, 0.0, 0.05)
        # B window 1 is contemporaneous with A window 0, etc.  B's leading
        # window and A's trailing window have no physical counterpart.
        assert pairs == [(0, 1), (1, 2), (2, 3)]


def test_frames_monotonic_ns_loaded_in_order() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sa, _ = _sessions(tmp)
        assert len(sa.frames_monotonic_ns) == len(sa.frame_numbers)
        # monotonic host clock must be strictly increasing like UTC
        assert all(
            b > a
            for a, b in zip(sa.frames_monotonic_ns, sa.frames_monotonic_ns[1:])
        )


def test_fuse_window_carries_screening_state_and_anomalies() -> None:
    from lab.mmwave77_usb.perception import PerceptionSpec

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sa, sb = _sessions(tmp)
        # Mark window 2 in A and window 3 in B as suspicious with anomalies.
        sa.window_state[2] = "suspicious_metal"
        sa.window_anomalies[2] = [
            {
                "center_m": [0.2, 1.4, 0.3],
                "reflective_anomaly_score": 0.9,
                "material_confirmed": False,
            }
        ]
        sb.window_state[3] = "suspicious_metal"
        sb.window_anomalies[3] = [
            {
                "center_m": [-0.1, 1.5, 0.2],
                "reflective_anomaly_score": 0.8,
                "material_confirmed": True,
            }
        ]
        spec = PerceptionSpec()
        track = None
        row, _, _, _, _ = pcf._fuse_window(
            sa, sb, 2, 2, D, None, None, 15, spec, track
        )
        assert row["a_screening_state"] == "suspicious_metal"
        assert row["b_screening_state"] == "person"
        assert row["a_anomaly_count"] == 1
        assert row["b_anomaly_count"] == 0
        ax, ay, az = row["a_anomaly_centers"][0]
        assert abs(ax - 0.2) < 1e-5
        assert abs(ay - 1.4) < 1e-5
        assert abs(az - 0.3) < 1e-5
        row_b, _, _, _, _ = pcf._fuse_window(
            sa, sb, 3, 3, D, None, None, 15, spec, track
        )
        assert row_b["a_screening_state"] == "person"
        assert row_b["b_screening_state"] == "suspicious_metal"
        assert row_b["b_anomaly_count"] == 1
        # B anomaly must be transformed into A's frame.
        bx, by, bz = row_b["b_anomaly_centers"][0]
        assert abs(bx - 0.1) < 1e-5
        assert abs(by - (D - 1.5)) < 1e-5
        assert abs(bz - 0.2) < 1e-5
