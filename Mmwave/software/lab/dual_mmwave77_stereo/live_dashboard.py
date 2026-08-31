"""Classic six-panel renderer for the calibrated dual-radar live stream."""
from __future__ import annotations

from collections import deque
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lab.dual_server_77ghz.live_contracts import FusedLiveFrame


class LiveDashboardRenderer:
    def __init__(self, *, sensor_distance_m: float, dpi: int = 100) -> None:
        self.distance_m = float(sensor_distance_m)
        self.dpi = int(dpi)
        self._person_history: deque[float] = deque(maxlen=120)
        self._reflective_history: deque[float] = deque(maxlen=120)

    @staticmethod
    def _xyz(rows: list[list[float]]) -> np.ndarray:
        if not rows:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32).reshape((-1, len(rows[0])))

    def render_status(self, state: str, detail: str = "") -> bytes:
        fig = plt.figure(figsize=(19.2, 10.8), dpi=self.dpi, facecolor="#030b16")
        fig.text(0.5, 0.57, f"Dual AWR1843 · {state}", ha="center", color="white", fontsize=28)
        fig.text(0.5, 0.49, detail, ha="center", color="#9fb3c8", fontsize=17)
        fig.text(0.5, 0.08, "Experimental radar evidence · no material or weapon confirmation", ha="center", color="#f6c85f", fontsize=14)
        return self._encode(fig)

    def render(self, frame: FusedLiveFrame) -> bytes:
        a = self._xyz(frame.points_a)
        b = self._xyz(frame.points_b)
        fused = self._xyz(frame.fused_points)
        active_tracks = [track for track in frame.tracks if track.missed_windows == 0]
        self._person_history.append(float(len(active_tracks)))
        self._reflective_history.append(float(len(frame.reflective_candidates)))

        fig = plt.figure(figsize=(19.2, 10.8), dpi=self.dpi, facecolor="#030b16")
        grid = fig.add_gridspec(2, 4, height_ratios=[1.45, 1.0], hspace=0.33, wspace=0.28)
        ax3d = fig.add_subplot(grid[0, :2], projection="3d")
        ax_top = fig.add_subplot(grid[0, 2])
        ax_side = fig.add_subplot(grid[0, 3])
        ax_azel = fig.add_subplot(grid[1, 0])
        ax_range = fig.add_subplot(grid[1, 1:3])
        ax_time = fig.add_subplot(grid[1, 3])
        axes = [ax3d, ax_top, ax_side, ax_azel, ax_range, ax_time]
        for axis in axes:
            axis.set_facecolor("#071426")
            axis.tick_params(colors="#a9bdd0", labelsize=8)
            axis.grid(True, color="#17314b", alpha=0.65)
            for spine in getattr(axis, "spines", {}).values():
                spine.set_color("#1f476a")

        def scatter_xyz(axis: Any, pts: np.ndarray, color: str, label: str, alpha: float = 0.75) -> None:
            if len(pts):
                axis.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, c=color, alpha=alpha, label=label)

        scatter_xyz(ax3d, a, "#58b7ff", "Radar A")
        scatter_xyz(ax3d, b, "#88d6ff", "Radar B", 0.62)
        for track in active_tracks:
            x, y, z = track.centroid_m
            ax3d.scatter([x], [y], [z], marker="x", s=95, c="#25e6cf")
            ax3d.text(x, y, z + 0.08, track.global_track_id, color="white", fontsize=9)
        for candidate in frame.reflective_candidates:
            x, y, z = candidate.center_m
            color = "#ff3b30" if candidate.persistent else "#ffd43b"
            ax3d.scatter([x], [y], [z], marker="D", s=55, c=color)
        ax3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", xlim=(-2.2, 2.2), ylim=(0, self.distance_m), zlim=(-1.7, 1.9))
        ax3d.set_title("Unified observed point cloud · A + B in A frame", color="white")
        if len(a) or len(b):
            ax3d.legend(loc="upper left", fontsize=8)

        if len(fused):
            ax_top.scatter(fused[:, 0], fused[:, 1], s=7, c="#58b7ff", alpha=0.72)
            ax_side.scatter(fused[:, 1], fused[:, 2], s=7, c="#58b7ff", alpha=0.72)
            ranges = np.linalg.norm(fused[:, :3], axis=1)
            azimuth = np.degrees(np.arctan2(fused[:, 0], np.maximum(fused[:, 1], 1e-6)))
            elevation = np.degrees(np.arcsin(np.clip(fused[:, 2] / np.maximum(ranges, 1e-6), -1, 1)))
            ax_azel.scatter(azimuth, elevation, s=7, c="#58b7ff", alpha=0.72)
        for candidate in frame.reflective_candidates:
            x, y, z = candidate.center_m
            color = "#ff3b30" if candidate.persistent else "#ffd43b"
            ax_top.scatter([x], [y], s=45, marker="D", c=color)
            ax_side.scatter([y], [z], s=45, marker="D", c=color)
        ax_top.set(title="Top view · fused x-y", xlabel="x (m)", ylabel="y (m)", xlim=(-2.2, 2.2), ylim=(0, self.distance_m))
        ax_side.set(title="Side view · fused y-z", xlabel="y (m)", ylabel="z (m)", xlim=(0, self.distance_m), ylim=(-1.7, 1.9))
        ax_azel.set(title="Elevation–azimuth · measured returns", xlabel="azimuth (deg)", ylabel="elevation (deg)", xlim=(-65, 65), ylim=(-45, 45))

        ax_range.plot(frame.range_profile_a, color="#58b7ff", linewidth=1.2, label="Radar A")
        ax_range.plot(frame.range_profile_b, color="#f6c85f", linewidth=1.2, label="Radar B")
        ax_range.set(title="Per-radar range-energy profiles", xlabel="range bin", ylabel="weighted return")
        ax_range.legend(fontsize=8)

        timeline = np.arange(len(self._person_history))
        ax_time.plot(timeline, self._person_history, color="#25e6cf", label="active global tracks")
        ax_time.plot(timeline, self._reflective_history, color="#ffd43b", label="reflective candidates")
        ax_time.set(title="Live evidence timeline", xlabel="live window", ylabel="count", ylim=(0, max(2.0, max(self._person_history, default=0) + 1, max(self._reflective_history, default=0) + 1)))
        ax_time.legend(fontsize=7)

        for axis in [ax_top, ax_side, ax_azel, ax_range, ax_time]:
            axis.title.set_color("white")
            axis.xaxis.label.set_color("#a9bdd0")
            axis.yaxis.label.set_color("#a9bdd0")
        quality = frame.quality
        fig.suptitle(
            f"Two facing AWR1843BOOST · calibrated live fusion · persons {len(active_tracks)}",
            color="white",
            fontsize=20,
            y=0.98,
        )
        fig.text(
            0.02,
            0.94,
            f"A frames {quality.frames_a} · B frames {quality.frames_b} · alignment {quality.alignment_error_ms if quality.alignment_error_ms is not None else '—'} ms · calibration {frame.calibration_id}",
            color="#9fb3c8",
            fontsize=10,
        )
        fig.text(0.02, 0.015, "Experimental reflectivity evidence · yellow transient / red persistent · no confirmed material or weapon classification", color="#f6c85f", fontsize=11)
        return self._encode(fig)

    def _encode(self, fig: Any) -> bytes:
        out = BytesIO()
        fig.savefig(out, format="jpeg", dpi=self.dpi, facecolor=fig.get_facecolor(), bbox_inches=None)
        plt.close(fig)
        return out.getvalue()
