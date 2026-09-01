"""Bounded rolling percentile window (no unbounded history)."""

from __future__ import annotations

from collections import deque


class RollingPercentiles:
    """Keep the last ``maxlen`` samples and report p50/p95/p99."""

    def __init__(self, maxlen: int = 256) -> None:
        self._samples: deque[float] = deque(maxlen=max(8, int(maxlen)))

    def add(self, value: float) -> None:
        self._samples.append(float(value))

    def __len__(self) -> int:
        return len(self._samples)

    def percentile(self, q: float) -> float | None:
        if not self._samples:
            return None
        ordered = sorted(self._samples)
        n = len(ordered)
        if n == 1:
            return float(ordered[0])
        q = min(100.0, max(0.0, float(q)))
        idx = min(n - 1, max(0, int(round((q / 100.0) * (n - 1)))))
        return float(ordered[idx])

    def snapshot(self) -> dict[str, float | None]:
        return {
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "n": float(len(self._samples)),
        }
