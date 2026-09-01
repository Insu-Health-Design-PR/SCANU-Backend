"""Latest-wins slot: at most one pending item; producer replaces, never queues."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestSlot(Generic[T]):
    """Single-slot mailbox. ``put`` overwrites unread items."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: T | None = None
        self._seq = 0
        self.replaced = 0

    def put(self, item: T) -> int:
        with self._lock:
            if self._item is not None:
                self.replaced += 1
            self._item = item
            self._seq += 1
            return self._seq

    def take(self) -> T | None:
        with self._lock:
            item = self._item
            self._item = None
            return item

    def peek(self) -> T | None:
        with self._lock:
            return self._item

    def __len__(self) -> int:
        with self._lock:
            return 0 if self._item is None else 1


class LatestJob(Generic[T]):
    """Run ``fn`` on a single worker. If busy, keep only the newest payload."""

    def __init__(self, fn: Callable[[T], None], *, name: str = "latest-job") -> None:
        self._fn = fn
        self._lock = threading.Lock()
        self._pending: T | None = None
        self._busy = False
        self.replaced = 0
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    def submit(self, payload: T) -> None:
        run_now = False
        with self._lock:
            if self._busy:
                if self._pending is not None:
                    self.replaced += 1
                self._pending = payload
                return
            self._busy = True
            run_now = True
        if run_now:
            self._pool.submit(self._drain, payload)

    def _drain(self, payload: T) -> None:
        current: T | None = payload
        while current is not None:
            try:
                self._fn(current)
            except Exception as exc:
                print(f"Warning: latest-job worker failed: {exc}", flush=True)
            with self._lock:
                current = self._pending
                self._pending = None
                if current is None:
                    self._busy = False

    def shutdown(self, wait: bool = False) -> None:
        try:
            self._pool.shutdown(wait=wait, cancel_futures=True)
        except TypeError:
            self._pool.shutdown(wait=wait)
