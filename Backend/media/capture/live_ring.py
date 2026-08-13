"""Latest-frame ring buffer."""

from threading import Lock


class LiveRingBuffer:
    def __init__(self):
        self._lock = Lock()
        self._frame = None

    def put(self, frame) -> None:
        with self._lock:
            self._frame = frame

    def get_latest(self):
        with self._lock:
            return self._frame
