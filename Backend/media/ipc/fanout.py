"""WebRTC fan-out from shared frame buffer."""

# TODO: broadcast frames to multiple WebRTC peers


class FrameFanout:
    def subscribe(self, peer_id: str) -> None:
        raise NotImplementedError

    def unsubscribe(self, peer_id: str) -> None:
        raise NotImplementedError
