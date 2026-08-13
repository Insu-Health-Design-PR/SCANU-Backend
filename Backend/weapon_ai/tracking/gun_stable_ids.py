"""Stable gun track IDs across frames."""

# TODO: maintain consistent gun object IDs


class GunStableIdTracker:
    def update(self, detections):
        raise NotImplementedError
