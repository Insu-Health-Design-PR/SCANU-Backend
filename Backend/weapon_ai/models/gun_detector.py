"""Gun / weapon detector model wrapper."""

# TODO: load gun detection weights


class GunDetector:
    def predict(self, frame):
        raise NotImplementedError
