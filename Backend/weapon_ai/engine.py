"""InferEngine: load models and process frames."""


class InferEngine:
    @classmethod
    def load(cls, profile: dict) -> "InferEngine":
        raise NotImplementedError

    def process_frame(self, frame):
        raise NotImplementedError
