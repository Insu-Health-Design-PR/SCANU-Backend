"""Single-image inference playground."""

# TODO: delegate to weapon_ai.pipelines.playground


class PlaygroundService:
    def infer_image(self, image_bytes: bytes) -> dict:
        raise NotImplementedError
