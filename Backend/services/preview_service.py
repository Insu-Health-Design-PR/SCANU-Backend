"""Pick BGR vs JPEG vs placeholder for preview."""

# TODO: integrate media.ipc.frame_reader


class PreviewService:
    def get_frame_bytes(self, source: str, fmt: str = "jpeg") -> bytes | None:
        raise NotImplementedError
