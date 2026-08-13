"""OpenCV VideoCapture wrapper."""

# TODO: wrap cv2.VideoCapture with reconnect logic


class OpenCVCapture:
    def __init__(self, device: str | int = 0):
        self.device = device

    def read(self):
        raise NotImplementedError

    def release(self) -> None:
        pass
