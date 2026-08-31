import numpy as np

from media.encode.jpeg import encode_preview_jpeg, is_valid_jpeg


def test_encode_preview_jpeg_downscales_and_is_valid():
    frame = np.zeros((2560, 1440, 3), dtype=np.uint8)
    frame[:] = (20, 40, 80)
    payload = encode_preview_jpeg(frame, quality=62, max_width=960)
    assert payload is not None
    assert is_valid_jpeg(payload)
    assert len(payload) < 250_000


def test_encode_preview_jpeg_downscales_portrait():
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    payload = encode_preview_jpeg(frame, max_width=960)
    assert payload is not None
    assert is_valid_jpeg(payload)
    assert len(payload) < 180_000


def test_encode_preview_jpeg_keeps_narrow_frames():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    payload = encode_preview_jpeg(frame, max_width=1280)
    assert payload is not None
    assert is_valid_jpeg(payload)
