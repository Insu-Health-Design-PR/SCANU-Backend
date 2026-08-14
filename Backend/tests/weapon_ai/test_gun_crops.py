"""Unit tests for batched person-crop firearm inference."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from weapon_ai.detection.gun_crops import (
    collect_person_gun_crops,
    mapped_gun_boxes_from_results,
    predict_gun_on_crops,
)


def test_collect_person_gun_crops_skips_tiny_and_keeps_order():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[20:80, 30:90] = 10
    frame[100:180, 150:250] = 20
    crops = collect_person_gun_crops(
        frame,
        [
            (0, (30, 20, 90, 80)),
            (1, (10, 10, 14, 14)),
            (2, (150, 100, 250, 180)),
        ],
        pad_frac=0.0,
        pad_px=0,
        min_box_px=16,
    )
    assert [c.ridx for c in crops] == [0, 2]
    assert crops[0].crop.shape[0] == 60
    assert crops[1].crop.shape[1] == 100


def test_predict_gun_on_crops_batches_when_enabled():
    calls: list[object] = []

    class _Det:
        def predict(self, source, **_kw):
            calls.append(source)
            n = len(source) if isinstance(source, list) else 1
            return [SimpleNamespace(boxes=None, names={0: "gun"}) for _ in range(n)]

    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    crops = collect_person_gun_crops(
        frame,
        [(0, (5, 5, 40, 40)), (1, (40, 40, 75, 75))],
        pad_frac=0.0,
        pad_px=0,
        min_box_px=8,
    )
    results = predict_gun_on_crops(
        _Det(), crops, conf=0.2, imgsz=640, device="cpu", batched=True
    )
    assert len(results) == 2
    assert len(calls) == 1
    assert isinstance(calls[0], list)
    assert len(calls[0]) == 2


def test_predict_gun_on_crops_sequential_fallback():
    calls: list[object] = []

    class _Det:
        def predict(self, source, **_kw):
            calls.append(source)
            return [SimpleNamespace(boxes=None, names={0: "gun"})]

    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    crops = collect_person_gun_crops(
        frame,
        [(0, (5, 5, 40, 40)), (1, (40, 40, 75, 75))],
        pad_frac=0.0,
        pad_px=0,
        min_box_px=8,
    )
    results = predict_gun_on_crops(
        _Det(), crops, conf=0.2, imgsz=640, device="cpu", batched=False
    )
    assert len(results) == 2
    assert len(calls) == 2
    assert all(isinstance(c, np.ndarray) for c in calls)


def test_mapped_gun_boxes_offset_into_frame():
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    from weapon_ai.detection.gun_crops import PersonGunCrop

    job = PersonGunCrop(
        ridx=3,
        person_xyxy=(100, 50, 160, 130),
        roi_xyxy=(90, 40, 170, 140),
        crop=crop,
    )

    class _Boxes:
        xyxy = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([[2.0, 3.0, 8.0, 9.0]])))
        cls = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([0])))
        conf = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([0.88])))

        def __len__(self) -> int:
            return 1

    raw = mapped_gun_boxes_from_results(
        [job],
        [SimpleNamespace(boxes=_Boxes(), names={0: "gun"})],
        names={0: "gun"},
    )
    assert len(raw) == 1
    gc, gx1, gy1, gx2, gy2, gnm, owner, cid, pr, pb = raw[0]
    assert gc == 0.88
    assert (gx1, gy1, gx2, gy2) == (92, 43, 98, 49)
    assert gnm == "gun" and owner == 3 and cid == 0
    assert (pr, pb) == (80, 100)
