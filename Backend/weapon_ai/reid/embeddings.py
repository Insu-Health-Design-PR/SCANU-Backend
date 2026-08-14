"""Person Re-ID embedding extractors (Jetson / desktop).

Backend preference (``backend='auto'``):
  1. TensorRT engine (``.engine``) when available
  2. ONNX Runtime CUDA / CPU (``.onnx``)
  3. Vendored OSNet (torchreid architecture) + MSMT17 Re-ID weights
  4. torchvision MobileNetV3-Small (legacy fallback)
  5. Mock embedder (tests / no torch)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default MSMT17-trained OSNet-x0.25 (Kaiyang Zhou / torchreid model zoo).
DEFAULT_OSNET_WEIGHTS = (
    Path(__file__).resolve().parents[2]
    / "trained_models"
    / "reid"
    / "osnet_x0_25_msmt17.pth"
)
DEFAULT_OSNET_ONNX = DEFAULT_OSNET_WEIGHTS.with_suffix(".onnx")
DEFAULT_OSNET_ENGINE = DEFAULT_OSNET_WEIGHTS.with_suffix(".engine")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = _l2_normalize(a)
    vb = _l2_normalize(b)
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0
    return float(np.dot(va, vb))


def _resolve_path(path: str | Path | None, default: Path) -> Path | None:
    if path is None or str(path).strip() == "":
        return default if default.is_file() else None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (Path(__file__).resolve().parents[2] / p).resolve()
    return p if p.is_file() else None


def _preprocess_bgr_nchw(
    bgr_crop: np.ndarray,
    *,
    input_h: int,
    input_w: int,
) -> np.ndarray:
    """BGR crop → float32 NCHW ImageNet-normalized tensor (1,3,H,W)."""
    import cv2

    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (int(input_w), int(input_h)), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
    return np.ascontiguousarray(x)


def _load_osnet_state_dict(model: Any, weight_path: Path) -> int:
    """Load torchreid-style checkpoint; ignore mismatched classifier keys."""
    import torch
    from collections import OrderedDict

    ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model_dict = model.state_dict()
    new_state = OrderedDict()
    matched = 0
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and tuple(model_dict[k].shape) == tuple(v.shape):
            new_state[k] = v
            matched += 1
    model_dict.update(new_state)
    model.load_state_dict(model_dict)
    return matched


class MockReIDEmbedder:
    """Deterministic embedder for tests/simulation (no torch)."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = int(dim)
        self.device = "cpu"
        self.backend = "mock"
        self.feature_dim = int(dim)

    def embed(self, bgr_crop: np.ndarray | None, *, identity_hint: int | None = None) -> np.ndarray:
        if identity_hint is not None:
            rng = np.random.default_rng(abs(int(identity_hint)) + 17)
            return _l2_normalize(rng.standard_normal(self.dim).astype(np.float32))
        if bgr_crop is None or getattr(bgr_crop, "size", 0) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        arr = np.asarray(bgr_crop, dtype=np.float32)
        h, w = arr.shape[:2]
        means = arr.reshape(-1, arr.shape[-1]).mean(axis=0) if arr.ndim == 3 else np.array([arr.mean()] * 3)
        feats = np.zeros(self.dim, dtype=np.float32)
        feats[:3] = means[:3] / 255.0
        feats[3] = h / 1000.0
        feats[4] = w / 1000.0
        feats[5] = float(arr.std() / 255.0) if arr.size else 0.0
        flat = arr.reshape(-1)[:: max(1, arr.size // (self.dim - 6))]
        n = min(self.dim - 6, flat.size)
        feats[6 : 6 + n] = flat[:n] / 255.0
        return _l2_normalize(feats)

    def embed_batch(self, crops: list[np.ndarray | None]) -> list[np.ndarray]:
        return [self.embed(c) for c in crops]


class _TrtReIDSession:
    """Minimal TensorRT inference wrapper for a fixed-shape OSNet engine."""

    def __init__(self, engine_path: Path) -> None:
        import tensorrt as trt

        self._trt = trt
        logger_trt = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger_trt)
        blob = Path(engine_path).read_bytes()
        engine = runtime.deserialize_cuda_engine(blob)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.engine = engine
        self.context = engine.create_execution_context()
        self._bindings: dict[str, Any] = {}
        self._allocate()

    def _allocate(self) -> None:
        import torch

        self._torch = torch
        n = self.engine.num_io_tensors
        self.input_name = None
        self.output_name = None
        for i in range(n):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            # Map TRT dtype → torch
            if dtype == self._trt.float32:
                tdt = torch.float32
            elif dtype == self._trt.float16:
                tdt = torch.float16
            else:
                tdt = torch.float32
            # Replace dynamic dims with 1 for default binding
            fixed = tuple(1 if (d is None or d < 0) else int(d) for d in shape)
            buf = torch.empty(fixed, dtype=tdt, device="cuda")
            self._bindings[name] = buf
            mode = self.engine.get_tensor_mode(name)
            if mode == self._trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = fixed
            else:
                self.output_name = name
                self.output_shape = fixed
        if not self.input_name or not self.output_name:
            raise RuntimeError("TRT engine missing input/output tensors")

    def infer(self, nchw: np.ndarray) -> np.ndarray:
        torch = self._torch
        shape = tuple(int(x) for x in nchw.shape)
        # Optimization profile requires explicit input shape every run when dynamic.
        self.context.set_input_shape(self.input_name, shape)
        out_shape = tuple(int(x) for x in self.context.get_tensor_shape(self.output_name))
        inp = self._bindings[self.input_name]
        out = self._bindings[self.output_name]
        if tuple(inp.shape) != shape:
            inp = torch.empty(shape, dtype=inp.dtype, device="cuda")
            self._bindings[self.input_name] = inp
        if tuple(out.shape) != out_shape:
            out = torch.empty(out_shape, dtype=out.dtype, device="cuda")
            self._bindings[self.output_name] = out
        inp.copy_(torch.from_numpy(nchw).to(device="cuda", dtype=inp.dtype))
        for name, buf in self._bindings.items():
            self.context.set_tensor_address(name, int(buf.data_ptr()))
        stream = torch.cuda.current_stream()
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()
        return self._bindings[self.output_name].detach().float().cpu().numpy()


class PersonReIDEmbedder:
    """Production Re-ID embedder: TensorRT → ONNX → OSNet(torch) → MobileNet."""

    def __init__(
        self,
        *,
        device: str = "auto",
        backend: str = "auto",
        model_name: str = "osnet_x0_25",
        weights_path: str | Path | None = None,
        onnx_path: str | Path | None = None,
        engine_path: str | Path | None = None,
        input_h: int = 256,
        input_w: int = 128,
    ) -> None:
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.model_name = str(model_name or "osnet_x0_25")
        self.backend = "mock"
        self.device = "cpu"
        self.feature_dim = 512
        self._model: Any = None
        self._torch: Any = None
        self._ort: Any = None
        self._trt: _TrtReIDSession | None = None
        self._mock: MockReIDEmbedder | None = None
        self.weights_path = _resolve_path(weights_path, DEFAULT_OSNET_WEIGHTS)
        self.onnx_path = _resolve_path(onnx_path, DEFAULT_OSNET_ONNX)
        self.engine_path = _resolve_path(engine_path, DEFAULT_OSNET_ENGINE)
        self._resolve(device=device, backend=backend)

    def _resolve(self, *, device: str, backend: str) -> None:
        want = (backend or "auto").strip().lower()
        try:
            import torch

            self._torch = torch
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = str(device)
        except Exception as exc:  # pragma: no cover
            logger.warning("Re-ID: torch unavailable (%s); using mock", exc)
            self._mock = MockReIDEmbedder()
            self.backend = "mock"
            return

        # 1) TensorRT engine
        if want in ("auto", "tensorrt", "trt", "engine") and self.engine_path and self.device.startswith("cuda"):
            try:
                self._trt = _TrtReIDSession(self.engine_path)
                self.backend = "tensorrt"
                self.feature_dim = int(np.prod(self._trt.output_shape[1:]) or 512)
                logger.info(
                    "Re-ID backend=tensorrt engine=%s out_dim=%s",
                    self.engine_path,
                    self.feature_dim,
                )
                return
            except Exception as exc:
                if want in ("tensorrt", "trt", "engine"):
                    logger.warning("TensorRT Re-ID failed (%s); falling back", exc)
                else:
                    logger.info("TensorRT Re-ID unavailable (%s); trying next backend", exc)

        # 2) ONNX Runtime
        if want in ("auto", "onnx", "onnxruntime") and self.onnx_path:
            try:
                import onnxruntime as ort

                providers = []
                if self.device.startswith("cuda"):
                    providers.append("CUDAExecutionProvider")
                providers.append("CPUExecutionProvider")
                sess = ort.InferenceSession(str(self.onnx_path), providers=providers)
                self._ort = sess
                self.backend = "onnx"
                out0 = sess.get_outputs()[0]
                dim = 512
                if out0.shape and len(out0.shape) >= 2 and isinstance(out0.shape[-1], int):
                    dim = int(out0.shape[-1])
                self.feature_dim = dim
                logger.info(
                    "Re-ID backend=onnx path=%s providers=%s dim=%s",
                    self.onnx_path,
                    sess.get_providers(),
                    dim,
                )
                return
            except Exception as exc:
                if want in ("onnx", "onnxruntime"):
                    logger.warning("ONNX Re-ID failed (%s); falling back", exc)
                else:
                    logger.info("ONNX Re-ID unavailable (%s); trying next backend", exc)

        # 3) Vendored OSNet (torchreid architecture) + MSMT17 weights
        if want in ("auto", "torchreid", "osnet"):
            try:
                self._load_osnet_torch()
                return
            except Exception as exc:
                if want in ("torchreid", "osnet"):
                    logger.warning("OSNet/torchreid failed (%s); falling back", exc)
                else:
                    logger.info("OSNet unavailable (%s); falling back to MobileNet", exc)

        # 4) MobileNetV3 fallback
        if want in ("auto", "torchvision", "mobilenet"):
            try:
                import torchvision

                weights = None
                try:
                    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
                    model = torchvision.models.mobilenet_v3_small(weights=weights)
                except Exception:
                    model = torchvision.models.mobilenet_v3_small(weights=None)
                model.classifier = self._torch.nn.Identity()
                model.eval()
                model.to(self.device)
                self._model = model
                self.backend = "torchvision"
                self.feature_dim = 576
                logger.info("Re-ID backend=torchvision mobilenet_v3_small device=%s", self.device)
                return
            except Exception as exc:
                logger.warning("MobileNet Re-ID failed (%s)", exc)

        self._mock = MockReIDEmbedder()
        self.backend = "mock"
        logger.warning("Re-ID: all backends failed; using mock")

    def _load_osnet_torch(self) -> None:
        from weapon_ai.reid.osnet import osnet_x0_25, osnet_x0_5, osnet_x0_75, osnet_x1_0

        builders = {
            "osnet_x0_25": osnet_x0_25,
            "osnet_x0_5": osnet_x0_5,
            "osnet_x0_75": osnet_x0_75,
            "osnet_x1_0": osnet_x1_0,
        }
        key = self.model_name if self.model_name in builders else "osnet_x0_25"
        # num_classes=1 → classifier discarded when loading MSMT17 weights
        model = builders[key](num_classes=1, pretrained=False, loss="softmax")
        if self.weights_path is None:
            raise FileNotFoundError(
                f"OSNet weights not found (expected {DEFAULT_OSNET_WEIGHTS})"
            )
        matched = _load_osnet_state_dict(model, self.weights_path)
        if matched < 10:
            raise RuntimeError(
                f"OSNet weight load matched only {matched} tensors from {self.weights_path}"
            )
        model.eval()
        model.to(self.device)
        self._model = model
        self.backend = "torchreid"
        self.feature_dim = int(getattr(model, "feature_dim", 512) or 512)
        logger.info(
            "Re-ID backend=torchreid %s weights=%s matched=%s device=%s dim=%s",
            key,
            self.weights_path.name,
            matched,
            self.device,
            self.feature_dim,
        )

    def embed(self, bgr_crop: np.ndarray | None, *, identity_hint: int | None = None) -> np.ndarray:
        if self.backend == "mock":
            assert self._mock is not None
            return self._mock.embed(bgr_crop, identity_hint=identity_hint)
        if bgr_crop is None or getattr(bgr_crop, "size", 0) == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        nchw = _preprocess_bgr_nchw(bgr_crop, input_h=self.input_h, input_w=self.input_w)

        if self.backend == "tensorrt":
            assert self._trt is not None
            out = self._trt.infer(nchw)
            return _l2_normalize(out.reshape(-1))

        if self.backend == "onnx":
            assert self._ort is not None
            inp_name = self._ort.get_inputs()[0].name
            out = self._ort.run(None, {inp_name: nchw})[0]
            return _l2_normalize(np.asarray(out).reshape(-1))

        # torch / torchreid / torchvision
        torch = self._torch
        tensor = torch.from_numpy(nchw).to(self.device)
        with torch.inference_mode():
            feat = self._model(tensor)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            vec = feat.detach().float().cpu().numpy().reshape(-1)
        return _l2_normalize(vec)

    def embed_batch(self, crops: list[np.ndarray | None]) -> list[np.ndarray]:
        """Batch embed (torch / onnx); falls back to per-crop for TRT fixed-batch."""
        if not crops:
            return []
        if self.backend in ("mock", "tensorrt") or len(crops) == 1:
            return [self.embed(c) for c in crops]
        valid_idx: list[int] = []
        batch_list: list[np.ndarray] = []
        outs: list[np.ndarray] = [
            np.zeros(self.feature_dim, dtype=np.float32) for _ in crops
        ]
        for i, c in enumerate(crops):
            if c is None or getattr(c, "size", 0) == 0:
                continue
            valid_idx.append(i)
            batch_list.append(
                _preprocess_bgr_nchw(c, input_h=self.input_h, input_w=self.input_w)[0]
            )
        if not batch_list:
            return outs
        nchw = np.stack(batch_list, axis=0).astype(np.float32)
        if self.backend == "onnx":
            assert self._ort is not None
            inp_name = self._ort.get_inputs()[0].name
            feat = np.asarray(self._ort.run(None, {inp_name: nchw})[0])
        else:
            torch = self._torch
            tensor = torch.from_numpy(nchw).to(self.device)
            with torch.inference_mode():
                feat_t = self._model(tensor)
                if isinstance(feat_t, (tuple, list)):
                    feat_t = feat_t[0]
                feat = feat_t.detach().float().cpu().numpy()
        for j, i in enumerate(valid_idx):
            outs[i] = _l2_normalize(feat[j].reshape(-1))
        return outs


class TrackEmbeddingCache:
    """Cache embeddings per (camera_id, local_track_id) with history + cadence."""

    def __init__(
        self,
        embedder: PersonReIDEmbedder | MockReIDEmbedder,
        *,
        interval_s: float = 0.5,
        max_history: int = 12,
        min_box_px: int = 40,
    ) -> None:
        self.embedder = embedder
        self.interval_s = max(0.05, float(interval_s))
        self.max_history = max(1, int(max_history))
        self.min_box_px = max(1, int(min_box_px))
        self._last_ts: dict[tuple[str, int], float] = {}
        self._history: dict[tuple[str, int], deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )

    def maybe_update(
        self,
        camera_id: str,
        local_track_id: int,
        bgr_frame: np.ndarray | None,
        bbox: tuple[int, int, int, int] | None,
        *,
        now: float | None = None,
        force: bool = False,
        identity_hint: int | None = None,
    ) -> np.ndarray | None:
        """Return latest embedding (possibly cached); update when due."""
        key = (str(camera_id), int(local_track_id))
        ts = float(time.time() if now is None else now)
        last = self._last_ts.get(key, 0.0)
        hist = self._history[key]
        due = force or (ts - last) >= self.interval_s or not hist
        if not due:
            return hist[-1] if hist else None
        if bgr_frame is None or bbox is None:
            return hist[-1] if hist else None
        x1, y1, x2, y2 = (int(v) for v in bbox)
        if (x2 - x1) < self.min_box_px or (y2 - y1) < self.min_box_px:
            return hist[-1] if hist else None
        h, w = bgr_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return hist[-1] if hist else None
        crop = bgr_frame[y1:y2, x1:x2]
        emb = self.embedder.embed(crop, identity_hint=identity_hint)
        hist.append(emb)
        self._last_ts[key] = ts
        return emb

    def representative(self, camera_id: str, local_track_id: int) -> np.ndarray | None:
        hist = self._history.get((str(camera_id), int(local_track_id)))
        if not hist:
            return None
        stacked = np.stack(list(hist), axis=0)
        return _l2_normalize(stacked.mean(axis=0))

    def drop(self, camera_id: str, local_track_id: int) -> None:
        key = (str(camera_id), int(local_track_id))
        self._history.pop(key, None)
        self._last_ts.pop(key, None)

    def prune_missing(self, camera_id: str, active_local_ids: set[int]) -> None:
        dead = [k for k in self._history if k[0] == camera_id and k[1] not in active_local_ids]
        for k in dead:
            self._history.pop(k, None)
            self._last_ts.pop(k, None)
