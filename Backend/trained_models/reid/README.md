# Person Re-ID models (OSNet / torchreid)

Primary model: **OSNet-x0.25** trained on MSMT17 (Kaiyang Zhou torchreid zoo).

| File | Purpose |
|------|---------|
| `osnet_x0_25_msmt17.pth` | PyTorch weights (source of truth) |
| `osnet_x0_25_msmt17.onnx` | ONNX export (optional) |
| `osnet_x0_25_msmt17.engine` | TensorRT FP16 engine (GPU-specific; rebuild per machine) |

Rebuild ONNX + TensorRT after moving GPUs:

```bash
python scripts/export_reid_osnet.py --build-engine --fp16
```

Runtime backend order (`embed_backend=auto`): TensorRT → ONNX → torch OSNet → MobileNet fallback.
