#!/usr/bin/env python3
"""Export OSNet Re-ID to ONNX and (optionally) TensorRT engine.

Usage:
  python scripts/export_reid_osnet.py
  python scripts/export_reid_osnet.py --fp16 --engine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_model(weights: Path, model_name: str):
    import torch
    from weapon_ai.reid.embeddings import _load_osnet_state_dict
    from weapon_ai.reid.osnet import osnet_x0_25, osnet_x0_5, osnet_x0_75, osnet_x1_0

    builders = {
        "osnet_x0_25": osnet_x0_25,
        "osnet_x0_5": osnet_x0_5,
        "osnet_x0_75": osnet_x0_75,
        "osnet_x1_0": osnet_x1_0,
    }
    key = model_name if model_name in builders else "osnet_x0_25"
    model = builders[key](num_classes=1, pretrained=False, loss="softmax")
    matched = _load_osnet_state_dict(model, weights)
    if matched < 10:
        raise SystemExit(f"Weight load matched only {matched} tensors from {weights}")
    model.eval()
    return model


def export_onnx(model, onnx_path: Path, *, h: int, w: int, opset: int) -> None:
    import torch

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove stale external-data leftovers from dynamo exporter.
    for extra in (onnx_path.with_suffix(".onnx.data"), Path(str(onnx_path) + ".data")):
        if extra.is_file():
            extra.unlink()
    dummy = torch.randn(1, 3, h, w, dtype=torch.float32)
    # Legacy exporter → single self-contained .onnx (TensorRT-friendly).
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=int(opset),
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        # Older torch without dynamo= kwarg
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=int(opset),
            do_constant_folding=True,
        )
    print(f"Wrote ONNX {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")


def build_engine(onnx_path: Path, engine_path: Path, *, fp16: bool, workspace_gb: float) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print("ONNX parse error:", parser.get_error(i))
        raise SystemExit("Failed to parse ONNX for TensorRT")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    # Optimize for batch 1..8
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    # Fixed H/W from ONNX; allow batch 1..8
    shape = tuple(inp.shape)
    h = int(shape[2]) if shape[2] > 0 else 256
    w = int(shape[3]) if shape[3] > 0 else 128
    profile.set_shape(inp.name, (1, 3, h, w), (1, 3, h, w), (8, 3, h, w))
    config.add_optimization_profile(profile)

    print("Building TensorRT engine (this can take a minute)…")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("TensorRT engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    print(f"Wrote engine {engine_path} ({engine_path.stat().st_size / 1e6:.2f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "trained_models/reid/osnet_x0_25_msmt17.pth",
    )
    ap.add_argument("--model", default="osnet_x0_25")
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--engine-out", type=Path, default=None, help="TensorRT engine output path")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    ap.add_argument("--build-engine", action="store_true", help="Also build TensorRT .engine")
    ap.add_argument("--skip-engine", action="store_true", help="ONNX only")
    args = ap.parse_args()

    weights = args.weights
    if not weights.is_file():
        raise SystemExit(f"Missing weights: {weights}")
    onnx_path = Path(args.onnx) if args.onnx else weights.with_suffix(".onnx")
    want_engine = bool(args.build_engine) and not bool(args.skip_engine)
    engine_path = Path(args.engine_out) if args.engine_out else weights.with_suffix(".engine")

    print(f"Loading {args.model} from {weights}")
    model = _build_model(weights, args.model)
    export_onnx(model, onnx_path, h=args.h, w=args.w, opset=args.opset)

    if want_engine:
        try:
            build_engine(
                onnx_path,
                engine_path,
                fp16=bool(args.fp16),
                workspace_gb=float(args.workspace_gb),
            )
        except Exception as exc:
            print(f"WARNING: TensorRT engine build failed ({exc})")
            print("ONNX is still usable; torch CUDA path remains the default fallback.")
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
