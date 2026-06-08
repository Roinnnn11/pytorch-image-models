"""Stage 3: Build TensorRT engines from ONNX using the Python TRT API.

trtexec binary has a CUDA runtime/driver mismatch on this host; the Python
tensorrt package (installed in deepburst env) runs against CUDA 12.8 correctly.

Model is selected via OPTIM_MODEL env var (default: vit_base_patch16_224).

Usage:
    python build_trt_engine.py --precision fp16
    python build_trt_engine.py --precision int8
    python build_trt_engine.py --precision int8 --onnx-suffix int8_qdq_inline.onnx
    python build_trt_engine.py --precision int8 --engine-suffix int8_v2.engine

Key design decisions (aligned with trtexec --fp16 --int8 best practice):

FP16:
  Weakly-typed network + BuilderFlag.FP16.

INT8 (Q/DQ ONNX):
  STRONGLY_TYPED + BuilderFlag.FP16 + BuilderFlag.INT8.
  - STRONGLY_TYPED: TRT reads QuantizeLinear/DequantizeLinear scale nodes and
    maps those ops to INT8 tensor-core GEMM kernels.
  - FP16 flag: non-quantized ops (LayerNorm, Softmax, residual adds, GELU) run
    FP16 instead of FP32. Without this flag those ops default to FP32 and
    dominate latency, eliminating most of the INT8 throughput advantage.
  - INT8 flag: enables INT8 tensor-core selection for Q/DQ-bracketed layers.
  In TRT 10.15 all three flags are compatible on STRONGLY_TYPED networks.

Optimization profile:
  opt=32 (matches eval/throughput batch), max=64 for sweep headroom.
  Previous opt=16 caused the kernel selector to under-optimize for bs=32.

Builder quality:
  builder_optimization_level=5  (exhaustive kernel search; default=3)
  avg_timing_iterations=8        (stable timing → better kernel selection)
  Persistent timing cache at engines/timing.cache, shared across builds.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorrt as trt
import common

# Timing cache shared across all engine builds for this model family.
_TIMING_CACHE = Path(__file__).parent / "engines" / "timing.cache"


def _load_timing_cache(config: trt.IBuilderConfig) -> None:
    if _TIMING_CACHE.exists():
        buf = _TIMING_CACHE.read_bytes()
        cache = config.create_timing_cache(buf)
        config.set_timing_cache(cache, ignore_mismatch=True)
        print(f"timing cache loaded: {_TIMING_CACHE} ({len(buf)//1024} KB)")
    else:
        cache = config.create_timing_cache(b"")
        config.set_timing_cache(cache, ignore_mismatch=False)
        print("timing cache: fresh")


def _save_timing_cache(config: trt.IBuilderConfig) -> None:
    cache = config.get_timing_cache()
    buf = cache.serialize()
    _TIMING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _TIMING_CACHE.write_bytes(bytes(buf))
    print(f"timing cache saved: {_TIMING_CACHE} ({buf.nbytes//1024} KB)")


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    opt_bs: int = 32,
    max_bs: int = 64,
) -> None:
    _, h, w = common.INPUT_SIZE

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    if precision == "int8":
        # WeaklyTyped + FP16 + INT8: non-quantized ops get FP16, Q/DQ ops get INT8.
        network_flags = 0
    else:
        network_flags = 0  # weakly-typed for FP16

    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    ok = parser.parse(onnx_path.read_bytes())
    if not ok:
        for i in range(parser.num_errors):
            print(f"ONNX parse error {i}: {parser.get_error(i)}")
        sys.exit(1)
    print(f"ONNX parsed OK: {network.num_layers} layers, "
          f"inputs={network.num_inputs}, outputs={network.num_outputs}")

    config = builder.create_builder_config()

    # Profile tuned to actual eval batch size so selected kernels are optimal
    # for the measurement point, not some smaller opt value.
    profile = builder.create_optimization_profile()
    profile.set_shape("input",
                      min=(1, 3, h, w),
                      opt=(opt_bs, 3, h, w),
                      max=(max_bs, 3, h, w))
    config.add_optimization_profile(profile)
    print(f"opt profile: min=1, opt={opt_bs}, max={max_bs}")

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("WARNING: platform does not report fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 build: weakly-typed + FP16 flag")

    elif precision == "int8":
        # TRT 10.15 hard rule: STRONGLY_TYPED network rejects ALL precision flags.
        # The correct path for Q/DQ ONNX with fast non-quantized fallback is:
        #   WeaklyTyped + FP16 + INT8  (equivalent to trtexec --fp16 --int8)
        # TRT uses the Q/DQ scale nodes to assign INT8 to those layers, and
        # FP16 flag ensures the non-quantized ops (LayerNorm, Softmax, adds,
        # GELU) run FP16 rather than FP32.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        print("INT8 build: weakly-typed + FP16 + INT8 flags (trtexec --fp16 --int8 equivalent)")

    # Exhaustive kernel search for best latency at the target batch size.
    config.builder_optimization_level = 5
    config.avg_timing_iterations = 8
    print("builder_optimization_level=5  avg_timing_iterations=8")

    _load_timing_cache(config)

    print(f"building engine ({precision}) -> {engine_path}")
    print("[opt level 5 takes longer — a few minutes is normal]")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: build_serialized_network returned None")
        sys.exit(1)

    _save_timing_cache(config)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"engine saved: {engine_path} ({engine_path.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp16", "int8"], required=True)
    parser.add_argument(
        "--onnx-suffix", default=None,
        help="Override ONNX file suffix, e.g. 'int8_qdq_inline.onnx'. "
             "If omitted, uses the default for the chosen precision.",
    )
    parser.add_argument(
        "--engine-suffix", default=None,
        help="Override engine file suffix, e.g. 'int8_v2.engine'. "
             "If omitted, uses '<precision>.engine'.",
    )
    parser.add_argument("--opt-bs", type=int, default=32,
                        help="Optimization profile opt batch size (default=32)")
    parser.add_argument("--max-bs", type=int, default=64,
                        help="Optimization profile max batch size (default=64)")
    args = parser.parse_args()

    if args.precision == "fp16":
        onnx_suffix   = args.onnx_suffix   or "fp32_dynbatch_inline.onnx"
        engine_suffix = args.engine_suffix or "fp16.engine"
    else:
        onnx_suffix   = args.onnx_suffix   or "int8_qdq_inline.onnx"
        engine_suffix = args.engine_suffix or "int8.engine"

    onnx_p = common.onnx_path(onnx_suffix)
    eng_p  = common.engine_path(engine_suffix)

    if not onnx_p.exists():
        print(f"ONNX not found: {onnx_p}")
        sys.exit(1)

    build_engine(onnx_p, eng_p, args.precision,
                 opt_bs=args.opt_bs, max_bs=args.max_bs)


if __name__ == "__main__":
    main()
