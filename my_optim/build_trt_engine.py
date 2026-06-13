"""Build corrected TensorRT FP16 or Q/DQ INT8 engines for CNN models.

Both paths use a weakly typed network. The INT8 path enables FP16 and INT8:
Q/DQ-bracketed layers can select INT8 kernels while unsupported or deliberately
unquantized operations can fall back to FP16 instead of FP32.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorrt as trt

import common


def _timing_cache_path() -> Path:
    return common.ENGINE_DIR / "timing.cache"


def _load_timing_cache(config: trt.IBuilderConfig) -> None:
    path = _timing_cache_path()
    if path.exists():
        data = path.read_bytes()
        cache = config.create_timing_cache(data)
        config.set_timing_cache(cache, ignore_mismatch=True)
        print(f"timing cache loaded: {path} ({len(data) // 1024} KB)")
    else:
        cache = config.create_timing_cache(b"")
        config.set_timing_cache(cache, ignore_mismatch=False)
        print("timing cache: fresh")


def _save_timing_cache(config: trt.IBuilderConfig) -> None:
    path = _timing_cache_path()
    serialized = bytes(config.get_timing_cache().serialize())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialized)
    print(f"timing cache saved: {path} ({len(serialized) // 1024} KB)")


def build_engine(
        onnx_path: Path,
        engine_path: Path,
        precision: str,
        opt_bs: int = 64,
        max_bs: int = 64,
        timing_iterations: int = 8,
) -> None:
    """Build one dynamic-batch TensorRT engine."""
    if not 1 <= opt_bs <= max_bs:
        raise ValueError("batch profile must satisfy 1 <= opt_bs <= max_bs")
    if timing_iterations <= 0:
        raise ValueError("timing_iterations must be positive")

    _, h, w = common.INPUT_SIZE
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    # Q/DQ nodes carry the quantization scales. Weak typing allows non-INT8
    # operations to use FP16 when both precision flags are enabled.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        for index in range(parser.num_errors):
            print(f"ONNX parse error {index}: {parser.get_error(index)}")
        sys.exit(1)
    print(
        f"ONNX parsed OK: {network.num_layers} layers, "
        f"inputs={network.num_inputs}, outputs={network.num_outputs}"
    )

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1, 3, h, w),
        opt=(opt_bs, 3, h, w),
        max=(max_bs, 3, h, w),
    )
    config.add_optimization_profile(profile)
    print(f"opt profile: min=1, opt={opt_bs}, max={max_bs}")

    if not builder.platform_has_fast_fp16:
        print("WARNING: platform does not report fast FP16")
    config.set_flag(trt.BuilderFlag.FP16)

    if precision == "int8":
        if not builder.platform_has_fast_int8:
            print("WARNING: platform does not report fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        print("INT8 build: weakly typed + FP16 + INT8 flags")
    else:
        print("FP16 build: weakly typed + FP16 flag")

    config.builder_optimization_level = 5
    config.avg_timing_iterations = timing_iterations
    print(
        "builder_optimization_level = 5 | "
        f"avg_timing_iterations = {timing_iterations}"
    )
    _load_timing_cache(config)

    print(f"building engine ({precision}) -> {engine_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: build_serialized_network returned None")
        sys.exit(1)

    _save_timing_cache(config)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"engine saved: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp16", "int8"], required=True)
    parser.add_argument(
        "--int8-variant",
        default=None,
        help="INT8 variant such as pct99p99, mse, or smoothquant.",
    )
    parser.add_argument(
        "--onnx-suffix",
        default=None,
        help="Override the model-prefixed ONNX suffix.",
    )
    parser.add_argument(
        "--engine-suffix",
        default=None,
        help="Override the model-prefixed engine suffix.",
    )
    parser.add_argument(
        "--opt-bs",
        type=int,
        default=64,
        help="Optimization profile batch size; match the throughput measurement.",
    )
    parser.add_argument("--max-bs", type=int, default=64)
    parser.add_argument("--timing-iterations", type=int, default=8)
    args = parser.parse_args()

    if args.precision == "fp16":
        default_onnx = "fp32_dynbatch_inline.onnx"
        default_engine = "fp16.engine"
    elif args.int8_variant:
        default_onnx = f"int8_qdq_{args.int8_variant}_inline.onnx"
        default_engine = f"int8_{args.int8_variant}.engine"
    else:
        default_onnx = "int8_qdq_inline.onnx"
        default_engine = "int8.engine"

    onnx_path = common.onnx_path(args.onnx_suffix or default_onnx)
    engine_path = common.engine_path(args.engine_suffix or default_engine)
    if not onnx_path.exists():
        print(f"ONNX not found: {onnx_path}")
        sys.exit(1)

    build_engine(
        onnx_path,
        engine_path,
        args.precision,
        opt_bs=args.opt_bs,
        max_bs=args.max_bs,
        timing_iterations=args.timing_iterations,
    )


if __name__ == "__main__":
    main()
