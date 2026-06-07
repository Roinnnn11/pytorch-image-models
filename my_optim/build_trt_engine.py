"""Build TensorRT engines from ONNX using the Python TRT API.

trtexec binary has a CUDA runtime/driver mismatch on this host; the Python
tensorrt package (installed in deepburst env) runs against the correct
CUDA 12.8 runtime and works fine.

Usage:
    python build_trt_engine.py --precision fp16   # resnet50_fp16.engine
    python build_trt_engine.py --precision int8   # resnet50_int8.engine (Q/DQ ONNX)

FP16 uses the legacy weakly-typed network + kFP16 flag (equivalent to trtexec --fp16).
INT8 uses STRONGLY_TYPED mode to consume the Q/DQ nodes from ModelOpt PTQ.
These modes are mutually exclusive in TRT 10.
"""
import argparse
import sys
from pathlib import Path

import tensorrt as trt


def build_engine(onnx_path: Path, engine_path: Path, precision: str):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    # TRT 10 rule: STRONGLY_TYPED + kFP16 flag is forbidden.
    # - FP16: use weakly-typed (legacy) network so kFP16 flag is accepted.
    # - INT8 (Q/DQ): use STRONGLY_TYPED so TRT consumes QuantizeLinear nodes.
    if precision == "int8":
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    else:
        network_flags = 0  # weakly-typed / legacy

    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = onnx_path.read_bytes()
    ok = parser.parse(onnx_bytes)
    if not ok:
        for i in range(parser.num_errors):
            print(f"ONNX parse error {i}: {parser.get_error(i)}")
        sys.exit(1)
    print(f"ONNX parsed OK: {network.num_layers} layers, inputs: "
          f"{network.num_inputs}, outputs: {network.num_outputs}")

    config = builder.create_builder_config()

    # Add an optimization profile to support batch sizes 1–64.
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1, 3, 224, 224),
        opt=(32, 3, 224, 224),
        max=(64, 3, 224, 224),
    )
    config.add_optimization_profile(profile)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("WARNING: platform does not advertise fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)

    elif precision == "int8":
        # Q/DQ ONNX: scale factors are embedded as QuantizeLinear/DequantizeLinear
        # nodes; STRONGLY_TYPED mode picks them up automatically. No extra flag needed.
        pass

    print(f"building engine ({precision}) -> {engine_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: build_serialized_network returned None")
        sys.exit(1)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"engine saved: {engine_path} ({engine_path.stat().st_size/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp16", "int8"], required=True)
    args = parser.parse_args()

    onnx_dir = Path(__file__).parent / "onnx"
    engine_dir = Path(__file__).parent / "engines"

    if args.precision == "fp16":
        # dynbatch_inline: exported with dynamic_axes so TRT profile works.
        onnx_path = onnx_dir / "resnet50_fp32_dynbatch_inline.onnx"
        engine_path = engine_dir / "resnet50_fp16.engine"
    else:
        onnx_path = onnx_dir / "resnet50_int8_qdq_inline.onnx"
        engine_path = engine_dir / "resnet50_int8.engine"

    if not onnx_path.exists():
        print(f"ONNX not found: {onnx_path}")
        sys.exit(1)

    build_engine(onnx_path, engine_path, args.precision)


if __name__ == "__main__":
    main()
