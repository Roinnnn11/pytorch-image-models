"""Compatibility entry point for the corrected TensorRT FP16 builder."""

import common
from build_trt_engine import build_engine


def main() -> None:
    onnx_path = common.onnx_path("fp32_dynbatch_inline.onnx")
    engine_path = common.engine_path("fp16.engine")
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX not found: {onnx_path}. Run export_onnx_fp32.py first."
        )
    build_engine(
        onnx_path,
        engine_path,
        precision="fp16",
        opt_bs=64,
        max_bs=64,
    )


if __name__ == "__main__":
    main()
