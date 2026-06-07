"""Stage 3: Export ResNet50 FP32 ONNX.

Uses timm exportable=True to disable non-traceable ops, then exports via
torch.onnx.export with opset 17 and dynamo=True (Torch 2.x dynamo exporter).

Outputs:
  onnx/resnet50_fp32.onnx
"""
import sys
from pathlib import Path

# Ensure repo root on path when run from my_optim/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import timm
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import common


def main():
    common.ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = str(common.ONNX_DIR / "resnet50_fp32.onnx")

    print("building model (exportable=True)...")
    model = common.build_model(exportable=True)

    # dynamo export requires CPU dummy input to avoid GPU/ONNX incompatibility
    # in older ONNX exporter code paths; with torch 2.10 dynamo exporter we
    # can pass a CUDA tensor directly.
    dummy = torch.randn(1, *common.INPUT_SIZE, device=common.DEVICE)

    print(f"exporting -> {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=True,
    )

    # Basic sanity check.
    import onnx
    m = onnx.load(onnx_path, load_external_data=True)
    onnx.checker.check_model(m)
    print(f"ONNX check OK | nodes={len(m.graph.node)} | output: {onnx_path}")

    # Save a self-contained inline copy for TRT (which cannot handle external data files).
    inline_path = str(common.ONNX_DIR / "resnet50_fp32_inline.onnx")
    onnx.save(m, inline_path)
    import os
    print(f"Inline copy: {inline_path} ({os.path.getsize(inline_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
