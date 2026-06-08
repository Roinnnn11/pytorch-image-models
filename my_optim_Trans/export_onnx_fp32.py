"""Stage 2: Export FP32 ONNX for the selected Transformer model.

Uses timm exportable=True (disables non-traceable ops such as the scaled-dot-
product-attention fallback) then exports via the legacy TorchScript exporter
(dynamo=False) with opset 17 and a dynamic batch axis. This path is verified
to work for both vit_base_patch16_224 and swin_tiny_patch4_window7_224, and
produces an ONNX that TensorRT 10 can parse without unsupported operators.

Note on Swin: window partition, shifted-window attention and relative-position
bias all trace correctly with exportable=True + opset 17 + legacy exporter.
The dynamo exporter bakes in a fixed shape; the legacy exporter honours
dynamic_axes, giving TRT a real dynamic batch dimension.

Outputs (under onnx/<model>/):
  <model>_fp32_dynbatch.onnx         (dynamic batch, may have external data)
  <model>_fp32_dynbatch_inline.onnx  (single-file, TRT-ready, used by FP16 build)

Usage:
    OPTIM_MODEL=vit_base_patch16_224 python export_onnx_fp32.py
    OPTIM_MODEL=swin_tiny_patch4_window7_224 python export_onnx_fp32.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import onnx

import common


def main():
    common.ONNX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"model: {common.MODEL_NAME}  input_size: {common.INPUT_SIZE}")
    print("building model (exportable=True)...")
    model = common.build_model(exportable=True)

    # Legacy TorchScript exporter needs CPU input for Swin (and is generally
    # more robust for export; dynamo=True bakes in fixed shapes).
    model_cpu = model.cpu()
    dummy_cpu = torch.randn(1, *common.INPUT_SIZE)

    dyn_p = str(common.onnx_path("fp32_dynbatch.onnx"))
    print(f"\nexporting dynamic-batch ONNX -> {dyn_p}")
    torch.onnx.export(
        model_cpu,
        dummy_cpu,
        dyn_p,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    m_dyn = onnx.load(dyn_p, load_external_data=True)
    print(f"ONNX loaded | nodes={len(m_dyn.graph.node)} | "
          f"size={os.path.getsize(dyn_p)/1e6:.1f} MB")

    # Verify TRT can parse (catches unsupported ops early).
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.ERROR)
        b = trt.Builder(logger)
        net = b.create_network(0)
        parser = trt.OnnxParser(net, logger)
        ok = parser.parse(open(dyn_p, "rb").read())
        if ok:
            print(f"TRT parse OK: {net.num_layers} layers")
        else:
            for i in range(parser.num_errors):
                print(f"TRT parse error {i}: {parser.get_error(i)}")
    except ImportError:
        print("tensorrt not available, skipping TRT parse check")

    # Save a single-file inline copy (no external data) that TRT can load
    # without needing the .data sidecar file.
    inline_p = str(common.onnx_path("fp32_dynbatch_inline.onnx"))
    onnx.save(m_dyn, inline_p)
    print(f"inline copy: {inline_p} ({os.path.getsize(inline_p)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
