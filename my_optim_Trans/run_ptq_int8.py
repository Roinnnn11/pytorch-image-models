"""Stage 4: INT8 Post-Training Quantization with NVIDIA ModelOpt.

Applies INT8_DEFAULT_CFG PTQ to the selected Transformer model (OPTIM_MODEL
env var) using 500 calibration images, then exports a Q/DQ ONNX that TRT
can consume in STRONGLY_TYPED mode for INT8 inference.

For Transformer models, INT8_DEFAULT_CFG quantizes Linear / MatMul / Conv
while typically leaving LayerNorm and Softmax at FP16 (they are not weight
nodes). This is safer than full INT8 for attention-based architectures.

If top1 drops more than 2% vs FP32, consider using SMOOTH_QUANT_CFG (or
manually falling back sensitive layers to FP16 — see run_sensitive_fallback.py).

Outputs (under onnx/<model>/):
  <model>_int8_qdq.onnx          (may have external data)
  <model>_int8_qdq_inline.onnx   (single-file, TRT-ready)

Usage:
    OPTIM_MODEL=vit_base_patch16_224 python run_ptq_int8.py
    OPTIM_MODEL=swin_tiny_patch4_window7_224 python run_ptq_int8.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import onnx

import common
import modelopt.torch.quantization as mtq


def main():
    common.ONNX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"model: {common.MODEL_NAME}  input_size: {common.INPUT_SIZE}")
    model = common.build_model(exportable=True)

    # 16 images per batch × 31 batches = 496 images from 500-image calib set.
    calib_loader = common.build_calib_loader(batch_size=16, workers=4)
    print(f"calibration images: {len(calib_loader.dataset)}")

    def calibration_loop(m):
        with torch.inference_mode():
            for i, (images, _) in enumerate(calib_loader):
                images = images.to(common.DEVICE, non_blocking=True)
                m(images)

    print("quantizing with INT8_DEFAULT_CFG...")
    quant_model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibration_loop)
    mtq.print_quant_summary(quant_model)

    # The legacy TorchScript exporter can segfault when the quantized model is
    # on CUDA (modelopt CUDA extension issue). Move to CPU first; the exported
    # graph is device-agnostic.
    qdq_p = str(common.onnx_path("int8_qdq.onnx"))
    dummy = torch.randn(1, *common.INPUT_SIZE)   # CPU tensor

    print("moving quantized model to CPU for ONNX export...")
    quant_model_cpu = quant_model.cpu()

    print(f"exporting Q/DQ ONNX -> {qdq_p}")
    torch.onnx.export(
        quant_model_cpu,
        dummy,
        qdq_p,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    # Load and save an inline (single-file) copy for TRT.
    # Skip onnx.checker.check_model — it can time out on Q/DQ graphs.
    m_onnx = onnx.load(qdq_p, load_external_data=True)
    print(f"ONNX loaded | nodes={len(m_onnx.graph.node)}")

    inline_p = str(common.onnx_path("int8_qdq_inline.onnx"))
    onnx.save(m_onnx, inline_p)
    print(f"inline ONNX: {inline_p} ({Path(inline_p).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
