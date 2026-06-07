"""Stage 5: INT8 Post-Training Quantization with NVIDIA ModelOpt.

Applies per-tensor INT8 PTQ to ResNet50 using 500 calibration images,
prints the quantization summary, and exports a Q/DQ ONNX that TRT can
consume for INT8 inference with no further calibration.

Outputs:
  onnx/resnet50_int8_qdq.onnx          (may have external data)
  onnx/resnet50_int8_qdq_inline.onnx   (single-file, TRT-ready)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import onnx

import common
import modelopt.torch.quantization as mtq


def main():
    common.ONNX_DIR.mkdir(parents=True, exist_ok=True)

    model = common.build_model(exportable=True)
    calib_loader = common.build_calib_loader(batch_size=32, workers=4)

    print(f"calibration images: {len(calib_loader.dataset)}")

    def calibration_loop(m):
        with torch.inference_mode():
            for i, (images, _) in enumerate(calib_loader):
                if i >= 100:   # 100 × 32 = 3200 fwd passes through 500 images
                    break
                images = images.to(common.DEVICE, non_blocking=True)
                m(images)

    print("quantizing with INT8_DEFAULT_CFG...")
    quant_model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibration_loop)
    mtq.print_quant_summary(quant_model)

    # Export Q/DQ ONNX. The legacy TorchScript exporter segfaults when the
    # quantized model is on CUDA (modelopt CUDA extension issue). Move to CPU
    # first; the exported graph is device-agnostic.
    qdq_path = str(common.ONNX_DIR / "resnet50_int8_qdq.onnx")
    dummy = torch.randn(1, *common.INPUT_SIZE)   # CPU tensor

    print(f"\nmoving quantized model to CPU for ONNX export...")
    quant_model_cpu = quant_model.cpu()

    print(f"exporting Q/DQ ONNX -> {qdq_path}")
    torch.onnx.export(
        quant_model_cpu,
        dummy,
        qdq_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    # Load and save an inline (single-file) copy for TRT.
    # Skip onnx.checker.check_model — it times out on Q/DQ graphs.
    m_onnx = onnx.load(qdq_path, load_external_data=True)
    print(f"ONNX loaded | nodes={len(m_onnx.graph.node)}")

    inline_path = str(common.ONNX_DIR / "resnet50_int8_qdq_inline.onnx")
    onnx.save(m_onnx, inline_path)
    print(f"inline ONNX: {inline_path} ({Path(inline_path).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
