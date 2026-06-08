"""Optional Stage 7: Sensitive-layer FP16 fallback for INT8.

Run this only if run_summary.py reports INT8 top1 drop > 2%.

Strategy: measure which modules lose the most accuracy when quantized by
temporarily excluding them from INT8 calibration (keep as FP16). Build a
revised Q/DQ ONNX with those sensitive modules at FP16 and re-export.

The most commonly sensitive Transformer layers are:
  patch embedding
  attention qkv projection
  attention output projection
  MLP fc1 / fc2
  final classifier head

Usage:
    OPTIM_MODEL=vit_base_patch16_224 python run_sensitive_fallback.py
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


# Candidate layer name fragments to keep in FP16.
# Ordered by typical sensitivity for ViT / Swin (most → least).
SENSITIVE_FRAGMENTS = [
    "patch_embed",          # patch embedding conv
    "head",                 # classifier head
    "attn.proj",            # attention output projection
    "attn.qkv",             # QKV
    "mlp.fc1",              # MLP first linear
    "mlp.fc2",              # MLP second linear
    "norm",                 # LayerNorm (usually not quantized by default, but explicit)
]


def make_fp16_config(excluded_fragments: list[str]) -> dict:
    """Build an INT8 quant config that leaves matching layers at FP16."""
    base_cfg = mtq.INT8_DEFAULT_CFG.copy()

    def override_fn(module_name: str, _module):
        for frag in excluded_fragments:
            if frag in module_name:
                return {"*weight_quantizer": {"enable": False},
                        "*input_quantizer": {"enable": False}}
        return None

    base_cfg["quant_cfg"] = {
        **base_cfg.get("quant_cfg", {}),
        "override_fn": override_fn,
    }
    return base_cfg


def evaluate_config(excluded_fragments: list[str], n_batches: int = 10):
    """Calibrate, quantize, and quick-evaluate on the first n_batches."""
    model = common.build_model(exportable=True)
    calib_loader = common.build_calib_loader(batch_size=16, workers=4)
    val_loader, _ = common.build_val_loader(batch_size=common.EVAL_BATCH_SIZE, workers=4)

    cfg = make_fp16_config(excluded_fragments)

    def calibration_loop(m):
        with torch.inference_mode():
            for images, _ in calib_loader:
                images = images.to(common.DEVICE, non_blocking=True)
                m(images)

    quant_model = mtq.quantize(model, cfg, forward_loop=calibration_loop)

    def infer(x):
        with torch.inference_mode():
            return quant_model(x)

    result = common.evaluate_accuracy(infer, val_loader, n_batches, desc="sensitive_eval")
    return result["top1"]


def main():
    import json

    # Load the FP32 top1 reference.
    baseline_path = common.RESULTS_DIR / "baseline.json"
    if not baseline_path.exists():
        print(f"baseline.json not found: {baseline_path}")
        sys.exit(1)
    with open(baseline_path) as f:
        baseline = json.load(f)
    fp32_top1 = baseline["fp32"]["top1"]

    # Load INT8 default top1.
    int8_path = common.RESULTS_DIR / "trt_int8.json"
    if not int8_path.exists():
        print(f"trt_int8.json not found: {int8_path}")
        sys.exit(1)
    with open(int8_path) as f:
        int8_result = json.load(f)
    int8_top1 = int8_result["top1"]

    drop = fp32_top1 - int8_top1
    print(f"model: {common.MODEL_NAME}")
    print(f"FP32 top1: {fp32_top1:.3f}%  INT8 top1: {int8_top1:.3f}%  drop: {drop:.3f}%")

    if drop <= 2.0:
        print("INT8 drop is within 2%, no fallback needed.")
        return

    print(f"\nDrop {drop:.2f}% > 2% — probing sensitive layers (10-batch quick eval)...")

    # Greedy forward pass: add layers back to FP16 until drop < 1%.
    kept_fp16 = []
    best_top1 = int8_top1

    for frag in SENSITIVE_FRAGMENTS:
        candidate = kept_fp16 + [frag]
        top1 = evaluate_config(candidate, n_batches=10)
        delta = fp32_top1 - top1
        print(f"  keep_fp16={candidate}  quick top1={top1:.2f}%  drop={delta:.2f}%")
        if top1 > best_top1:
            best_top1 = top1
            kept_fp16 = candidate
        if delta < 1.0:
            print(f"  → drop below 1%, stopping greedy search")
            break

    print(f"\nFinal FP16 fallback set: {kept_fp16}")
    print(f"Expected quick top1: {best_top1:.2f}%  (drop from FP32: {fp32_top1-best_top1:.2f}%)")

    # Export final mixed-precision Q/DQ ONNX.
    print("\nExporting mixed-precision Q/DQ ONNX...")
    model = common.build_model(exportable=True)
    calib_loader = common.build_calib_loader(batch_size=16, workers=4)
    cfg = make_fp16_config(kept_fp16)

    def calibration_loop(m):
        with torch.inference_mode():
            for images, _ in calib_loader:
                images = images.to(common.DEVICE, non_blocking=True)
                m(images)

    quant_model = mtq.quantize(model, cfg, forward_loop=calibration_loop)
    mtq.print_quant_summary(quant_model)

    qdq_p = str(common.onnx_path("int8_qdq_mixed.onnx"))
    dummy = torch.randn(1, *common.INPUT_SIZE)
    quant_model_cpu = quant_model.cpu()

    torch.onnx.export(
        quant_model_cpu, dummy, qdq_p,
        input_names=["input"], output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    m_onnx = onnx.load(qdq_p, load_external_data=True)
    inline_p = str(common.onnx_path("int8_qdq_mixed_inline.onnx"))
    onnx.save(m_onnx, inline_p)
    print(f"mixed-precision ONNX: {inline_p} ({Path(inline_p).stat().st_size/1e6:.1f} MB)")
    print(f"Next: build_trt_engine.py --precision int8 (pointing at mixed ONNX)")
    print(f"      or rename {inline_p} to replace int8_qdq_inline.onnx")


if __name__ == "__main__":
    main()
