"""Stage 5: INT8 Post-Training Quantization with NVIDIA ModelOpt.

Applies per-tensor INT8 PTQ to the selected model (OPTIM_MODEL env var) using
500 calibration images, prints the quantization summary, and exports a Q/DQ
ONNX that TRT can consume for INT8 inference with no further calibration.

Calibration method is selected via --calib:
  max          (default) per-tensor MaxCalibrator — fast but sensitive to outliers
  percentile   two-phase: max first, then re-collect histogram + compute_amax
               at --percentile (default 99.9). Truncates outliers from hard-swish/
               depthwise activations that MaxCalibrator inflates.
  mse          per-tensor MSE calibrator — minimises round-trip error
  smoothquant  INT8_SMOOTHQUANT_CFG — migrates activation difficulty to weights

Outputs (under onnx/<model>/):
  <model>_int8_qdq_<calib>.onnx          (may have external data)
  <model>_int8_qdq_<calib>_inline.onnx   (single-file, TRT-ready)

  When --calib=max, the legacy names (no suffix) are used for backwards compat.
"""
import argparse
import copy
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import onnx

import common
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.quantization.calib import HistogramCalibrator


def _apply_percentile_calib(quant_model, calib_loader, percentile: float):
    """Phase-2: swap input_quantizers to HistogramCalibrator, recollect, patch amax.

    Weights keep their per-channel MaxCalibrator (already optimal).
    Only activation (input_quantizer) scales are replaced.
    """
    # Enable histogram collection on all input quantizers.
    patched = 0
    for n, m in quant_model.named_modules():
        if isinstance(m, TensorQuantizer) and "input_quantizer" in n:
            m._calibrator = HistogramCalibrator(num_bits=8, axis=None, torch_hist=True)
            m.enable_calib()
            m.disable_quant()
            patched += 1
    print(f"  histogram calib enabled on {patched} input quantizers")

    # Re-run forward on calibration data to fill histograms.
    with torch.inference_mode():
        for i, (images, _) in enumerate(calib_loader):
            if i >= 100:
                break
            images = images.to(common.DEVICE, non_blocking=True)
            quant_model(images)

    # Replace amax with percentile value and re-enable quantization.
    loaded = 0
    for n, m in quant_model.named_modules():
        if isinstance(m, TensorQuantizer) and isinstance(
            getattr(m, "_calibrator", None), HistogramCalibrator
        ):
            new_amax = m._calibrator.compute_amax("percentile", percentile=percentile)
            if new_amax is not None:
                m._amax = new_amax.to(m._amax.device) if m._amax is not None else new_amax
                loaded += 1
            m.enable_quant()
            m.disable_calib()
    print(f"  percentile-{percentile} amax loaded for {loaded} input quantizers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calib",
        choices=["max", "percentile", "mse", "smoothquant"],
        default="max",
        help="Activation calibration strategy (default: max)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.9,
        help="Percentile for --calib=percentile (default: 99.9)",
    )
    args = parser.parse_args()

    common.ONNX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"model: {common.MODEL_NAME}  input_size: {common.INPUT_SIZE}")
    print(f"calibration strategy: {args.calib}"
          + (f"  percentile={args.percentile}" if args.calib == "percentile" else ""))

    model = common.build_model(exportable=True)
    calib_loader = common.build_calib_loader(batch_size=32, workers=4)
    print(f"calibration images: {len(calib_loader.dataset)}")

    def calibration_loop(m):
        with torch.inference_mode():
            for i, (images, _) in enumerate(calib_loader):
                if i >= 100:
                    break
                images = images.to(common.DEVICE, non_blocking=True)
                m(images)

    # Choose quantization config.
    if args.calib == "smoothquant":
        quant_cfg = mtq.INT8_SMOOTHQUANT_CFG
        print("quantizing with INT8_SMOOTHQUANT_CFG...")
    elif args.calib == "mse":
        quant_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
        quant_cfg["algorithm"] = "mse"
        print("quantizing with INT8_DEFAULT_CFG + algorithm=mse...")
    else:
        # max or percentile both start with INT8_DEFAULT_CFG + max collect
        quant_cfg = mtq.INT8_DEFAULT_CFG
        print("quantizing with INT8_DEFAULT_CFG (max)...")

    quant_model = mtq.quantize(model, quant_cfg, forward_loop=calibration_loop)

    # Phase 2: percentile re-calibration of activation scales.
    if args.calib == "percentile":
        print(f"\nphase-2: replacing input_quantizer scales with percentile-{args.percentile}...")
        _apply_percentile_calib(quant_model, calib_loader, args.percentile)

    mtq.print_quant_summary(quant_model)

    # Derive output file suffix.  max keeps the legacy name (no suffix) for
    # backwards compatibility so existing engines remain valid.
    if args.calib == "max":
        suffix = "int8_qdq"
    elif args.calib == "percentile":
        pct_tag = str(args.percentile).replace(".", "p")
        suffix = f"int8_qdq_pct{pct_tag}"
    else:
        suffix = f"int8_qdq_{args.calib}"

    # Export Q/DQ ONNX on CPU (modelopt CUDA extension segfaults on GPU).
    qdq_p = str(common.onnx_path(f"{suffix}.onnx"))
    dummy = torch.randn(1, *common.INPUT_SIZE)   # CPU tensor

    print(f"\nmoving quantized model to CPU for ONNX export...")
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
    m_onnx = onnx.load(qdq_p, load_external_data=True)
    print(f"ONNX loaded | nodes={len(m_onnx.graph.node)}")

    inline_p = str(common.onnx_path(f"{suffix}_inline.onnx"))
    onnx.save(m_onnx, inline_p)
    print(f"inline ONNX: {inline_p} ({Path(inline_p).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
