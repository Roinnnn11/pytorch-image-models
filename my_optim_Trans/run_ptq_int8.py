"""Export named ViT INT8 Q/DQ candidates with NVIDIA ModelOpt.

Each candidate is calibrated once. The resulting Q/DQ ONNX is reused to build
the fixed batch-size TensorRT engines, so batch size never changes the scales.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))

import modelopt.torch.quantization as mtq
import onnx
import torch
from torch.utils.data import DataLoader, Subset

import common
from my_optim.experiment_utils import class_balanced_indices


def make_quant_config(calib: str, smoothquant_alpha: float) -> tuple[dict, str]:
    """Return a detached ModelOpt config and a filesystem-safe candidate name."""
    if calib == "max":
        return copy.deepcopy(mtq.INT8_DEFAULT_CFG), "max"
    if calib == "mse":
        config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
        config["algorithm"] = "mse"
        return config, "mse"
    if not 0.0 <= smoothquant_alpha <= 1.0:
        raise ValueError("smoothquant alpha must be in [0, 1]")
    config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
    config["algorithm"] = {"method": "smoothquant", "alpha": smoothquant_alpha}
    alpha_tag = f"{smoothquant_alpha:g}".replace(".", "p")
    return config, f"smoothquant_a{alpha_tag}"


def build_balanced_calib_loader(sample_size: int, seed: int, batch_size: int, workers: int):
    """Use a deterministic class-balanced ImageNet validation subset."""
    _, dataset = common.build_val_loader(batch_size=batch_size, workers=0)
    sample_size = min(sample_size, len(dataset))
    indices = class_balanced_indices(dataset.targets, sample_size, seed)
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    ), indices


def export_qdq(quant_model, candidate: str) -> tuple[Path, Path]:
    """Export a dynamic-batch Q/DQ model and a self-contained inline copy."""
    stem = f"int8_qdq_{candidate}"
    qdq_path = common.onnx_path(f"{stem}.onnx")
    inline_path = common.onnx_path(f"{stem}_inline.onnx")
    dummy = torch.randn(1, *common.INPUT_SIZE)
    quant_model_cpu = quant_model.cpu()
    torch.onnx.export(
        quant_model_cpu,
        dummy,
        str(qdq_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    model_onnx = onnx.load(str(qdq_path), load_external_data=True)
    onnx.save(model_onnx, str(inline_path))
    return qdq_path, inline_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", choices=["max", "mse", "smoothquant"], default="max")
    parser.add_argument("--smoothquant-alpha", type=float, default=0.5)
    parser.add_argument("--calib-samples", type=int, default=3000)
    parser.add_argument("--calib-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.calib_samples <= 0 or args.calib_batch_size <= 0:
        parser.error("calibration sizes must be positive")
    if common.MODEL_NAME != "vit_base_patch16_224":
        parser.error("this optimized matrix currently targets vit_base_patch16_224 only")

    quant_config, candidate = make_quant_config(args.calib, args.smoothquant_alpha)
    common.ONNX_DIR.mkdir(parents=True, exist_ok=True)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = common.build_model(exportable=True)
    calib_loader, indices = build_balanced_calib_loader(
        args.calib_samples,
        args.seed,
        args.calib_batch_size,
        args.workers,
    )

    def calibration_loop(candidate_model):
        with torch.inference_mode():
            for images, _ in calib_loader:
                candidate_model(images.to(common.DEVICE, non_blocking=True))

    print(f"candidate={candidate} calibration_images={len(indices)} seed={args.seed}")
    quant_model = mtq.quantize(model, quant_config, forward_loop=calibration_loop)
    mtq.print_quant_summary(quant_model)
    qdq_path, inline_path = export_qdq(quant_model, candidate)
    metadata = {
        "model": common.MODEL_NAME,
        "candidate": candidate,
        "calibration": args.calib,
        "smoothquant_alpha": args.smoothquant_alpha if args.calib == "smoothquant" else None,
        "calibration_samples": len(indices),
        "seed": args.seed,
        "qdq_onnx": str(qdq_path),
        "inline_onnx": str(inline_path),
    }
    metadata_path = common.RESULTS_DIR / f"ptq_{candidate}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"inline ONNX: {inline_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
