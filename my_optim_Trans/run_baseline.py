"""Stage 1: FP32 baseline and Torch FP16 (autocast) accuracy + latency.

Model is selected via OPTIM_MODEL env var (default: vit_base_patch16_224).

Establishes the reference numbers every later stage is compared against:
  - FP32 top1/top5 over the full val set
  - FP32 latency at bs=1 and bs=EVAL_BATCH_SIZE
  - Torch FP16 (inference_mode + autocast) top1/top5 and latency

Writes results/<model>/baseline.json.

Usage:
    OPTIM_MODEL=vit_base_patch16_224 python run_baseline.py
    OPTIM_MODEL=swin_tiny_patch4_window7_224 python run_baseline.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=common.EVAL_BATCH_SIZE,
                        help="eval and throughput batch size (default from common.EVAL_BATCH_SIZE=32)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="limit eval batches (None = full val set)")
    args = parser.parse_args()

    cfg = common.get_data_config()
    model = common.build_model()
    loader, dataset = common.build_val_loader(args.batch_size, args.workers, cfg)

    print(f"GPU: {common.gpu_name()}")
    print(f"model: {common.MODEL_NAME}  val images: {len(dataset)} | "
          f"input_size: {cfg['input_size']} | "
          f"crop_pct: {cfg['crop_pct']} | interp: {cfg['interpolation']}")

    results = {
        "model": common.MODEL_NAME,
        "gpu": common.gpu_name(),
        "input_size": list(cfg["input_size"]),
        "batch_size_eval": args.batch_size,
        "n_val": len(dataset),
        "data_config": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in cfg.items()},
    }

    # ---- FP32 accuracy ----
    def fp32_infer(x):
        return model(x)

    print("\n=== FP32 accuracy ===")
    results["fp32"] = common.evaluate_accuracy(
        fp32_infer, loader, args.max_batches, desc="fp32")
    print(results["fp32"])

    # ---- Torch FP16 (autocast) accuracy ----
    def fp16_infer(x):
        with torch.autocast("cuda", dtype=torch.float16):
            return model(x)

    print("\n=== Torch FP16 (autocast) accuracy ===")
    results["torch_fp16"] = common.evaluate_accuracy(
        fp16_infer, loader, args.max_batches, desc="fp16")
    print(results["torch_fp16"])

    # ---- Latency: FP32 and FP16 at bs=1 and bs=EVAL_BATCH_SIZE ----
    print("\n=== latency ===")
    results["latency"] = {}
    for bs in common.LATENCY_BATCHES:
        results["latency"][f"fp32_bs{bs}"] = common.measure_latency(fp32_infer, bs)
        results["latency"][f"fp16_bs{bs}"] = common.measure_latency(fp16_infer, bs)
        print(f"bs={bs} fp32: {results['latency'][f'fp32_bs{bs}']['latency_ms_mean']:.3f} ms | "
              f"fp16: {results['latency'][f'fp16_bs{bs}']['latency_ms_mean']:.3f} ms")

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "baseline.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
