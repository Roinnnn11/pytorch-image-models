"""TensorRT engine inference: accuracy eval + latency benchmark.

Loads a serialized .engine file, runs it over the ImageNet val set (or a
subset), and records top1/top5 and per-batch latency. Used for both the
FP16 and INT8 engines.

Usage:
    python run_trt_eval.py --engine engines/resnet50_fp16.engine --tag trt_fp16
    python run_trt_eval.py --engine engines/resnet50_int8.engine --tag trt_int8
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt

# Ensure timm is importable from the parent repo directory.
sys.path.insert(0, str(Path(__file__).parent.parent))
import common


class TRTInferencer:
    """Thin wrapper around a TRT engine for synchronous FP32-I/O inference."""

    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate device buffers for input and output.
        # TRT 10 API: use tensor names, not binding indices.
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """images: NCHW float32 CUDA tensor. Returns logits float32 CUDA tensor."""
        bs = images.shape[0]
        n_classes = 1000
        out = torch.empty(bs, n_classes, dtype=torch.float32, device="cuda")

        self.context.set_input_shape(self.input_name, images.shape)
        self.context.set_tensor_address(self.input_name, images.data_ptr())
        self.context.set_tensor_address(self.output_name, out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="path to .engine file")
    parser.add_argument("--tag", required=True, help="result key prefix")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    engine_path = Path(args.engine)
    if not engine_path.is_absolute():
        engine_path = Path(__file__).parent / engine_path

    print(f"loading engine: {engine_path}")
    infer = TRTInferencer(str(engine_path))

    loader, dataset = common.build_val_loader(args.batch_size, args.workers)

    print(f"\n=== {args.tag} accuracy ===")
    acc = common.evaluate_accuracy(infer, loader, args.max_batches, desc=args.tag)
    print(acc)

    print(f"\n=== {args.tag} latency ===")
    latency = {}
    for bs in (1, 64):
        latency[f"bs{bs}"] = common.measure_latency(infer, bs)
        print(f"bs={bs}: {latency[f'bs{bs}']['latency_ms_mean']:.3f} ms  "
              f"({latency[f'bs{bs}']['throughput_img_s']:.0f} img/s)")

    result = {"tag": args.tag, "engine": str(engine_path), **acc, "latency": latency}

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / f"{args.tag}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
