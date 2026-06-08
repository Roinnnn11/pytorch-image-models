"""Stage 5: TensorRT engine inference — accuracy eval + latency benchmark.

Loads a serialized .engine file, runs it over the ImageNet val set, and
records top1/top5 and per-batch latency. Used for both FP16 and INT8 engines.

Model is selected via OPTIM_MODEL env var (default: vit_base_patch16_224).

Usage:
    OPTIM_MODEL=vit_base_patch16_224 python run_trt_eval.py --precision fp16
    OPTIM_MODEL=vit_base_patch16_224 python run_trt_eval.py --precision int8
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import tensorrt as trt

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
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
        # TRT 10 API: use tensor names, not binding indices.
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """images: NCHW float32 CUDA tensor. Returns logits float32 CUDA tensor."""
        bs = images.shape[0]
        out = torch.empty(bs, 1000, dtype=torch.float32, device="cuda")
        self.context.set_input_shape(self.input_name, images.shape)
        self.context.set_tensor_address(self.input_name, images.data_ptr())
        self.context.set_tensor_address(self.output_name, out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp16", "int8"], required=True)
    parser.add_argument("--batch-size", type=int, default=common.EVAL_BATCH_SIZE,
                        help="eval and throughput batch size (default=32)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    tag = f"trt_{args.precision}"
    eng_p = common.engine_path(f"{args.precision}.engine")

    if not eng_p.exists():
        print(f"engine not found: {eng_p}")
        sys.exit(1)

    print(f"model: {common.MODEL_NAME}  engine: {eng_p}")
    infer = TRTInferencer(str(eng_p))
    loader, dataset = common.build_val_loader(args.batch_size, args.workers)

    print(f"\n=== {tag} accuracy ===")
    acc = common.evaluate_accuracy(infer, loader, args.max_batches, desc=tag)
    print(acc)

    print(f"\n=== {tag} latency ===")
    latency = {}
    for bs in common.LATENCY_BATCHES:
        latency[f"bs{bs}"] = common.measure_latency(infer, bs)
        print(f"bs={bs}: {latency[f'bs{bs}']['latency_ms_mean']:.3f} ms  "
              f"({latency[f'bs{bs}']['throughput_img_s']:.0f} img/s)")

    result = {"tag": tag, "engine": str(eng_p), **acc, "latency": latency}

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / f"{tag}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
