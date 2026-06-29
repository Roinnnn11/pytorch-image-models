"""Evaluate dynamic or fixed-batch TensorRT ViT engines."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import tensorrt as trt
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))
import common


class LayerProfiler(trt.IProfiler):
    """Collect TensorRT per-layer time during a separate profiling run."""

    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layers = []

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self.layers.append({"layer": layer_name, "time_ms": float(ms)})


class TRTInferencer:
    """Asynchronous FP32-I/O TensorRT runner with output-buffer reuse."""

    def __init__(self, engine_path: str, profile_layers: bool = False):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as file:
            self.engine = runtime.deserialize_cuda_engine(file.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self._output_cache = {}
        self._last_shape = None
        self._graph = None
        self._graph_input = None
        self._graph_output = None
        self.profiler = LayerProfiler() if profile_layers else None
        if self.profiler is not None:
            self.context.profiler = self.profiler

    def _enqueue(self, images: torch.Tensor) -> torch.Tensor:
        shape = tuple(images.shape)
        if shape != self._last_shape:
            if not self.context.set_input_shape(self.input_name, shape):
                raise ValueError(f"engine does not support input shape {shape}")
            self._last_shape = shape
        key = (shape[0], images.device.index)
        out = self._output_cache.get(key)
        if out is None:
            out = torch.empty(shape[0], 1000, dtype=torch.float32, device=images.device)
            self._output_cache[key] = out
        self.context.set_tensor_address(self.input_name, images.data_ptr())
        self.context.set_tensor_address(self.output_name, out.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        return out

    def enable_cuda_graph(self, batch_size: int) -> None:
        """Capture fixed-buffer inference; callers still copy into the stable input."""
        self._graph_input = torch.empty(batch_size, *common.INPUT_SIZE, device="cuda")
        self._graph_output = self._enqueue(self._graph_input)
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._graph_output = self._enqueue(self._graph_input)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if self._graph is None:
            return self._enqueue(images)
        if images.shape != self._graph_input.shape:
            raise ValueError("CUDA Graph runner only supports its captured fixed batch")
        self._graph_input.copy_(images)
        self._graph.replay()
        return self._graph_output


def pad_fixed_batch(images: torch.Tensor, fixed_bs: int) -> tuple[torch.Tensor, int]:
    """Pad an undersized final batch without dropping real validation images."""
    actual_bs = images.shape[0]
    if actual_bs > fixed_bs:
        raise ValueError(f"actual batch {actual_bs} exceeds fixed engine batch {fixed_bs}")
    if actual_bs == fixed_bs:
        return images, actual_bs
    padded = images.new_zeros((fixed_bs, *images.shape[1:]))
    padded[:actual_bs].copy_(images)
    return padded, actual_bs


@torch.inference_mode()
def evaluate_fixed_accuracy(infer, loader, fixed_bs: int, max_batches=None, desc="eval") -> dict:
    """Evaluate every image through a fixed engine, padding only the final batch."""
    top1 = top5 = evaluated_samples = 0
    started = time.perf_counter()
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(common.DEVICE, non_blocking=True)
        targets = targets.to(common.DEVICE, non_blocking=True)
        padded, actual_bs = pad_fixed_batch(images, fixed_bs)
        logits = infer(padded)[:actual_bs]
        torch.cuda.synchronize()
        prediction = logits.topk(5, dim=1, largest=True, sorted=True).indices
        correct = prediction.eq(targets.view(-1, 1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        evaluated_samples += actual_bs
        if batch_index % 50 == 0:
            print(f"[{desc}] batch={batch_index} n={evaluated_samples}", flush=True)
    elapsed = time.perf_counter() - started
    if evaluated_samples == 0:
        raise ValueError("evaluation loader produced no samples")
    return {
        "top1": 100.0 * top1 / evaluated_samples,
        "top5": 100.0 * top5 / evaluated_samples,
        "n": evaluated_samples,
        "evaluated_samples": evaluated_samples,
        "eval_seconds": elapsed,
        "e2e_throughput_img_s": evaluated_samples / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp16", "int8"], required=True)
    parser.add_argument("--batch-size", type=int, default=common.EVAL_BATCH_SIZE)
    parser.add_argument("--fixed-bs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--engine-suffix", default=None)
    parser.add_argument("--result-tag", default=None)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--profile-layers", action="store_true")
    args = parser.parse_args()
    if args.fixed_bs is not None and args.fixed_bs <= 0:
        parser.error("--fixed-bs must be positive")
    if args.fixed_bs is not None:
        args.batch_size = args.fixed_bs

    engine_suffix = args.engine_suffix or f"{args.precision}.engine"
    tag = args.result_tag or f"trt_{Path(engine_suffix).stem}"
    engine_path = common.engine_path(engine_suffix)
    if not engine_path.exists():
        parser.error(f"engine not found: {engine_path}")

    infer = TRTInferencer(str(engine_path), profile_layers=args.profile_layers)
    graph_error = None
    if args.use_cuda_graph:
        if args.fixed_bs is None:
            parser.error("--use-cuda-graph requires --fixed-bs")
        try:
            infer.enable_cuda_graph(args.fixed_bs)
        except (RuntimeError, ValueError) as error:
            graph_error = str(error)
            print(f"CUDA Graph unavailable; using enqueue_v3: {error}")

    loader, _ = common.build_val_loader(args.batch_size, args.workers)
    if args.fixed_bs is None:
        accuracy = common.evaluate_accuracy(infer, loader, args.max_batches, desc=tag)
        accuracy["evaluated_samples"] = accuracy["n"]
        accuracy["e2e_throughput_img_s"] = accuracy["n"] / accuracy["eval_seconds"]
        latency_batches = common.LATENCY_BATCHES
    else:
        accuracy = evaluate_fixed_accuracy(
            infer,
            loader,
            args.fixed_bs,
            args.max_batches,
            desc=tag,
        )
        latency_batches = (args.fixed_bs,)

    latency = {}
    for batch_size in latency_batches:
        latency[f"bs{batch_size}"] = common.measure_latency(infer, batch_size)
        metric = latency[f"bs{batch_size}"]
        print(
            f"bs={batch_size}: {metric['latency_ms_mean']:.3f} ms "
            f"({metric['throughput_img_s']:.0f} img/s)"
        )

    result = {
        "tag": tag,
        "engine": str(engine_path),
        "fixed_batch_size": args.fixed_bs,
        "cuda_graph_requested": args.use_cuda_graph,
        "cuda_graph_enabled": args.use_cuda_graph and graph_error is None,
        "cuda_graph_error": graph_error,
        **accuracy,
        "latency": latency,
    }
    if infer.profiler is not None:
        result["layer_profile"] = infer.profiler.layers
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = common.RESULTS_DIR / f"{tag}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
