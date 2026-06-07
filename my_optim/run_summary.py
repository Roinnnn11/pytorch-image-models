"""Stage 7: Final comparison report — FP32 vs Torch FP16 vs TRT FP16 vs TRT INT8.

Reads the JSON results files from every stage and computes:
  - top1 / top5 accuracy drop vs FP32
  - latency and throughput at bs=1 and bs=64
  - speedup ratios relative to FP32
  - logits MSE and cosine similarity (FP32 vs each precision)

Writes results/summary.json and prints a markdown table.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import common

RESULTS = common.RESULTS_DIR


def load(name: str) -> dict:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def compute_logit_similarity(fp32_model, other_infer_fn, n_batches=10):
    """Compare logits on the same val images (FP32 reference vs other)."""
    loader, _ = common.build_val_loader(batch_size=64, workers=4)
    mses, cosines = [], []
    with torch.inference_mode():
        for i, (imgs, _) in enumerate(loader):
            if i >= n_batches:
                break
            imgs = imgs.to(common.DEVICE, non_blocking=True)
            ref = fp32_model(imgs).float()
            other = other_infer_fn(imgs)
            if not torch.is_tensor(other):
                other = torch.as_tensor(other, device=common.DEVICE)
            other = other.float()
            mse = torch.mean((ref - other) ** 2).item()
            cos = torch.nn.functional.cosine_similarity(ref, other, dim=1).mean().item()
            mses.append(mse)
            cosines.append(cos)
    return {
        "logits_mse": sum(mses) / len(mses),
        "logits_cosine": sum(cosines) / len(cosines),
    }


def load_trt_infer(engine_path: str):
    """Return an infer callable for a TRT engine."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    inp_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)

    def infer(x):
        bs = x.shape[0]
        out = torch.empty(bs, 1000, dtype=torch.float32, device="cuda")
        ctx.set_input_shape(inp_name, x.shape)
        ctx.set_tensor_address(inp_name, x.data_ptr())
        ctx.set_tensor_address(out_name, out.data_ptr())
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out

    return infer


def main():
    # Load per-stage results.
    baseline = load("baseline")
    trt_fp16 = load("trt_fp16")
    trt_int8 = load("trt_int8")

    if not baseline:
        print("ERROR: baseline.json not found, run run_baseline.py first")
        sys.exit(1)

    fp32_top1 = baseline["fp32"]["top1"]
    fp32_top5 = baseline["fp32"]["top5"]
    fp32_lat1 = baseline["latency"]["fp32_bs1"]["latency_ms_mean"]
    fp32_lat64 = baseline["latency"]["fp32_bs64"]["latency_ms_mean"]
    fp32_tput64 = baseline["latency"]["fp32_bs64"]["throughput_img_s"]

    rows = []

    def add_row(tag, top1, top5, lat1, lat64, tput64):
        rows.append({
            "tag": tag,
            "top1": top1,
            "top5": top5,
            "top1_drop": fp32_top1 - top1,
            "lat_bs1_ms": lat1,
            "lat_bs64_ms": lat64,
            "tput_bs64": tput64,
            "speedup_bs1": fp32_lat1 / lat1 if lat1 else None,
            "speedup_bs64": fp32_lat64 / lat64 if lat64 else None,
        })

    # FP32 baseline
    add_row(
        "FP32 (torch)",
        fp32_top1, fp32_top5,
        fp32_lat1, fp32_lat64, fp32_tput64,
    )

    # Torch FP16 (autocast)
    tf16 = baseline.get("torch_fp16", {})
    if tf16:
        add_row(
            "Torch FP16 (autocast)",
            tf16["top1"], tf16["top5"],
            baseline["latency"]["fp16_bs1"]["latency_ms_mean"],
            baseline["latency"]["fp16_bs64"]["latency_ms_mean"],
            baseline["latency"]["fp16_bs64"]["throughput_img_s"],
        )

    # TRT FP16
    if trt_fp16:
        add_row(
            "TRT FP16",
            trt_fp16.get("top1", 0), trt_fp16.get("top5", 0),
            trt_fp16["latency"]["bs1"]["latency_ms_mean"],
            trt_fp16["latency"]["bs64"]["latency_ms_mean"],
            trt_fp16["latency"]["bs64"]["throughput_img_s"],
        )

    # TRT INT8
    if trt_int8:
        add_row(
            "TRT INT8 (PTQ)",
            trt_int8.get("top1", 0), trt_int8.get("top5", 0),
            trt_int8["latency"]["bs1"]["latency_ms_mean"],
            trt_int8["latency"]["bs64"]["latency_ms_mean"],
            trt_int8["latency"]["bs64"]["throughput_img_s"],
        )

    # Compute logit similarity (FP32 vs TRT FP16/INT8) on 10 batches.
    print("computing logit similarity (FP32 vs TRT FP16/INT8, 10 batches)...")
    fp32_model = common.build_model()
    sim = {}

    if Path(str(common.ENGINE_DIR / "resnet50_fp16.engine")).exists():
        trt_fp16_fn = load_trt_infer(str(common.ENGINE_DIR / "resnet50_fp16.engine"))
        sim["trt_fp16_vs_fp32"] = compute_logit_similarity(fp32_model, trt_fp16_fn)
        print("TRT FP16 logit similarity:", sim["trt_fp16_vs_fp32"])

    if Path(str(common.ENGINE_DIR / "resnet50_int8.engine")).exists():
        trt_int8_fn = load_trt_infer(str(common.ENGINE_DIR / "resnet50_int8.engine"))
        sim["trt_int8_vs_fp32"] = compute_logit_similarity(fp32_model, trt_int8_fn)
        print("TRT INT8 logit similarity:", sim["trt_int8_vs_fp32"])

    # Print markdown summary table.
    print("\n## Quantization Results — ResNet50 on ImageNet val (50k)")
    print(f"GPU: {common.gpu_name()}, batch eval bs=64, latency bs=1 and bs=64\n")

    hdr = ("Precision", "top1%", "top5%", "Δtop1", "lat bs=1 ms",
           "lat bs=64 ms", "tput img/s", "speedup×(bs=64)")
    row_fmt = "| {:22s} | {:6s} | {:6s} | {:6s} | {:11s} | {:12s} | {:10s} | {:14s} |"
    sep = "|" + "-" * 24 + "|" + ("-" * 8 + "|") * 7
    print(row_fmt.format(*hdr))
    print(sep)
    for r in rows:
        print(row_fmt.format(
            r["tag"],
            f"{r['top1']:.3f}",
            f"{r['top5']:.3f}",
            f"{r['top1_drop']:+.3f}",
            f"{r['lat_bs1_ms']:.3f}",
            f"{r['lat_bs64_ms']:.3f}",
            f"{r['tput_bs64']:.0f}",
            f"{r['speedup_bs64']:.2f}x" if r["speedup_bs64"] else "—",
        ))

    print("\n### Logit similarity (vs FP32, 640 images)")
    for k, v in sim.items():
        print(f"  {k}: MSE={v['logits_mse']:.4f}  cosine={v['logits_cosine']:.6f}")

    summary = {"rows": rows, "logit_similarity": sim, "gpu": common.gpu_name()}
    out = RESULTS / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
