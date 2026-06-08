"""Stage 6: Final comparison report — FP32 / Torch FP16 / TRT FP16 / TRT INT8.

Model is selected via OPTIM_MODEL env var (default: vit_base_patch16_224).

Reads the JSON results files from every stage and computes:
  - top1 / top5 accuracy drop vs FP32
  - latency and throughput at bs=1 and bs=EVAL_BATCH_SIZE
  - speedup ratios relative to FP32
  - logits MSE and cosine similarity (FP32 vs each precision, 10 batches)

Writes results/<model>/summary.json and prints a markdown table.

Usage:
    OPTIM_MODEL=vit_base_patch16_224 python run_summary.py
    OPTIM_MODEL=swin_tiny_patch4_window7_224 python run_summary.py
    python run_summary.py --all   # print both models if both results exist
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))
import common

_BASE = Path(__file__).parent
RESULTS_BASE = _BASE / "results"


def load(model_name: str, name: str) -> dict:
    p = RESULTS_BASE / model_name / f"{name}.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


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


def compute_logit_similarity(fp32_model, other_infer_fn, n_batches: int = 10):
    """Compare logits on the same val images (FP32 reference vs other)."""
    loader, _ = common.build_val_loader(batch_size=common.EVAL_BATCH_SIZE, workers=4)
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


def summarize_model(model_name: str):
    """Print summary table and write summary.json for one model."""
    # Override common's MODEL_NAME so helpers use the right paths.
    common.MODEL_NAME = model_name
    common.RESULTS_DIR = _BASE / "results" / model_name
    common.ENGINE_DIR = _BASE / "engines" / model_name
    common._INPUT_SIZE = None  # force re-resolve for this model

    baseline = load(model_name, "baseline")
    trt_fp16 = load(model_name, "trt_fp16")
    trt_int8 = load(model_name, "trt_int8")

    if not baseline:
        print(f"  [{model_name}] baseline.json not found, skipping")
        return

    bs_tput = common.EVAL_BATCH_SIZE   # throughput column batch size
    bs_key = f"bs{bs_tput}"

    fp32_top1 = baseline["fp32"]["top1"]
    fp32_top5 = baseline["fp32"]["top5"]
    fp32_lat1 = baseline["latency"]["fp32_bs1"]["latency_ms_mean"]
    fp32_lat_tput = baseline["latency"][f"fp32_{bs_key}"]["latency_ms_mean"]
    fp32_tput = baseline["latency"][f"fp32_{bs_key}"]["throughput_img_s"]

    rows = []

    def add_row(tag, top1, top5, lat1, lat_tput, tput):
        rows.append({
            "tag": tag,
            "top1": top1,
            "top5": top5,
            "top1_drop": fp32_top1 - top1,
            "lat_bs1_ms": lat1,
            f"lat_bs{bs_tput}_ms": lat_tput,
            f"tput_bs{bs_tput}": tput,
            "speedup_bs1": fp32_lat1 / lat1 if lat1 else None,
            f"speedup_bs{bs_tput}": fp32_lat_tput / lat_tput if lat_tput else None,
        })

    add_row("FP32 (torch)", fp32_top1, fp32_top5, fp32_lat1, fp32_lat_tput, fp32_tput)

    tf16 = baseline.get("torch_fp16", {})
    if tf16:
        add_row(
            "Torch FP16",
            tf16["top1"], tf16["top5"],
            baseline["latency"]["fp16_bs1"]["latency_ms_mean"],
            baseline["latency"][f"fp16_{bs_key}"]["latency_ms_mean"],
            baseline["latency"][f"fp16_{bs_key}"]["throughput_img_s"],
        )

    if trt_fp16:
        add_row(
            "TRT FP16",
            trt_fp16.get("top1", 0), trt_fp16.get("top5", 0),
            trt_fp16["latency"]["bs1"]["latency_ms_mean"],
            trt_fp16["latency"][bs_key]["latency_ms_mean"],
            trt_fp16["latency"][bs_key]["throughput_img_s"],
        )

    if trt_int8:
        add_row(
            "TRT INT8 (PTQ)",
            trt_int8.get("top1", 0), trt_int8.get("top5", 0),
            trt_int8["latency"]["bs1"]["latency_ms_mean"],
            trt_int8["latency"][bs_key]["latency_ms_mean"],
            trt_int8["latency"][bs_key]["throughput_img_s"],
        )

    # Logit similarity (FP32 vs TRT precisions).
    print(f"  computing logit similarity for {model_name}...")
    fp32_model = common.build_model()
    sim = {}

    fp16_eng = common.ENGINE_DIR / f"{model_name}_fp16.engine"
    int8_eng = common.ENGINE_DIR / f"{model_name}_int8.engine"

    if fp16_eng.exists():
        sim["trt_fp16_vs_fp32"] = compute_logit_similarity(
            fp32_model, load_trt_infer(str(fp16_eng)))
        print(f"  TRT FP16 logit similarity: {sim['trt_fp16_vs_fp32']}")

    if int8_eng.exists():
        sim["trt_int8_vs_fp32"] = compute_logit_similarity(
            fp32_model, load_trt_infer(str(int8_eng)))
        print(f"  TRT INT8 logit similarity: {sim['trt_int8_vs_fp32']}")

    # Print markdown table.
    print(f"\n## {model_name} — Quantization Results (ImageNet val 50k)")
    print(f"GPU: {common.gpu_name()}, eval bs={bs_tput}, latency bs=1 and bs={bs_tput}\n")

    hdr = ("Precision", "top1%", "top5%", "Δtop1", "lat bs=1 ms",
           f"lat bs={bs_tput} ms", "tput img/s", f"speedup×(bs={bs_tput})")
    row_fmt = "| {:22s} | {:6s} | {:6s} | {:7s} | {:11s} | {:12s} | {:10s} | {:16s} |"
    sep = "|" + "-" * 24 + "|" + ("-" * 8 + "|") * 7
    print(row_fmt.format(*hdr))
    print(sep)
    for r in rows:
        spup = r.get(f"speedup_bs{bs_tput}")
        print(row_fmt.format(
            r["tag"],
            f"{r['top1']:.3f}",
            f"{r['top5']:.3f}",
            f"{r['top1_drop']:+.3f}",
            f"{r['lat_bs1_ms']:.3f}",
            f"{r[f'lat_bs{bs_tput}_ms']:.3f}",
            f"{r[f'tput_bs{bs_tput}']:.0f}",
            f"{spup:.2f}x" if spup else "—",
        ))

    if sim:
        print(f"\n### Logit similarity vs FP32 ({10 * bs_tput} images)")
        for k, v in sim.items():
            print(f"  {k}: MSE={v['logits_mse']:.5f}  cosine={v['logits_cosine']:.6f}")

    # INT8 drop warning.
    for r in rows:
        if "INT8" in r["tag"] and r["top1_drop"] > 2.0:
            print(f"\n⚠  INT8 top1 drop {r['top1_drop']:.2f}% > 2% threshold. "
                  "Consider running run_sensitive_fallback.py for this model.")

    summary = {
        "model": model_name,
        "gpu": common.gpu_name(),
        "eval_batch_size": bs_tput,
        "rows": rows,
        "logit_similarity": sim,
    }
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {out}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="summarize all models found in results/")
    args = parser.parse_args()

    if args.all:
        models = sorted(p.name for p in RESULTS_BASE.iterdir()
                        if p.is_dir() and (p / "baseline.json").exists())
        if not models:
            print("No results found in results/")
            sys.exit(1)
    else:
        models = [common.MODEL_NAME]

    for m in models:
        print(f"\n{'='*60}\n{m}\n{'='*60}")
        summarize_model(m)


if __name__ == "__main__":
    main()
