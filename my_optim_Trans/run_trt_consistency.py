"""Compare logits from fixed-batch engines built from the same Q/DQ ONNX."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))
import common
from my_optim.experiment_utils import parse_fixed_batches
from run_trt_eval import TRTInferencer


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--batches", default="1,4,16,32")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    batches = parse_fixed_batches(args.batches)

    torch.manual_seed(args.seed)
    sample = torch.randn(1, *common.INPUT_SIZE, device=common.DEVICE)
    reference = None
    comparisons = []
    for batch in batches:
        engine = common.engine_path(f"int8_{args.candidate}_bs{batch}.engine")
        if not engine.exists():
            raise FileNotFoundError(engine)
        infer = TRTInferencer(str(engine))
        inputs = sample.expand(batch, *sample.shape[1:]).contiguous()
        logits = infer(inputs)[0].detach().clone()
        torch.cuda.synchronize()
        if reference is None:
            reference = logits
        comparisons.append({
            "batch_size": batch,
            "max_abs_logit_error_vs_bs1": float((logits - reference).abs().max().item()),
            "cosine_similarity_vs_bs1": float(
                functional.cosine_similarity(logits[None], reference[None]).item()
            ),
        })
        del infer
        torch.cuda.empty_cache()

    result = {
        "model": common.MODEL_NAME,
        "candidate": args.candidate,
        "seed": args.seed,
        "comparisons": comparisons,
    }
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = common.RESULTS_DIR / f"consistency_{args.candidate}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
