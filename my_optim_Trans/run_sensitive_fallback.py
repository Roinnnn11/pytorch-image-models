"""Search for a reproducible mixed-precision fallback configuration.

Every candidate is calibrated identically and evaluated on the same
deterministic, class-balanced ImageNet subset. The FP32 and default fake-quant
references are also measured on that subset, so their accuracy values are
directly comparable.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))

import onnx
import torch
from torch.utils.data import SubsetRandomSampler

import common
import modelopt.torch.quantization as mtq
from my_optim.experiment_utils import (
    class_balanced_indices,
    make_modelopt_exclusion_rules,
)


# Related projections are tested as groups because disabling only one half of a
# Transformer sub-block can miss interactions between its paired operations.
SENSITIVE_GROUPS = [
    ("mlp", ["mlp.fc1", "mlp.fc2"]),
    ("attention", ["attn.qkv", "attn.proj"]),
    ("patch_embed", ["patch_embed"]),
    ("head", ["head"]),
]


def make_fp16_config(excluded_fragments: list[str]) -> dict:
    """Build an INT8 config whose matching quantizers are disabled."""
    config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    config["quant_cfg"].update(make_modelopt_exclusion_rules(excluded_fragments))
    return config


def build_search_loader(
        sample_size: int,
        seed: int,
        batch_size: int,
        workers: int,
):
    """Build one deterministic class-balanced loader shared by every candidate."""
    initial_loader, dataset = common.build_val_loader(batch_size=batch_size, workers=0)
    del initial_loader

    sample_size = min(sample_size, len(dataset))
    indices = class_balanced_indices(dataset.targets, sample_size=sample_size, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    sampler = SubsetRandomSampler(indices, generator=generator)
    loader, _ = common.build_val_loader(
        batch_size=batch_size,
        workers=workers,
        sampler=sampler,
    )
    return loader, indices


def evaluate_fp32(loader) -> dict:
    """Evaluate the exportable FP32 model on the search subset."""
    model = common.build_model(exportable=True)
    result = common.evaluate_accuracy(model, loader, desc="fallback_fp32")
    del model
    torch.cuda.empty_cache()
    return result


def quantize_model(excluded_fragments: list[str]):
    """Calibrate and quantize one fresh model with the requested exclusions."""
    model = common.build_model(exportable=True)
    calib_loader = common.build_calib_loader(batch_size=16, workers=4)
    config = make_fp16_config(excluded_fragments)

    def calibration_loop(candidate_model):
        with torch.inference_mode():
            for images, _ in calib_loader:
                images = images.to(common.DEVICE, non_blocking=True)
                candidate_model(images)

    return mtq.quantize(model, config, forward_loop=calibration_loop)


def evaluate_config(excluded_fragments: list[str], loader, desc: str) -> dict:
    """Evaluate one fake-quant candidate on the shared search subset."""
    quant_model = quantize_model(excluded_fragments)
    result = common.evaluate_accuracy(quant_model, loader, desc=desc)
    del quant_model
    torch.cuda.empty_cache()
    return result


def export_mixed_model(excluded_fragments: list[str]) -> Path:
    """Export the selected mixed-precision model as a Q/DQ ONNX file."""
    quant_model = quantize_model(excluded_fragments)
    mtq.print_quant_summary(quant_model)

    qdq_path = common.onnx_path("int8_qdq_mixed.onnx")
    inline_path = common.onnx_path("int8_qdq_mixed_inline.onnx")
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
    return inline_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=common.EVAL_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--target-drop",
        type=float,
        default=1.0,
        help="Stop when subset top1 drop from the subset FP32 baseline is at most this value.",
    )
    args = parser.parse_args()

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.ONNX_DIR.mkdir(parents=True, exist_ok=True)

    loader, indices = build_search_loader(
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(
        f"model: {common.MODEL_NAME} | balanced search samples: {len(indices)} "
        f"| seed: {args.seed}"
    )

    fp32 = evaluate_fp32(loader)
    default_int8 = evaluate_config([], loader, desc="fallback_default_int8")
    current_top1 = default_int8["top1"]
    kept_fragments = []
    remaining_groups = list(SENSITIVE_GROUPS)
    trials = []

    print(
        f"subset FP32 top1={fp32['top1']:.3f}% | "
        f"default fake-quant top1={current_top1:.3f}% | "
        f"drop={fp32['top1'] - current_top1:.3f}%"
    )

    while remaining_groups and fp32["top1"] - current_top1 > args.target_drop:
        round_trials = []
        for group_name, fragments in remaining_groups:
            candidate_fragments = kept_fragments + fragments
            result = evaluate_config(
                candidate_fragments,
                loader,
                desc=f"fallback_{group_name}",
            )
            trial = {
                "group": group_name,
                "added_fragments": fragments,
                "candidate_fragments": candidate_fragments,
                "top1": result["top1"],
                "top5": result["top5"],
                "drop_from_subset_fp32": fp32["top1"] - result["top1"],
            }
            round_trials.append(trial)
            trials.append(trial)
            print(
                f"  candidate={group_name:12s} top1={result['top1']:.3f}% "
                f"drop={trial['drop_from_subset_fp32']:.3f}%"
            )

        best_trial = max(round_trials, key=lambda item: item["top1"])
        if best_trial["top1"] <= current_top1:
            print("No remaining fallback group improves subset top1; stopping.")
            break

        kept_fragments = best_trial["candidate_fragments"]
        current_top1 = best_trial["top1"]
        remaining_groups = [
            group for group in remaining_groups if group[0] != best_trial["group"]
        ]
        print(
            f"accepted={best_trial['group']} | top1={current_top1:.3f}% | "
            f"drop={fp32['top1'] - current_top1:.3f}%"
        )

    search_result = {
        "model": common.MODEL_NAME,
        "sample_size": len(indices),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "target_drop": args.target_drop,
        "subset_fp32": fp32,
        "subset_default_fake_quant": default_int8,
        "selected_fragments": kept_fragments,
        "selected_top1": current_top1,
        "selected_drop_from_subset_fp32": fp32["top1"] - current_top1,
        "trials": trials,
    }
    result_path = common.RESULTS_DIR / "fallback_search.json"
    result_path.write_text(json.dumps(search_result, indent=2), encoding="utf-8")
    print(f"search results saved: {result_path}")

    if not kept_fragments:
        print("No fallback group was selected; keeping the default INT8 model.")
        return

    inline_path = export_mixed_model(kept_fragments)
    print(f"mixed-precision ONNX: {inline_path}")
    print(
        "Next: python build_trt_engine.py --precision int8 "
        "--onnx-suffix int8_qdq_mixed_inline.onnx "
        "--engine-suffix int8_mixed.engine"
    )
    print(
        "Then: python run_trt_eval.py --precision int8 "
        "--engine-suffix int8_mixed.engine "
        "--result-tag trt_int8_mixed"
    )


if __name__ == "__main__":
    main()
