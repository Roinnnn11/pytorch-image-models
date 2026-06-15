"""Search architecture-aware MobileNetV3 INT8/FP16 fallback presets.

ModelOpt remains the PTQ/Q-DQ backend. This script manually defines a bounded
search space from MobileNetV3 structure, evaluates every preset on the same
class-balanced ImageNet subset, and exports the smallest preset that satisfies
the requested Top1-drop constraint.
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
    select_accuracy_constrained_candidate,
)


# This intentionally stays small: every preset requires a fresh ModelOpt
# calibration. The ordered early-block presets test the failure pattern already
# observed in MobileNetV3, while depthwise and SE isolate architectural causes.
MOBILENET_CANDIDATES = [
    {"name": "full_int8", "fragments": [], "fallback_cost": 0},
    {"name": "stem_fp16", "fragments": ["conv_stem"], "fallback_cost": 1},
    {
        "name": "se_fp16",
        "fragments": ["se.conv_reduce", "se.conv_expand"],
        "fallback_cost": 2,
    },
    {
        "name": "blocks_0_1_fp16",
        "fragments": ["blocks.0", "blocks.1"],
        "fallback_cost": 2,
    },
    {
        "name": "blocks_0_3_fp16",
        "fragments": ["blocks.0", "blocks.1", "blocks.2", "blocks.3"],
        "fallback_cost": 4,
    },
    {
        "name": "blocks_0_5_fp16",
        "fragments": [
            "blocks.0",
            "blocks.1",
            "blocks.2",
            "blocks.3",
            "blocks.4",
            "blocks.5",
        ],
        "fallback_cost": 6,
    },
    {"name": "depthwise_fp16", "fragments": ["conv_dw"], "fallback_cost": 7},
    {
        "name": "blocks_0_5_stem_fp16",
        "fragments": [
            "conv_stem",
            "blocks.0",
            "blocks.1",
            "blocks.2",
            "blocks.3",
            "blocks.4",
            "blocks.5",
        ],
        "fallback_cost": 7,
    },
]


def _validate_model() -> None:
    if not common.MODEL_NAME.startswith("mobilenetv3_large_100"):
        raise ValueError(
            "This bounded search currently supports mobilenetv3_large_100 only; "
            f"got {common.MODEL_NAME!r}."
        )


def make_fp16_config(excluded_fragments: list[str]) -> dict:
    config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    config["quant_cfg"].update(make_modelopt_exclusion_rules(excluded_fragments))
    return config


def build_search_loader(sample_size: int, seed: int, batch_size: int, workers: int):
    _, dataset = common.build_val_loader(batch_size=batch_size, workers=0)
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


def quantize_model(excluded_fragments: list[str], calib_batch_size: int):
    model = common.build_model(exportable=True)
    calib_loader = common.build_calib_loader(batch_size=calib_batch_size, workers=4)
    config = make_fp16_config(excluded_fragments)

    def calibration_loop(candidate_model):
        with torch.inference_mode():
            for images, _ in calib_loader:
                images = images.to(common.DEVICE, non_blocking=True)
                candidate_model(images)

    return mtq.quantize(model, config, forward_loop=calibration_loop)


def evaluate_candidate(candidate: dict, loader, calib_batch_size: int) -> dict:
    quant_model = quantize_model(candidate["fragments"], calib_batch_size)
    result = common.evaluate_accuracy(
        quant_model,
        loader,
        desc=f"search_{candidate['name']}",
    )
    del quant_model
    torch.cuda.empty_cache()
    return {
        **candidate,
        "top1": result["top1"],
        "top5": result["top5"],
        "n": result["n"],
    }


def export_selected_model(selected: dict, calib_batch_size: int) -> Path:
    quant_model = quantize_model(selected["fragments"], calib_batch_size)
    mtq.print_quant_summary(quant_model)

    qdq_path = common.onnx_path("int8_qdq_search_selected.onnx")
    inline_path = common.onnx_path("int8_qdq_search_selected_inline.onnx")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--calib-batch-size", type=int, default=32)
    parser.add_argument(
        "--target-drop",
        type=float,
        default=0.5,
        help="Maximum acceptable Top1 drop on the balanced search subset.",
    )
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    _validate_model()
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.ONNX_DIR.mkdir(parents=True, exist_ok=True)

    loader, indices = build_search_loader(
        args.sample_size,
        args.seed,
        args.batch_size,
        args.workers,
    )
    fp32_model = common.build_model(exportable=True)
    fp32 = common.evaluate_accuracy(fp32_model, loader, desc="search_fp32")
    del fp32_model
    torch.cuda.empty_cache()

    print(
        f"model={common.MODEL_NAME} samples={len(indices)} "
        f"subset_fp32={fp32['top1']:.3f}% target_drop={args.target_drop:.3f}"
    )
    trials = []
    for candidate in MOBILENET_CANDIDATES:
        trial = evaluate_candidate(candidate, loader, args.calib_batch_size)
        trial["drop_from_subset_fp32"] = fp32["top1"] - trial["top1"]
        trials.append(trial)
        print(
            f"candidate={trial['name']:26s} top1={trial['top1']:.3f}% "
            f"drop={trial['drop_from_subset_fp32']:.3f} "
            f"cost={trial['fallback_cost']}"
        )

    selected = dict(select_accuracy_constrained_candidate(
        trials,
        fp32_top1=fp32["top1"],
        target_drop=args.target_drop,
    ))
    selected["meets_target"] = selected["drop_from_subset_fp32"] <= args.target_drop

    search_result = {
        "model": common.MODEL_NAME,
        "search_type": "bounded_architecture_presets",
        "sample_size": len(indices),
        "seed": args.seed,
        "target_drop": args.target_drop,
        "subset_fp32": fp32,
        "selected": selected,
        "trials": trials,
    }
    result_path = common.RESULTS_DIR / "fallback_search.json"
    result_path.write_text(json.dumps(search_result, indent=2), encoding="utf-8")
    print(
        f"selected={selected['name']} top1={selected['top1']:.3f}% "
        f"drop={selected['drop_from_subset_fp32']:.3f} "
        f"meets_target={selected['meets_target']}"
    )
    print(f"search results saved: {result_path}")

    if args.no_export:
        return
    inline_path = export_selected_model(selected, args.calib_batch_size)
    print(f"selected mixed-precision ONNX: {inline_path}")


if __name__ == "__main__":
    main()
