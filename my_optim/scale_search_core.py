"""Shared ModelOpt activation-scale search used by CNN and Transformer scripts.

ModelOpt first performs ordinary INT8 PTQ. The search then changes only each
activation TensorQuantizer's calibrated ``_amax``. For symmetric INT8,
``scale = amax / 127``. The adjusted values are therefore embedded into the
Q/DQ constants when the final model is exported to ONNX.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import modelopt.torch.quantization as mtq
import onnx
import torch
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from torch.utils.data import SubsetRandomSampler

from my_optim.experiment_utils import class_balanced_indices
from my_optim.scale_search_utils import (
    parse_scale_ratios,
    quantizer_matches,
    select_best_scale_trial,
)


def _build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--calib-batch-size", type=int, default=16)
    parser.add_argument(
        "--ratios",
        default="1.0,0.9,0.8,0.7,0.6",
        help="Comma-separated multipliers applied to ModelOpt activation amax.",
    )
    parser.add_argument("--no-export", action="store_true")
    return parser


def _build_search_loader(common, sample_size: int, seed: int, batch_size: int, workers: int):
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


def _quantize_once(common, calib_batch_size: int, workers: int):
    model = common.build_model(exportable=True)
    calib_loader = common.build_calib_loader(
        batch_size=calib_batch_size,
        workers=workers,
    )

    def calibration_loop(candidate_model):
        with torch.inference_mode():
            for images, _ in calib_loader:
                images = images.to(common.DEVICE, non_blocking=True)
                candidate_model(images)

    return mtq.quantize(
        model,
        mtq.INT8_DEFAULT_CFG,
        forward_loop=calibration_loop,
    )


def _activation_quantizers(quant_model) -> dict[str, TensorQuantizer]:
    quantizers = {}
    for name, module in quant_model.named_modules():
        if (
            isinstance(module, TensorQuantizer)
            and "input_quantizer" in name
            and getattr(module, "_amax", None) is not None
        ):
            quantizers[name] = module
    if not quantizers:
        raise RuntimeError("ModelOpt produced no activation input quantizers with amax")
    return quantizers


def _set_group_ratio(
        quantizers: Mapping[str, TensorQuantizer],
        base_amax: Mapping[str, torch.Tensor],
        names: Iterable[str],
        ratio: float,
) -> None:
    for name in names:
        quantizer = quantizers[name]
        device = quantizer._amax.device
        quantizer._amax = base_amax[name].to(device) * ratio


def _tensor_json(tensor: torch.Tensor):
    value = tensor.detach().cpu().tolist()
    return value


def _scale_snapshot(
        quantizers: Mapping[str, TensorQuantizer],
) -> dict[str, dict]:
    snapshot = {}
    for name, quantizer in quantizers.items():
        amax = quantizer._amax.detach()
        snapshot[name] = {
            "amax": _tensor_json(amax),
            "scale": _tensor_json(amax / 127.0),
        }
    return snapshot


def _export_qdq(common, quant_model) -> Path:
    qdq_path = common.onnx_path("int8_qdq_scale_search.onnx")
    inline_path = common.onnx_path("int8_qdq_scale_search_inline.onnx")
    dummy = torch.randn(1, *common.INPUT_SIZE)

    print("moving searched quantized model to CPU for ONNX export...")
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


def run_scale_search(
        common,
        groups: Sequence[Mapping[str, object]],
        expected_model_prefixes: Sequence[str],
        description: str,
) -> None:
    """Run greedy architecture-group activation amax search and export Q/DQ."""
    parser = _build_parser(description)
    args = parser.parse_args()
    if not any(common.MODEL_NAME.startswith(prefix) for prefix in expected_model_prefixes):
        expected = ", ".join(expected_model_prefixes)
        raise ValueError(f"expected model prefix in [{expected}], got {common.MODEL_NAME!r}")

    ratios = parse_scale_ratios(args.ratios)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.ONNX_DIR.mkdir(parents=True, exist_ok=True)
    loader, indices = _build_search_loader(
        common,
        args.sample_size,
        args.seed,
        args.batch_size,
        args.workers,
    )

    print(
        f"model={common.MODEL_NAME} samples={len(indices)} "
        f"ratios={ratios} objective=Top1"
    )
    quant_model = _quantize_once(common, args.calib_batch_size, args.workers)
    quantizers = _activation_quantizers(quant_model)
    base_amax = {
        name: quantizer._amax.detach().clone()
        for name, quantizer in quantizers.items()
    }
    print(f"searchable activation quantizers: {len(quantizers)}")

    default_metrics = common.evaluate_accuracy(
        quant_model,
        loader,
        desc="scale_default_int8",
    )
    assigned_names = set()
    selected_ratios = {}
    group_results = []
    current_metrics = default_metrics

    for group in groups:
        group_name = str(group["name"])
        fragments = list(group["fragments"])
        matched_names = [
            name
            for name in quantizers
            if name not in assigned_names and quantizer_matches(name, fragments)
        ]
        if not matched_names:
            group_results.append({
                "name": group_name,
                "fragments": fragments,
                "matched_quantizers": [],
                "skipped": True,
            })
            print(f"group={group_name}: no matching activation quantizers, skipped")
            continue

        trials = []
        for ratio in ratios:
            _set_group_ratio(quantizers, base_amax, matched_names, ratio)
            metrics = common.evaluate_accuracy(
                quant_model,
                loader,
                desc=f"scale_{group_name}_{ratio:g}",
            )
            trial = {"ratio": ratio, **metrics}
            trials.append(trial)
            print(
                f"group={group_name:18s} ratio={ratio:.4f} "
                f"top1={metrics['top1']:.3f}% top5={metrics['top5']:.3f}%"
            )

        selected = dict(select_best_scale_trial(trials))
        selected_ratio = float(selected["ratio"])
        _set_group_ratio(quantizers, base_amax, matched_names, selected_ratio)
        assigned_names.update(matched_names)
        selected_ratios[group_name] = selected_ratio
        current_metrics = {
            key: selected[key]
            for key in ("top1", "top5", "n", "eval_seconds")
            if key in selected
        }
        group_results.append({
            "name": group_name,
            "fragments": fragments,
            "matched_quantizers": matched_names,
            "selected_ratio": selected_ratio,
            "trials": trials,
        })
        print(
            f"selected group={group_name} ratio={selected_ratio:.4f} "
            f"top1={selected['top1']:.3f}%"
        )

    search_result = {
        "model": common.MODEL_NAME,
        "search_type": "architecture_group_activation_amax_coordinate_search",
        "scale_definition": "symmetric_int8_scale = amax / 127",
        "sample_size": len(indices),
        "seed": args.seed,
        "ratios": ratios,
        "default_int8": default_metrics,
        "selected_int8": current_metrics,
        "selected_group_ratios": selected_ratios,
        "groups": group_results,
        "activation_quantizers": _scale_snapshot(quantizers),
    }
    result_path = common.RESULTS_DIR / "scale_search.json"
    result_path.write_text(json.dumps(search_result, indent=2), encoding="utf-8")
    print(f"scale search results saved: {result_path}")

    if args.no_export:
        return
    mtq.print_quant_summary(quant_model)
    inline_path = _export_qdq(common, quant_model)
    print(f"scale-searched Q/DQ ONNX: {inline_path}")
