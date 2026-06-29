"""Dependency-free helpers shared by quantization experiment scripts."""

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence


def parse_fixed_batches(value: str) -> List[int]:
    """Parse a unique, positive comma-separated batch-size list."""
    try:
        batches = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("batch sizes must be comma-separated integers") from error
    if not batches or any(batch <= 0 for batch in batches):
        raise ValueError("batch sizes must be positive")
    if len(set(batches)) != len(batches):
        raise ValueError("batch sizes must be unique")
    return batches


def padded_batch_size(actual: int, fixed: int) -> int:
    """Validate that an actual batch can be padded to a fixed engine batch."""
    if actual <= 0 or fixed <= 0 or actual > fixed:
        raise ValueError("expected 0 < actual <= fixed")
    return fixed


def pareto_front(
        trials: Sequence[Mapping],
        fp32_top1: float,
        max_drop: float,
) -> List[Mapping]:
    """Return accuracy-eligible, non-dominated latency/throughput trials."""
    if max_drop < 0:
        raise ValueError("max_drop cannot be negative")
    eligible = [
        trial for trial in trials
        if fp32_top1 - float(trial["top1"]) <= max_drop
    ]
    front = []
    for candidate in eligible:
        dominated = any(
            other is not candidate
            and float(other["top1"]) >= float(candidate["top1"])
            and float(other["latency_ms"]) <= float(candidate["latency_ms"])
            and float(other["throughput"]) >= float(candidate["throughput"])
            and (
                float(other["top1"]) > float(candidate["top1"])
                or float(other["latency_ms"]) < float(candidate["latency_ms"])
                or float(other["throughput"]) > float(candidate["throughput"])
            )
            for other in eligible
        )
        if not dominated:
            front.append(candidate)
    return sorted(
        front,
        key=lambda trial: (
            -float(trial["top1"]),
            float(trial["latency_ms"]),
            str(trial["name"]),
        ),
    )


def class_balanced_indices(
        targets: Sequence[int],
        sample_size: int,
        seed: int = 42,
) -> List[int]:
    """Select a deterministic sample distributed as evenly as possible by class."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(targets):
        raise ValueError("sample_size cannot exceed the dataset size")

    indices_by_class = defaultdict(list)
    for index, target in enumerate(targets):
        indices_by_class[target].append(index)
    if not indices_by_class:
        raise ValueError("targets cannot be empty")

    rng = random.Random(seed)
    class_order = sorted(indices_by_class)
    rng.shuffle(class_order)
    for indices in indices_by_class.values():
        rng.shuffle(indices)

    selected = []
    offsets = {target: 0 for target in class_order}
    while len(selected) < sample_size:
        made_progress = False
        for target in class_order:
            offset = offsets[target]
            class_indices = indices_by_class[target]
            if offset >= len(class_indices):
                continue
            selected.append(class_indices[offset])
            offsets[target] += 1
            made_progress = True
            if len(selected) == sample_size:
                break
        if not made_progress:
            break

    rng.shuffle(selected)
    return selected


def make_modelopt_exclusion_rules(fragments: Iterable[str]) -> Dict[str, dict]:
    """Create ordered ModelOpt quantizer rules that keep matching modules floating point."""
    rules = {}
    for fragment in fragments:
        if not fragment:
            raise ValueError("module fragments cannot be empty")
        rules[f"*{fragment}*weight_quantizer"] = {"enable": False}
        rules[f"*{fragment}*input_quantizer"] = {"enable": False}
    return rules


def select_accuracy_constrained_candidate(
        trials: Sequence[Mapping],
        fp32_top1: float,
        target_drop: float,
) -> Mapping:
    """Choose the smallest preset fallback that meets the accuracy constraint.

    ``fallback_cost`` is an architecture-level estimate supplied by the search
    script. If no candidate meets the target, return the highest-accuracy trial
    so the experiment still produces a useful fallback.
    """
    if not trials:
        raise ValueError("trials cannot be empty")
    if target_drop < 0:
        raise ValueError("target_drop cannot be negative")

    acceptable = [
        trial for trial in trials
        if fp32_top1 - float(trial["top1"]) <= target_drop
    ]
    if acceptable:
        return min(
            acceptable,
            key=lambda trial: (
                int(trial["fallback_cost"]),
                -float(trial["top1"]),
                str(trial["name"]),
            ),
        )
    return max(
        trials,
        key=lambda trial: (
            float(trial["top1"]),
            -int(trial["fallback_cost"]),
            str(trial["name"]),
        ),
    )
