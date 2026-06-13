"""Dependency-free helpers shared by quantization experiment scripts."""

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence


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
