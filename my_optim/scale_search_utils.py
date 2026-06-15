"""Dependency-free helpers for activation scale search."""

from typing import Iterable, Mapping, Sequence


def parse_scale_ratios(value: str) -> list[float]:
    """Parse a comma-separated list of positive amax multipliers."""
    if not value.strip():
        raise ValueError("scale ratios cannot be empty")
    ratios = [float(item.strip()) for item in value.split(",")]
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("scale ratios must be positive")
    return list(dict.fromkeys(ratios))


def quantizer_matches(name: str, fragments: Iterable[str]) -> bool:
    """Return whether a quantizer name belongs to an architecture group."""
    return any(fragment in name for fragment in fragments)


def select_best_scale_trial(trials: Sequence[Mapping]) -> Mapping:
    """Select by Top1, then Top5, then the ratio closest to the PTQ default."""
    if not trials:
        raise ValueError("trials cannot be empty")
    return max(
        trials,
        key=lambda trial: (
            float(trial["top1"]),
            float(trial.get("top5", 0.0)),
            -abs(float(trial["ratio"]) - 1.0),
        ),
    )
