# Quantization Pipeline Corrections Design

## Goal

Correct the quantization experiment harness without changing timm model
implementations. The corrected harness must compare backends fairly, apply
ModelOpt selective quantization through supported configuration rules, and use
the same representative validation subset when ranking fallback candidates.

## Scope

- Remove synchronization from TensorRT inference callables. Synchronization
  belongs to the benchmark or evaluation boundary.
- Add deterministic, class-balanced validation sampling for quick experiments.
- Compare each fallback candidate against FP32 and default fake-quant baselines
  evaluated on exactly the same subset.
- Express ModelOpt exclusions as ordered wildcard rules in `quant_cfg`.
- Bring the CNN TensorRT builder to the same configurable FP16+INT8 build,
  profile, optimization-level, and timing-cache approach as the Transformer
  builder.
- Add tests that run without CUDA, TensorRT, ImageNet, or ModelOpt installed.
- Update reports to distinguish historical measurements from corrected results
  that still need to be rerun.

## Non-Goals

- Change ResNet, MobileNet, ViT, or Swin architecture code.
- Generate replacement performance numbers without the original GPU and
  ImageNet environment.
- Merge the CNN and Transformer experiment directories into a new framework.

## Design

`my_optim/experiment_utils.py` will contain environment-independent helpers:

- `class_balanced_indices(targets, sample_size, seed)` returns deterministic
  indices spread across classes.
- `make_modelopt_exclusion_rules(fragments)` returns ordered ModelOpt wildcard
  rules that disable input and weight quantizers for matching modules.

Both `common.py` modules will support an optional sampler and will validate that
an evaluation processed at least one sample. The Transformer fallback script
will construct one balanced subset, evaluate FP32 and default fake-quant on it,
then evaluate every candidate on that same subset.

TensorRT callables remain asynchronous. Accuracy evaluation synchronizes once
after inference before reading logits. Latency measurement owns CUDA event
synchronization. This keeps backend timing boundaries consistent.

The CNN builder will use weakly typed Q/DQ parsing with FP16 and INT8 flags for
the corrected INT8 path, expose profile batch sizes, enable optimization level
5 and timing iterations, and persist a timing cache.

## Validation

- Unit tests for balanced sampling and generated ModelOpt wildcard rules.
- Source-level regression tests ensuring TensorRT callables do not synchronize.
- Source-level regression tests for CNN builder defaults and INT8 flags.
- Python compilation and focused standard-library `unittest` execution.
- No claims that corrected GPU performance passes until the pipeline is rerun
  in the original CUDA/TensorRT environment.
