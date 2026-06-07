# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Install: `python -m pip install -e .`
- Run tests: `pytest tests/`
- Run specific test: `pytest tests/test_models.py::test_specific_function -v`
- Run tests in parallel: `pytest -n 4 tests/`
- Filter tests: `pytest -k "substring-to-match" tests/`
- Validate a model: `python validate.py /path/to/data --model resnet50`
- Benchmark: `python benchmark.py --model resnet50`

## Code Style Guidelines
- Line length: 120 chars
- Indentation: 4-space hanging indents; arguments get an extra indent level; closing paren and colon go on their own line ("sadface" style)
- Typing: PEP 484 type annotations in all function signatures
- Docstrings: Google style — do not repeat type annotations or defaults
- Imports: stdlib → third-party → local
- Naming: `snake_case` for functions/variables, `PascalCase` for classes
- Error handling: `try/except` with specific exception types
- Conditional expressions: parentheses around complex conditions

## Architecture Overview

### Package Structure

`timm/` contains six main subpackages:

| Subpackage | Purpose |
|---|---|
| `timm/models/` | 100+ architecture definitions and the model registry/factory |
| `timm/layers/` | Reusable building blocks (attention, norm, activation, positional encoding) |
| `timm/data/` | Dataset readers, transforms, augmentation, data loaders |
| `timm/optim/` | 20+ optimizer implementations and parameter grouping utilities |
| `timm/scheduler/` | LR schedulers (cosine, polynomial, tanh, step, plateau) |
| `timm/loss/` | Loss functions supporting label smoothing and soft targets |
| `timm/task/` | Task abstractions (`ClassificationTask`, `DistillationTask`) for training |
| `timm/utils/` | ModelEMA, checkpointing, mixed-precision scaler, distributed helpers |

Root-level scripts (`train.py`, `validate.py`, `inference.py`, `benchmark.py`) wire these together for end-to-end workflows.

### Model Registry & Factory

The model system centres on three files in `timm/models/`:

- **`_registry.py`** — global dicts mapping model names to factory functions and `PretrainedCfg` objects. The `@register_model` decorator populates these at import time.
- **`_pretrained.py`** — `PretrainedCfg` dataclass holding weight source (URL / HF Hub / local file), input normalization, class count, and metadata. `DefaultCfg` groups multiple pretrained variants per architecture.
- **`_factory.py`** — `create_model(name, pretrained, **kwargs)` is the main public entry point. It resolves the name, calls the registered factory function, then optionally loads pretrained weights via `_builder.py`.

Model names follow `{architecture}.{tag}` (e.g. `resnet50.tv_in1k`, `vit_base_patch16_224.augreg_in21k_ft_in1k`). Calling `timm.create_model('resnet50', pretrained=True)` is the canonical usage.

### Adding a New Architecture

1. Create `timm/models/my_model.py`.
2. Define model classes and a `default_cfgs` dict of `PretrainedCfg` entries.
3. Decorate each public variant function with `@register_model`.
4. Import the module in `timm/models/__init__.py`.

The `build_model_with_cfg()` helper in `_builder.py` handles pretrained weight loading, class-count adaptation, and feature extraction wiring — prefer it over rolling your own weight-loading logic.

### Feature Extraction

Two mechanisms exist for extracting intermediate features:

- **Hook-based** (`_features.py`): wraps a model to return a list/dict of feature maps via forward hooks. Used by `create_feature_extractor()`.
- **FX-based** (`_features_fx.py`): uses `torch.fx` to trace the model graph and splice out named nodes. More reliable for complex control flow.

`FeatureListNet` / `FeatureDictNet` are the wrapper classes produced.

### Data Pipeline

`timm/data/` resolves config → applies transforms → wraps in a loader:

1. `resolve_data_config(model)` — reads the model's `pretrained_cfg` to infer mean/std/input size.
2. `create_transform(cfg)` — builds a `torchvision.transforms` pipeline with optional RandAugment / AutoAugment / AugMix / RandomErasing.
3. `create_loader(dataset, cfg)` — wraps in a `DataLoader` with optional Mixup/CutMix collation and prefetching.

### Pretrained Weights

Weights are hosted on Hugging Face Hub under `timm/<model-name>` or at legacy `torch.hub` URLs. `safetensors` is the preferred format. Local paths and in-memory state dicts are also supported via `PretrainedCfg`.

### Layer Configuration

`timm/layers/` uses a global config object (`set_layer_config(scriptable=True, exportable=True, ...)`) to toggle JIT-compatible code paths. This affects factory functions like `create_act_layer()` and `create_attn()`. Set this before building a model when targeting `torch.script` or ONNX export.
