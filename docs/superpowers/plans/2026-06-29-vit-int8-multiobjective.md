# ViT INT8 Multi-Objective Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable ViT-B/16 INT8 experiment matrix that searches bounded PTQ candidates and fixed batch-size TensorRT engines under a two-percentage-point Top-1 constraint.

**Architecture:** Keep model quantization, engine construction, evaluation, and Pareto selection as separate file-based stages. Extend the existing scripts with small testable interfaces, and add one dependency-free launcher that prints commands and optionally executes them on the server.

**Tech Stack:** Python 3, unittest, PyTorch, NVIDIA ModelOpt, ONNX, TensorRT 10.15, Bash

---

## File Structure

- `my_optim/experiment_utils.py`: pure validation, naming, padding, accuracy filtering, and Pareto helpers.
- `my_optim_Trans/run_ptq_int8.py`: export named Max, MSE, and SmoothQuant Q/DQ candidates.
- `my_optim_Trans/build_trt_engine.py`: support fixed `min=opt=max` profiles and engine metadata.
- `my_optim_Trans/run_trt_eval.py`: fixed-engine padding, cached outputs, GPU and E2E measurements.
- `my_optim_Trans/run_vit_int8_matrix.py`: resumable command matrix and result summarization.
- `my_optim_Trans/run_vit_int8_matrix.sh`: one-command server entry point.
- `tests/test_quantization_experiments.py`: dependency-free behavior and source regression tests.
- `my_optim_Trans/VIT_INT8_MATRIX.md`: server commands, artifacts, and result interpretation.

### Task 1: Pure experiment helpers

**Files:**
- Modify: `tests/test_quantization_experiments.py`
- Modify: `my_optim/experiment_utils.py`

- [ ] **Step 1: Write failing tests for fixed batches, padding, and Pareto selection**

```python
def test_parse_fixed_batches_rejects_duplicates_and_non_positive_values(self):
    utils = importlib.import_module("my_optim.experiment_utils")
    self.assertEqual(utils.parse_fixed_batches("1,4,16,32"), [1, 4, 16, 32])
    for value in ("", "1,1", "0,4", "1,-4"):
        with self.assertRaises(ValueError):
            utils.parse_fixed_batches(value)

def test_padded_batch_size_rounds_up_without_dropping_samples(self):
    utils = importlib.import_module("my_optim.experiment_utils")
    self.assertEqual(utils.padded_batch_size(17, 32), 32)
    self.assertEqual(utils.padded_batch_size(32, 32), 32)

def test_pareto_front_filters_accuracy_and_dominated_trials(self):
    utils = importlib.import_module("my_optim.experiment_utils")
    trials = [
        {"name": "fast", "top1": 84.0, "latency_ms": 1.0, "throughput": 3000},
        {"name": "dominated", "top1": 83.5, "latency_ms": 1.2, "throughput": 2500},
        {"name": "accurate", "top1": 85.0, "latency_ms": 1.5, "throughput": 2700},
        {"name": "ineligible", "top1": 82.0, "latency_ms": 0.8, "throughput": 4000},
    ]
    front = utils.pareto_front(trials, fp32_top1=85.1, max_drop=2.0)
    self.assertEqual([trial["name"] for trial in front], ["accurate", "fast"])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m unittest tests.test_quantization_experiments -v`

Expected: failures because `parse_fixed_batches`, `padded_batch_size`, and `pareto_front` do not exist.

- [ ] **Step 3: Implement minimal pure helpers**

```python
def parse_fixed_batches(value: str) -> List[int]:
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
    if actual <= 0 or fixed <= 0 or actual > fixed:
        raise ValueError("expected 0 < actual <= fixed")
    return fixed

def pareto_front(trials, fp32_top1: float, max_drop: float):
    eligible = [trial for trial in trials if fp32_top1 - float(trial["top1"]) <= max_drop]
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
    return sorted(front, key=lambda trial: (-float(trial["top1"]), float(trial["latency_ms"]), str(trial["name"])))
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m unittest tests.test_quantization_experiments -v`

Expected: all focused tests pass.

### Task 2: Named ModelOpt PTQ candidates

**Files:**
- Modify: `tests/test_quantization_experiments.py`
- Modify: `my_optim_Trans/run_ptq_int8.py`

- [ ] **Step 1: Add source-level tests for candidate CLI and SmoothQuant alpha**

```python
def test_vit_ptq_exports_named_calibration_candidates(self):
    source = _source("my_optim_Trans/run_ptq_int8.py")
    self.assertIn('choices=["max", "mse", "smoothquant"]', source)
    self.assertIn("--smoothquant-alpha", source)
    self.assertIn('"method": "smoothquant"', source)
    self.assertIn("int8_qdq_{candidate}", source)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m unittest tests.test_quantization_experiments.QuantizationExperimentTests.test_vit_ptq_exports_named_calibration_candidates -v`

Expected: failure because the current exporter only uses `INT8_DEFAULT_CFG`.

- [ ] **Step 3: Add `--calib`, `--smoothquant-alpha`, and named output selection**

Implement `make_quant_config(calib, alpha)` using a deep copy. For `mse`, set
`config["algorithm"] = "mse"`. For SmoothQuant, copy
`INT8_SMOOTHQUANT_CFG` and set
`config["algorithm"] = {"method": "smoothquant", "alpha": alpha}`. Name
artifacts `int8_qdq_max`, `int8_qdq_mse`, and
`int8_qdq_smoothquant_a0p5`; preserve the legacy default alias only for Max.

- [ ] **Step 4: Run focused tests and compile the exporter**

Run: `python -m unittest tests.test_quantization_experiments -v`

Run: `python -m py_compile my_optim_Trans/run_ptq_int8.py`

Expected: exit code 0 for both commands.

### Task 3: Fixed-shape engine construction

**Files:**
- Modify: `tests/test_quantization_experiments.py`
- Modify: `my_optim_Trans/build_trt_engine.py`

- [ ] **Step 1: Add a failing fixed-profile regression test**

```python
def test_transformer_builder_supports_fixed_min_opt_max_profile(self):
    source = _source("my_optim_Trans/build_trt_engine.py")
    self.assertIn("--fixed-bs", source)
    self.assertIn("min_bs = opt_bs = max_bs = fixed_bs", source)
    self.assertIn('"fixed_batch_size"', source)
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `python -m unittest tests.test_quantization_experiments.QuantizationExperimentTests.test_transformer_builder_supports_fixed_min_opt_max_profile -v`

Expected: failure because `--fixed-bs` is absent.

- [ ] **Step 3: Add fixed profile and metadata output**

Add optional `fixed_bs` to `build_engine`. When supplied, assign
`min_bs = opt_bs = max_bs = fixed_bs`; otherwise retain the dynamic defaults.
Use these values in `profile.set_shape`. Write `<engine>.json` after a successful
build with model, precision, ONNX, engine, fixed batch, profile shapes,
TensorRT version, and GPU name.

- [ ] **Step 4: Run tests and compile**

Run: `python -m unittest tests.test_quantization_experiments -v`

Run: `python -m py_compile my_optim_Trans/build_trt_engine.py`

Expected: exit code 0.

### Task 4: Fixed-engine evaluation and E2E metrics

**Files:**
- Modify: `tests/test_quantization_experiments.py`
- Modify: `my_optim_Trans/run_trt_eval.py`

- [ ] **Step 1: Add failing regression tests for buffer reuse and padding**

```python
def test_transformer_eval_reuses_outputs_and_pads_fixed_final_batch(self):
    source = _source("my_optim_Trans/run_trt_eval.py")
    self.assertIn("self._output_cache", source)
    self.assertIn("--fixed-bs", source)
    self.assertIn("pad_fixed_batch", source)
    self.assertIn('"e2e_throughput_img_s"', source)
    self.assertIn('"evaluated_samples"', source)
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `python -m unittest tests.test_quantization_experiments.QuantizationExperimentTests.test_transformer_eval_reuses_outputs_and_pads_fixed_final_batch -v`

Expected: failure on the current per-call `torch.empty` implementation.

- [ ] **Step 3: Implement cached output and fixed-batch padding**

Cache output tensors by `(batch, device)`. Add `pad_fixed_batch(images,
fixed_bs)` that allocates a fixed-size tensor, copies real images into its
prefix, and returns `(padded, actual_bs)`. Trim `logits[:actual_bs]` before
accuracy accounting. Add E2E wall-clock measurement around the full validation
loop and persist exact evaluated sample count.

- [ ] **Step 4: Restrict latency sweep to the engine's supported shape**

When `--fixed-bs` is supplied, call `measure_latency` only for that batch.
Keep the current `(1, EVAL_BATCH_SIZE)` sweep for dynamic engines.

- [ ] **Step 5: Run tests and compile**

Run: `python -m unittest tests.test_quantization_experiments -v`

Run: `python -m py_compile my_optim_Trans/run_trt_eval.py`

Expected: exit code 0.

### Task 5: Resumable matrix and Pareto report

**Files:**
- Create: `my_optim_Trans/run_vit_int8_matrix.py`
- Create: `my_optim_Trans/run_vit_int8_matrix.sh`
- Modify: `tests/test_quantization_experiments.py`

- [ ] **Step 1: Add failing tests for matrix commands and result filtering**

Test that the source contains the candidates `max`, `mse`, SmoothQuant alpha
values `0.3,0.5,0.7,0.9`, fixed batches `1,4,16,32`, `--max-top1-drop`,
`--dry-run`, reuse checks, and `pareto_front`.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_quantization_experiments -v`

Expected: failure because the launcher does not exist.

- [ ] **Step 3: Implement the launcher**

The launcher validates `OPTIM_MODEL == "vit_base_patch16_224"`, generates PTQ,
build, and eval subprocess commands, skips existing successful artifacts unless
`--force` is passed, and supports `--dry-run`. It always includes existing
`scale_search` and `mlp_fallback` ONNX files when present. After runs, it loads
result JSON files, obtains FP32 Top-1 from `baseline.json`, filters candidates
using `--max-top1-drop`, calls `pareto_front`, and atomically writes
`vit_int8_matrix.json` and `vit_int8_matrix.md`.

- [ ] **Step 4: Add the shell entry point**

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export OPTIM_MODEL=vit_base_patch16_224
PY="${PY:-python}"
exec "$PY" run_vit_int8_matrix.py "$@"
```

- [ ] **Step 5: Run tests, compile, and dry-run**

Run: `python -m unittest tests.test_quantization_experiments -v`

Run: `python -m py_compile my_optim_Trans/run_vit_int8_matrix.py`

Run: `python my_optim_Trans/run_vit_int8_matrix.py --dry-run`

Expected: tests and compilation exit 0; dry-run prints candidate commands
without importing ModelOpt, ONNX, TensorRT, or CUDA.

### Task 6: Documentation and final verification

**Files:**
- Create: `my_optim_Trans/VIT_INT8_MATRIX.md`
- Modify: `docs/superpowers/plans/2026-06-29-vit-int8-multiobjective.md`

- [ ] **Step 1: Document server prerequisites, quick run, and artifacts**

Document the default command, dry-run, candidate-only runs, `--force`, expected
disk/build cost, accuracy threshold, fixed-batch artifacts, and the distinction
between GPU-only and E2E throughput.

- [ ] **Step 2: Run the full local verification suite**

Run: `python -m unittest tests.test_quantization_experiments -v`

Run: `python -m py_compile my_optim/experiment_utils.py my_optim_Trans/run_ptq_int8.py my_optim_Trans/build_trt_engine.py my_optim_Trans/run_trt_eval.py my_optim_Trans/run_vit_int8_matrix.py`

Run: `git diff --check`

Expected: all tests pass, compilation exits 0, and diff check prints nothing.

- [ ] **Step 3: Review scope and record unverified GPU work**

Confirm no timm model files changed. State explicitly that CUDA Graph capture,
TensorRT builds, full ImageNet accuracy, and speed measurements require the
target server and have not been claimed locally.
