# Quantization Pipeline Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the quantization experiment scripts fair, reproducible, and consistent across CNN and Transformer models.

**Architecture:** Add small dependency-free experiment helpers, keep TensorRT synchronization at measurement boundaries, use one deterministic balanced validation subset for fallback comparisons, and align the CNN builder with the proven Transformer builder settings.

**Tech Stack:** Python, PyTorch, NVIDIA ModelOpt, TensorRT, unittest

---

### Task 1: Experiment helper tests

**Files:**
- Create: `tests/test_quantization_experiments.py`
- Create: `my_optim/experiment_utils.py`

- [x] Write tests for deterministic class-balanced indices and ModelOpt wildcard exclusion rules.
- [x] Run the focused tests and confirm they fail because the helper module does not exist.
- [x] Implement the minimal dependency-free helper functions.
- [x] Run the focused tests and confirm they pass.

### Task 2: Fair TensorRT timing

**Files:**
- Modify: `my_optim/common.py`
- Modify: `my_optim/run_trt_eval.py`
- Modify: `my_optim/run_summary.py`
- Modify: `my_optim_Trans/common.py`
- Modify: `my_optim_Trans/run_trt_eval.py`
- Modify: `my_optim_Trans/run_summary.py`
- Test: `tests/test_quantization_experiments.py`

- [x] Write source-regression tests that reject synchronization inside TensorRT callables.
- [x] Confirm the tests fail on the current wrappers.
- [x] Remove wrapper synchronization and synchronize once in accuracy evaluation.
- [x] Confirm focused tests pass.

### Task 3: Correct fallback selection

**Files:**
- Modify: `my_optim_Trans/common.py`
- Modify: `my_optim_Trans/run_sensitive_fallback.py`
- Test: `tests/test_quantization_experiments.py`

- [x] Add tests for sampler plumbing and wildcard configuration shape.
- [x] Confirm the current behavior fails.
- [x] Build one class-balanced subset and evaluate FP32, default fake-quant, and all fallback candidates on it.
- [x] Use ordered wildcard rules instead of `override_fn`.
- [x] Save the search results and selected fragments to JSON.

### Task 4: Align the CNN TensorRT builder

**Files:**
- Modify: `my_optim/build_trt_engine.py`
- Test: `tests/test_quantization_experiments.py`

- [x] Add regression assertions for FP16+INT8 flags, configurable profile batches, optimization level, timing iterations, and timing cache.
- [x] Confirm the assertions fail against the old builder.
- [x] Port the relevant builder behavior from `my_optim_Trans/build_trt_engine.py`.
- [x] Confirm focused tests pass.

### Task 5: Documentation and verification

**Files:**
- Modify: `my_optim/quantization_acceleration_notes_detailed.md`
- Modify: `my_optim_Trans/REPORT.md`
- Modify: `my_optim_Trans/run_all.sh`

- [x] Mark old CNN INT8 and fallback conclusions as historical pending corrected reruns.
- [x] Add `set -o pipefail` to the shell pipeline.
- [x] Run focused unittest, Python compilation, and `git diff --check`.
- [x] Review the final diff for unrelated changes.
