# ViT INT8 Multi-Objective Optimization Design

## Goal

Improve `vit_base_patch16_224` TensorRT INT8 inference for batch sizes 1, 4,
16, and 32 while limiting ImageNet Top-1 loss to at most 2.0 percentage
points relative to the PyTorch FP32 baseline. The experiment must report both
GPU-only performance and end-to-end performance and must select Pareto-optimal
configurations instead of assuming that the most heavily quantized model is
the fastest.

## Current Baseline

The repository already provides:

- NVIDIA ModelOpt INT8 PTQ with explicit Q/DQ ONNX export.
- A dynamic TensorRT engine with profile `min=1, opt=32, max=64`, FP16 and
  INT8 builder flags, optimization level 5, eight timing iterations, and a
  persistent timing cache.
- Architecture-group activation-scale search.
- Accuracy-driven FP16 fallback search that identifies ViT MLP layers as the
  main sensitive group.
- ImageNet accuracy and CUDA-event latency measurements.

The corrected full-ImageNet baseline on RTX 3080 Ti is 85.104% Top-1. Full
INT8 reaches 83.312% and the MLP-FP16 mixed engine reaches 85.034%. The
existing dynamic engines report 1.80 ms at batch 1 and 1,979 images/s at batch
32 for full INT8, or 1.87 ms and 2,367 images/s for the mixed engine.

## Success Criteria

The implementation is successful when it can be run on the target server to:

1. Build and evaluate fixed-shape engines for batches 1, 4, 16, and 32.
2. Compare the existing default PTQ model with MSE calibration, SmoothQuant
   alpha candidates, scale-search output, and the existing MLP fallback.
3. Reuse one calibrated Q/DQ model across all batch-specific engine builds.
4. Reject final candidates whose full ImageNet Top-1 loss exceeds 2.0
   percentage points.
5. Report GPU-only latency/throughput, end-to-end throughput, Top-1, Top-5,
   and cross-engine numerical consistency.
6. Produce a machine-readable summary and a concise Markdown Pareto table.

Absolute speedup is not a local acceptance criterion because this workstation
does not have the target CUDA, TensorRT, GPU, or ImageNet environment.

## Considered Approaches

### 1. Reproduce layer-local MSE scale search only

For every quantizer, test multiple clipping ratios and minimize local
round-trip reconstruction error. This closely follows the comparison method
provided by the user. It is straightforward but expensive, and local tensor
MSE does not always predict final Top-1 or TensorRT speed. It also does not
address the current dynamic-engine performance limitation.

### 2. Engine-first multi-objective search (selected)

Combine fixed-shape TensorRT builds with a bounded set of quantization
candidates, then select configurations using measured accuracy and runtime.
This directly addresses both major uncertainty sources: quantization quality
and deployment-engine quality. It reuses the existing scale and fallback
searches instead of replacing them.

### 3. Quantization-aware training

Fine-tune the ViT with fake quantization enabled. QAT is the fallback if PTQ
cannot meet the accuracy constraint, but it requires a training recipe,
training data, checkpoints, and much more server time. It is excluded from the
first implementation.

## Architecture

The work remains inside `my_optim_Trans/` and the dependency-free experiment
helpers in `my_optim/experiment_utils.py`. No timm model implementation is
modified.

The pipeline has four independent stages:

1. **Quantize and export**: produce one Q/DQ ONNX artifact per quantization
   candidate.
2. **Build**: compile one fixed-shape TensorRT engine per candidate and batch
   size, with `min=opt=max`.
3. **Evaluate**: measure accuracy, GPU-only runtime, and end-to-end runtime
   using explicit and consistent boundaries.
4. **Select**: filter by the 2.0-point accuracy constraint and compute the
   non-dominated configurations for latency, throughput, and accuracy.

Stages communicate through files and JSON metadata so failed or interrupted
server runs can resume without recalibrating successful candidates.

## Quantization Candidates

The bounded first-round search contains:

- `max`: the existing `INT8_DEFAULT_CFG` baseline.
- `mse`: `INT8_DEFAULT_CFG` with ModelOpt MSE calibration.
- `smoothquant`: `INT8_SMOOTHQUANT_CFG` with alpha values 0.3, 0.5, 0.7,
  and 0.9.
- `scale_search`: the existing ViT group-wise amax search result.
- `mlp_fallback`: the existing MLP `fc1` and `fc2` FP16 fallback.

Calibration uses a deterministic, class-balanced subset of 3,000 ImageNet
validation images by default. The subset size and seed are command-line
options. A candidate is calibrated once and its exported Q/DQ scales are
reused by every batch-specific TensorRT build.

A quick subset evaluation ranks candidates before expensive engine builds.
The final accuracy decision always uses the complete 50,000-image validation
set. Search-subset accuracy is never reported as final accuracy.

## TensorRT Engine Matrix

For every surviving Q/DQ ONNX candidate, build four engines:

```text
batch size: 1, 4, 16, 32
profile:    min = opt = max = batch size
precision:  FP16 fallback + explicit Q/DQ INT8
builder:    optimization level 5, eight timing iterations, timing cache
```

The current dynamic `min=1, opt=32, max=64` engine remains as the control.
Engine filenames encode candidate and batch size. Build metadata records the
TensorRT version, GPU name, profile, flags, ONNX path, and timing-cache path.

Fixed profiles are separate engines rather than multiple profiles in one
engine. This prevents one profile or execution context from contaminating
batch-specific measurements and allows tactics available only for
`MIN=OPT=MAX`.

## Runtime and Measurement

The TensorRT runner caches execution contexts and output buffers by input
shape. It avoids allocating a new output tensor on every inference. Input
shape and tensor addresses are updated only when required.

For a fixed engine, an undersized final validation batch is padded to the
engine batch size and the padded logits are discarded. No validation image is
dropped, including the final batch for batch size 32.

Two timing modes are reported:

- **GPU-only**: warmed-up TensorRT execution measured with CUDA events. This
  excludes DataLoader, CPU transforms, and host-to-device transfer.
- **End-to-end**: wall-clock time around data loading, transfer, inference,
  synchronization, and result consumption.

CUDA Graph is an optional fixed-batch benchmark mode. It uses persistent input
and output buffers and is reported separately from ordinary enqueue results.
If graph capture is unsupported, the candidate records the failure and
continues with ordinary TensorRT execution.

Accuracy evaluation and performance benchmarking remain separate runs so
profiling and synchronization do not distort throughput.

## Numerical Consistency Checks

The same deterministic input batch is evaluated through every fixed engine
created from the same Q/DQ ONNX. The report records maximum absolute logit
error and cosine similarity against the batch-1 engine.

Meaningful batch-dependent accuracy differences are treated as a warning, not
as an expected consequence of fixed-engine compilation. The report also
records the exact number of evaluated images to expose accidental `drop_last`
or last-batch errors.

## Profiling

A separate opt-in profiling run records per-layer TensorRT time as JSON. It is
used to identify:

- Q/DQ and reformat overhead.
- FP32 fallback that should have used FP16.
- unfused attention or MLP patterns.
- INT8 layers whose tactic is slower than the mixed-precision alternative.

Profiling is diagnostic and is not enabled during official latency runs.

## Pareto Selection

Candidates with full-validation Top-1 loss greater than 2.0 points are
ineligible. Among eligible candidates, a result is Pareto-optimal when no
other result has all of:

- equal or better Top-1 accuracy,
- equal or lower batch-1 GPU latency,
- equal or higher batch-4, batch-16, or batch-32 throughput for the selected
  deployment objective.

The summary explicitly identifies:

- lowest batch-1 latency,
- highest batch-4 end-to-end throughput,
- highest batch-16 GPU throughput,
- highest batch-32 GPU throughput,
- best-accuracy eligible configuration.

## Error Handling and Resumability

- Missing ImageNet, ModelOpt, ONNX, CUDA, or TensorRT dependencies produce a
  clear stage-specific error.
- Existing successful ONNX and engine artifacts are reused unless `--force`
  is supplied.
- Each subprocess writes its result atomically after success; partial files do
  not count as completed stages.
- One failed candidate does not discard completed candidates. The matrix
  summary records the failure and continues where safe.
- Candidate names, batch sizes, seeds, calibration sizes, and accuracy limits
  are validated before GPU work starts.

## Testing and Verification

Local tests require no GPU, TensorRT, ModelOpt, or ImageNet. They cover:

- fixed-profile argument validation and engine naming,
- last-batch padding and output trimming,
- Pareto filtering and the 2.0-point accuracy constraint,
- candidate command generation and resumability decisions,
- source-level checks for fixed `min=opt=max` profiles and cached outputs.

Local verification includes focused unit tests, Python compilation, shell
syntax checks where available, and diff whitespace validation. GPU correctness
and speed claims remain pending until the generated server script is run in
the RTX 3080 Ti environment.

## Deliverables

- Extended ViT PTQ export supporting the bounded candidate set.
- Fixed-batch TensorRT engine construction.
- Fixed-engine inference with last-batch padding and buffer reuse.
- Optional CUDA Graph and per-layer profiling modes.
- A resumable ViT INT8 experiment-matrix launcher.
- JSON and Markdown Pareto summaries.
- CPU-only regression tests and server run documentation.

## Non-Goals

- Modifying timm ViT architecture code.
- Supporting Swin or CNN models in this iteration.
- Adding QAT, pruning, structured sparsity, FP8, or INT4.
- Claiming speedups before the server benchmark runs.
- Reproducing another experiment's absolute numbers without matching its GPU,
  TensorRT version, preprocessing, and timing boundaries.
