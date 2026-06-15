#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PY:-python}"
export OPTIM_MODEL="${OPTIM_MODEL:-mobilenetv3_large_100}"

"$PY" run_sensitive_fallback.py \
  --sample-size "${SEARCH_SAMPLES:-2000}" \
  --target-drop "${TARGET_DROP:-0.5}"

"$PY" build_trt_engine.py \
  --precision int8 \
  --onnx-suffix int8_qdq_search_selected_inline.onnx \
  --engine-suffix int8_search.engine \
  --opt-bs 32 \
  --max-bs 64

"$PY" run_trt_eval.py \
  --precision int8 \
  --engine-suffix int8_search.engine \
  --result-tag trt_int8_search \
  --throughput-batch-size 32
