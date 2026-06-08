#!/usr/bin/env bash
# run_all.sh — Full quantization pipeline for ViT + Swin in the deepburst env.
#
# Runs all stages in sequence:
#   Stage 1: FP32 + Torch FP16 baseline (accuracy + latency)
#   Stage 2: ONNX export (fp32 dynamic-batch, TRT-ready)
#   Stage 3: TRT FP16 engine build
#   Stage 4: TRT FP16 eval (accuracy + latency)
#   Stage 5: INT8 PTQ (ModelOpt Q/DQ ONNX export)
#   Stage 6: TRT INT8 engine build
#   Stage 7: TRT INT8 eval
#   Stage 8: Summary table
#
# Usage:
#   bash run_all.sh [vit|swin|both]   (default: both)
#
# To run a single stage for a specific model:
#   OPTIM_MODEL=vit_base_patch16_224 python run_baseline.py
#   OPTIM_MODEL=vit_base_patch16_224 python export_onnx_fp32.py
#   ... etc.
#
# Conda env: deepburst
# Dataset: data/ (symlink to ../my_optim/data — 50k val + 500 calib)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY=/data1/liurongying/miniconda3/envs/deepburst/bin/python
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

export HF_ENDPOINT=https://hf-mirror.com

TARGET="${1:-both}"

run_model() {
    local MODEL="$1"
    echo ""
    echo "========================================================"
    echo "  $MODEL"
    echo "========================================================"
    export OPTIM_MODEL="$MODEL"

    echo "[1/8] FP32 + Torch FP16 baseline..."
    $PY "$SCRIPT_DIR/run_baseline.py" 2>&1 | tee "$LOG_DIR/${MODEL}_baseline.log"

    echo "[2/8] ONNX export (fp32 dynamic-batch)..."
    $PY "$SCRIPT_DIR/export_onnx_fp32.py" 2>&1 | tee "$LOG_DIR/${MODEL}_onnx.log"

    echo "[3/8] TRT FP16 engine build..."
    $PY "$SCRIPT_DIR/build_trt_engine.py" --precision fp16 2>&1 | tee "$LOG_DIR/${MODEL}_trt_fp16_build.log"

    echo "[4/8] TRT FP16 eval..."
    $PY "$SCRIPT_DIR/run_trt_eval.py" --precision fp16 2>&1 | tee "$LOG_DIR/${MODEL}_trt_fp16_eval.log"

    echo "[5/8] INT8 PTQ (ModelOpt QDQ export)..."
    $PY "$SCRIPT_DIR/run_ptq_int8.py" 2>&1 | tee "$LOG_DIR/${MODEL}_ptq_int8.log"

    echo "[6/8] TRT INT8 engine build..."
    $PY "$SCRIPT_DIR/build_trt_engine.py" --precision int8 2>&1 | tee "$LOG_DIR/${MODEL}_trt_int8_build.log"

    echo "[7/8] TRT INT8 eval..."
    $PY "$SCRIPT_DIR/run_trt_eval.py" --precision int8 2>&1 | tee "$LOG_DIR/${MODEL}_trt_int8_eval.log"

    echo "[8/8] Summary..."
    $PY "$SCRIPT_DIR/run_summary.py" 2>&1 | tee "$LOG_DIR/${MODEL}_summary.log"
}

case "$TARGET" in
    vit)
        run_model vit_base_patch16_224
        ;;
    swin)
        run_model swin_tiny_patch4_window7_224
        ;;
    both|*)
        run_model vit_base_patch16_224
        run_model swin_tiny_patch4_window7_224
        echo ""
        echo "=== Combined summary ==="
        $PY "$SCRIPT_DIR/run_summary.py" --all 2>&1 | tee "$LOG_DIR/combined_summary.log"
        ;;
esac

echo ""
echo "All done. Logs in $LOG_DIR/"
