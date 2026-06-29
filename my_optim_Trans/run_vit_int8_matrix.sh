#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export OPTIM_MODEL=vit_base_patch16_224
PY="${PY:-python}"
exec "$PY" run_vit_int8_matrix.py "$@"
