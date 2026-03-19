#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gpuadmin/Jins/Classification_with_nano17/force_estimation"
DATA_BASE="$ROOT/data/current_dataset/output/bodypart"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:- 2}"
FOLDS="${FOLDS:- 6}"
EXTRA_ARGS=("$@")

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "Using single GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for fold in $FOLDS; do
  DATA_DIR="$DATA_BASE/fold${fold}"

  [ -f "$DATA_DIR/train.h5" ] || { echo "missing $DATA_DIR/train.h5"; exit 1; }
  [ -f "$DATA_DIR/val.h5" ] || { echo "missing $DATA_DIR/val.h5"; exit 1; }
  [ -f "$DATA_DIR/test.h5" ] || { echo "missing $DATA_DIR/test.h5"; exit 1; }

  echo "===== Classification Fold ${fold} start ====="

  "$PYTHON_BIN" run_classification.py \
    data.data_dir="$DATA_DIR" \
    "${EXTRA_ARGS[@]}"

  echo "===== Classification Fold ${fold} end ====="
done
