#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gpuadmin/Jins/Classification_with_nano17/force_estimation"
DATA_BASE="$ROOT/data/current_dataset/output/bodypart"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
EXTRA_ARGS=("$@")

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "Using single GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for fold in 4 5 6; do
  DATA_DIR="$DATA_BASE/fold${fold}"

  [ -f "$DATA_DIR/train.h5" ] || { echo "missing $DATA_DIR/train.h5"; exit 1; }
  [ -f "$DATA_DIR/val.h5" ] || { echo "missing $DATA_DIR/val.h5"; exit 1; }
  [ -f "$DATA_DIR/test.h5" ] || { echo "missing $DATA_DIR/test.h5"; exit 1; }

  echo "===== Regression Fold ${fold} start ====="

  "$PYTHON_BIN" run_regression.py \
    data.data_dir="$DATA_DIR" \
    save_directory="./saved_models/6fold_regression/fold${fold}" \
    wandb.group="12_dataset_6fold_regression_fold${fold}" \
    "${EXTRA_ARGS[@]}"

  echo "===== Regression Fold ${fold} end ====="
done
