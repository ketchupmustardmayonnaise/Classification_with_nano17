#!/usr/bin/env bash
set -e

REPO_ROOT="/home/gpuadmin/Jins/Classification_with_nano17"
ROOT="/home/gpuadmin/Jins/Classification_with_nano17/force_estimation/data/current_dataset/12_dataset"
OUT_BASE="/home/gpuadmin/Jins/Classification_with_nano17/force_estimation/data/current_dataset/dataset/12_dataset_h5"
MODULE="force_estimation.data.current_dataset.build_h5_cross_subject"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"

"$PYTHON_BIN" -m "$MODULE" \
  --root "$ROOT" \
  --out_dir "$OUT_BASE/fold1" \
  --split loso \
  --test_subjects "subject9 subject13 subject20 subject22" \
  --val_subjects "subject11 subject17 subject18 subject24" \
  --otsu_k 0.0

"$PYTHON_BIN" -m "$MODULE" \
  --root "$ROOT" \
  --out_dir "$OUT_BASE/fold2" \
  --split loso \
  --test_subjects "subject11 subject17 subject18 subject24" \
  --val_subjects "subject10 subject12 subject21 subject23" \
  --otsu_k 0.0

"$PYTHON_BIN" -m "$MODULE" \
  --root "$ROOT" \
  --out_dir "$OUT_BASE/fold3" \
  --split loso \
  --test_subjects "subject10 subject12 subject21 subject23" \
  --val_subjects "subject6 subject7 subject14 subject15" \
  --otsu_k 0.0

"$PYTHON_BIN" -m "$MODULE" \
  --root "$ROOT" \
  --out_dir "$OUT_BASE/fold4" \
  --split loso \
  --test_subjects "subject6 subject7 subject14 subject15" \
  --val_subjects "subject1 subject2 subject4 subject16" \
  --otsu_k 0.0

"$PYTHON_BIN" -m "$MODULE" \
  --root "$ROOT" \
  --out_dir "$OUT_BASE/fold5" \
  --split loso \
  --test_subjects "subject1 subject2 subject4 subject16" \
  --val_subjects "subject3 subject5 subject8 subject19" \
  --otsu_k 0.0

"$PYTHON_BIN" -m "$MODULE" \
  --root "$ROOT" \
  --out_dir "$OUT_BASE/fold6" \
  --split loso \
  --test_subjects "subject3 subject5 subject8 subject19" \
  --val_subjects "subject9 subject13 subject20 subject22" \
  --otsu_k 0.0
