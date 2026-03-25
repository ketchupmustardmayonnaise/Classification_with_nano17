#!/usr/bin/env bash
set -e

OUT_DIR_NAME="output"
DATASET_DIR_NAME="24_subject_dataset"

# 현재 sh 파일의 절대경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 상위 3개 폴더를 ROOT로 설정
ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
OUT_BASE="$ROOT/force_estimation/data/current_dataset/$OUT_DIR_NAME"

DATASET_ROOT="$ROOT/force_estimation/data/current_dataset/$DATASET_DIR_NAME"
MODULE="force_estimation.data.current_dataset.build_h5_cross_subject"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ROOT로 이동
cd "$ROOT"

"$PYTHON_BIN" -m "$MODULE" \
  --root "$DATASET_ROOT" \
  --out_dir "$OUT_BASE/fingerpad/fold1" \
  --split loso \
  --test_subjects "subject7 subject8 subject13 subject23" \
  --val_subjects "subject1 subject4 subject18 subject24" \
  --use_only_fingerpad True \
  --otsu_k 0.0


"$PYTHON_BIN" -m "$MODULE" \
  --root "$DATASET_ROOT" \
  --out_dir "$OUT_BASE/fingerpad/fold2" \
  --split loso \
  --test_subjects "subject1 subject4 subject18 subject24" \
  --val_subjects "subject3 subject9 subject12 subject17" \
  --use_only_fingerpad True \
  --otsu_k 0.0


"$PYTHON_BIN" -m "$MODULE" \
  --root "$DATASET_ROOT" \
  --out_dir "$OUT_BASE/fingerpad/fold3" \
  --split loso \
  --test_subjects "subject3 subject9 subject12 subject17" \
  --val_subjects "subject2 subject6 subject20 subject22" \
  --use_only_fingerpad True \
  --otsu_k 0.0


"$PYTHON_BIN" -m "$MODULE" \
  --root "$DATASET_ROOT" \
  --out_dir "$OUT_BASE/fingerpad/fold4" \
  --split loso \
  --test_subjects "subject2 subject6 subject20 subject22" \
  --val_subjects "subject11 subject16 subject19 subject21" \
  --use_only_fingerpad True \
  --otsu_k 0.0


"$PYTHON_BIN" -m "$MODULE" \
  --root "$DATASET_ROOT" \
  --out_dir "$OUT_BASE/fingerpad/fold5" \
  --split loso \
  --test_subjects "subject11 subject16 subject19 subject21" \
  --val_subjects "subject5 subject10 subject14 subject15" \
  --use_only_fingerpad True \
  --otsu_k 0.0


"$PYTHON_BIN" -m "$MODULE" \
  --root "$DATASET_ROOT" \
  --out_dir "$OUT_BASE/fingerpad/fold6" \
  --split loso \
  --test_subjects "subject5 subject10 subject14 subject15" \
  --val_subjects "subject7 subject8 subject13 subject23" \
  --use_only_fingerpad True \
  --otsu_k 0.0


# ====================================


"$PYTHON_BIN" -m "$MODULE" \
  --root "$DATASET_ROOT" \
  --out_dir "$OUT_BASE/bodypart/fold1" \
  --split loso \
  --test_subjects "subject7 subject8 subject13 subject23" \
  --val_subjects "subject1 subject4 subject18 subject24" \
  --use_only_fingerpad False \
  --otsu_k 0.0


"$PYTHON_BIN" -m "$MODULE" \
  --root "$DATASET_ROOT" \
  --out_dir "$OUT_BASE/bodypart/fold2" \
  --split loso \
  --test_subjects "subject1 subject4 subject18 subject24" \
  --val_subjects "subject3 subject9 subject12 subject17" \
  --use_only_fingerpad False \
  --otsu_k 0.0

"$PYTHON_BIN" -m "$MODULE" \
  --root "$DATASET_ROOT" \
  --out_dir "$OUT_BASE/bodypart/fold3" \
  --split loso \
  --test_subjects "subject3 subject9 subject12 subject17" \
  --val_subjects "subject2 subject6 subject20 subject22" \
  --use_only_fingerpad False \
  --otsu_k 0.0

"$PYTHON_BIN" -m "$MODULE" \
  --root "$DATASET_ROOT" \
  --out_dir "$OUT_BASE/bodypart/fold4" \
  --split loso \
  --test_subjects "subject2 subject6 subject20 subject22" \
  --val_subjects "subject11 subject16 subject19 subject21" \
  --use_only_fingerpad False \
  --otsu_k 0.0

"$PYTHON_BIN" -m "$MODULE" \
  --root "$DATASET_ROOT" \
  --out_dir "$OUT_BASE/bodypart/fold5" \
  --split loso \
  --test_subjects "subject11 subject16 subject19 subject21" \
  --val_subjects "subject5 subject10 subject14 subject15" \
  --use_only_fingerpad False \
  --otsu_k 0.0

"$PYTHON_BIN" -m "$MODULE" \
  --root "$DATASET_ROOT" \
  --out_dir "$OUT_BASE/bodypart/fold6" \
  --split loso \
  --test_subjects "subject5 subject10 subject14 subject15" \
  --val_subjects "subject7 subject8 subject13 subject23" \
  --use_only_fingerpad False \
  --otsu_k 0.0