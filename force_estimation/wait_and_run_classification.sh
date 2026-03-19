#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gpuadmin/Jins/Classification_with_nano17/force_estimation"
GPU_ID="${GPU_ID:-0}"
FOLDS="${FOLDS:-4 5 6}"
WAIT_INTERVAL="${WAIT_INTERVAL:-30}"
CLASS_LOG_DIR="${CLASS_LOG_DIR:-$ROOT/logs}"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <regression_pid>"
  exit 1
fi

REG_PID="$1"

cd "$ROOT"
mkdir -p "$CLASS_LOG_DIR"

echo "Watching regression PID: $REG_PID"
echo "Classification will use GPU_ID=$GPU_ID, FOLDS=$FOLDS"

while kill -0 "$REG_PID" 2>/dev/null; do
  sleep "$WAIT_INTERVAL"
done

echo "Regression PID $REG_PID exited. Starting classification."

CLASS_LOG="$CLASS_LOG_DIR/classification_after_regression_$(date +%F_%H-%M-%S).log"
nohup env GPU_ID="$GPU_ID" FOLDS="$FOLDS" ./run_classification_6fold.sh > "$CLASS_LOG" 2>&1 &
CLASS_PID=$!

echo "Started classification PID: $CLASS_PID"
echo "Classification log: $CLASS_LOG"
