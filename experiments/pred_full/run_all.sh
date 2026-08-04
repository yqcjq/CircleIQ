#!/bin/bash
# pred_full 全链路。用法: bash run_all.sh [main|pre]
set -e
P=${1:-main}
cd ~/circleiq/pred_full
PY=~/circleiq/venv/bin/python
suffix=$([ "$P" = "main" ] && echo "" || echo "_$P")

echo "=== build ($P) ==="
$PY build_dataset.py --partition $P --workers 10 2>&1 | tee build$suffix.log
echo "=== baselines ==="
$PY baselines.py --partition $P 2>&1 | tee baselines$suffix.log
echo "=== gbdt ==="
$PY train_gbdt.py --partition $P 2>&1 | tee gbdt$suffix.log
echo "=== gru ==="
$PY train_gru.py --partition $P 2>&1 | tee gru$suffix.log
echo "=== evaluate ==="
$PY evaluate.py --partition $P 2>&1 | tee eval$suffix.log
echo "ALL DONE ($P)"
