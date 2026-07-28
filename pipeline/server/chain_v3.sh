#!/bin/bash
# v2 重跑:分段μ Hawkes 全量 -> 滚动验证 -> 调制 -> 策略 -> 案例
# 用法: nohup bash chain_v3.sh > chain_v3.log 2>&1 &
cd /root/data
source ~/circleiq/venv/bin/activate
export POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMBA_NUM_THREADS=2

echo "=== V3: selftest $(date +%H:%M:%S) ==="
python3 hawkes_fit.py || { echo "SELFTEST FAILED"; exit 1; }

echo "=== V3: hawkes full $(date +%H:%M:%S) ==="
rm -rf hawkes
python3 run_hawkes.py --category all --pool 12 --dims 10 --workers 8 > hawkes_v3.log 2>&1 || { echo "HAWKES FAILED"; exit 1; }
tail -2 hawkes_v3.log

echo "=== V3: stage_cde $(date +%H:%M:%S) ==="
bash stage_cde.sh > stage_cde_v3.log 2>&1 || { echo "CDE FAILED"; tail -5 stage_cde_v3.log; exit 1; }
tail -3 stage_cde_v3.log

echo "CHAIN_V3_DONE $(date +%H:%M:%S)"
