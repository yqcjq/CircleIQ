#!/bin/bash
# 数据补齐后的总重跑链:阶段A(最终数据) -> Hawkes全量(最终划分) -> 阶段CDE
# 用法: nohup bash chain_all.sh > chain_all.log 2>&1 &
cd /root/data
source ~/circleiq/venv/bin/activate
# 线程上限:8-12 进程并行时防 BLAS/polars/numba 超订(24核 load 曾冲到 134)
export POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMBA_NUM_THREADS=2

echo "=== CHAIN: stage_a $(date +%H:%M:%S) ==="
bash stage_a.sh > stage_a_final.log 2>&1 || { echo "STAGE_A FAILED"; exit 1; }

echo "=== CHAIN: hawkes full $(date +%H:%M:%S) ==="
rm -rf hawkes
python3 run_hawkes.py --category all --pool 12 --dims 10 --workers 8 > hawkes_full_final.log 2>&1 || { echo "HAWKES FAILED"; exit 1; }

echo "=== CHAIN: stage_cde $(date +%H:%M:%S) ==="
bash stage_cde.sh > stage_cde.log 2>&1 || { echo "CDE FAILED"; exit 1; }

echo "CHAIN_ALL_DONE $(date +%H:%M:%S)"
