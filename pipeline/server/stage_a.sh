#!/bin/bash
# 阶段A全量:边表 -> 画像 -> 单议题Leiden -> 稳定圈(K网格+防泄漏版) -> A5评估
# 用法: nohup bash stage_a.sh > stage_a.log 2>&1 &
set -e
cd /root/data
source ~/circleiq/venv/bin/activate

echo "=== [1/6] build_edges $(date +%H:%M:%S) ==="
python3 build_edges.py --workers 4

echo "=== [2/6] user_profiles $(date +%H:%M:%S) ==="
python3 user_profiles.py

echo "=== [3/6] leiden_per_topic $(date +%H:%M:%S) ==="
python3 leiden_per_topic.py

echo "=== [4/6] stable_circles K=3 $(date +%H:%M:%S) ==="
python3 stable_circles.py --k 3

echo "=== [5/6] stable_circles K=3 pre-only(防泄漏) $(date +%H:%M:%S) ==="
python3 stable_circles.py --k 3 --pre-only

echo "=== [6/6] eval_stable $(date +%H:%M:%S) ==="
python3 eval_stable.py --k 3

echo "STAGE A ALL DONE $(date +%H:%M:%S)"
