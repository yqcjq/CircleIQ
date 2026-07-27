#!/bin/bash
# 阶段C/D/E收尾:计数验证 -> 内容调制分析 -> 策略搜索 -> 大V反事实案例
# 用法: nohup bash stage_cde.sh > stage_cde.log 2>&1 &
set -e
cd /root/data
source ~/circleiq/venv/bin/activate
export POLARS_MAX_THREADS=3

echo "=== [1/4] validate_rolling $(date +%H:%M:%S) ==="
python3 validate_rolling.py --workers 12

echo "=== [2/4] content modulation $(date +%H:%M:%S) ==="
python3 analyze_content_modulation.py

echo "=== [3/4] strategy top8 $(date +%H:%M:%S) ==="
python3 strategy_optimizer.py --auto-top 8 --workers 8

echo "=== [4/4] case studies $(date +%H:%M:%S) ==="
python3 case_study.py --name hot__20260313162141
python3 case_study.py --name hot__20260319181003

echo "STAGE CDE ALL DONE $(date +%H:%M:%S)"
