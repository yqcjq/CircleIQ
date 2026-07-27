#!/bin/bash
# 杀链条+全部计算进程(embed 除外)
CHAIN_PID=$(ps -eo pid,cmd | grep "[c]hain_all.sh" | awk '{print $1}')
[ -n "$CHAIN_PID" ] && kill -9 $CHAIN_PID 2>/dev/null
pkill -9 -f "[r]un_hawkes" 2>/dev/null
pkill -9 -f "[s]tage_a.sh" 2>/dev/null
pkill -9 -f "[s]tage_cde.sh" 2>/dev/null
# spawn workers,但保留 embed(它不是 spawn_main)
pkill -9 -f "[s]pawn_main" 2>/dev/null
sleep 1
echo "remaining compute:"; ps -eo pid,cmd | grep -cE "[r]un_hawkes|[s]pawn_main|[c]hain_all"
