#!/bin/bash
# 持续上传已完成的 core parquet 到 ppio-gpu:~/data/
# 只上传 stats json 已落盘(即 zip 处理完成)的文件;major -> hot -> case 优先级
# 用法: nohup bash pipeline/uploader.sh > /Users/ppio/Desktop/CircleIQ-data/upload.log 2>&1 &
set -u
DATA=/Users/ppio/Desktop/CircleIQ-data
DONE_LIST="$DATA/.uploaded"
touch "$DONE_LIST"
ssh ppio-gpu 'mkdir -p ~/data/core/{major,hot,case} ~/data/text/{major,hot,case} ~/data/stats ~/data/edges' || exit 1

while true; do
  # 按优先级列出已完成的 zip
  for pref in major hot case; do
    for sj in "$DATA"/stats/${pref}__*.json; do
      [ -e "$sj" ] || continue
      base=$(basename "$sj" .json)             # cat__topic__zipstem
      rel=${base#${pref}__}                     # topic__zipstem
      core="$DATA/core/$pref/$rel.parquet"
      grep -qxF "$core" "$DONE_LIST" && continue
      [ -e "$core" ] || continue
      if rsync -a --partial --timeout=120 "$core" "ppio-gpu:~/data/core/$pref/" 2>>"$DATA/upload.err"; then
        echo "$core" >> "$DONE_LIST"
        echo "[$(date +%H:%M:%S)] up core/$pref/$rel.parquet ($(du -m "$core" | cut -f1)MB)"
      else
        echo "[$(date +%H:%M:%S)] FAIL core/$pref/$rel.parquet, retry next loop"
        sleep 10
      fi
    done
  done
  # 全部处理完且全部传完 -> 收尾
  if grep -q "ALL DONE" "$DATA/preprocess.log" 2>/dev/null; then
    n_stats=$(ls "$DATA"/stats/*.json 2>/dev/null | wc -l | tr -d ' ')
    n_up=$(wc -l < "$DONE_LIST" | tr -d ' ')
    if [ "$n_up" -ge "$n_stats" ]; then
      rsync -a "$DATA/stats/" ppio-gpu:~/data/stats/
      [ -e "$DATA/catalog.parquet" ] && rsync -a "$DATA/catalog.parquet" ppio-gpu:~/data/
      echo "[$(date +%H:%M:%S)] UPLOAD COMPLETE: $n_up files"
      break
    fi
  fi
  sleep 30
done
