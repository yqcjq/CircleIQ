"""生成 catalog.parquet:每个 zip 的类别/议题/事件名/处理统计。

事件名来自 数据集目录、示例与字段说明.json;统计来自 stats/*.json(存在则合并)。
"""
import json
from pathlib import Path

import pandas as pd

RAW = Path("/Users/ppio/Desktop/第八届传播数据挖掘竞赛")
DATA = Path("/Users/ppio/Desktop/CircleIQ-data")
CATALOG_JSON = Path("/Users/ppio/Desktop/project/2026-7-CircleIQ/数据集目录、示例与字段说明.json")


def main():
    entries = json.load(open(CATALOG_JSON))
    name_map = {}  # zip stem -> (类别, 事件名)
    for e in entries:
        if set(e.keys()) != {"类别", "话题/事件", "文件名"}:
            continue
        cat, name, files = e["类别"], e["话题/事件"], e["文件名"]
        for f in files.replace(";", ";").split(";"):
            f = f.strip().removesuffix(".zip")
            if f:
                name_map[f] = (cat, name)

    rows = []
    for sj in sorted((DATA / "stats").glob("*.json")):
        s = json.loads(sj.read_text())
        stem = s["zip"]
        cat_label, event_name = name_map.get(stem, (None, None))
        if event_name is None:  # major 议题按 topic 名匹配
            cat_label, event_name = name_map.get(s["topic"], (None, s["topic"]))
        rows.append({
            "category": s["category"], "topic": s["topic"], "zip": stem,
            "event_name": event_name, "catalog_label": cat_label,
            "n_csv": s["n_csv"], "rows": s["rows"], "n_users": s["n_users"],
            "reposts": s["reposts"], "sentiment_filled": s["sentiment_filled"],
            "dup_mid_dropped": s["dup_mid_dropped"], "bad_ts": s["bad_ts"],
            "ts_min": s["ts_min"], "ts_max": s["ts_max"],
            "core_mb": s["core_mb"], "text_mb": s["text_mb"], "md5_computed": s["md5_computed"],
        })
    df = pd.DataFrame(rows).sort_values(["category", "topic", "zip"])
    df.to_parquet(DATA / "catalog.parquet", index=False)
    print(f"catalog: {len(df)} zips, named: {df.event_name.notna().sum()}")
    print(df.groupby("category")[["rows", "n_users", "core_mb", "text_mb"]].sum())


if __name__ == "__main__":
    main()
