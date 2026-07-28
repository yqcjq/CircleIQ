"""事件级内容特征(供"内容如何调制传播参数"的回归分析)。

每个议题/事件: 情绪分布、敏感占比、主行业、转发占比、大V/机构事件占比、事件量、时长。
输出: ~/data/event_features.parquet
用法: python3 content_features.py
"""
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

DATA = Path.home() / "data"
t0 = time.time()

groups = defaultdict(list)
for cat_dir in (DATA / "core").iterdir():
    if cat_dir.is_dir():
        for p in sorted(cat_dir.glob("*.parquet")):
            groups[(cat_dir.name, p.stem.split("__", 1)[0])].append(str(p))

rows = []
SENTS = ["愤怒", "悲伤", "恐惧", "惊奇", "喜悦", "中性"]
for (cat, topic), files in sorted(groups.items()):
    df = pl.concat([pl.read_parquet(f, columns=[
        "md5_mid", "ts", "sentiment", "info_attr", "industry", "is_original", "auth_tier"])
        for f in files])
    df = df.filter(pl.col("md5_mid").is_null() | (pl.int_range(pl.len()).over("md5_mid") == 0))
    n = df.height
    if n == 0:
        continue
    sent = df["sentiment"].drop_nulls()
    n_sent = max(1, len(sent))
    sc = sent.value_counts()
    sdict = {r["sentiment"]: r["count"] for r in sc.to_dicts()}
    ind = (df["industry"].drop_nulls().str.split("，").explode().value_counts()
           .sort("count", descending=True))
    top_ind = ind["industry"][0] if ind.height else None
    ts = df["ts"].drop_nulls()
    row = {
        "category": cat, "topic": topic, "n_events": n,
        "sentiment_coverage": round(len(sent) / n, 4),
        **{f"sent_{s}": round(sdict.get(s, 0) / n_sent, 4) for s in SENTS},
        "sensitive_share": round(float((df["info_attr"] == "敏感").sum() / n), 4),
        "repost_share": round(float((~df["is_original"]).sum() / n), 4),
        "bigv_share": round(float((df["auth_tier"] == "bigv").sum() / n), 4),
        "org_share": round(float((df["auth_tier"] == "org").sum() / n), 4),
        "top_industry": top_ind,
        "span_days": round(float((ts.max() - ts.min()) / 86400), 1) if len(ts) else None,
    }
    rows.append(row)

out = pl.DataFrame(rows)
out.write_parquet(DATA / "event_features.parquet")
print(f"event_features: {out.height} topics [{time.time()-t0:.0f}s]")
print(out.head(5))
print("FEATURES DONE")
