"""A5:稳定互动圈评估(服务器端)。

输入: partition_stable_K{K}.parquet + user_profiles.parquet + partition_topic/*.parquet + edges/*.parquet
输出: ~/data/eval_stable_K{K}.json —— 供报告用的全部统计
维度:
  1. 基本统计: 圈数/大小分布/覆盖
  2. 画像组成: 每大圈 top 省份/认证/行业 + 熵(vs 全体基线)
  3. 单议题激活: 每个重大议题各稳定圈的事件占比
  4. 稳定圈 vs 单议题圈 ARI(公共用户上)
  5. 稳定用户 vs 非稳定用户的行为差异(粉丝/发帖/认证)
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path.home() / "data"


def entropy(counter):
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    return -sum(c / tot * math.log(c / tot) for c in counter.values() if c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--top-circles", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()

    part = pl.read_parquet(DATA / f"partition_stable_K{args.k}{args.suffix}.parquet")
    prof = pl.read_parquet(DATA / "user_profiles.parquet")
    report = {"K": args.k, "suffix": args.suffix}

    sizes = part.group_by("circle_id").len().sort("len", descending=True)
    report["n_users"] = part.height
    report["n_circles"] = sizes.height
    report["size_dist"] = {
        "ge1000": int((sizes["len"] >= 1000).sum()), "ge100": int((sizes["len"] >= 100).sum()),
        "ge10": int((sizes["len"] >= 10).sum()), "top20": sizes.head(20).to_dicts(),
    }

    # 2. 画像组成(top N 圈)
    top_ids = sizes.head(args.top_circles)["circle_id"].to_list()
    pj = part.join(prof, on="md5_author", how="left")
    base_prov = Counter({r["province"]: r["len"] for r in
                         pj.group_by("province").len().to_dicts() if r["province"]})
    report["baseline_province_entropy"] = entropy(base_prov)
    circles = []
    for cid in top_ids:
        sub = pj.filter(pl.col("circle_id") == cid)
        prov = Counter({r["province"]: r["len"] for r in sub.group_by("province").len().to_dicts() if r["province"]})
        auth = Counter({r["auth_tier"]: r["len"] for r in sub.group_by("auth_tier").len().to_dicts() if r["auth_tier"]})
        ind = Counter({r["industry_top"]: r["len"] for r in sub.group_by("industry_top").len().to_dicts() if r["industry_top"]})
        circles.append({
            "circle_id": int(cid), "size": sub.height,
            "top_province": prov.most_common(3), "province_entropy": round(entropy(prov), 3),
            "auth_dist": dict(auth.most_common(5)),
            "top_industry": ind.most_common(3),
            "mean_log_followers": round(float(sub["log_followers"].mean() or 0), 2),
            "mean_n_posts": round(float(sub["n_posts"].mean() or 0), 1),
            "mean_n_major_topics": round(float(sub["n_major_topics"].mean() or 0), 2),
        })
    report["top_circles"] = circles

    # 3. 单议题激活模式
    activation = {}
    for f in sorted((DATA / "core" / "major").glob("*.parquet")):
        topic = f.stem.split("__", 1)[0]
        ev = pl.read_parquet(f, columns=["md5_author"]).join(part, on="md5_author", how="left")
        n = ev.height
        by = ev.filter(pl.col("circle_id").is_in(top_ids)).group_by("circle_id").len()
        stable_all = int(ev["circle_id"].is_not_null().sum())
        prev = activation.get(topic, {"events": 0, "stable_events": 0, "by_circle": {}})
        prev["events"] += n
        prev["stable_events"] += stable_all
        for r in by.to_dicts():
            prev["by_circle"][str(r["circle_id"])] = prev["by_circle"].get(str(r["circle_id"]), 0) + r["len"]
        activation[topic] = prev
    for t, a in activation.items():
        a["stable_share"] = round(a["stable_events"] / max(1, a["events"]), 4)
    report["activation"] = activation

    # 4. ARI vs 单议题圈
    from sklearn.metrics import adjusted_rand_score
    aris = {}
    for f in sorted((DATA / "partition_topic").glob("major__*.parquet")):
        tp = pl.read_parquet(f).rename({"circle_id": "topic_circle"})
        both = part.join(tp, on="md5_author", how="inner")
        if both.height >= 1000:
            aris[f.stem] = {
                "common_users": both.height,
                "ari": round(adjusted_rand_score(both["circle_id"].to_list(),
                                                 both["topic_circle"].to_list()), 4),
            }
    report["ari_vs_topic"] = aris

    # 5. 稳定 vs 非稳定用户
    stable_set = part["md5_author"]
    pj2 = prof.with_columns(pl.col("md5_author").is_in(stable_set).alias("is_stable"))
    cmp = pj2.group_by("is_stable").agg(
        pl.len().alias("users"),
        pl.col("log_followers").mean().round(2).alias("mean_log_followers"),
        pl.col("n_posts").mean().round(1).alias("mean_posts"),
        pl.col("n_major_topics").mean().round(2).alias("mean_major_topics"),
        (pl.col("auth_tier") == "bigv").mean().round(4).alias("bigv_rate"),
        (pl.col("auth_tier") == "org").mean().round(4).alias("org_rate"),
    )
    report["stable_vs_rest"] = cmp.to_dicts()

    report["secs"] = round(time.time() - t0, 1)
    out = DATA / f"eval_stable_K{args.k}{args.suffix}.json"
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:3000])
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
