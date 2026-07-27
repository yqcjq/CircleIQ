"""服务器端:core parquet -> 每议题互动边表(24 核并行)。

边定义(与本地 POC 一致):src=转发者, dst=被转发者(父帖作者), 无向权重=互动次数。
- 常规:md5_parent_mid -> mid_to_author 映射
- fallback(体重管理等无父mid的议题):根微博作者名 -> 作者名->md5 映射(method='name')
- 时间切分:weight_pre = 边在 ts < BOUNDARY 前的互动次数(供防泄漏训练用)

用法: python3 build_edges.py [--workers 4]
输入: ~/data/core/<cat>/<topic>__<zip>.parquet
输出: ~/data/edges/<cat>__<topic>.parquet + ~/data/edges_summary.json
"""
import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DATA = Path.home() / "data"
BOUNDARY = 1751328000  # 2025-07-01 00:00 (naive epoch,与预处理一致)

def topic_groups():
    groups = defaultdict(list)
    for cat_dir in (DATA / "core").iterdir():
        if not cat_dir.is_dir():
            continue
        for p in sorted(cat_dir.glob("*.parquet")):
            topic = p.stem.split("__", 1)[0]
            groups[(cat_dir.name, topic)].append(str(p))
    return groups


def build_topic_edges(cat, topic, files):
    import polars as pl
    t0 = time.time()
    cols = ["md5_author", "md5_mid", "md5_parent_mid", "is_original", "ts",
            "root_author_name", "author_name"]
    df = pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical")
    n_raw = df.height
    # 跨 zip 去重(同一 topic 多 zip 可能重复抓取):保留每个 md5_mid 首次出现
    df = df.filter(pl.col("md5_mid").is_null() | (pl.int_range(pl.len()).over("md5_mid") == 0))
    n_posts = df.height
    n_users = df["md5_author"].n_unique()

    mid_author = df.filter(pl.col("md5_mid").is_not_null() & pl.col("md5_author").is_not_null()) \
                   .select(["md5_mid", "md5_author"]).unique(subset="md5_mid", keep="first")

    reposts = df.filter(~pl.col("is_original") & pl.col("md5_parent_mid").is_not_null()) \
                .select(["md5_author", "md5_parent_mid", "ts"]) \
                .join(mid_author.rename({"md5_mid": "md5_parent_mid", "md5_author": "dst"}),
                      on="md5_parent_mid", how="inner") \
                .rename({"md5_author": "src"}) \
                .filter(pl.col("src") != pl.col("dst")) \
                .select(["src", "dst", "ts"])
    n_edge_events = reposts.height
    method = "parent_mid"

    # fallback:父mid全缺 & 有根作者名(体重管理)
    if n_edge_events == 0:
        nm = df.filter(pl.col("author_name").is_not_null() & pl.col("md5_author").is_not_null()) \
               .group_by("author_name").agg(pl.col("md5_author").mode().first().alias("dst"))
        reposts = df.filter(~pl.col("is_original") & pl.col("root_author_name").is_not_null()) \
                    .select(["md5_author", "root_author_name", "ts"]) \
                    .join(nm.rename({"author_name": "root_author_name"}), on="root_author_name", how="inner") \
                    .rename({"md5_author": "src"}) \
                    .filter(pl.col("src") != pl.col("dst")) \
                    .select(["src", "dst", "ts"])
        n_edge_events = reposts.height
        method = "root_name"

    edges = reposts.group_by(["src", "dst"]).agg(
        pl.len().alias("weight"),
        (pl.col("ts") < BOUNDARY).sum().alias("weight_pre"),
        pl.col("ts").min().alias("ts_min"),
        pl.col("ts").max().alias("ts_max"),
    ).with_columns(pl.lit(topic).alias("topic"), pl.lit(cat).alias("category"),
                   pl.lit(method).alias("method"))
    out = DATA / "edges" / f"{cat}__{topic}.parquet"
    edges.write_parquet(out, compression="zstd")
    return {"category": cat, "topic": topic, "n_files": len(files), "rows_raw": n_raw,
            "rows_dedup": n_posts, "n_users": n_users, "edge_events": n_edge_events,
            "n_edges": edges.height, "method": method, "secs": round(time.time() - t0, 1)}


def run(job):
    cat, topic, files = job
    try:
        return build_topic_edges(cat, topic, files)
    except Exception:
        return {"category": cat, "topic": topic, "error": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.environ.setdefault("POLARS_MAX_THREADS", "6")

    groups = topic_groups()
    jobs = [(c, t, fs) for (c, t), fs in groups.items()]
    jobs.sort(key=lambda j: -sum(Path(f).stat().st_size for f in j[2]))
    (DATA / "edges").mkdir(exist_ok=True)
    print(f"topics: {len(jobs)}", flush=True)

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if "error" in r:
                print(f"[{i}/{len(jobs)}] ERROR {r['topic']}\n{r['error']}", flush=True)
            else:
                print(f"[{i}/{len(jobs)} {time.time()-t0:.0f}s] {r['category']}/{r['topic']}: "
                      f"posts={r['rows_dedup']:,} users={r['n_users']:,} edges={r['n_edges']:,} ({r['method']}) {r['secs']}s", flush=True)
    with open(DATA / "edges_summary.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("EDGES DONE", flush=True)


if __name__ == "__main__":
    main()
