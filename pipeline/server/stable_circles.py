"""阶段A:跨议题稳定互动圈识别(服务器端)。

Step 1: 读全部议题边表 -> 无向 pair 聚合: n_topics / n_major / total_weight / n_topics_pre
Step 2: K 网格统计(K=2..6)
Step 3: 选定 K 在稳定图上跑 Leiden(3 seeds 并行,取模块度最高)
Step 4: 输出 stable_pairs.parquet / stable_edges_K{K}.parquet / partition_stable_K{K}.parquet / stable_report.json

用法: python3 stable_circles.py [--k 3] [--pre-only]  # --pre-only 只用 2025-07-01 前的互动(防泄漏版)
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DATA = Path.home() / "data"


def leiden_run(args):
    """独立进程:一个 seed 的 Leiden。返回 (seed, modularity, membership)"""
    edges_file, seed, weight_col = args
    import random

    import igraph as ig
    import polars as pl
    ig.set_random_number_generator(random.Random(seed))
    e = pl.read_parquet(edges_file)
    g = ig.Graph.TupleList(
        ((r[0], r[1]) for r in e.select(["u", "v"]).iter_rows()), directed=False)
    w = e[weight_col].to_list()
    part = g.community_leiden(objective_function="modularity", weights=w, n_iterations=-1)
    # membership 对齐节点名
    names = [v["name"] for v in g.vs]
    return seed, part.modularity, names, list(part.membership)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pre-only", action="store_true")
    ap.add_argument("--weight", default="n_topics", choices=["n_topics", "total_weight"])
    args = ap.parse_args()
    import polars as pl

    t0 = time.time()
    files = sorted((DATA / "edges").glob("*.parquet"))
    print(f"edge files: {len(files)}", flush=True)
    frames = []
    for f in files:
        e = pl.read_parquet(f, columns=["src", "dst", "weight", "weight_pre", "topic", "category"])
        frames.append(e)
    edges = pl.concat(frames)
    print(f"edge rows (directed, per-topic): {edges.height:,}  [{time.time()-t0:.0f}s]", flush=True)

    wcol = "weight_pre" if args.pre_only else "weight"
    edges = edges.filter(pl.col(wcol) > 0)

    # 无向 pair
    edges = edges.with_columns(
        pl.min_horizontal("src", "dst").alias("u"),
        pl.max_horizontal("src", "dst").alias("v"),
    )
    pairs = edges.group_by(["u", "v"]).agg(
        pl.col("topic").n_unique().alias("n_topics"),
        pl.col("topic").filter(pl.col("category") == "major").n_unique().alias("n_major"),
        pl.col(wcol).sum().alias("total_weight"),
    )
    print(f"unique pairs: {pairs.height:,}  [{time.time()-t0:.0f}s]", flush=True)
    suffix = "_pre" if args.pre_only else ""
    pairs.write_parquet(DATA / f"stable_pairs{suffix}.parquet", compression="zstd")

    # K 网格
    grid = {}
    for k in [2, 3, 4, 5, 6]:
        sel = pairs.filter(pl.col("n_topics") >= k)
        n_users = pl.concat([sel["u"], sel["v"]]).n_unique()
        grid[k] = {"edges": sel.height, "users": n_users}
        print(f"K={k}: stable_edges={sel.height:,} users={n_users:,}", flush=True)

    K = args.k
    stable = pairs.filter(pl.col("n_topics") >= K)
    sf = DATA / f"stable_edges_K{K}{suffix}.parquet"
    stable.write_parquet(sf, compression="zstd")

    # Leiden 3 seeds 并行
    print(f"Leiden on K={K} graph: {stable.height:,} edges ...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for r in ex.map(leiden_run, [(str(sf), s, args.weight) for s in (1, 2, 3)]):
            results.append(r)
            print(f"  seed={r[0]} modularity={r[1]:.4f} circles={len(set(r[3]))}", flush=True)
    best = max(results, key=lambda r: r[1])
    seed, mod, names, membership = best

    part = pl.DataFrame({"md5_author": names, "circle_id": membership})
    sizes = part.group_by("circle_id").len().sort("len", descending=True)
    part.write_parquet(DATA / f"partition_stable_K{K}{suffix}.parquet", compression="zstd")

    # 稳定性:seed 间 ARI
    from sklearn.metrics import adjusted_rand_score
    m = {r[0]: dict(zip(r[2], r[3])) for r in results}
    common = list(set(m[1]) & set(m[2]) & set(m[3]))
    ari12 = adjusted_rand_score([m[1][x] for x in common], [m[2][x] for x in common])
    ari13 = adjusted_rand_score([m[1][x] for x in common], [m[3][x] for x in common])

    report = {
        "pre_only": args.pre_only, "K": K, "weight": args.weight,
        "k_grid": grid, "n_stable_edges": stable.height,
        "n_stable_users": part.height, "modularity": mod, "best_seed": seed,
        "n_circles": sizes.height,
        "n_circles_ge10": int((sizes["len"] >= 10).sum()),
        "top20_sizes": sizes.head(20).to_dicts(),
        "seed_ari": {"1v2": ari12, "1v3": ari13},
        "secs": round(time.time() - t0, 1),
    }
    with open(DATA / f"stable_report_K{K}{suffix}.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "top20_sizes"}, ensure_ascii=False, indent=1), flush=True)
    print("STABLE DONE", flush=True)


if __name__ == "__main__":
    main()
