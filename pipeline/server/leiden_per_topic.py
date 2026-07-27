"""每个重大议题单独跑 Leiden(论文对比基线 + A5 评估素材)。

输出: ~/data/partition_topic/<topic>.parquet + leiden_per_topic.json
7 议题并行(leidenalg 单图单线程,跨议题进程池并行)。
"""
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DATA = Path.home() / "data"


def run_topic(edge_file):
    import igraph as ig
    import polars as pl
    t0 = time.time()
    name = Path(edge_file).stem  # cat__topic
    e = pl.read_parquet(edge_file)
    e = e.with_columns(pl.min_horizontal("src", "dst").alias("u"),
                       pl.max_horizontal("src", "dst").alias("v"))
    und = e.group_by(["u", "v"]).agg(pl.col("weight").sum())
    g = ig.Graph.TupleList(((r[0], r[1]) for r in und.select(["u", "v"]).iter_rows()), directed=False)
    w = und["weight"].to_list()
    best = None
    for seed in (1, 2, 3):
        import random
        ig.set_random_number_generator(random.Random(seed))
        part = g.community_leiden(objective_function="modularity", weights=w, n_iterations=-1)
        if best is None or part.modularity > best[0]:
            best = (part.modularity, list(part.membership))
    mod, membership = best
    names = [v["name"] for v in g.vs]
    out = DATA / "partition_topic" / f"{name}.parquet"
    out.parent.mkdir(exist_ok=True)
    pl.DataFrame({"md5_author": names, "circle_id": membership}).write_parquet(out)
    import collections
    sizes = sorted(collections.Counter(membership).values(), reverse=True)
    return {"topic": name, "nodes": g.vcount(), "edges": g.ecount(), "modularity": mod,
            "n_circles": len(sizes), "n_circles_ge10": sum(s >= 10 for s in sizes),
            "top5_sizes": sizes[:5], "secs": round(time.time() - t0, 1)}


def main():
    files = sorted((DATA / "edges").glob("major__*.parquet"))
    print(f"major topics: {len(files)}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=7) as ex:
        for r in ex.map(run_topic, [str(f) for f in files]):
            results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    with open(DATA / "leiden_per_topic.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("PER-TOPIC LEIDEN DONE", flush=True)


if __name__ == "__main__":
    main()
