"""按议题/事件构建圈层维度事件流并批量拟合 Hawkes(服务器端,多进程)。

事件定义: 每条帖子(原创+转发)= 一个事件,维度 = 作者所属稳定圈(top D-1 圈,其余归 dim "other")
用法:
  python3 run_hawkes.py --partition partition_stable_K3.parquet --category hot --dims 10 \
      --min-events 3000 [--val-frac 0.2] [--workers 12]
输出: ~/data/hawkes/<cat>__<topic>.json (μ/A/β/LL/验证) + 汇总 hawkes_summary_<cat>.json
"""
import argparse
import json
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DATA = Path.home() / "data"


def topic_groups(category):
    groups = defaultdict(list)
    for cat_dir in (DATA / "core").iterdir():
        if not cat_dir.is_dir() or (category != "all" and cat_dir.name != category):
            continue
        for p in sorted(cat_dir.glob("*.parquet")):
            groups[(cat_dir.name, p.stem.split("__", 1)[0])].append(str(p))
    return groups


def fit_topic(job):
    cat, topic, files, part_file, pool, D, min_events, min_dim_events, val_frac = job
    import numpy as np
    import polars as pl
    from hawkes_fit import fit_hawkes
    t0 = time.time()
    df = pl.concat([pl.read_parquet(f, columns=["md5_author", "md5_mid", "ts"]) for f in files])
    df = df.filter(pl.col("md5_mid").is_null() | (pl.int_range(pl.len()).over("md5_mid") == 0))
    df = df.filter(pl.col("ts").is_not_null() & pl.col("md5_author").is_not_null())
    if df.height < min_events:
        return {"category": cat, "topic": topic, "skip": f"events={df.height}<{min_events}"}

    part = pl.read_parquet(part_file)
    df = df.join(part, on="md5_author", how="left")  # circle_id null = 非稳定用户

    # 维度: 全局候选圈池(pool,固定顺序)中,该议题内事件数过门槛的圈 + other
    # -> 各事件的 α 矩阵在共享圈上跨事件可比(见 docs/plan-revision-2026-07-27.md 修正2)
    cnt = {r["circle_id"]: r["len"] for r in
           df.filter(pl.col("circle_id").is_in(pool)).group_by("circle_id").len().to_dicts()}
    thresh = max(min_dim_events, int(df.height * 0.002))
    active = [c for c in pool if cnt.get(c, 0) >= thresh][: D - 1]
    if not active:
        return {"category": cat, "topic": topic, "skip": f"no pool circle >= {thresh} events"}
    cmap = {c: i for i, c in enumerate(active)}
    other = len(active)
    K = other + 1
    df = df.with_columns(
        pl.col("circle_id").replace_strict(cmap, default=other).alias("dim"))

    stable_share = float((df["dim"] != other).sum() / df.height)
    df = df.sort("ts")
    times = df["ts"].to_numpy().astype("float64")
    dims = df["dim"].to_numpy().astype("int64")
    # 时间窗:掐头去尾 0.5%,避免零星早/晚点拉长窗口
    T0, T1 = float(np.quantile(times, 0.005)) - 1, float(np.quantile(times, 0.995)) + 1
    keep = (times >= T0) & (times <= T1)
    times, dims = times[keep], dims[keep]

    res = fit_hawkes(times, dims, K, T0, T1, val_frac=val_frac)
    out = {
        "category": cat, "topic": topic, "n_events": int(len(times)), "D": K,
        "dim_circles": {str(i): int(c) for c, i in cmap.items()}, "other_dim": other,
        "stable_event_share": round(stable_share, 4),
        "T0": T0, "T1": T1, "span_days": round((T1 - T0) / 86400, 1),
        "beta": res["beta"], "beta_halflife_min": round(np.log(2) / res["beta"] / 60, 1),
        "mu": res["mu"].tolist(), "A": res["A"].tolist(),
        "n_by_dim": res["n_by_dim"],
        "ll_train": res["ll_train"], "ll_poisson": res["ll_poisson"],
        "ll_gain_per_event": res["ll_gain_per_event"],
        "branching_max_row_sum": res["branching_max_row_sum"],
        "ll_val": res.get("ll_val"), "ll_val_poisson": res.get("ll_val_poisson"),
        "n_val": res.get("n_val"), "secs": round(time.time() - t0, 1),
    }
    (DATA / "hawkes").mkdir(exist_ok=True)
    (DATA / "hawkes" / "events").mkdir(exist_ok=True)
    np.savez_compressed(DATA / "hawkes" / "events" / f"{cat}__{topic}.npz",
                        times=times, dims=dims)
    with open(DATA / "hawkes" / f"{cat}__{topic}.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="partition_stable_K3.parquet")
    ap.add_argument("--category", default="hot", choices=["major", "hot", "case", "all"])
    ap.add_argument("--pool", type=int, default=12, help="全局候选圈数(按稳定圈大小)")
    ap.add_argument("--dims", type=int, default=10, help="单事件最大维度(含 other)")
    ap.add_argument("--min-events", type=int, default=3000)
    ap.add_argument("--min-dim-events", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    import polars as pl
    part = pl.read_parquet(DATA / args.partition)
    pool = (part.group_by("circle_id").len().sort("len", descending=True)
                .head(args.pool)["circle_id"].to_list())
    print(f"global circle pool (top{args.pool} by size): {pool}", flush=True)

    groups = topic_groups(args.category)
    jobs = [(c, t, fs, str(DATA / args.partition), pool, args.dims, args.min_events,
             args.min_dim_events, args.val_frac)
            for (c, t), fs in sorted(groups.items())]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"topics: {len(jobs)}", flush=True)
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fit_topic, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                r = fut.result()
            except Exception:
                r = {"category": j[0], "topic": j[1], "error": traceback.format_exc()}
            results.append(r)
            if "error" in r:
                print(f"[{i}/{len(jobs)}] ERROR {r['topic']}\n{r['error'][-500:]}", flush=True)
            elif "skip" in r:
                print(f"[{i}/{len(jobs)}] skip {r['topic']}: {r['skip']}", flush=True)
            else:
                print(f"[{i}/{len(jobs)} {time.time()-t0:.0f}s] {r['category']}/{r['topic']}: "
                      f"n={r['n_events']:,} stable={r['stable_event_share']:.0%} "
                      f"β半衰={r['beta_halflife_min']}min gain={r['ll_gain_per_event']:.3f} "
                      f"val={'%.0f' % r['ll_val'] if r['ll_val'] is not None else '-'}", flush=True)
    with open(DATA / f"hawkes_summary_{args.category}.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("HAWKES DONE", flush=True)


if __name__ == "__main__":
    main()
