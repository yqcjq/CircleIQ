"""构建筛选实验数据集:core parquet + 稳定圈分区 -> 每事件多粒度流。

产物(~/circleiq/pred_screen/out/):
  events/<name>.npz :
    counts[6,nb] int32 小时六通道计数, b0(unix小时)
    ts_all int64 / ch_all int8            全帖秒级流(Hawkes 用)
    p_ts int64 / p_gid int32              圈0/圈5成员发帖流(IC 种子/节点特征用)
    r_ts int64 / r_src,r_dst int32        转发交互流,成员级编码(TGN/EvolveGCN 用)
  members.npz       : 圈0/圈5成员全局索引规模 + 圈内稳定图边(gid, gid, weight)
  anchors.parquet   : (event, circle, k, split, y6, y24)
  panel.json        : 事件/配对元信息
通道: 0=c0 1=c5 2=其他稳定圈 3=圈外大V 4=圈外机构 5=散户
成员编码: >=0 圈成员全局idx(圈0 在前);负数聚合 -1 stab -2 bigv -3 org -4 other
用法: python3 build_dataset.py [--workers 10]
"""
import argparse
import json
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from common import (BIN, CAPS, CIRCLES, CTX_MIN_BINS, DATA, OUT,
                    PAIR_MIN_TOTAL, STEP_BINS, event_split, subsample)

# 通道编码
CH_NAMES = ["c0", "c5", "stab", "bigv", "org", "other"]
# 成员编码:>=0 圈成员全局idx;负数=聚合通道 -1 stab -2 bigv -3 org -4 other


def build_members():
    import polars as pl
    part = pl.read_parquet(DATA / "partition_stable_K3.parquet")
    m0 = part.filter(pl.col("circle_id") == 0)["md5_author"].to_list()
    m5 = part.filter(pl.col("circle_id") == 5)["md5_author"].to_list()
    gid = {a: i for i, a in enumerate(m0)}
    gid.update({a: len(m0) + i for i, a in enumerate(m5)})
    edges = pl.read_parquet(DATA / "stable_edges_K3.parquet")
    sets = {0: set(m0), 5: set(m5)}
    intra = {}
    for c in CIRCLES:
        s = sets[c]
        e = edges.filter(pl.col("u").is_in(list(s)) & pl.col("v").is_in(list(s)))
        intra[c] = np.array([[gid[u], gid[v], w] for u, v, w in
                             zip(e["u"], e["v"], e["total_weight"])], dtype=np.int64)
    np.savez_compressed(OUT / "members.npz",
                        n0=len(m0), n5=len(m5),
                        intra0=intra[0], intra5=intra[5])
    circle_of = part.select("md5_author", "circle_id")
    return gid, circle_of, len(m0), len(m5)


def process_event(job):
    name, files = job
    import polars as pl
    cols = ["md5_author", "md5_mid", "md5_parent_mid", "ts", "is_original", "auth_tier"]
    df = pl.concat([pl.read_parquet(f, columns=cols) for f in files])
    df = df.filter(pl.col("md5_mid").is_null() | (pl.int_range(pl.len()).over("md5_mid") == 0))
    df = df.filter(pl.col("ts").is_not_null() & pl.col("md5_author").is_not_null())
    if df.height < 500:
        return {"name": name, "skip": f"rows={df.height}"}
    df = df.join(PART, on="md5_author", how="left")
    ch = (pl.when(pl.col("circle_id") == 0).then(0)
            .when(pl.col("circle_id") == 5).then(1)
            .when(pl.col("circle_id").is_not_null()).then(2)
            .when(pl.col("auth_tier") == "bigv").then(3)
            .when(pl.col("auth_tier") == "org").then(4)
            .otherwise(5))
    df = df.with_columns(ch.alias("ch"))

    ts = df["ts"].to_numpy()
    T0, T1 = np.quantile(ts, 0.005), np.quantile(ts, 0.995)
    keep = (ts >= T0) & (ts <= T1)
    df = df.filter(pl.Series(keep))
    ts = df["ts"].to_numpy()
    chv = df["ch"].to_numpy().astype(np.int8)

    b0 = int(T0 // BIN)
    nb = int(T1 // BIN) - b0 + 1
    if nb < CTX_MIN_BINS + 30:
        return {"name": name, "skip": f"nb={nb}"}
    kbin = (ts // BIN).astype(np.int64) - b0
    counts = np.zeros((6, nb), dtype=np.int32)
    np.add.at(counts, (chv, kbin), 1)

    # 全帖秒级流(Hawkes/SIR 用)与圈成员发帖流
    df = df.join(GIDF, on="md5_author", how="left").sort("ts")
    ts_all = df["ts"].to_numpy().astype(np.int64)
    ch_all = df["ch"].to_numpy().astype(np.int8)
    pg = df.filter(pl.col("gid").is_not_null())
    p_ts = pg["ts"].to_numpy().astype(np.int64)
    p_gid = pg["gid"].to_numpy().astype(np.int32)

    # 转发交互流(成员级):src=转发者, dst=父帖作者
    # 编码:>=0 圈成员全局idx;-1 其他稳定圈;-2 圈外大V;-3 圈外机构;-4 散户
    mid_author = (df.filter(pl.col("md5_mid").is_not_null())
                    .select(["md5_mid", "md5_author"]).unique(subset="md5_mid", keep="first"))
    rp = (df.filter(~pl.col("is_original") & pl.col("md5_parent_mid").is_not_null())
            .select(["md5_author", "md5_parent_mid", "ts", "circle_id", "auth_tier"])
            .join(mid_author.rename({"md5_mid": "md5_parent_mid", "md5_author": "dst_a"}),
                  on="md5_parent_mid", how="inner")
            .join(PART.rename({"md5_author": "dst_a", "circle_id": "dst_cid"}),
                  on="dst_a", how="left")
            .join(GIDF.rename({"gid": "src_gid"}), on="md5_author", how="left")
            .join(GIDF.rename({"md5_author": "dst_a", "gid": "dst_gid"}),
                  on="dst_a", how="left"))
    if rp.height:
        rp = rp.with_columns(
            pl.coalesce(
                pl.col("src_gid"),
                pl.when(pl.col("circle_id").is_not_null()).then(-1)
                  .when(pl.col("auth_tier") == "bigv").then(-2)
                  .when(pl.col("auth_tier") == "org").then(-3)
                  .otherwise(-4)).alias("r_src"),
            pl.coalesce(
                pl.col("dst_gid"),
                pl.when(pl.col("dst_cid").is_not_null()).then(-1).otherwise(-4)).alias("r_dst"),
        ).sort("ts").filter((pl.col("ts") >= T0) & (pl.col("ts") <= T1))
        r_ts = rp["ts"].to_numpy().astype(np.int64)
        r_src = rp["r_src"].to_numpy().astype(np.int32)
        r_dst = rp["r_dst"].to_numpy().astype(np.int32)
    else:
        r_ts = np.zeros(0, np.int64)
        r_src = np.zeros(0, np.int32)
        r_dst = np.zeros(0, np.int32)

    np.savez_compressed(OUT / "events" / f"{name}.npz",
                        counts=counts, b0=np.int64(b0),
                        ts_all=ts_all, ch_all=ch_all, p_ts=p_ts, p_gid=p_gid,
                        r_ts=r_ts, r_src=r_src, r_dst=r_dst)
    tot = counts.sum(1)
    return {"name": name, "nb": nb, "n_rows": int(df.height), "n_reposts": int(len(r_ts)),
            "totals": {CH_NAMES[i]: int(tot[i]) for i in range(6)}}


GIDF = None
PART = None


def _init(gid_items, part_df):
    global GIDF, PART
    import polars as pl
    GIDF = pl.DataFrame({"md5_author": [a for a, _ in gid_items],
                         "gid": [g for _, g in gid_items]},
                        schema={"md5_author": pl.String, "gid": pl.Int64})
    PART = part_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    (OUT / "events").mkdir(parents=True, exist_ok=True)

    import polars as pl
    t0 = time.time()
    gid, circle_of, n0, n5 = build_members()
    print(f"members: c0={n0} c5={n5} (global idx built) {time.time()-t0:.0f}s", flush=True)

    from collections import defaultdict
    groups = defaultdict(list)
    for cat_dir in sorted((DATA / "core").iterdir()):
        if cat_dir.is_dir():
            for p in sorted(cat_dir.glob("*.parquet")):
                groups[f"{cat_dir.name}__{p.stem.split('__', 1)[0]}"].append(str(p))
    jobs = sorted(groups.items(), key=lambda kv: -sum(Path(f).stat().st_size for f in kv[1]))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"events: {len(jobs)}", flush=True)

    metas = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=_init, initargs=(list(gid.items()), circle_of)) as ex:
        futs = {ex.submit(process_event, (n, fs)): n for n, fs in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                metas.append(fut.result())
            except Exception:
                metas.append({"name": futs[fut], "error": traceback.format_exc()[-400:]})
            if i % 40 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)} {time.time()-t0:.0f}s]", flush=True)

    ok = [m for m in metas if "totals" in m]
    err = [m for m in metas if "error" in m]
    for e in err[:3]:
        print("ERROR", e["name"], e["error"][-200:], flush=True)

    # 配对 + 锚点 + 目标
    split = event_split([m["name"] for m in ok])
    rows = []
    pairs = []
    for m in ok:
        name = m["name"]
        z = np.load(OUT / "events" / f"{name}.npz")
        counts = z["counts"]
        nb = counts.shape[1]
        ks = np.arange(CTX_MIN_BINS, nb - 24, STEP_BINS)
        if len(ks) < 3:
            continue
        # 活动门(仅历史):近7天全通道有帖
        act = counts.sum(0)
        cact = np.concatenate([[0], np.cumsum(act)])
        lo = np.maximum(0, ks - 168)
        alive = (cact[ks] - cact[lo]) >= 1
        ks = ks[alive]
        if len(ks) < 3:
            continue
        sp = split[name]
        ks = subsample(ks, CAPS[sp])
        for ci, c in enumerate(CIRCLES):
            tot = m["totals"][f"c{c}"]
            if tot < PAIR_MIN_TOTAL:
                continue
            cc = np.concatenate([[0], np.cumsum(counts[ci])])
            y6 = cc[np.minimum(ks + 6, nb)] - cc[ks]
            y24 = cc[np.minimum(ks + 24, nb)] - cc[ks]
            pairs.append({"event": name, "circle": c, "split": sp, "total": tot,
                          "n_anchors": int(len(ks)),
                          "nz24": float((y24 > 0).mean())})
            for k, a, b in zip(ks, y6, y24):
                rows.append({"event": name, "circle": c, "k": int(k), "split": sp,
                             "y6": int(a), "y24": int(b)})
    pl.DataFrame(rows).write_parquet(OUT / "anchors.parquet")
    panel = {"n_events_ok": len(ok), "n_pairs": len(pairs),
             "members": {"n0": n0, "n5": n5},
             "split_counts": {s: sum(1 for p in pairs if p["split"] == s) for s in ("train", "val", "test")},
             "pairs": pairs, "events": {m["name"]: m for m in ok}}
    (OUT / "panel.json").write_text(json.dumps(panel, ensure_ascii=False))
    df = pl.DataFrame(rows)
    print(f"pairs={len(pairs)} anchors={df.height} "
          f"(train/val/test={[df.filter(pl.col('split')==s).height for s in ('train','val','test')]})")
    print(f"y24 nonzero rate: {float((df['y24']>0).mean()):.2%}; "
          f"y24 p50/p90/max among nonzero: "
          f"{np.percentile(df.filter(pl.col('y24')>0)['y24'].to_numpy(), [50,90,100]).tolist()}")
    print("BUILD DONE", flush=True)


if __name__ == "__main__":
    main()
