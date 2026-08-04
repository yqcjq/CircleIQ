"""构建大规模预测数据集:core parquet + 稳定圈分区 -> 每事件全通道流。

产物(~/circleiq/pred_full/out[_pre]/):
  events/<name>.npz :
    counts[C+4, nb] int32   小时级全通道计数(0..C-1 top圈 / C rest / C+1 bigv / C+2 org / C+3 other)
    scnt[C+4, 6, nb] int16  小时级 通道x情绪 计数(舆论线用;情绪缺失帖不计入)
    b0 int64                第一个小时 bin 的 unix 小时索引
    p_ts/p_uid/p_cid        top圈成员发帖流(活跃成员特征用),按 ts 排序
    rt_ts/rt_cid/rt_root    top圈成员带根帖流(级联集中度特征用),root 为事件内因子化编码
  members.json      : 圈 id/规模/类型
  anchors.parquet   : (event, circle, k, split, y6, y24)
  panel.json        : 事件/配对元信息
用法: python3 build_dataset.py [--partition main|pre] [--workers 10]
"""
import argparse
import json
import multiprocessing as mp
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from common import (BIN, CAPS, CIRCLE_TYPE, CTX_MIN_BINS, DATA, MIN_CIRCLE,
                    PAIR_MIN_TOTAL, SENTS, STEP_BINS, event_split, out_dir,
                    subsample)

PART_FILES = {"main": "partition_stable_K3.parquet",
              "pre": "partition_stable_K3_pre.parquet"}

# worker 全局(spawn 初始化)
G = {}


def _init(part_file, circles):
    import polars as pl
    part = pl.read_parquet(DATA / part_file)
    cmap = {c: i for i, c in enumerate(circles)}
    C = len(circles)
    part = part.with_columns(
        pl.col("circle_id").replace_strict(cmap, default=C).alias("cidx"))
    # top 圈成员 -> 全局 uid(活跃成员去重用)
    topm = part.filter(pl.col("cidx") < C).sort("cidx")
    G["part"] = part.select("md5_author", "cidx")
    G["uid"] = topm.select("md5_author").with_row_index("uid").with_columns(
        pl.col("uid").cast(pl.Int64))
    G["C"] = C
    G["smap"] = {s: i for i, s in enumerate(SENTS)}


def process_event(job):
    name, files = job
    import polars as pl
    C = G["C"]
    cols = ["md5_author", "md5_mid", "md5_root_mid", "ts", "auth_tier", "sentiment"]
    df = pl.concat([pl.read_parquet(f, columns=cols) for f in files])
    df = df.filter(pl.col("md5_mid").is_null() | (pl.int_range(pl.len()).over("md5_mid") == 0))
    df = df.filter(pl.col("ts").is_not_null() & pl.col("md5_author").is_not_null())
    if df.height < 500:
        return {"name": name, "skip": f"rows={df.height}"}
    df = df.join(G["part"], on="md5_author", how="left")
    ch = (pl.when(pl.col("cidx").is_not_null()).then(pl.col("cidx"))
            .when(pl.col("auth_tier") == "bigv").then(C + 1)
            .when(pl.col("auth_tier") == "org").then(C + 2)
            .otherwise(C + 3)).cast(pl.Int32)
    df = df.with_columns(ch.alias("ch"))

    ts = df["ts"].to_numpy()
    T0, T1 = np.quantile(ts, 0.005), np.quantile(ts, 0.995)
    df = df.filter(pl.Series((ts >= T0) & (ts <= T1))).sort("ts")

    b0 = int(T0 // BIN)
    nb = int(T1 // BIN) - b0 + 1
    if nb < CTX_MIN_BINS + 30:
        return {"name": name, "skip": f"nb={nb}"}
    ts = df["ts"].to_numpy()
    chv = df["ch"].to_numpy().astype(np.int32)
    kbin = (ts // BIN).astype(np.int64) - b0
    counts = np.zeros((C + 4, nb), dtype=np.int32)
    np.add.at(counts, (chv, kbin), 1)

    # 通道x情绪x小时(舆论线)
    sv = (df["sentiment"].replace_strict(G["smap"], default=-1, return_dtype=pl.Int32)
          .fill_null(-1).to_numpy().astype(np.int32))
    m = sv >= 0
    scnt = np.zeros((C + 4, 6, nb), dtype=np.int32)
    np.add.at(scnt, (chv[m], sv[m], kbin[m]), 1)

    # top 圈成员发帖流(uid 去重活跃成员;join 不保序,需重排)
    pg = (df.filter(pl.col("cidx") < C)
            .join(G["uid"], on="md5_author", how="inner").sort("ts"))
    p_ts = pg["ts"].to_numpy().astype(np.int64)
    p_uid = pg["uid"].to_numpy().astype(np.int32)
    p_cid = pg["cidx"].to_numpy().astype(np.int16)

    # top 圈成员带根帖流(级联集中度)
    rg = df.filter((pl.col("cidx") < C) & pl.col("md5_root_mid").is_not_null())
    rt_ts = rg["ts"].to_numpy().astype(np.int64)
    rt_cid = rg["cidx"].to_numpy().astype(np.int16)
    _, rt_root = np.unique(rg["md5_root_mid"].to_numpy(), return_inverse=True)
    rt_root = rt_root.astype(np.int32)

    np.savez_compressed(G["out"] / "events" / f"{name}.npz",
                        counts=counts, scnt=scnt, b0=np.int64(b0),
                        p_ts=p_ts, p_uid=p_uid, p_cid=p_cid,
                        rt_ts=rt_ts, rt_cid=rt_cid, rt_root=rt_root)
    tot = counts.sum(1)
    return {"name": name, "nb": nb, "n_rows": int(df.height),
            "tot_by_cidx": tot[:C].tolist(),
            "tot_agg": {"rest": int(tot[C]), "bigv": int(tot[C + 1]),
                        "org": int(tot[C + 2]), "other": int(tot[C + 3])}}


def _init_full(part_file, circles, out):
    _init(part_file, circles)
    G["out"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="main", choices=["main", "pre"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    OUT = out_dir(args.partition)
    (OUT / "events").mkdir(parents=True, exist_ok=True)

    import polars as pl
    t0 = time.time()
    part_file = PART_FILES[args.partition]
    part = pl.read_parquet(DATA / part_file)
    sizes = part.group_by("circle_id").len().sort("len", descending=True)
    top = sizes.filter(pl.col("len") >= MIN_CIRCLE)
    circles = top["circle_id"].to_list()
    C = len(circles)
    members = {
        "partition": part_file, "C": C,
        "circles": [{"circle_id": int(c), "size": int(s),
                     "type": (CIRCLE_TYPE.get(c, "other_small") if args.partition == "main" else "na")}
                    for c, s in zip(top["circle_id"], top["len"])],
        "n_rest_stable": int(part.height - top["len"].sum())}
    (OUT / "members.json").write_text(json.dumps(members, ensure_ascii=False))
    print(f"partition={part_file} C={C} top users={int(top['len'].sum())} "
          f"rest={members['n_rest_stable']}", flush=True)

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
                             initializer=_init_full, initargs=(part_file, circles, OUT)) as ex:
        futs = {ex.submit(process_event, (n, fs)): n for n, fs in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                metas.append(fut.result())
            except Exception:
                metas.append({"name": futs[fut], "error": traceback.format_exc()[-400:]})
            if i % 40 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)} {time.time()-t0:.0f}s]", flush=True)

    ok = [m for m in metas if "tot_by_cidx" in m]
    for e in [m for m in metas if "error" in m][:3]:
        print("ERROR", e["name"], e["error"][-200:], flush=True)

    split = event_split([m["name"] for m in ok])
    rows, pairs = [], []
    for m in ok:
        name = m["name"]
        z = np.load(OUT / "events" / f"{name}.npz")
        counts = z["counts"]
        nb = counts.shape[1]
        ks = np.arange(CTX_MIN_BINS, nb - 24, STEP_BINS)
        if len(ks) < 3:
            continue
        act = counts.sum(0)
        cact = np.concatenate([[0], np.cumsum(act)])
        alive = (cact[ks] - cact[np.maximum(0, ks - 168)]) >= 1
        ks = ks[alive]
        if len(ks) < 3:
            continue
        sp = split[name]
        ks = subsample(ks, CAPS[sp])
        for cidx, tot in enumerate(m["tot_by_cidx"]):
            if tot < PAIR_MIN_TOTAL:
                continue
            c = circles[cidx]
            cc = np.concatenate([[0], np.cumsum(counts[cidx])])
            y6 = cc[np.minimum(ks + 6, nb)] - cc[ks]
            y24 = cc[np.minimum(ks + 24, nb)] - cc[ks]
            pairs.append({"event": name, "circle": int(c), "cidx": cidx, "split": sp,
                          "total": int(tot), "n_anchors": int(len(ks)),
                          "nz24": float((y24 > 0).mean())})
            for k, a, b in zip(ks, y6, y24):
                rows.append({"event": name, "circle": int(c), "cidx": cidx,
                             "k": int(k), "split": sp, "y6": int(a), "y24": int(b)})
    pl.DataFrame(rows).write_parquet(OUT / "anchors.parquet")
    panel = {"n_events_ok": len(ok), "n_pairs": len(pairs), "C": C,
             "split_counts": {s: sum(1 for p in pairs if p["split"] == s)
                              for s in ("train", "val", "test")},
             "pairs": pairs,
             "events": {m["name"]: {k: m[k] for k in ("nb", "n_rows", "tot_agg")}
                        for m in ok}}
    (OUT / "panel.json").write_text(json.dumps(panel, ensure_ascii=False))
    df = pl.DataFrame(rows)
    print(f"pairs={len(pairs)} anchors={df.height} "
          f"(train/val/test={[df.filter(pl.col('split')==s).height for s in ('train','val','test')]})")
    print(f"y24 nonzero rate: {float((df['y24']>0).mean()):.2%}")
    print(f"circles with >=1 pair: {df['circle'].n_unique()}/{C}")
    print("BUILD DONE", flush=True)


if __name__ == "__main__":
    main()
