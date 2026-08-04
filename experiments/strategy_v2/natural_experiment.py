"""历史自然实验(P0-4):模型预测的注入增量 vs 真实大V进场后的再加速。

设计(应对进场内生性):
- 处理组:事件内粉丝数 top 档大V的"首帖时刻" t_e(此前该作者未发过帖,
  且 t_e 前后各留 24h 窗)
- 对照组:同事件内无大V首帖(±1h)的时刻,按进场前 24h 活动量最近邻匹配
- DiD:excess = (后24h - 前24h);DiD = excess_treat - excess_ctrl
- 预测:闭式 24h 簇大小(注入 bigv 维,m=该作者进场后 1h 内发帖数)
- 汇总:per-entry 散点(predicted vs DiD)+ Spearman + 符号一致率
用法: python3 natural_experiment.py [--workers 10]
输出: ~/data/strategy_v2/natural_experiment.json
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

DATA = Path.home() / "data"
sys.path.insert(0, str(DATA))
OUT = DATA / "strategy_v2"
W = 24 * 3600.0


def process_event(name):
    import polars as pl
    from closed_form import finite_horizon_counts
    fit = json.loads((DATA / "hawkes" / f"{name}.json").read_text())
    ev = np.load(DATA / "hawkes" / "events" / f"{name}.npz")
    times = ev["times"].astype(float)
    T0f, T1f = fit["T0"], fit["T1"]
    if T1f - T0f < 4 * W:
        return []
    A1, A2 = np.asarray(fit["A1"]), np.asarray(fit["A2"])
    n24 = finite_horizon_counts(A1, float(fit["beta1"]), np.asarray(fit["A2"]),
                                float(fit["beta2"]), W)
    if not np.isfinite(n24).all():
        return []
    pred_per_post = float(n24[fit["bigv_dim"]].sum())

    cat, topic = name.split("__", 1)
    files = sorted((DATA / "core" / cat).glob(f"{topic}__*.parquet"))
    df = pl.concat([pl.read_parquet(f, columns=["md5_author", "md5_mid", "ts",
                                                "auth_tier", "n_followers"])
                    for f in files])
    df = df.filter(pl.col("md5_mid").is_null() | (pl.int_range(pl.len()).over("md5_mid") == 0))
    df = df.filter(pl.col("ts").is_not_null())
    bv = df.filter((pl.col("auth_tier") == "bigv") & pl.col("n_followers").is_not_null())
    if bv.height < 5:
        return []
    fw_thresh = float(bv["n_followers"].quantile(0.9))
    firsts = (bv.sort("ts").group_by("md5_author").first()
                .filter((pl.col("n_followers") >= fw_thresh)
                        & (pl.col("ts") > T0f + W) & (pl.col("ts") < T1f - W))
                .sort("n_followers", descending=True).head(5))
    if firsts.height == 0:
        return []

    ts_sorted = np.sort(times)

    def win_count(t, lo, hi):
        return int(np.searchsorted(ts_sorted, t + hi) - np.searchsorted(ts_sorted, t + lo))

    bv_first_ts = np.sort(firsts["ts"].to_numpy().astype(float))
    all_bv_ts = np.sort(bv["ts"].to_numpy().astype(float))

    entries = []
    for r in firsts.iter_rows(named=True):
        t_e = float(r["ts"])
        m = int(bv.filter((pl.col("md5_author") == r["md5_author"])
                          & pl.col("ts").is_between(t_e, t_e + 3600)).height)
        entries.append({
            "t": t_e, "m": max(1, m), "followers": int(r["n_followers"]),
            "pre24": win_count(t_e, -W, 0), "post24": win_count(t_e, 0, W)})

    # 对照候选:均匀网格,排除任意大V帖 ±1h
    grid = np.arange(T0f + W, T1f - W, 1800.0)
    idx = np.searchsorted(all_bv_ts, grid)
    near = np.zeros(len(grid), bool)
    for off in (0, 1):
        j = np.clip(idx - off, 0, len(all_bv_ts) - 1)
        near |= np.abs(all_bv_ts[j] - grid) < 3600
    ctrl_ts = grid[~near]
    if len(ctrl_ts) < 5:
        return []
    ctrl = [{"t": float(t), "pre24": win_count(t, -W, 0), "post24": win_count(t, 0, W)}
            for t in ctrl_ts[:: max(1, len(ctrl_ts) // 200)]]
    ctrl_pre = np.array([c["pre24"] for c in ctrl], float)

    rows = []
    for e in entries:
        # 最近邻匹配(log 空间)
        d = np.abs(np.log1p(ctrl_pre) - np.log1p(e["pre24"]))
        j = int(np.argmin(d))
        did = (e["post24"] - e["pre24"]) - (ctrl[j]["post24"] - ctrl[j]["pre24"])
        rows.append({"name": name, "rho": float(np.abs(np.linalg.eigvals(A1 + A2)).max()),
                     "t": e["t"], "followers": e["followers"], "m": e["m"],
                     "pre24": e["pre24"], "post24": e["post24"],
                     "ctrl_pre24": ctrl[j]["pre24"], "ctrl_post24": ctrl[j]["post24"],
                     "match_gap_log": round(float(d[j]), 3),
                     "did": int(did), "pred": round(pred_per_post * e["m"], 1)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    names = sorted(f.stem for f in (DATA / "hawkes").glob("*.json")
                   if not f.stem.startswith("hawkes") and not f.stem.startswith("major"))
    print(f"events: {len(names)}", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(process_event, n): n for n in names}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.extend(fut.result())
            except Exception:
                print("ERR", futs[fut], traceback.format_exc()[-200:], flush=True)
            if i % 50 == 0:
                print(f"[{i}/{len(names)} {time.time()-t0:.0f}s]", flush=True)

    pred = np.array([r["pred"] for r in rows])
    did = np.array([r["did"] for r in rows], float)
    from scipy.stats import spearmanr
    ok = np.isfinite(pred) & np.isfinite(did)
    sp, pv = spearmanr(pred[ok], did[ok])
    # 匹配质量过滤后的稳健版
    good = ok & np.array([r["match_gap_log"] < 0.3 for r in rows])
    sp2, pv2 = spearmanr(pred[good], did[good])
    summary = {
        "n_entries": int(ok.sum()), "n_good_match": int(good.sum()),
        "spearman": round(float(sp), 3), "p": float(pv),
        "spearman_good": round(float(sp2), 3), "p_good": float(pv2),
        "sign_agree_good": round(float((np.sign(did[good]) == 1).mean()), 3),
        "did_med_good": float(np.median(did[good])),
        "pred_med_good": float(np.median(pred[good]))}
    (OUT / "natural_experiment.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))
    print("NATEXP DONE", flush=True)


if __name__ == "__main__":
    main()
