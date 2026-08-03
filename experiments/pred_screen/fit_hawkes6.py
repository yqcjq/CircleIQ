"""六通道 Hawkes(锚点前重拟合):对每个 test/val 锚点,用锚点前秒级流拟合双核 Hawkes,
前向模拟 (t, t+H] 取圈通道计数中位数。回归族主力候选。

维度 = 6 通道(c0, c5, stab, bigv, org, other),与数据集通道一致。
为控算力:每 (event) 只在测试锚点子集上跑;拟合窗口截尾 14 天(长事件太长的历史稀释 μ)。
用法: python3 fit_hawkes6.py [--splits test val] [--workers 12] [--n-runs 100]
"""
import argparse
import json
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import BIN, HORIZONS, OUT, load_anchors, save_preds

FIT_WINDOW_S = 14 * 86400  # 拟合窗:锚点前最多 14 天
MIN_FIT_EVENTS = 300


def run_event(job):
    event, ks, circles, n_runs = job
    import numpy as np
    from hawkes_lib import fit_hawkes
    from simulate_lib import _simulate_many, history_state
    z = np.load(OUT / "events" / f"{event}.npz")
    ts_all = z["ts_all"].astype(np.float64)
    ch_all = z["ch_all"].astype(np.int64)
    b0 = int(z["b0"])
    K = 6
    out = []
    cost = 0.0
    for k in sorted(set(ks)):
        t_anchor = float((b0 + k) * BIN)
        lo = t_anchor - FIT_WINDOW_S
        m = (ts_all < t_anchor) & (ts_all >= lo)
        tt, dd = ts_all[m], ch_all[m]
        if len(tt) < MIN_FIT_EVENTS:
            # 历史太薄:退化为速率外推
            n7 = ((ts_all < t_anchor) & (ts_all >= t_anchor - 7 * 86400)).sum()
            for hname, H in HORIZONS.items():
                for c, ci in zip(circles, [0 if c == 0 else 1 for c in circles]):
                    frac = ((ch_all[m] == ci).sum() / max(1, len(tt)))
                    out.append({"event": event, "circle": c, "k": int(k), "horizon": hname,
                                "yhat": float(n7 / 7 / 24 * H * frac), "mode": "fallback"})
            continue
        t0 = time.time()
        T0f, T1f = float(tt[0]) - 1, t_anchor
        try:
            res = fit_hawkes(tt, dd, K, T0f, T1f, val_frac=0.0, mu_bins=4)
        except Exception:
            for hname, H in HORIZONS.items():
                for c in circles:
                    out.append({"event": event, "circle": c, "k": int(k), "horizon": hname,
                                "yhat": 0.0, "mode": "fit_error"})
            continue
        s1 = history_state(tt, dd, K, res["beta1"], t_anchor)
        s2 = history_state(tt, dd, K, res["beta2"], t_anchor)
        # μ 用最后一段(拟合窗内最新背景;窗内全为锚点前历史,无泄漏)
        mu_loc = res["mu"][:, -1][:, None].copy()
        Hmax = max(HORIZONS.values()) * BIN
        sim = _simulate_many(mu_loc, np.array([-1e18, 1e18]),
                             res["A1"], res["beta1"], res["A2"], res["beta2"],
                             s1, s2, t_anchor, t_anchor + Hmax, -1.0, 0, 0,
                             n_runs, 1234, 300_000)
        cost += time.time() - t0
        # sim 是各维总计数;要按视界分,需要重模拟或返回时间。粗做:6h 用二次模拟
        sim6 = _simulate_many(mu_loc, np.array([-1e18, 1e18]),
                              res["A1"], res["beta1"], res["A2"], res["beta2"],
                              s1, s2, t_anchor, t_anchor + 6 * BIN, -1.0, 0, 0,
                              n_runs, 4321, 300_000)
        for c in circles:
            ci = 0 if c == 0 else 1
            out.append({"event": event, "circle": c, "k": int(k), "horizon": "h24",
                        "yhat": float(np.median(sim[:, ci])), "mode": "hawkes"})
            out.append({"event": event, "circle": c, "k": int(k), "horizon": "h6",
                        "yhat": float(np.median(sim6[:, ci])), "mode": "hawkes"})
    return out, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-runs", type=int, default=100)
    args = ap.parse_args()

    an = load_anchors().filter(__import__("polars").col("split").is_in(args.splits))
    jobs = {}
    for row in an.iter_rows(named=True):
        jobs.setdefault(row["event"], {"ks": set(), "circles": set()})
        jobs[row["event"]]["ks"].add(row["k"])
        jobs[row["event"]]["circles"].add(row["circle"])
    joblist = [(e, sorted(v["ks"]), sorted(v["circles"]), args.n_runs) for e, v in jobs.items()]
    joblist.sort(key=lambda j: -len(j[1]))
    print(f"events: {len(joblist)}, anchor-fits: {sum(len(j[1]) for j in joblist)}", flush=True)

    rows, total_cost = [], 0.0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(run_event, j): j[0] for j in joblist}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r, c = fut.result()
                rows.extend(r)
                total_cost += c
            except Exception:
                print("ERR", futs[fut], traceback.format_exc()[-300:], flush=True)
            if i % 20 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)} {time.time()-t0:.0f}s]", flush=True)

    # join 真值
    key = {(r["event"], r["circle"], r["k"]): (r["y6"], r["y24"]) for r in
           load_anchors().iter_rows(named=True)}
    for r in rows:
        y6, y24 = key[(r["event"], r["circle"], r["k"])]
        r["y"] = y6 if r["horizon"] == "h6" else y24
    save_preds("hawkes6", rows)
    meta = {"total_fit_sim_cost_s": round(total_cost, 1), "wall_s": round(time.time() - t0, 1),
            "n_rows": len(rows)}
    (OUT / "preds" / "hawkes6_meta.json").write_text(json.dumps(meta))
    print(meta)


if __name__ == "__main__":
    main()
