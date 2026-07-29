"""独立级联 IC(圈内稳定图上的蒙特卡洛):锚点近期活跃成员为种子,在圈内互动图上
以边权归一的激活概率逐跳传播,预测视界内圈内新增发帖人数 ≈ 激活数 × 人均帖率。

结构性局限(筛选目的):IC 是"一次性触达"模型,无时间轴;把 24h 窗当作一轮级联的
收敛集合,天然难匹配计数目标。p_scale 在 val 上校准。
用法: python3 fit_ic.py [--splits val test] [--workers 12] [--n-mc 60]
"""
import argparse
import multiprocessing as mp
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import HORIZONS, OUT, load_anchors, save_preds

ACTIVE_WINDOW_H = {"h6": 12, "h24": 48}  # 种子=近x小时发过帖的成员
P_SCALES = [0.02, 0.05, 0.1, 0.2, 0.4]


def load_graphs():
    z = np.load(OUT / "members.npz")
    n0 = int(z["n0"])
    graphs = {}
    for c, key in ((0, "intra0"), (5, "intra5")):
        e = z[key]
        adj = defaultdict(list)
        if len(e):
            wmax = e[:, 2].max()
            for u, v, w in e:
                p = w / wmax
                adj[int(u)].append((int(v), float(p)))
                adj[int(v)].append((int(u), float(p)))
        graphs[c] = adj
    return graphs, n0


def ic_expected_activations(adj, seeds, p_scale, n_mc, rng):
    if not seeds:
        return 0.0
    tot = 0
    for _ in range(n_mc):
        active = set(seeds)
        frontier = list(seeds)
        while frontier:
            nxt = []
            for u in frontier:
                for v, p in adj.get(u, ()):
                    if v not in active and rng.random() < p * p_scale:
                        active.add(v)
                        nxt.append(v)
            frontier = nxt
        tot += len(active) - len(seeds)
    return tot / n_mc


def run_event(job):
    event, ks, circles, n_mc = job
    graphs, n0 = GRAPHS
    z = np.load(OUT / "events" / f"{event}.npz")
    p_ts, p_gid = z["p_ts"], z["p_gid"]
    b0 = int(z["b0"])
    counts = z["counts"]
    rng = np.random.default_rng(hash(event) % 2**31)
    out = []
    for k in ks:
        t_anchor = (b0 + k) * 3600
        for c in circles:
            ci = 0 if c == 0 else 1
            in_circle = (p_gid < n0) if c == 0 else (p_gid >= n0)
            # 人均发帖率(锚点前 72h 的圈通道帖数 / 活跃成员数)
            lo72 = max(0, k - 72)
            posts72 = counts[ci, lo72:k].sum()
            m72 = (p_ts < t_anchor) & (p_ts >= t_anchor - 72 * 3600) & in_circle
            actives72 = len(np.unique(p_gid[m72]))
            rate = posts72 / max(1, actives72)
            for hname, H in HORIZONS.items():
                wa = ACTIVE_WINDOW_H[hname]
                m = (p_ts < t_anchor) & (p_ts >= t_anchor - wa * 3600) & in_circle
                seeds = list(np.unique(p_gid[m]))
                row = {"event": event, "circle": c, "k": int(k), "horizon": hname,
                       "n_seeds": len(seeds), "rate": float(rate)}
                for ps in P_SCALES:
                    act = ic_expected_activations(graphs[c], seeds, ps, n_mc, rng)
                    # 预测帖数 = (种子继续发 + 新激活) × 人均率 × 视界折算
                    exp_users = len(seeds) + act
                    row[f"yhat_p{ps}"] = float(exp_users * rate * (H / 72))
                out.append(row)
    return out


GRAPHS = None


def _init():
    global GRAPHS
    GRAPHS = load_graphs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-mc", type=int, default=60)
    args = ap.parse_args()
    import polars as pl
    an = load_anchors()
    sub = an.filter(pl.col("split").is_in(args.splits))
    jobs = {}
    for row in sub.iter_rows(named=True):
        jobs.setdefault(row["event"], {"ks": set(), "circles": set()})
        jobs[row["event"]]["ks"].add(row["k"])
        jobs[row["event"]]["circles"].add(row["circle"])
    joblist = [(e, sorted(v["ks"]), sorted(v["circles"]), args.n_mc) for e, v in jobs.items()]
    print(f"events: {len(joblist)}", flush=True)
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=_init) as ex:
        futs = {ex.submit(run_event, j): j[0] for j in joblist}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.extend(fut.result())
            except Exception:
                print("ERR", futs[fut], traceback.format_exc()[-200:], flush=True)
            if i % 30 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)} {time.time()-t0:.0f}s]", flush=True)

    # val 上按 MALE 选 p_scale,test 用选中的
    key = {(r["event"], r["circle"], r["k"]): (r["y6"], r["y24"], r["split"])
           for r in an.iter_rows(named=True)}
    import numpy as np
    best_ps = {}
    for hname in HORIZONS:
        val_rows = [r for r in rows if r["horizon"] == hname
                    and key[(r["event"], r["circle"], r["k"])][2] == "val"]
        scores = {}
        for ps in P_SCALES:
            y = np.array([key[(r["event"], r["circle"], r["k"])][0 if hname == "h6" else 1]
                          for r in val_rows], float)
            yh = np.array([r[f"yhat_p{ps}"] for r in val_rows], float)
            scores[ps] = float(np.mean(np.abs(np.log1p(yh) - np.log1p(y)))) if len(y) else 9e9
        best_ps[hname] = min(scores, key=scores.get)
        print(f"{hname}: val MALE by p_scale = { {k: round(v,3) for k,v in scores.items()} } "
              f"-> pick {best_ps[hname]}")
    final = []
    for r in rows:
        y6, y24, _ = key[(r["event"], r["circle"], r["k"])]
        final.append({"event": r["event"], "circle": r["circle"], "k": r["k"],
                      "horizon": r["horizon"],
                      "y": y6 if r["horizon"] == "h6" else y24,
                      "yhat": r[f"yhat_p{best_ps[r['horizon']]}"],
                      "n_seeds": r["n_seeds"]})
    save_preds("ic", final)


if __name__ == "__main__":
    main()
