"""SIR 滑窗拟合(逐锚点):对圈通道的小时新增帖数拟合离散 SIR,前向积分预测。

把"发帖"视作感染事件:新增帖 ≈ β S I / N(每小时),恢复率 γ。
参数经网格搜索(R0 × γ半衰 × 有效人口倍数),窗口 = 锚点前 14 天,按 log1p-SSE 选优。
多峰/再燃事件预期表现差——这正是筛选要暴露的结构性缺陷。
用法: python3 fit_sir.py [--splits val test] [--workers 12]
"""
import argparse
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import BIN, HORIZONS, OUT, load_anchors, save_preds

WINDOW_H = 14 * 24
R0_GRID = [0.6, 0.9, 1.1, 1.5, 2.5, 4.0]
GAMMA_HL_H = [2, 6, 12, 24, 72]
NMULT_GRID = [1.2, 2.0, 4.0, 10.0]


def sir_forward(S0, I0, N, beta, gamma, steps):
    """离散 Euler(1h 步),返回每步新增感染。"""
    S, I = S0, I0
    new = np.zeros(steps)
    for t in range(steps):
        inf = beta * S * I / N if N > 0 else 0.0
        inf = min(inf, S)
        rec = gamma * I
        S -= inf
        I += inf - rec
        I = max(I, 0.0)
        new[t] = inf
    return new


def fit_one(counts_hist):
    """counts_hist: 窗口内小时新增(圈通道)。返回 (beta,gamma,N,S0,I0)。"""
    cum = counts_hist.sum()
    if cum < 5:
        return None
    best, best_sse = None, np.inf
    obs = np.log1p(counts_hist)
    for ghl in GAMMA_HL_H:
        gamma = np.log(2) / ghl
        # I0: 按 γ 衰减折算的近期活跃量
        w = np.exp(-gamma * np.arange(len(counts_hist) - 1, -1, -1))
        I0 = float((counts_hist * w).sum())
        for r0 in R0_GRID:
            beta = r0 * gamma
            for nm in NMULT_GRID:
                N = max(cum * nm, 10.0)
                S0 = max(N - cum, 1.0)
                sim = sir_forward(S0, I0, N, beta, gamma, len(counts_hist))
                sse = float(((np.log1p(sim) - obs) ** 2).sum())
                if sse < best_sse:
                    best_sse, best = sse, (beta, gamma, N, S0, I0)
    return best


def run_event(job):
    event, ks, circles = job
    z = np.load(OUT / "events" / f"{event}.npz")
    counts = z["counts"]
    out = []
    for k in ks:
        lo = max(0, k - WINDOW_H)
        for c in circles:
            ci = 0 if c == 0 else 1
            hist = counts[ci, lo:k].astype(float)
            fit = fit_one(hist)
            if fit is None:
                for hname in HORIZONS:
                    out.append({"event": event, "circle": c, "k": int(k),
                                "horizon": hname, "yhat": 0.0})
                continue
            beta, gamma, N, S0, I0 = fit
            # 用窗口末状态重放到锚点,再前向预测
            replay = sir_forward(S0, I0, N, beta, gamma, len(hist))
            used = replay.sum()
            S_t = max(S0 - used, 0.0)
            I_t = I0 * np.exp(-gamma * len(hist)) + replay[-min(24, len(replay)):].sum() * 0.5
            fut = sir_forward(S_t, max(I_t, hist[-6:].sum() / 6), N, beta, gamma,
                              max(HORIZONS.values()))
            cumfut = np.concatenate([[0], np.cumsum(fut)])
            for hname, H in HORIZONS.items():
                out.append({"event": event, "circle": c, "k": int(k), "horizon": hname,
                            "yhat": float(cumfut[H])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    import polars as pl
    an = load_anchors()
    sub = an.filter(pl.col("split").is_in(args.splits))
    jobs = {}
    for row in sub.iter_rows(named=True):
        jobs.setdefault(row["event"], {"ks": set(), "circles": set()})
        jobs[row["event"]]["ks"].add(row["k"])
        jobs[row["event"]]["circles"].add(row["circle"])
    joblist = [(e, sorted(v["ks"]), sorted(v["circles"])) for e, v in jobs.items()]
    print(f"events: {len(joblist)}", flush=True)
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(run_event, j): j[0] for j in joblist}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.extend(fut.result())
            except Exception:
                print("ERR", futs[fut], traceback.format_exc()[-200:], flush=True)
            if i % 30 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)} {time.time()-t0:.0f}s]", flush=True)
    key = {(r["event"], r["circle"], r["k"]): (r["y6"], r["y24"]) for r in an.iter_rows(named=True)}
    for r in rows:
        y6, y24 = key[(r["event"], r["circle"], r["k"])]
        r["y"] = y6 if r["horizon"] == "h6" else y24
    save_preds("sir", rows)


if __name__ == "__main__":
    main()
