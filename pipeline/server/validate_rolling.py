"""阶段D验证v2:滚动短视界计数预测(方法修正)。

v1 的"最后20%长窗前向模拟"验证与常数μ假设不匹配(热点事件自然冷却→系统性高估)。
v2 改为操作性任务:在事件时间轴 20%-90% 均匀取 8 个锚点 t,
给定 t 之前的**真实历史**,预测 (t, t+2h] 的事件总数(模拟 100 runs 均值),
与基线对比:
  - poisson_train: 用 t 之前全历史平均速率外推
  - naive: 用 (t-2h, t] 的实际计数持续
按事件汇总 8 锚点的中位 APE。
输出: ~/data/validation_rolling.json
用法: python3 validate_rolling.py [--horizon-h 2] [--workers 12]
"""
import argparse
import json
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DATA = Path.home() / "data"


def validate_one(job):
    name, horizon_h, n_runs = job
    import numpy as np
    from simulate import _simulate_many, history_state
    fit = json.loads((DATA / "hawkes" / f"{name}.json").read_text())
    ev = np.load(DATA / "hawkes" / "events" / f"{name}.npz")
    times, dims = ev["times"].astype(float), ev["dims"].astype(np.int64)
    K, T0, T1 = fit["D"], fit["T0"], fit["T1"]
    mu, A, beta = np.array(fit["mu"]), np.array(fit["A"]), fit["beta"]
    H = horizon_h * 3600.0
    span = T1 - T0
    if span <= H * 4:
        return {"name": name, "skip": "window too short"}

    anchors = [T0 + f * span for f in np.linspace(0.2, 0.9, 8)]
    rows = []
    for t in anchors:
        hist = times < t
        n_hist = int(hist.sum())
        if n_hist < 100:
            continue
        actual = int(((times >= t) & (times < t + H)).sum())
        state0 = history_state(times[hist], dims[hist], K, beta, t)
        sim = _simulate_many(mu, A, beta, state0, t, t + H, -1.0, 0, 0, n_runs, 99, 200_000)
        pred = float(sim.sum(1).mean())
        rate = n_hist / (t - T0)
        poisson = rate * H
        naive = int(((times >= t - H) & (times < t)).sum())
        rows.append({"t_frac": round((t - T0) / span, 2), "actual": actual,
                     "hawkes": round(pred, 1), "poisson": round(poisson, 1), "naive": naive,
                     "q10": float(np.quantile(sim.sum(1), 0.1)),
                     "q90": float(np.quantile(sim.sum(1), 0.9))})
    if not rows:
        return {"name": name, "skip": "no valid anchors"}

    def med_ape(key):
        import numpy as np
        return float(np.median([abs(r[key] - r["actual"]) / max(1, r["actual"]) for r in rows]))
    cov = float(np.mean([r["q10"] <= r["actual"] <= r["q90"] for r in rows]))
    return {"name": name, "n_anchors": len(rows),
            "ape_hawkes": round(med_ape("hawkes"), 4), "ape_poisson": round(med_ape("poisson"), 4),
            "ape_naive": round(med_ape("naive"), 4), "coverage": round(cov, 3), "anchors": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-h", type=float, default=2.0)
    ap.add_argument("--n-runs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    names = sorted(p.stem for p in (DATA / "hawkes" / "events").glob("*.npz"))
    print(f"events: {len(names)}", flush=True)
    jobs = [(n, args.horizon_h, args.n_runs) for n in names]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(validate_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception:
                results.append({"name": futs[fut][0], "error": traceback.format_exc()[-300:]})
            if i % 50 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)} {time.time()-t0:.0f}s]", flush=True)
    with open(DATA / "validation_rolling.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    import numpy as np
    ok = [r for r in results if "ape_hawkes" in r]
    med = {k: float(np.median([r[k] for r in ok])) for k in ("ape_hawkes", "ape_poisson", "ape_naive")}
    win = float(np.mean([r["ape_hawkes"] <= min(r["ape_poisson"], r["ape_naive"]) for r in ok]))
    cov = float(np.median([r["coverage"] for r in ok]))
    print(json.dumps({"n": len(ok), "median_ape": med, "hawkes_wins": win, "median_coverage": cov}, indent=1))
    print("ROLLING VALIDATION DONE", flush=True)


if __name__ == "__main__":
    main()
