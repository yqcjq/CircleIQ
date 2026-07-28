"""阶段D:留出窗口计数验证(全事件定量,多进程)。

对每个已拟合事件:
  从 t_split(训练/验证分界)起,用拟合参数前向模拟(不看验证事件),
  预测验证窗总事件数与各维事件数,对比:
    - actual: 实际计数
    - poisson: 训练窗平均速率外推
    - naive: 训练窗最后同长时段的计数(持续性基线)
输出: ~/data/validation_counts.json(逐事件)
用法: python3 validate_counts.py [--n-runs 200] [--workers 12]
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
    name, n_runs = job
    import numpy as np
    from simulate import _simulate_many, history_state
    fit = json.loads((DATA / "hawkes" / f"{name}.json").read_text())
    if fit.get("ll_val") is None:
        return {"name": name, "skip": "no val window"}
    ev = np.load(DATA / "hawkes" / "events" / f"{name}.npz")
    times, dims = ev["times"].astype(float), ev["dims"].astype(np.int64)
    K, T0, T1 = fit["D"], fit["T0"], fit["T1"]
    mu, A, beta = np.array(fit["mu"]), np.array(fit["A"]), fit["beta"]
    t_split = T0 + (T1 - T0) * 0.8  # 与 run_hawkes val_frac=0.2 一致
    dur_val = T1 - t_split

    tr = times <= t_split
    va = ~tr
    actual_total = int(va.sum())
    actual_by = np.bincount(dims[va], minlength=K)

    state0 = history_state(times[tr], dims[tr], K, beta, t_split)
    sim = _simulate_many(mu, A, beta, state0, t_split, t_split + dur_val, -1.0, 0, 0, n_runs, 777,
                         500_000)
    pred_total = float(sim.sum(1).mean())
    pred_by = sim.mean(0)

    rate_train = tr.sum() / (t_split - T0)
    poisson_total = float(rate_train * dur_val)
    naive_mask = (times > t_split - dur_val) & tr
    naive_total = int(naive_mask.sum())

    def ape(pred, act):
        return abs(pred - act) / max(1, act)

    return {
        "name": name, "n_train": int(tr.sum()), "actual_val": actual_total,
        "pred_hawkes": round(pred_total, 1), "pred_poisson": round(poisson_total, 1),
        "pred_naive": naive_total,
        "ape_hawkes": round(ape(pred_total, actual_total), 4),
        "ape_poisson": round(ape(poisson_total, actual_total), 4),
        "ape_naive": round(ape(naive_total, actual_total), 4),
        "dim_mae_hawkes": round(float(np.abs(pred_by - actual_by).mean()), 2),
        "pred_q10": float(np.quantile(sim.sum(1), 0.1)),
        "pred_q90": float(np.quantile(sim.sum(1), 0.9)),
        "covered": bool(np.quantile(sim.sum(1), 0.1) <= actual_total <= np.quantile(sim.sum(1), 0.9)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    names = sorted(p.stem for p in (DATA / "hawkes" / "events").glob("*.npz"))
    if args.limit:
        names = names[: args.limit]
    print(f"events to validate: {len(names)}", flush=True)
    jobs = [(n, args.n_runs) for n in names]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(validate_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception:
                r = {"name": futs[fut][0], "error": traceback.format_exc()[-300:]}
            results.append(r)
            if i % 25 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)} {time.time()-t0:.0f}s]", flush=True)
    with open(DATA / "validation_counts.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    import numpy as np
    ok = [r for r in results if "ape_hawkes" in r]
    if ok:
        med = {k: float(np.median([r[k] for r in ok])) for k in ("ape_hawkes", "ape_poisson", "ape_naive")}
        cov = float(np.mean([r["covered"] for r in ok]))
        win = float(np.mean([r["ape_hawkes"] <= min(r["ape_poisson"], r["ape_naive"]) for r in ok]))
        print(json.dumps({"n": len(ok), "median_ape": med, "q10_q90_coverage": cov,
                          "hawkes_wins_share": win}, indent=1))
    print("VALIDATION DONE", flush=True)


if __name__ == "__main__":
    main()
