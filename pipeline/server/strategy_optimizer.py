"""阶段E:策略搜索(服务器端)。

对每个已拟合议题/事件:
  在 t_start = T0 + 30% span 处,枚举策略 (目标圈层维 d, 注入量 m, 延迟 delay),
  用 Hawkes 反事实模拟评估 24h 视界内:
    induced = E[总事件增量] - m(诱发量,不含注入本身)
    roi     = induced / m
    runaway = P(总量 > 1.5×base_q95)
    backfire_other = other 维(圈外大众)增量占比
输出: ~/data/strategy/<cat>__<topic>.json + strategy_summary.json

用法: python3 strategy_optimizer.py --targets hot__20260313162141,hot__xxx [--m 1,5,20]
      python3 strategy_optimizer.py --auto-top 8   # 自动选验证效果最好的 8 个事件
"""
import argparse
import json
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DATA = Path.home() / "data"
HORIZON = 24 * 3600.0


def optimize_topic(job):
    name, ms, delays, n_runs = job
    import numpy as np
    from simulate import simulate_counterfactual
    t0 = time.time()
    fit = json.loads((DATA / "hawkes" / f"{name}.json").read_text())
    ev = np.load(DATA / "hawkes" / "events" / f"{name}.npz")
    times, dims = ev["times"], ev["dims"]
    params = {"mu": fit["mu"], "A": fit["A"], "beta": fit["beta"]}
    K = fit["D"]
    T0f, T1f = fit["T0"], fit["T1"]
    t_start = T0f + (T1f - T0f) * 0.3
    hist = (times[times < t_start], dims[times < t_start])

    rows = []
    base_cache = None
    for d in range(K):
        for m in ms:
            for delay in delays:
                res = simulate_counterfactual(params, hist, t_start, HORIZON,
                                              inject=(t_start + delay, d, m), n_runs=n_runs)
                if base_cache is None:
                    base_cache = {"base_mean_total": float(res["base_mean"].sum()),
                                  "base_q95_total": res["base_q95_total"]}
                induced = res["delta_total"] - m
                other = fit["other_dim"]
                rows.append({
                    "dim": d, "dim_label": fit.get("dim_labels", {}).get(str(d), str(d)),
                    "m": m, "delay_h": delay / 3600,
                    "induced": round(induced, 1), "roi": round(induced / m, 2),
                    "runaway_prob": round(res["runaway_prob"], 3),
                    "delta_other_share": round(float(res["delta_mean"][other] / max(res["delta_total"], 1e-9)), 3),
                })
    rows.sort(key=lambda r: -r["roi"])
    out = {"name": name, "K": K, "t_start_frac": 0.3, "horizon_h": 24,
           "n_runs": n_runs, **(base_cache or {}), "grid": rows,
           "top5": rows[:5], "secs": round(time.time() - t0, 1)}
    (DATA / "strategy").mkdir(exist_ok=True)
    with open(DATA / "strategy" / f"{name}.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="")
    ap.add_argument("--auto-top", type=int, default=0, help="按验证LL增益自动选前N个已拟合事件")
    ap.add_argument("--m", default="1,5,20")
    ap.add_argument("--delays", default="0,21600")
    ap.add_argument("--n-runs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.targets:
        names = args.targets.split(",")
    else:
        cands = []
        for f in (DATA / "hawkes").glob("*.json"):
            if f.stem.startswith("hawkes"):
                continue
            r = json.loads(f.read_text())
            if r.get("ll_val") is not None and r.get("ll_val_poisson") is not None \
                    and (DATA / "hawkes" / "events" / f"{f.stem}.npz").exists():
                cands.append((r["ll_val"] - r["ll_val_poisson"], f.stem))
        cands.sort(reverse=True)
        names = [n for _, n in cands[: args.auto_top or 8]]
    ms = [int(x) for x in args.m.split(",")]
    delays = [float(x) for x in args.delays.split(",")]
    print(f"targets: {names}", flush=True)

    jobs = [(n, ms, delays, args.n_runs) for n in names]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(optimize_topic, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                r = fut.result()
                results.append({"name": r["name"], "top5": r["top5"], "base": r.get("base_mean_total")})
                print(f"{r['name']}: top ROI={r['top5'][0]['roi']} "
                      f"(dim={r['top5'][0]['dim']} m={r['top5'][0]['m']}) {r['secs']}s", flush=True)
            except Exception:
                print(f"ERROR {j[0]}\n{traceback.format_exc()[-400:]}", flush=True)
    with open(DATA / "strategy_summary.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("STRATEGY DONE", flush=True)


if __name__ == "__main__":
    main()
