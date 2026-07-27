"""阶段D案例:真实大V发帖的反事实模拟(What-if)。

流程:
  1. 在事件流中定位一个高影响力时刻:认证=bigv/org 且粉丝数最高的作者在窗口
     [T0+10%, T0+60%] 内的首帖 t_kol
  2. actual   : t_kol 后 24h 每小时真实事件数
  3. factual  : 以 t_kol 为界,历史含该帖 -> 前向模拟(200 runs)
  4. counterfactual: 历史剔除该帖(该作者 t_kol 时刻的注入不发生)-> 前向模拟
  5. 输出曲线 JSON 供本地画图
用法: python3 case_study.py --name hot__20260313162141
"""
import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path.home() / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--horizon-h", type=float, default=24)
    ap.add_argument("--n-runs", type=int, default=300)
    args = ap.parse_args()
    import polars as pl
    from simulate import _simulate_many, history_state

    fit = json.loads((DATA / "hawkes" / f"{args.name}.json").read_text())
    ev = np.load(DATA / "hawkes" / "events" / f"{args.name}.npz")
    times, dims = ev["times"].astype(float), ev["dims"].astype(np.int64)
    mu, A, beta = np.array(fit["mu"]), np.array(fit["A"]), fit["beta"]
    K, T0, T1 = fit["D"], fit["T0"], fit["T1"]

    cat, topic = args.name.split("__", 1)
    files = sorted((DATA / "core" / cat).glob(f"{topic}__*.parquet"))
    df = pl.concat([pl.read_parquet(f, columns=["md5_author", "md5_mid", "ts", "auth_tier", "n_followers"]) for f in files])
    df = df.filter(pl.col("md5_mid").is_null() | (pl.int_range(pl.len()).over("md5_mid") == 0))
    lo, hi = T0 + 0.10 * (T1 - T0), T0 + 0.60 * (T1 - T0)
    kol = (df.filter(pl.col("ts").is_between(lo, hi) & pl.col("auth_tier").is_in(["bigv", "org"]))
             .sort("n_followers", descending=True).head(1))
    if kol.height == 0:
        print(json.dumps({"error": "no KOL post in window"}))
        return
    t_kol = float(kol["ts"][0])
    kol_author = kol["md5_author"][0]
    # KOL 事件在流中的维度:找 times 最接近 t_kol 的该作者事件
    part = pl.read_parquet(DATA / "partition_stable_K3.parquet").filter(pl.col("md5_author") == kol_author)
    circle = int(part["circle_id"][0]) if part.height else None
    dim_map = {int(v): int(k) for k, v in fit["dim_circles"].items()}
    if circle in dim_map:
        d_kol = dim_map[circle]
    elif kol["auth_tier"][0] == "bigv":
        d_kol = fit.get("bigv_dim", fit["other_dim"])
    elif kol["auth_tier"][0] == "org":
        d_kol = fit.get("org_dim", fit["other_dim"])
    else:
        d_kol = fit["other_dim"]

    horizon = args.horizon_h * 3600
    hist_mask = times < t_kol
    ht, hd = times[hist_mask], dims[hist_mask]
    state_with = history_state(ht, hd, K, beta, t_kol)
    state_with[d_kol] += beta  # 该帖本身在 t_kol 注入

    n_bins = int(args.horizon_h)
    # factual / counterfactual 模拟(按小时分箱需要事件级输出;简化:用 12 段×2h)
    from simulate import _simulate_one
    def run_many(state0, seed0):
        curves = np.zeros((args.n_runs, n_bins))
        for r in range(args.n_runs):
            st, sd = _simulate_one(mu, A, beta, state0.copy(), t_kol, t_kol + horizon,
                                   -1.0, 0, 0, seed0 + r)
            if len(st):
                b = np.minimum(((st - t_kol) / 3600).astype(int), n_bins - 1)
                np.add.at(curves[r], b, 1)
        return curves

    cw = run_many(state_with, 1000)
    cwo = run_many(history_state(ht, hd, K, beta, t_kol), 5000)

    va = (times >= t_kol) & (times < t_kol + horizon)
    actual = np.zeros(n_bins)
    b = np.minimum(((times[va] - t_kol) / 3600).astype(int), n_bins - 1)
    np.add.at(actual, b, 1)

    out = {
        "name": args.name, "t_kol": t_kol, "kol_followers": int(kol["n_followers"][0]),
        "kol_auth": kol["auth_tier"][0], "kol_dim": d_kol, "kol_circle": circle,
        "horizon_h": args.horizon_h,
        "actual_hourly": actual.tolist(),
        "factual_mean": cw.mean(0).tolist(), "factual_p10": np.quantile(cw, 0.1, 0).tolist(),
        "factual_p90": np.quantile(cw, 0.9, 0).tolist(),
        "counterfactual_mean": cwo.mean(0).tolist(),
        "counterfactual_p10": np.quantile(cwo, 0.1, 0).tolist(),
        "counterfactual_p90": np.quantile(cwo, 0.9, 0).tolist(),
        "delta_total_mean": float(cw.sum(1).mean() - cwo.sum(1).mean()),
    }
    (DATA / "case_studies").mkdir(exist_ok=True)
    with open(DATA / "case_studies" / f"{args.name}.json", "w") as f:
        json.dump(out, f)
    print(json.dumps({k: v for k, v in out.items() if not k.endswith(("mean", "p10", "p90", "hourly"))},
                     ensure_ascii=False))
    print("CASE DONE")


if __name__ == "__main__":
    main()
