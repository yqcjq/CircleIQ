"""策略反事实的闭式计算(P0-1):替代 MC 网格搜索的数学重构。

第一性原理:线性 Hawkes 的期望是线性代数,不需要蒙特卡洛。
- 无穷视界:注入维 d 的期望簇大小向量 g_d = (I-N^T)^{-1} e_d,N = A1+A2(分支矩阵)
  仅 rho(N)<1 时有效;近临界时被 1/(1-rho) 放大,继承 α 全部误差
- 有限视界 T(与模拟同口径,24h):期望强度 λ(t) 满足线性更新方程,双指数核下化为
  2K 维线性 ODE:z=(u1,u2), z'=Mz, M=[[b1(A1^T-I), b1 A1^T],[b2 A2^T, b2(A2^T-I)]],
  z(0)=(b1 A1^T e_d, b2 A2^T e_d);n(T) = e_d + ∫λdt 用增广矩阵指数一次算出全部种子维。
  对超临界事件依然有限、无 cap 伪影——这是"模拟目标"的精确实现。
- 敏感度:A 元素独立 ×U(0.8,1.2) 扰动 50 次,per-事件通道排序 Kendall τ。
- MC 对照验收(--validate):分 rho 档抽事件,比较闭式 24h Δ 与 simulate_counterfactual。

用法: python3 closed_form.py [--validate] [--workers 8]
输出: ~/data/strategy_v2/closed_form_all.json / mc_validation.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path.home() / "data"
sys.path.insert(0, str(DATA))
OUT = DATA / "strategy_v2"
HORIZON = 24 * 3600.0


def finite_horizon_counts(A1, b1, A2, b2, T):
    """返回 n[K, K]:n[d] = 种子在维 d 时 T 内期望计数向量(含种子)。

    推导:u1=E[s1], u2=E[s2](两核衰减和的期望),λ̄=A1ᵀu1+A2ᵀu2,
    u1' = -b1 u1 + b1 λ̄,u2' = -b2 u2 + b2 λ̄,u1(0)=b1 e_d, u2(0)=b2 e_d;
    n(T) = e_d + ∫₀ᵀ λ̄ dt。T→∞ 亚临界极限 = (I-Nᵀ)⁻¹ e_d(已验证代数一致)。"""
    from scipy.linalg import expm
    K = A1.shape[0]
    I = np.eye(K)
    Maug = np.zeros((3 * K, 3 * K))
    Maug[:K, :K] = b1 * (A1.T - I)
    Maug[:K, K:2 * K] = b1 * A2.T
    Maug[K:2 * K, :K] = b2 * A1.T
    Maug[K:2 * K, K:2 * K] = b2 * (A2.T - I)
    Maug[2 * K:, :K] = A1.T            # w' = λ̄
    Maug[2 * K:, K:2 * K] = A2.T
    E = expm(Maug * T)
    Z0 = np.zeros((3 * K, K))          # 每列 = 种子维 d 的初值
    Z0[:K] = b1 * I
    Z0[K:2 * K] = b2 * I
    W = (E @ Z0)[2 * K:]               # ∫λ̄ dt,列 d
    return np.eye(K) + W.T             # n[d, j]


def analyze_event(name, n_perturb=50, seed=7):
    fit = json.loads((DATA / "hawkes" / f"{name}.json").read_text())
    A1, A2 = np.asarray(fit["A1"]), np.asarray(fit["A2"])
    b1, b2 = float(fit["beta1"]), float(fit["beta2"])
    K = fit["D"]
    N = A1 + A2
    rho = float(np.abs(np.linalg.eigvals(N)).max())
    labels = [fit["dim_labels"][str(i)] for i in range(K)]

    out = {"name": name, "K": K, "rho": round(rho, 4), "labels": labels,
           "dim_circles": fit.get("dim_circles", {}),
           "bigv_dim": fit.get("bigv_dim"), "org_dim": fit.get("org_dim"),
           "other_dim": fit.get("other_dim"),
           "stable_event_share": fit.get("stable_event_share"),
           "amplification": round(1.0 / max(1e-9, 1 - rho), 1) if rho < 1 else None}

    # 无穷视界(亚临界)
    if rho < 0.999:
        Ginf = np.linalg.solve(np.eye(K) - N.T, np.eye(K))   # 列 d = g_d
        out["inf_total"] = np.round(Ginf.sum(0), 3).tolist()
        out["inf_comp"] = np.round(Ginf.T, 4).tolist()        # [d][j]
    # 有限视界 24h(全事件可算;强超临界数值溢出时标记)
    n24 = finite_horizon_counts(A1, b1, A2, b2, HORIZON)
    if not np.isfinite(n24).all():
        out["overflow"] = True
        return out
    out["h24_total"] = np.round(n24.sum(1), 3).tolist()       # 注入 d 的期望总计数(含种子)
    out["h24_comp"] = np.round(n24, 4).tolist()
    # 溢出分解修复:非目标维合计占诱发量比例(d 行,去掉种子)
    induced = n24 - np.eye(K)
    tot = induced.sum(1)
    spill = 1.0 - np.diag(induced) / np.maximum(tot, 1e-12)
    out["h24_spill_share"] = np.round(spill, 4).tolist()
    out["h24_other_share"] = np.round(
        induced[:, fit["other_dim"]] / np.maximum(tot, 1e-12), 4).tolist()

    # 敏感度:排序稳定性
    rng = np.random.default_rng(seed)
    base_rank = np.argsort(-n24.sum(1))
    taus = []
    from scipy.stats import kendalltau
    for _ in range(n_perturb):
        f1 = rng.uniform(0.8, 1.2, size=A1.shape)
        n_p = finite_horizon_counts(A1 * f1, b1, A2 * f1, b2, HORIZON)
        tau, _ = kendalltau(-n24.sum(1), -n_p.sum(1))
        taus.append(tau)
    out["kendall_tau_med"] = round(float(np.median(taus)), 3)
    out["kendall_tau_p10"] = round(float(np.percentile(taus, 10)), 3)
    return out


def mc_validate(events_by_rho, n_runs=300):
    """闭式 vs MC 对照:每事件注入 m=20 到 3 个代表维,t_start=T0+0.3span。"""
    from simulate import params_from_fit, simulate_counterfactual
    rows = []
    for name in events_by_rho:
        fit = json.loads((DATA / "hawkes" / f"{name}.json").read_text())
        ev = np.load(DATA / "hawkes" / "events" / f"{name}.npz")
        times, dims = ev["times"], ev["dims"]
        P = params_from_fit(fit)
        A1, A2 = np.asarray(fit["A1"]), np.asarray(fit["A2"])
        b1, b2 = float(fit["beta1"]), float(fit["beta2"])
        K = fit["D"]
        rho = float(np.abs(np.linalg.eigvals(A1 + A2)).max())
        n24 = finite_horizon_counts(A1, b1, A2, b2, HORIZON)
        T0f, T1f = fit["T0"], fit["T1"]
        t_start = T0f + (T1f - T0f) * 0.3
        hist = (times[times < t_start], dims[times < t_start])
        m = 20
        test_dims = [0, fit["bigv_dim"], fit["other_dim"]]
        for d in test_dims:
            t0 = time.time()
            res = simulate_counterfactual(P, hist, t_start, HORIZON,
                                          inject=(t_start, d, m), n_runs=n_runs,
                                          max_events=500_000)
            cf = m * n24[d].sum()
            se = float((res["inj_totals"].std() + res["base_totals"].std())
                       / np.sqrt(n_runs))
            rows.append({
                "name": name, "rho": round(rho, 4), "dim": int(d),
                "label": fit["dim_labels"][str(d)], "m": m,
                "closed_form": round(float(cf), 1),
                "mc_delta": round(float(res["delta_total"]), 1),
                "mc_se": round(se, 1),
                "z": round((res["delta_total"] - cf) / max(se, 1e-9), 2),
                "cap_hit": res["base_cap_hit"],
                "cap_hit_inj": float((res["inj_totals"] >= 500_000 * 0.98).mean()),
                "secs": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n-perturb", type=int, default=50)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    names = sorted(f.stem for f in (DATA / "hawkes").glob("*.json")
                   if not f.stem.startswith("hawkes"))
    print(f"events: {len(names)}", flush=True)
    t0 = time.time()
    results, errors = [], []
    for i, n in enumerate(names, 1):
        try:
            results.append(analyze_event(n, args.n_perturb))
        except Exception as e:
            errors.append({"name": n, "error": str(e)})
        if i % 50 == 0:
            print(f"[{i}/{len(names)} {time.time()-t0:.0f}s]", flush=True)
    rhos = np.array([r["rho"] for r in results])
    summary = {
        "n": len(results), "errors": errors,
        "rho_pcts": {p: round(float(np.percentile(rhos, p)), 4)
                     for p in (10, 50, 90, 99)},
        "n_supercritical": int((rhos >= 1).sum()),
        "n_near_critical": int((rhos >= 0.97).sum()),
        "kendall_med_overall": round(float(np.median(
            [r["kendall_tau_med"] for r in results])), 3)}
    (OUT / "closed_form_all.json").write_text(
        json.dumps({"summary": summary, "events": results}, ensure_ascii=False))
    print("summary:", json.dumps(summary, ensure_ascii=False), flush=True)

    if args.validate:
        by_rho = sorted(results, key=lambda r: r["rho"])
        idx = [int(q * (len(by_rho) - 1)) for q in (0.05, 0.3, 0.6, 0.85, 0.95, 0.99)]
        sel = [by_rho[i]["name"] for i in idx]
        print("validate on:", [(by_rho[i]["name"], by_rho[i]["rho"]) for i in idx], flush=True)
        rows = mc_validate(sel)
        (OUT / "mc_validation.json").write_text(json.dumps(rows, ensure_ascii=False))
    print("CLOSED FORM DONE", flush=True)


if __name__ == "__main__":
    main()
