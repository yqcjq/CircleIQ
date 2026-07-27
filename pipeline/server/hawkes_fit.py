"""多维 Hawkes 拟合(服务器端,numba + 多进程)。

模型: λ_k(t) = μ_k + Σ_j α_{jk} Σ_{t^j_i<t} β e^{-β(t-t^j_i)}
- β 走网格(固定 β 下 LL 对 (μ,α) 凸,且按目标维 k 分离 -> K 个独立小凸问题)
- 每个目标维用 scipy L-BFGS-B + 解析梯度
- 输出: μ 向量、α 矩阵、最优 β、LL、vs 齐次 Poisson 的提升、验证窗 LL

用法(被 run_hawkes.py 调用,也可单测):
  from hawkes_fit import fit_hawkes
  res = fit_hawkes(times, dims, K, T0, T1, betas=[...])
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize


@njit(cache=True)
def _precompute_R(times, dims, K, beta):
    """R[i, j] = Σ_{l: t_l<t_i, d_l=j} β e^{-β(t_i-t_l)}  (单遍扫描)"""
    n = len(times)
    R = np.zeros((n, K))
    state = np.zeros(K)  # 当前各源维的衰减和
    last_t = times[0]
    for i in range(n):
        dt = times[i] - last_t
        if dt > 0:
            decay = np.exp(-beta * dt)
            for j in range(K):
                state[j] *= decay
            last_t = times[i]
        R[i] = state
        state[dims[i]] += beta
    return R


@njit(cache=True)
def _tail_integral(times, dims, K, beta, T1):
    """G[j] = Σ_{l: d_l=j} (1 - e^{-β(T1-t_l)}) —— 激发核到窗尾的积分"""
    G = np.zeros(K)
    for l in range(len(times)):
        G[dims[l]] += 1.0 - np.exp(-beta * (T1 - times[l]))
    return G


def _fit_dim(args):
    """单目标维 k 的凸 MLE: 参数 (μ_k1..μ_kB, α_{1k}..α_{Kk}),μ 分段常数"""
    R_k, G, bin_idx_k, durs, n_k, K, l2 = args
    B = len(durs)

    def negll(p):
        mu_b, a = p[:B], p[B:]
        lam = mu_b[bin_idx_k] + R_k @ a
        lam = np.maximum(lam, 1e-12)
        ll = np.log(lam).sum() - mu_b @ durs - a @ G - l2 * (a @ a)
        inv = 1.0 / lam
        g_mu = np.bincount(bin_idx_k, weights=inv, minlength=B) - durs
        g_a = R_k.T @ inv - G - 2 * l2 * a
        return -ll, -np.concatenate((g_mu, g_a))

    counts_b = np.bincount(bin_idx_k, minlength=B).astype(float)
    x0 = np.concatenate((counts_b / np.maximum(durs, 1.0) * 0.5 + 1e-9, np.full(K, 0.01)))
    bounds = [(1e-10, None)] * B + [(0.0, 10.0)] * K
    res = minimize(negll, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500})
    return res.x, -res.fun


def fit_hawkes(times, dims, K, T0, T1, betas=None, l2=1e-3, val_frac=0.0, mu_bins=6):
    """times: 升序秒级 float64;dims: int64 [0,K);窗口 [T0,T1]。
    μ 为分段常数(mu_bins 段),对付事件的非平稳背景;固定 β 下每维仍是凸问题。
    val_frac>0 时,最后一段时间作验证(不参与拟合,报告验证 LL/事件)。"""
    times = np.asarray(times, dtype=np.float64)
    dims = np.asarray(dims, dtype=np.int64)
    if betas is None:
        betas = [1 / 120, 1 / 300, 1 / 600, 1 / 1800, 1 / 7200, 1 / 21600, 1 / 86400]  # 2min..1d

    t_split = T1 if val_frac <= 0 else T0 + (T1 - T0) * (1 - val_frac)
    tr = times <= t_split
    times_tr, dims_tr = times[tr], dims[tr]
    Tspan = t_split - T0

    B = mu_bins
    edges = np.linspace(T0, t_split, B + 1)
    durs = np.diff(edges)
    bin_idx = np.clip(np.searchsorted(edges, times_tr, side="right") - 1, 0, B - 1)

    n_by_dim = np.bincount(dims_tr, minlength=K)
    best = None
    for beta in betas:
        R = _precompute_R(times_tr, dims_tr, K, beta)
        G = _tail_integral(times_tr, dims_tr, K, beta, t_split)
        mu = np.zeros((K, B))
        A = np.zeros((K, K))  # A[j,k]
        ll = 0.0
        for k in range(K):
            mask = dims_tr == k
            if mask.sum() == 0:
                mu[k, :] = 1e-10
                continue
            p, llk = _fit_dim((R[mask], G, bin_idx[mask], durs, int(mask.sum()), K, l2))
            mu[k, :], A[:, k] = p[:B], p[B:]
            ll += llk
        if best is None or ll > best["ll"]:
            best = {"beta": beta, "mu": mu, "A": A, "ll": ll}

    # 基线1:齐次 Poisson;基线2:同样分段的 Poisson(结构增益的公平对照)
    ll_pois = 0.0
    ll_pois_pw = 0.0
    for k in range(K):
        if n_by_dim[k] > 0:
            rate = n_by_dim[k] / Tspan
            ll_pois += n_by_dim[k] * np.log(rate) - rate * Tspan
        cb = np.bincount(bin_idx[dims_tr == k], minlength=B).astype(float)
        pos = cb > 0
        ll_pois_pw += float((cb[pos] * np.log(cb[pos] / durs[pos])).sum() - cb.sum())

    out = {
        "beta": best["beta"], "mu": best["mu"], "A": best["A"],
        "mu_edges": edges, "ll_train": best["ll"], "ll_poisson": ll_pois,
        "ll_poisson_piecewise": ll_pois_pw,
        "n_train": int(len(times_tr)), "n_by_dim": n_by_dim.tolist(),
        "ll_gain_per_event": float((best["ll"] - ll_pois) / max(1, len(times_tr))),
        "ll_gain_vs_piecewise": float((best["ll"] - ll_pois_pw) / max(1, len(times_tr))),
        "branching_max_row_sum": float(best["A"].sum(axis=1).max()),
    }

    # 验证窗 LL(用训练好的参数;背景用最后一段的 μ 外推)
    if val_frac > 0:
        va = times > t_split
        out["n_val"] = int(va.sum())
        if va.sum() > 0:
            beta, mu, A = best["beta"], best["mu"], best["A"]
            mu_last = mu[:, -1]
            R_all = _precompute_R(times, dims, K, beta)
            lam_val = mu_last[dims[va]] + np.einsum("ij,ij->i", R_all[va], A[:, dims[va]].T)
            lam_val = np.maximum(lam_val, 1e-12)
            G_val = _tail_integral(times, dims, K, beta, T1) - _tail_integral(times_tr, dims_tr, K, beta, t_split)
            integral = mu_last.sum() * (T1 - t_split) + (A.sum(axis=1) * G_val).sum()
            out["ll_val"] = float(np.log(lam_val).sum() - integral)
            rate = n_by_dim.sum() / Tspan
            out["ll_val_poisson"] = float(va.sum() * np.log(max(rate, 1e-12)) - rate * (T1 - t_split))
    return out


if __name__ == "__main__":
    # 自测:模拟一个 2 维 Hawkes,检查参数还原
    rng = np.random.default_rng(0)
    beta_true, mu_true = 1 / 100, np.array([0.02, 0.01])
    A_true = np.array([[0.3, 0.2], [0.0, 0.4]])
    T = 200000.0
    events = []
    state = np.zeros(2)
    t, last = 0.0, 0.0
    while t < T:
        lam_bar = mu_true.sum() + state.sum() * 1.05 + 1e-9
        t += rng.exponential(1 / lam_bar)
        state *= np.exp(-beta_true * (t - last)); last = t
        lam = mu_true + A_true.T @ state
        if rng.random() < lam.sum() / lam_bar:
            k = rng.choice(2, p=lam / lam.sum())
            events.append((t, k))
            state[k] += beta_true
    times = np.array([e[0] for e in events]); dims = np.array([e[1] for e in events])
    print(f"simulated {len(events)} events")
    res = fit_hawkes(times, dims, 2, 0.0, T, betas=[1/50, 1/100, 1/200], val_frac=0.2, mu_bins=4)
    print("beta:", res["beta"], "(true 0.01)")
    print("mu (bins mean):", res["mu"].mean(axis=1).round(4), "(true", mu_true, ")")
    print("A:\n", res["A"].round(3), "\n(true\n", A_true, ")")
    print("ll gain/event:", round(res["ll_gain_per_event"], 4),
          "vs piecewise-poisson:", round(res["ll_gain_vs_piecewise"], 4),
          "val ll:", res.get("ll_val"))
