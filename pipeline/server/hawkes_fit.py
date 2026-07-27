"""多维 Hawkes 拟合(服务器端,numba + 多进程)。

模型(双时间尺度激发 + 分段常数背景):
  λ_k(t) = μ_k(t) + Σ_j Σ_{t^j_i<t} [α1_{jk} β1 e^{-β1(t-t^j_i)} + α2_{jk} β2 e^{-β2(t-t^j_i)}]
- 快核 β1(分钟级)抓转发爆发微结构;慢核 β2(小时级)抓持续发酵,承载中视界预测信息
- (β1,β2) 走网格;固定核下 LL 对 (μ_bins, α1, α2) 凹且按目标维分离 -> 每维小凸问题
- μ 分段常数(mu_bins 段)对付事件级非平稳背景

用法(被 run_hawkes.py 调用,也可单测):
  from hawkes_fit import fit_hawkes
  res = fit_hawkes(times, dims, K, T0, T1)
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
    """单目标维 k 的凸 MLE: 参数 (μ_k1..μ_kB, α1_{1k}..α1_{Kk}, α2_{1k}..α2_{Kk})"""
    R1_k, R2_k, G1, G2, bin_idx_k, durs, K, l2 = args
    B = len(durs)

    def negll(p):
        mu_b, a1, a2 = p[:B], p[B:B + K], p[B + K:]
        lam = mu_b[bin_idx_k] + R1_k @ a1 + R2_k @ a2
        lam = np.maximum(lam, 1e-12)
        ll = np.log(lam).sum() - mu_b @ durs - a1 @ G1 - a2 @ G2 - l2 * (a1 @ a1 + a2 @ a2)
        inv = 1.0 / lam
        g_mu = np.bincount(bin_idx_k, weights=inv, minlength=B) - durs
        g_a1 = R1_k.T @ inv - G1 - 2 * l2 * a1
        g_a2 = R2_k.T @ inv - G2 - 2 * l2 * a2
        return -ll, -np.concatenate((g_mu, g_a1, g_a2))

    counts_b = np.bincount(bin_idx_k, minlength=B).astype(float)
    x0 = np.concatenate((counts_b / np.maximum(durs, 1.0) * 0.5 + 1e-9,
                         np.full(K, 0.01), np.full(K, 0.01)))
    bounds = [(1e-10, None)] * B + [(0.0, 10.0)] * (2 * K)
    res = minimize(negll, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500})
    return res.x, -res.fun


def fit_hawkes(times, dims, K, T0, T1, beta_pairs=None, l2=1e-3, val_frac=0.0, mu_bins=6):
    """times: 升序秒级 float64;dims: int64 [0,K);窗口 [T0,T1]。
    双时间尺度核 + 分段 μ。返回 A1(快核激发阵)/A2(慢核激发阵)/beta1/beta2/mu[K,B]。"""
    times = np.asarray(times, dtype=np.float64)
    dims = np.asarray(dims, dtype=np.int64)
    if beta_pairs is None:
        fast = [1 / 120, 1 / 600]            # 半衰 1.4 / 6.9 分钟
        slow = [1 / 7200, 1 / 21600, 1 / 86400]  # 半衰 1.4 / 4.2 / 16.6 小时
        beta_pairs = [(f, s) for f in fast for s in slow]

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
    for b1, b2 in beta_pairs:
        R1 = _precompute_R(times_tr, dims_tr, K, b1)
        R2 = _precompute_R(times_tr, dims_tr, K, b2)
        G1 = _tail_integral(times_tr, dims_tr, K, b1, t_split)
        G2 = _tail_integral(times_tr, dims_tr, K, b2, t_split)
        mu = np.zeros((K, B))
        A1 = np.zeros((K, K))
        A2 = np.zeros((K, K))
        ll = 0.0
        for k in range(K):
            mask = dims_tr == k
            if mask.sum() == 0:
                mu[k, :] = 1e-10
                continue
            p, llk = _fit_dim((R1[mask], R2[mask], G1, G2, bin_idx[mask], durs, K, l2))
            mu[k, :], A1[:, k], A2[:, k] = p[:B], p[B:B + K], p[B + K:]
            ll += llk
        if best is None or ll > best["ll"]:
            best = {"beta1": b1, "beta2": b2, "mu": mu, "A1": A1, "A2": A2, "ll": ll}

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

    A_tot = best["A1"] + best["A2"]
    out = {
        "beta1": best["beta1"], "beta2": best["beta2"],
        "mu": best["mu"], "A1": best["A1"], "A2": best["A2"], "A": A_tot,
        "mu_edges": edges, "ll_train": best["ll"], "ll_poisson": ll_pois,
        "ll_poisson_piecewise": ll_pois_pw,
        "n_train": int(len(times_tr)), "n_by_dim": n_by_dim.tolist(),
        "ll_gain_per_event": float((best["ll"] - ll_pois) / max(1, len(times_tr))),
        "ll_gain_vs_piecewise": float((best["ll"] - ll_pois_pw) / max(1, len(times_tr))),
        "branching_max_row_sum": float(A_tot.sum(axis=1).max()),
    }

    # 验证窗 LL(背景用最后一段 μ 外推)
    if val_frac > 0:
        va = times > t_split
        out["n_val"] = int(va.sum())
        if va.sum() > 0:
            b1, b2, mu, A1, A2 = best["beta1"], best["beta2"], best["mu"], best["A1"], best["A2"]
            mu_last = mu[:, -1]
            R1a = _precompute_R(times, dims, K, b1)
            R2a = _precompute_R(times, dims, K, b2)
            lam_val = (mu_last[dims[va]] + np.einsum("ij,ij->i", R1a[va], A1[:, dims[va]].T)
                       + np.einsum("ij,ij->i", R2a[va], A2[:, dims[va]].T))
            lam_val = np.maximum(lam_val, 1e-12)
            G1v = _tail_integral(times, dims, K, b1, T1) - _tail_integral(times_tr, dims_tr, K, b1, t_split)
            G2v = _tail_integral(times, dims, K, b2, T1) - _tail_integral(times_tr, dims_tr, K, b2, t_split)
            integral = mu_last.sum() * (T1 - t_split) + (A1.sum(axis=1) * G1v).sum() + (A2.sum(axis=1) * G2v).sum()
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
    res = fit_hawkes(times, dims, 2, 0.0, T,
                     beta_pairs=[(1/50, 1/500), (1/100, 1/1000), (1/100, 1/2000)],
                     val_frac=0.2, mu_bins=4)
    print("beta1:", res["beta1"], "beta2:", res["beta2"], "(true 单核 0.01)")
    print("mu (bins mean):", res["mu"].mean(axis=1).round(4), "(true", mu_true, ")")
    print("A1+A2:\n", res["A"].round(3), "\n(true\n", A_true, ")")
    print("ll gain/event:", round(res["ll_gain_per_event"], 4),
          "vs piecewise-poisson:", round(res["ll_gain_vs_piecewise"], 4),
          "val ll:", res.get("ll_val"))
