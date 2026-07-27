"""Hawkes 反事实模拟器(双时间尺度核 + 分段μ,Ogata thinning,numba 多线程)。

能力:
- 给定拟合参数 (μ[K,B], edges, A1/β1 快核, A2/β2 慢核) 与历史事件,前向模拟传播轨迹
- What-if: 在 t_inj 向维度 d_inj 注入 m 条事件,对比 base/injected 的期望增量与风险

用法(库): from simulate import simulate_counterfactual, history_state, params_from_fit
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _simulate_one(mu2d, edges, A1, b1, A2, b2, s1_0, s2_0,
                  t_start, t_end, inject_t, inject_dim, inject_m, seed, max_events=2_000_000):
    """单次模拟。s1_0/s2_0: 两个核在 t_start 的衰减和。μ 段内常数,
    候选点越过段边界/注入点时推进重算上界(μ 可上跳),thinning 无偏。"""
    np.random.seed(seed)
    K = mu2d.shape[0]
    B = mu2d.shape[1]
    s1 = s1_0.copy()
    s2 = s2_0.copy()
    t = t_start
    out_t = np.empty(max_events)
    out_d = np.empty(max_events, dtype=np.int64)
    n = 0
    inj_done = inject_t < t_start
    if inj_done and inject_m > 0:
        s1[inject_dim] += b1 * inject_m * np.exp(-b1 * (t_start - inject_t))
        s2[inject_dim] += b2 * inject_m * np.exp(-b2 * (t_start - inject_t))
    while t < t_end and n < max_events:
        if not inj_done and t >= inject_t:
            s1[inject_dim] += b1 * inject_m
            s2[inject_dim] += b2 * inject_m
            for _ in range(inject_m):
                if n < max_events:
                    out_t[n] = inject_t
                    out_d[n] = inject_dim
                    n += 1
            inj_done = True
        b = B - 1
        for bi in range(B):
            if t < edges[bi + 1]:
                b = bi
                break
        lam_k = mu2d[:, b] + A1.T @ s1 + A2.T @ s2
        lam_bar = lam_k.sum() * 1.02 + 1e-12
        dt = np.random.exponential(1.0 / lam_bar)
        t_next = t + dt
        if not inj_done and t < inject_t <= t_next:
            s1 *= np.exp(-b1 * (inject_t - t))
            s2 *= np.exp(-b2 * (inject_t - t))
            t = inject_t
            continue
        if b < B - 1 and t_next > edges[b + 1]:
            s1 *= np.exp(-b1 * (edges[b + 1] - t))
            s2 *= np.exp(-b2 * (edges[b + 1] - t))
            t = edges[b + 1]
            continue
        t = t_next
        if t >= t_end:
            break
        s1 *= np.exp(-b1 * dt)
        s2 *= np.exp(-b2 * dt)
        lam_k2 = mu2d[:, b] + A1.T @ s1 + A2.T @ s2
        lam_tot2 = lam_k2.sum()
        if np.random.random() < lam_tot2 / lam_bar:
            u = np.random.random() * lam_tot2
            acc = 0.0
            k = K - 1
            for j in range(K):
                acc += lam_k2[j]
                if u <= acc:
                    k = j
                    break
            out_t[n] = t
            out_d[n] = k
            n += 1
            s1[k] += b1
            s2[k] += b2
    return out_t[:n], out_d[:n]


@njit(parallel=True, cache=True)
def _simulate_many(mu2d, edges, A1, b1, A2, b2, s1_0, s2_0,
                   t_start, t_end, inject_t, inject_dim, inject_m, n_runs, seed0, max_events=2_000_000):
    """并行 n_runs 次,返回每 run 各维事件数 [n_runs, K]"""
    K = mu2d.shape[0]
    counts = np.zeros((n_runs, K), dtype=np.int64)
    for r in prange(n_runs):
        ts, ds = _simulate_one(mu2d, edges, A1, b1, A2, b2, s1_0, s2_0, t_start, t_end,
                               inject_t, inject_dim, inject_m, seed0 + r, max_events)
        for i in range(len(ds)):
            counts[r, ds[i]] += 1
    return counts


def history_state(times, dims, K, beta, t_now):
    """历史事件在 t_now 的衰减和(单核)"""
    state = np.zeros(K)
    for t, d in zip(times, dims):
        if t < t_now:
            state[d] += beta * np.exp(-beta * (t_now - t))
    return state


def params_from_fit(fit):
    """从 run_hawkes 的 JSON 构造模拟参数(兼容旧单核格式)。"""
    mu = np.asarray(fit["mu"], dtype=np.float64)
    if mu.ndim == 1:
        mu = mu[:, None]
        edges = np.array([-1e18, 1e18])
    else:
        edges = np.asarray(fit["mu_edges"], dtype=np.float64)
    if "A1" in fit:
        return {"mu2d": mu, "edges": edges,
                "A1": np.asarray(fit["A1"], float), "beta1": float(fit["beta1"]),
                "A2": np.asarray(fit["A2"], float), "beta2": float(fit["beta2"])}
    K = mu.shape[0]
    return {"mu2d": mu, "edges": edges,
            "A1": np.asarray(fit["A"], float), "beta1": float(fit["beta"]),
            "A2": np.zeros((K, K)), "beta2": 1.0 / 3600}


def simulate_counterfactual(params, history, t_start, horizon_s, inject=None, n_runs=200, seed=42,
                            max_events=200_000):
    """params: params_from_fit 的输出;history: (times, dims) ndarray;
    inject: None 或 (t_inj, dim, m)。返回 base/injected 的各维事件数分布统计。"""
    mu2d, edges = params["mu2d"], params["edges"]
    A1, b1, A2, b2 = params["A1"], params["beta1"], params["A2"], params["beta2"]
    K = mu2d.shape[0]
    ht, hd = history
    ht = np.asarray(ht, float)
    hd = np.asarray(hd, np.int64)
    s1_0 = history_state(ht, hd, K, b1, t_start)
    s2_0 = history_state(ht, hd, K, b2, t_start)
    t_end = t_start + horizon_s

    base = _simulate_many(mu2d, edges, A1, b1, A2, b2, s1_0, s2_0,
                          t_start, t_end, -1.0, 0, 0, n_runs, seed, max_events)
    out = {"base_mean": base.mean(0), "base_median_total": float(np.median(base.sum(1))),
           "base_q95_total": float(np.quantile(base.sum(1), 0.95)),
           "base_cap_hit": float((base.sum(1) >= max_events * 0.98).mean())}
    if inject is not None:
        t_inj, d_inj, m = inject
        inj = _simulate_many(mu2d, edges, A1, b1, A2, b2, s1_0, s2_0,
                             t_start, t_end, float(t_inj), int(d_inj), int(m),
                             n_runs, seed + 10_000, max_events)
        out["inj_mean"] = inj.mean(0)
        out["delta_mean"] = out["inj_mean"] - out["base_mean"]
        out["delta_total"] = float(out["delta_mean"].sum())
        out["delta_median_total"] = float(np.median(inj.sum(1)) - np.median(base.sum(1)))
        thresh = out["base_q95_total"] * 1.5
        out["runaway_prob"] = float((inj.sum(1) > thresh).mean())
        out["inj_totals"] = inj.sum(1)
        out["base_totals"] = base.sum(1)
    return out
