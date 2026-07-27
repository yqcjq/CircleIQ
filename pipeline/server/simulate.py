"""Hawkes 反事实模拟器(Ogata thinning,numba 多线程)。

能力:
- 给定拟合参数 (μ, A, β) 与历史事件,向前模拟 [t_start, t_end] 的传播轨迹
- What-if: 在 t_inj 向维度 d_inj 注入 m 条事件(策略=让某圈层发声),对比 base/injected 的期望增量
- 风险: 失控概率 P(总量 > q95_base × 1.5)、各圈层增量分布

用法(库): from simulate import simulate_counterfactual
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _simulate_one(mu2d, edges, A, beta, state0, t_start, t_end, inject_t, inject_dim, inject_m, seed, max_events=2_000_000):
    """单次模拟(分段常数 μ)。mu2d: [K,B];edges: [B+1](最后一段向未来外推)。
    state0: 各源维在 t_start 时刻的衰减和。两事件间 λ 单调衰减(μ 段内不变),
    候选点越过段边界时推进到边界重算上界,保证 thinning 无偏。"""
    np.random.seed(seed)
    K = mu2d.shape[0]
    B = mu2d.shape[1]
    state = state0.copy()
    t = t_start
    out_t = np.empty(max_events)
    out_d = np.empty(max_events, dtype=np.int64)
    n = 0
    inj_done = inject_t < t_start
    if inj_done and inject_m > 0:
        state[inject_dim] += beta * inject_m * np.exp(-beta * (t_start - inject_t))
    while t < t_end and n < max_events:
        if not inj_done and t >= inject_t:
            state[inject_dim] += beta * inject_m
            for _ in range(inject_m):
                if n < max_events:
                    out_t[n] = inject_t
                    out_d[n] = inject_dim
                    n += 1
            inj_done = True
        # 当前段
        b = B - 1
        for bi in range(B):
            if t < edges[bi + 1]:
                b = bi
                break
        lam_k = mu2d[:, b] + A.T @ state
        lam_bar = lam_k.sum() * 1.02 + 1e-12
        dt = np.random.exponential(1.0 / lam_bar)
        t_next = t + dt
        if not inj_done and t < inject_t <= t_next:
            state *= np.exp(-beta * (inject_t - t))
            t = inject_t
            continue
        if b < B - 1 and t_next > edges[b + 1]:
            state *= np.exp(-beta * (edges[b + 1] - t))
            t = edges[b + 1]
            continue
        t = t_next
        if t >= t_end:
            break
        state *= np.exp(-beta * dt)
        lam_k2 = mu2d[:, b] + A.T @ state
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
            state[k] += beta
    return out_t[:n], out_d[:n]


@njit(parallel=True, cache=True)
def _simulate_many(mu2d, edges, A, beta, state0, t_start, t_end, inject_t, inject_dim, inject_m, n_runs, seed0, max_events=2_000_000):
    """并行 n_runs 次,返回每 run 各维事件数 [n_runs, K]"""
    K = mu2d.shape[0]
    counts = np.zeros((n_runs, K), dtype=np.int64)
    for r in prange(n_runs):
        ts, ds = _simulate_one(mu2d, edges, A, beta, state0, t_start, t_end,
                               inject_t, inject_dim, inject_m, seed0 + r, max_events)
        for i in range(len(ds)):
            counts[r, ds[i]] += 1
    return counts


def as_mu2d(params):
    """params["mu"] 兼容常数向量与 [K,B] 分段矩阵,返回 (mu2d, edges)。"""
    mu = np.asarray(params["mu"], dtype=np.float64)
    if mu.ndim == 1:
        return mu[:, None], np.array([-1e18, 1e18])
    edges = np.asarray(params["mu_edges"], dtype=np.float64)
    return mu, edges


def history_state(times, dims, K, beta, t_now):
    """历史事件在 t_now 的衰减和"""
    state = np.zeros(K)
    for t, d in zip(times, dims):
        if t < t_now:
            state[d] += beta * np.exp(-beta * (t_now - t))
    return state


def simulate_counterfactual(params, history, t_start, horizon_s, inject=None, n_runs=200, seed=42,
                            max_events=200_000):
    """params: dict(mu[, mu_edges], A, beta);history: (times, dims) ndarray;
    inject: None 或 (t_inj, dim, m)。返回 base/injected 的各维事件数分布统计。
    max_events: 单次模拟事件数上限(超临界分支比时防爆,命中上限的 run 本身就是失控信号)。"""
    mu2d, edges = as_mu2d(params)
    A = np.asarray(params["A"], dtype=np.float64)
    beta = float(params["beta"])
    K = mu2d.shape[0]
    ht, hd = history
    state0 = history_state(np.asarray(ht, float), np.asarray(hd, np.int64), K, beta, t_start)
    t_end = t_start + horizon_s

    base = _simulate_many(mu2d, edges, A, beta, state0, t_start, t_end, -1.0, 0, 0, n_runs, seed, max_events)
    out = {"base_mean": base.mean(0), "base_q95_total": float(np.quantile(base.sum(1), 0.95)),
           "base_cap_hit": float((base.sum(1) >= max_events * 0.98).mean())}
    if inject is not None:
        t_inj, d_inj, m = inject
        inj = _simulate_many(mu2d, edges, A, beta, state0, t_start, t_end, float(t_inj), int(d_inj), int(m),
                             n_runs, seed + 10_000, max_events)
        out["inj_mean"] = inj.mean(0)
        out["delta_mean"] = out["inj_mean"] - out["base_mean"]
        out["delta_total"] = float(out["delta_mean"].sum())
        thresh = out["base_q95_total"] * 1.5
        out["runaway_prob"] = float((inj.sum(1) > thresh).mean())
        out["inj_totals"] = inj.sum(1)
        out["base_totals"] = base.sum(1)
    return out
