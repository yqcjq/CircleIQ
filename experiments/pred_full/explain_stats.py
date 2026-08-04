"""解释性文档(33/34/35-2-explain)的补充统计:在服务器上一次算完,输出 JSON 供本地做图。

产出 /root/data/explain_stats.json:
  task_profile     : 一个代表性(事件,圈)对的完整轨迹+锚点(讲清"任务是什么")
  zero_inflation   : y24 分布直方(讲清零膨胀)
  gate_examples    : gru_gated 门控前后对比的典型锚点(讲清门控为什么有效)
  pred_cases       : 2 个 test 事件 × 大圈的 naive/GRU/gated 预测轨迹(直观对比)
  branching_intuit : 一个代表性事件的 A 矩阵 + 每通道注入的诱发链分解(讲清闭式)
  induced_flow     : 打法表通道 -> 诱发去向的构成(桑基图数据)
  sent_variance    : 情绪构成方差分解:事件间 vs 事件内时间(讲清为什么事件基线赢)
  natexp_case      : 一次典型大V进场的前后窗口轨迹(讲清 DiD)
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path.home() / "data"
PF = Path.home() / "circleiq" / "pred_full" / "out"
sys.path.insert(0, str(DATA))

out = {}
names = {f"{r['category']}__{r['topic']}": r["event_name"]
         for r in pl.read_parquet(DATA / "catalog.parquet").iter_rows(named=True)}
out["event_names"] = names

# ---- 1. 任务剖面:找一个 test 事件、圈3,截 400h 轨迹 + 锚点 ----
an = pl.read_parquet(PF / "anchors.parquet")
panel = json.loads((PF / "panel.json").read_text())
cand = (an.filter((pl.col("split") == "test") & (pl.col("circle") == 3))
          .group_by("event").agg(pl.col("y24").max(), pl.len())
          .filter(pl.col("len") >= 10).sort("y24", descending=True))
ev_name = cand["event"][1]  # 第二大,避免极端
z = np.load(PF / "events" / f"{ev_name}.npz")
counts = z["counts"]
g = an.filter((pl.col("event") == ev_name) & (pl.col("circle") == 3))
ks = g["k"].to_list()
c3 = counts[3]  # cidx: 圈按大小排,circle 3 是最大圈 -> cidx 0? 需查
# cidx 对齐:panel pairs 里有 circle 与 cidx
cidx_of = {p["circle"]: p["cidx"] for p in panel["pairs"] if p["event"] == ev_name}
tgt = counts[cidx_of[3]]
hi = min(len(tgt), (max(ks) + 48))
out["task_profile"] = {
    "event": ev_name, "event_name": names.get(ev_name, ev_name),
    "circle": 3, "hourly": tgt[:hi].tolist(),
    "total_hourly": counts.sum(0)[:hi].tolist(),
    "anchors": [int(k) for k in ks if k < hi],
    "y24": {int(r["k"]): int(r["y24"]) for r in g.iter_rows(named=True) if r["k"] < hi}}

# ---- 2. 零膨胀:test y24 直方 ----
te = an.filter(pl.col("split") == "test")
y = te["y24"].to_numpy()
bins = [0, 1, 10, 100, 1000, 10 ** 9]
labels = ["0", "1-9", "10-99", "100-999", ">=1000"]
hist = [int(((y >= bins[i]) & (y < bins[i + 1])).sum()) for i in range(5)]
out["zero_inflation"] = {"labels": labels, "counts": hist, "n": int(len(y)),
                         "nonzero_med": float(np.median(y[y > 0])),
                         "max": int(y.max())}

# ---- 3/4. 预测案例轨迹:2 个 test 事件,naive/gru/gru_gated 的 h24 预测 vs 真实 ----
preds = {m: pl.read_parquet(PF / "preds" / f"{m}.parquet")
           .filter((pl.col("horizon") == "h24") & (pl.col("split") == "test"))
         for m in ("naive", "gru", "gru_gated")}
cases = []
for evn, circ in ((ev_name, 3), (cand["event"][0], 3)):
    rows = {}
    for m, df in preds.items():
        sub = (df.filter((pl.col("event") == evn) & (pl.col("circle") == circ))
                 .sort("k"))
        rows[m] = {int(k): float(v) for k, v in zip(sub["k"], sub["yhat"])}
        ys = {int(k): float(v) for k, v in zip(sub["k"], sub["y"])}
    if rows["naive"]:
        cases.append({"event": evn, "event_name": names.get(evn, evn),
                      "circle": circ, "y": ys, **{m: rows[m] for m in rows}})
out["pred_cases"] = cases

# ---- 5. 分支直觉 + 诱发构成(闭式解释) ----
from closed_form import finite_horizon_counts
cf_all = json.loads((DATA / "strategy_v2" / "closed_form_all.json").read_text())
# 选一个 rho 中位、K=10 的事件
evs = sorted((e for e in cf_all["events"] if e["K"] >= 9 and "h24_total" in e),
             key=lambda e: abs(e["rho"] - 0.96))
e0 = evs[0]
fit = json.loads((DATA / "hawkes" / f"{e0['name']}.json").read_text())
A = np.array(fit["A"])
out["branching_intuit"] = {
    "event": e0["name"], "event_name": names.get(e0["name"], e0["name"]),
    "labels": e0["labels"], "rho": e0["rho"],
    "A": np.round(A, 3).tolist(),
    "h24_total": e0["h24_total"]}
# 诱发几何级数分解:注入 bigv,第 1/2/3+ 代的期望量
d = fit["bigv_dim"]
K = fit["D"]
e_d = np.zeros(K); e_d[d] = 1
gen1 = A.T @ e_d
gen2 = A.T @ gen1
n24 = finite_horizon_counts(np.array(fit["A1"]), fit["beta1"],
                            np.array(fit["A2"]), fit["beta2"], 24 * 3600.0)
total = n24[d] - e_d
out["branching_intuit"]["generations"] = {
    "gen1": np.round(gen1, 3).tolist(),
    "gen2": np.round(gen2, 3).tolist(),
    "total_induced": np.round(total, 3).tolist(),
    "gen1_sum": round(float(gen1.sum()), 2),
    "gen2_sum": round(float(gen2.sum()), 2),
    "total_sum": round(float(total.sum()), 2)}

# ---- 6. 诱发去向(池化,通道->目标组构成) ----
flow = {}
CIRCLE_TYPE = {3: "hub", 5: "hub", 11: "hub", 0: "hub", 14: "hub", 1: "hub", 10: "hub",
               18: "org_matrix", 12: "org_matrix", 9: "media_kol",
               28: "interest", 20: "interest", 30: "regional", 41: "regional",
               8: "regional", 26: "organized_risk", 2: "organized_risk"}
for e in cf_all["events"]:
    if "h24_comp" not in e or e["name"].startswith("major"):
        continue
    K = len(e["labels"])
    dimc = {int(k): v for k, v in e["dim_circles"].items()}
    comp = np.array(e["h24_comp"])  # [d, j] 含种子
    comp = comp - np.eye(K)

    def kind(j):
        if j in dimc:
            return "圈_" + CIRCLE_TYPE.get(dimc[j], "other")
        if j == e["bigv_dim"]:
            return "大V"
        if j == e["org_dim"]:
            return "机构"
        return "散户"
    for d in range(K):
        src = kind(d)
        tot = comp[d].sum()
        if tot <= 0:
            continue
        tgt_ = {}
        for j in range(K):
            tgt_[kind(j)] = tgt_.get(kind(j), 0.0) + float(comp[d, j])
        f = flow.setdefault(src, {})
        for kk, vv in tgt_.items():
            f.setdefault(kk, []).append(vv / tot)
out["induced_flow"] = {src: {k: round(float(np.median(v)), 4) for k, v in tg.items()}
                       for src, tg in flow.items()}

# ---- 7. 情绪方差分解:L1(事件间) vs L1(事件内时间) ----
rng = np.random.default_rng(7)
ev_shares, within = [], []
for evn in list(panel["events"])[:200]:
    z = np.load(PF / "events" / f"{evn}.npz")
    s = z["scnt"].sum(0).astype(float)  # [6, nb]
    tot = s.sum(1)
    if tot.sum() < 500:
        continue
    share_ev = tot / tot.sum()
    ev_shares.append(share_ev)
    # 事件内:随机取 20 个 24h 窗的构成 与全事件构成的 L1
    nb = s.shape[1]
    if nb < 72:
        continue
    cums = np.concatenate([np.zeros((6, 1)), np.cumsum(s, 1)], 1)
    for k in rng.choice(np.arange(24, nb - 24), size=min(20, nb - 48), replace=False):
        w = cums[:, k + 24] - cums[:, k]
        if w.sum() >= 20:
            within.append(float(np.abs(w / w.sum() - share_ev).sum()))
ev_shares = np.array(ev_shares)
grand = ev_shares.mean(0)
between = [float(np.abs(s - grand).sum()) for s in ev_shares]
out["sent_variance"] = {
    "between_med": round(float(np.median(between)), 3),
    "within_med": round(float(np.median(within)), 3),
    "n_events": len(ev_shares), "n_windows": len(within),
    "grand_share": np.round(grand, 4).tolist(),
    "between_all": np.round(between, 3).tolist()[:200],
    "within_sample": np.round(rng.choice(within, 200), 3).tolist()}

# ---- 8. 自然实验案例:挑一个匹配质量好、DiD 大的进场 ----
ne = json.loads((DATA / "strategy_v2" / "natural_experiment.json").read_text())
good = [r for r in ne["rows"] if r["match_gap_log"] < 0.1 and r["did"] > 500]
good.sort(key=lambda r: -r["followers"])
r0 = good[0]
cat_, topic_ = r0["name"].split("__", 1)
files = sorted((DATA / "core" / cat_).glob(f"{topic_}__*.parquet"))
df = pl.concat([pl.read_parquet(f, columns=["md5_mid", "ts"]) for f in files])
df = df.filter(pl.col("md5_mid").is_null() | (pl.int_range(pl.len()).over("md5_mid") == 0))
ts_all = np.sort(df["ts"].drop_nulls().to_numpy())
t_e = r0["t"]
hrs = np.arange(-48, 48)
curve = [int(np.searchsorted(ts_all, t_e + (h + 1) * 3600.0)
             - np.searchsorted(ts_all, t_e + h * 3600.0)) for h in hrs]
out["natexp_case"] = {"row": r0, "event_name": names.get(r0["name"], r0["name"]),
                      "hours": hrs.tolist(), "hourly": curve}

(DATA / "explain_stats.json").write_text(json.dumps(out, ensure_ascii=False))
print("keys:", list(out))
print("EXPLAIN STATS DONE")
