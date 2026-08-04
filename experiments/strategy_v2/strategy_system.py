"""策略体系构建(P0-3):类型 x 通道 x 情绪 的打法表 + 风险侧时机。

三块:
1. 类型池化:312 事件的闭式 24h Δ(含种子)按 [圈层类型 x 注入通道] 池化中位数,
   溢出构成同池化 -> "打法表"。疑似组织化圈(26/2)单列为监测对象。
2. 情绪条件:按事件主导情绪分组池化(同 analyze_content_modulation 口径,
   dominant = 非中性里占比最高且 >0.15,否则中性主导)-> 情绪 x 通道 Δ。
3. 风险侧时机:对代表性事件,在多个锚点状态(低潮/上升/峰值,按历史 1h 强度分位)
   注入 m=20,MC 模拟失控概率 P(24h 总量 > 1.5 x base_q95) -> 时机决定风险而非收益。
用法: python3 strategy_system.py [--risk-events 12] [--n-runs 200]
输出: ~/data/strategy_v2/{playbook.json, risk_timing.json}
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

CIRCLE_TYPE = {}
for _c in (3, 5, 11, 0, 14, 1, 10):
    CIRCLE_TYPE[_c] = "hub"
for _c in (18, 12):
    CIRCLE_TYPE[_c] = "org_matrix"
CIRCLE_TYPE[9] = "media_kol"
for _c in (28, 20):
    CIRCLE_TYPE[_c] = "interest"
for _c in (30, 41, 8):
    CIRCLE_TYPE[_c] = "regional"
for _c in (26, 2):
    CIRCLE_TYPE[_c] = "organized_risk"


def dominant_sentiment(row):
    sents = {"愤怒": row["sent_愤怒"], "悲伤": row["sent_悲伤"], "恐惧": row["sent_恐惧"],
             "惊奇": row["sent_惊奇"], "喜悦": row["sent_喜悦"]}
    top = max(sents, key=sents.get)
    return top if sents[top] > 0.15 else "中性主导"


def build_playbook(cf):
    import polars as pl
    ef = pl.read_parquet(DATA / "event_features.parquet")
    dom = {f"{r['category']}__{r['topic']}": dominant_sentiment(r)
           for r in ef.iter_rows(named=True)}

    # 池化容器: (type_or_agg_channel) -> list of (delta, spill, other_share, rho)
    rows = []
    for e in cf["events"]:
        if "h24_total" not in e:
            continue
        name = e["name"]
        cat = name.split("__")[0]
        if cat == "major":
            continue  # 大议题窗口跨年,Δ口径与事件不可比,剔除(同 P0-3)
        tot = np.array(e["h24_total"])
        spill = np.array(e["h24_spill_share"])
        oth = np.array(e["h24_other_share"])
        dimc = {int(k): int(v) for k, v in e["dim_circles"].items()}
        for d in range(e["K"]):
            if d in dimc:
                ch_type = CIRCLE_TYPE.get(dimc[d], "other_small")
                ch_kind = f"circle_{ch_type}"
            elif d == e["bigv_dim"]:
                ch_kind = "bigv"
            elif d == e["org_dim"]:
                ch_kind = "org"
            else:
                ch_kind = "other"
            rows.append({"event": name, "channel": ch_kind,
                         "circle": dimc.get(d), "rho": e["rho"],
                         "delta24": float(tot[d]),      # 每注入1帖的期望总计数(含种子)
                         "spill": float(spill[d]), "other_share": float(oth[d]),
                         "sent": dom.get(name, "?"),
                         "amplif": e.get("amplification")})
    df = pl.DataFrame(rows)

    def pool(g):
        return {"n": g.height,
                "delta24_med": round(float(g["delta24"].median()), 2),
                "delta24_p25": round(float(g["delta24"].quantile(0.25)), 2),
                "delta24_p75": round(float(g["delta24"].quantile(0.75)), 2),
                "spill_med": round(float(g["spill"].median()), 3),
                "other_share_med": round(float(g["other_share"].median()), 3)}

    playbook = {"by_channel": {}, "by_channel_sent": {}, "n_events": df["event"].n_unique()}
    for (ch,), g in df.group_by(["channel"]):
        playbook["by_channel"][ch] = pool(g)
    for (ch, s), g in df.group_by(["channel", "sent"]):
        if g.height >= 8:
            playbook["by_channel_sent"][f"{ch}|{s}"] = pool(g)
    return playbook, df


def risk_timing(names, n_runs, m=20):
    """低潮/上升/峰值锚点的失控概率对比。锚点由 1h 滚动计数分位挑选。"""
    from simulate import params_from_fit, simulate_counterfactual
    out = []
    for name in names:
        fit = json.loads((DATA / "hawkes" / f"{name}.json").read_text())
        ev = np.load(DATA / "hawkes" / "events" / f"{name}.npz")
        times, dims = ev["times"].astype(float), ev["dims"]
        P = params_from_fit(fit)
        T0f, T1f = fit["T0"], fit["T1"]
        # 1h 计数曲线
        hrs = ((times - T0f) // 3600).astype(int)
        curve = np.bincount(hrs)
        span_h = len(curve)
        lo_b, hi_b = int(span_h * 0.1), int(span_h * 0.9)
        seg = curve[lo_b:hi_b]
        if len(seg) < 30:
            continue
        anchors = {}
        # 低潮:分位 <=20%;上升:峰前且 60-90 分位;峰值:全局峰
        peak_rel = int(np.argmax(seg))
        q20, q60 = np.quantile(seg, [0.2, 0.6])
        lows = np.where(seg <= q20)[0]
        rises = [i for i in np.where((seg >= q60) & (seg <= np.quantile(seg, 0.9)))[0]
                 if i < peak_rel]
        if len(lows):
            anchors["low"] = int(lows[len(lows) // 2])
        if rises:
            anchors["rising"] = int(rises[-1])
        anchors["peak"] = peak_rel
        d_inj = fit.get("bigv_dim")
        for phase, rel in anchors.items():
            t_anchor = T0f + (lo_b + rel + 0.5) * 3600
            hist = (times[times < t_anchor], dims[times < t_anchor])
            if len(hist[0]) < 50:
                continue
            t0 = time.time()
            res = simulate_counterfactual(P, hist, t_anchor, 24 * 3600.0,
                                          inject=(t_anchor, d_inj, m),
                                          n_runs=n_runs, max_events=500_000)
            base_med = float(np.median(res["base_totals"]))
            out.append({
                "name": name, "phase": phase, "anchor_hour_count": int(seg[rel]),
                "base_median_24h": base_med,
                "delta_total": round(res["delta_total"], 1),
                "runaway_prob_inj": round(res["runaway_prob"], 3),
                "runaway_prob_base": round(
                    float((res["base_totals"] > res["base_q95_total"] * 1.5).mean()), 3),
                "cap_hit": res["base_cap_hit"], "secs": round(time.time() - t0, 1)})
            print(out[-1], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk-events", type=int, default=12)
    ap.add_argument("--n-runs", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cf = json.loads((OUT / "closed_form_all.json").read_text())

    playbook, df = build_playbook(cf)
    (OUT / "playbook.json").write_text(json.dumps(playbook, ensure_ascii=False, indent=1))
    df.write_parquet(OUT / "playbook_rows.parquet")
    print(json.dumps(playbook["by_channel"], ensure_ascii=False, indent=1), flush=True)

    # 风险时机:验证LL增益最高的非major事件,分 rho 档覆盖
    evs = [e for e in cf["events"] if not e["name"].startswith("major") and "h24_total" in e]
    evs.sort(key=lambda e: e["rho"])
    idx = np.linspace(0, len(evs) - 1, args.risk_events).astype(int)
    names = [evs[i]["name"] for i in idx]
    print("risk timing on:", names, flush=True)
    rt = risk_timing(names, args.n_runs)
    (OUT / "risk_timing.json").write_text(json.dumps(rt, ensure_ascii=False))
    print("STRATEGY SYSTEM DONE", flush=True)


if __name__ == "__main__":
    main()
