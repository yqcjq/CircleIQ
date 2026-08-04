"""33/34/35 报告图:大规模预测验证 + 策略闭式 + 舆论线。

用法: python make_figs_full.py  (在 pipeline/ 目录,读 experiments/*/results/)
产出: docs/figures/33-*.png, 34-*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle
from figstyle import CAT, DIV_NEG, DIV_POS, INK, INK2, MUTED

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "figures"
PF = ROOT / "experiments" / "pred_full" / "results"
SV = ROOT / "experiments" / "strategy_v2" / "results"

MODEL_LABEL = {
    "gru": "GRU", "gbdt": "GBDT", "naive": "naive", "snaive": "季节naive",
    "ewma": "EWMA", "poisson": "Poisson", "gru_gated": "GRU+门控",
    "gbdt_2s": "GBDT两段式", "gbdt_casc": "GBDT+级联", "gbdt_casc_2s": "GBDT级联+两段",
    "gbdt_pois": "GBDT-Poisson", "gru_wl": "GRU加权",
}
CH_LABEL = {"circle_hub": "枢纽圈", "circle_org_matrix": "机构矩阵圈",
            "circle_media_kol": "媒体/KOL圈", "circle_interest": "兴趣圈",
            "circle_regional": "地域圈", "circle_organized_risk": "疑似组织化圈",
            "circle_other_small": "其他小圈", "bigv": "圈外大V",
            "org": "圈外机构", "other": "圈外散户"}


def fig_male_ladder(r):
    """总榜:h6/h24 双面板横条。"""
    show = ["gru_gated", "gbdt_casc_2s", "gbdt_2s", "gru", "gbdt_casc", "gbdt",
            "naive", "ewma", "snaive"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=False)
    for ax, h in zip(axes, ("h6", "h24")):
        mm = r["metrics"][h]
        ms = [m for m in show if m in mm]
        ms.sort(key=lambda m: -mm[m]["male"])
        vals = [mm[m]["male"] for m in ms]
        cols = [CAT[0] if m not in ("naive", "snaive", "ewma") else figstyle.BASE
                for m in ms]
        bars = ax.barh(range(len(ms)), vals, color=cols, height=0.62)
        ax.set_yticks(range(len(ms)), [MODEL_LABEL[m] for m in ms])
        nv = mm["naive"]["male"]
        ax.axvline(nv, color=INK2, lw=1.0, ls="--")
        ax.text(nv, len(ms) - 0.2, f" naive={nv:.3f}", color=INK2, fontsize=8.5, va="bottom")
        for b, v in zip(bars, vals):
            ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                    va="center", fontsize=8.5, color=INK2)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_title(f"{h} 视界")
        ax.set_xlabel("MALE(log 误差,越低越好)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("大规模验证总榜:66 圈 × 382 事件,test 12,222 锚点", y=1.02)
    fig.savefig(FIGS / "33-full_male.png")
    plt.close(fig)


def fig_strata(r):
    show = ["gru_gated", "gbdt_casc", "gru", "gbdt_2s", "naive"]
    labels = {"0-0": "y=0(60%样本)", "1-9": "1–9", "10-99": "10–99", "100-inf": "≥100(爆发)"}
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    strata = list(labels)
    x = np.arange(len(strata))
    w = 0.15
    for i, m in enumerate(show):
        vals = [r["strata"]["h24"][s][m] for s in strata]
        col = figstyle.BASE if m == "naive" else CAT[i]
        ax.bar(x + (i - 2) * w, vals, w * 0.92, label=MODEL_LABEL[m], color=col)
    ax.set_xticks(x, [labels[s] for s in strata])
    ax.set_ylabel("MALE(h24)")
    ax.set_xlabel("真实 24h 量级分层")
    ax.set_ylim(0, 1.95)
    ax.set_title("分层表现:门控赢在零层,级联特征赢在爆发层")
    ax.legend(ncol=3, loc="upper left", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    fig.savefig(FIGS / "33-full_strata.png")
    plt.close(fig)


def fig_bytype(r):
    show = ["gru_gated", "gru", "gbdt_2s", "naive"]
    tl = {"hub": "枢纽圈群", "org_matrix": "机构矩阵", "media_kol": "媒体/KOL",
          "interest": "兴趣圈", "regional": "地域圈",
          "organized_risk": "疑似组织化", "other_small": "其他小圈"}
    types = [t for t in tl if t in r["by_type"]["h24"]]
    types.sort(key=lambda t: -r["by_type"]["h24"][t]["n"])
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    x = np.arange(len(types))
    w = 0.19
    for i, m in enumerate(show):
        vals = [r["by_type"]["h24"][t][m] for t in types]
        col = figstyle.BASE if m == "naive" else CAT[i]
        ax.bar(x + (i - 1.5) * w, vals, w * 0.9, label=MODEL_LABEL[m], color=col)
    names = [f"{tl[t]}\nn={r['by_type']['h24'][t]['n']}" for t in types]
    ax.set_xticks(x, names, fontsize=8.5)
    ax.set_ylabel("MALE(h24)")
    ax.set_title("圈层类型分组:七类圈层中六类学习模型显著占优")
    ax.legend(ncol=4, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    fig.savefig(FIGS / "33-full_bytype.png")
    plt.close(fig)


def fig_leakage(rm, rp):
    """main vs _pre:核心模型的 MALE 与 vs-naive 边际。"""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    show = ["gru", "gbdt", "naive"]
    for ax, h in zip(axes, ("h6", "h24")):
        x = np.arange(len(show))
        w = 0.32
        v_m = [rm["metrics"][h][m]["male"] for m in show]
        v_p = [rp["metrics"][h][m]["male"] for m in show]
        ax.bar(x - w / 2, v_m, w * 0.92, label="主口径(全窗圈)", color=CAT[0])
        ax.bar(x + w / 2, v_p, w * 0.92, label="防泄漏(_pre 圈)", color=CAT[2])
        for xi, (a, b) in zip(x, zip(v_m, v_p)):
            ax.text(xi - w / 2, a + 0.006, f"{a:.3f}", ha="center", fontsize=8, color=INK2)
            ax.text(xi + w / 2, b + 0.006, f"{b:.3f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x, [MODEL_LABEL[m] for m in show])
        ax.set_title(f"{h}")
        ax.set_ylabel("MALE" if h == "h6" else "")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8.5)
    fig.suptitle("泄漏审计:圈成员仅用 2025-07-01 前互动定义,结论不变", y=1.02)
    fig.savefig(FIGS / "33-full_leakage.png")
    plt.close(fig)


def fig_sentiment(s):
    show = ["blend", "event", "gru_sent", "persist", "markov"]
    lbl = {"persist": "圈持续", "event": "事件历史", "markov": "马尔可夫",
           "gru_sent": "GRU情绪头", "blend": "blend(0.7事件+0.3持续)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ms = sorted(show, key=lambda m: -s["models"][m]["L1"])
    v1 = [s["models"][m]["L1"] for m in ms]
    v2 = [s["models"][m]["dom_hit"] for m in ms]
    best_l1 = min(show, key=lambda m: s["models"][m]["L1"])
    best_hit = max(show, key=lambda m: s["models"][m]["dom_hit"])
    cols = [CAT[0] if m in (best_l1, best_hit) else figstyle.BASE for m in ms]
    b = axes[0].barh(range(len(ms)), v1, color=cols, height=0.6)
    axes[0].set_yticks(range(len(ms)), [lbl[m] for m in ms])
    for bb, v in zip(b, v1):
        axes[0].text(v + 0.008, bb.get_y() + 0.3, f"{v:.3f}", fontsize=8.5, color=INK2)
    axes[0].set_xlabel("情绪构成 L1 距离(越低越好)")
    axes[0].set_xlim(0, max(v1) * 1.18)
    axes[0].grid(axis="y", visible=False)
    b = axes[1].barh(range(len(ms)), v2, color=cols, height=0.6)
    axes[1].set_yticks(range(len(ms)), ["" for _ in ms])
    for bb, v in zip(b, v2):
        axes[1].text(v + 0.008, bb.get_y() + 0.3, f"{v:.3f}", fontsize=8.5, color=INK2)
    axes[1].set_xlabel("主导情绪命中率(越高越好)")
    axes[1].set_xlim(0, 1.0)
    axes[1].grid(axis="y", visible=False)
    fig.suptitle(f"舆论线:圈内 24h 情绪构成预测(test n={s['n_test']})", y=1.02)
    fig.savefig(FIGS / "35-sentiment.png")
    plt.close(fig)


def fig_closed_vs_mc(mc):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    cf = np.array([r["closed_form"] for r in mc])
    md = np.array([r["mc_delta"] for r in mc])
    se = np.array([r["mc_se"] for r in mc])
    rho = np.array([r["rho"] for r in mc])
    near = rho >= 0.97
    ax.errorbar(cf[~near], md[~near], yerr=1.96 * se[~near], fmt="o", ms=5.5,
                color=CAT[0], ecolor=figstyle.BASE, capsize=2, label="ρ<0.97")
    ax.errorbar(cf[near], md[near], yerr=1.96 * se[near], fmt="s", ms=5.5,
                color=CAT[1], ecolor=figstyle.BASE, capsize=2, label="ρ≥0.97(近临界)")
    lim = [min(cf.min(), 1) * 0.7, max(cf.max(), md.max()) * 1.4]
    ax.plot(lim, lim, ls="--", lw=1, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("闭式期望增量(24h,m=20)")
    ax.set_ylabel("MC 模拟增量(300 runs)")
    ax.set_title("闭式 Δ 与蒙特卡洛一致(18 组,|z|≤1.63)")
    ax.legend(fontsize=8.5)
    fig.savefig(FIGS / "34-closed_vs_mc.png")
    plt.close(fig)


def fig_playbook(p):
    chs = sorted(p["by_channel"], key=lambda c: p["by_channel"][c]["delta24_med"])
    chs = [c for c in chs if p["by_channel"][c]["n"] >= 30]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
    y = np.arange(len(chs))
    med = [p["by_channel"][c]["delta24_med"] for c in chs]
    p25 = [p["by_channel"][c]["delta24_p25"] for c in chs]
    p75 = [p["by_channel"][c]["delta24_p75"] for c in chs]
    risk_col = {c: (CAT[7] if c == "circle_organized_risk" else CAT[0]) for c in chs}
    axes[0].hlines(y, p25, p75, color=[risk_col[c] for c in chs], lw=3, alpha=0.35)
    axes[0].scatter(med, y, s=42, color=[risk_col[c] for c in chs], zorder=3)
    axes[0].set_yticks(y, [CH_LABEL[c] for c in chs])
    axes[0].set_xlabel("每帖期望 24h 总计数(中位,IQR)")
    axes[0].set_title("效果:注入 1 帖的期望簇大小")
    axes[0].grid(axis="y", visible=False)
    spill = [p["by_channel"][c]["spill_med"] for c in chs]
    axes[1].barh(y, spill, height=0.55, color=[risk_col[c] for c in chs])
    axes[1].set_xlabel("溢出份额(诱发量中非注入通道占比)")
    axes[1].set_title("溢出:全维分解(修复后口径)")
    axes[1].set_xlim(0, 1.0)
    axes[1].grid(axis="y", visible=False)
    fig.suptitle("通道打法表(305 个热点/案例事件池化;红=仅监测不引导)", y=1.03)
    fig.savefig(FIGS / "34-playbook.png")
    plt.close(fig)


def fig_risk_timing(rt):
    phases = ["low", "rising", "peak"]
    pl_ = {"low": "低潮期", "rising": "上升期", "peak": "峰值期"}
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    rng = np.random.default_rng(3)
    for i, ph in enumerate(phases):
        vals = [r["runaway_prob_inj"] - r["runaway_prob_base"] for r in rt if r["phase"] == ph]
        xs = i + rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(xs, vals, s=36, color=CAT[0], alpha=0.75)
        med = float(np.median(vals))
        ax.hlines(med, i - 0.22, i + 0.22, color=INK, lw=2)
        ax.text(i + 0.26, med, f"中位 {med:.2f}", fontsize=8.5, color=INK2, va="center")
    ax.set_xticks(range(3), [pl_[p] for p in phases])
    ax.set_ylabel("超额失控概率(注入-基线)")
    ax.set_title("时机的真实角色在风险侧:低潮期注入的失控风险最高")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIGS / "34-risk_timing.png")
    plt.close(fig)


def fig_natexp(ne):
    rows = [r for r in ne["rows"] if r["match_gap_log"] < 0.3]
    pred = np.array([r["pred"] for r in rows])
    did = np.array([r["did"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    pos = did > 0
    ax.scatter(pred[pos], did[pos], s=14, alpha=0.45, color=CAT[0], label="DiD>0(76.6%)")
    ax.scatter(pred[~pos], -did[~pos], s=14, alpha=0.45, color=CAT[1],
               label="DiD<0(绝对值)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("模型预测增量(闭式,bigv 通道 × 进场帖数)")
    ax.set_ylabel("|DiD|:进场后24h超额(相对匹配对照)")
    s = ne["summary"]
    ax.set_title(f"自然实验:方向弱一致(Spearman {s['spearman_good']}, p<0.001),量级不可比")
    ax.legend(fontsize=8.5)
    fig.savefig(FIGS / "34-natexp.png")
    plt.close(fig)


def main():
    figstyle.apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    rm = json.loads((PF / "full_results.json").read_text())
    rp = json.loads((PF / "full_results_pre.json").read_text())
    s = json.loads((PF / "sentiment_results.json").read_text())
    mc = json.loads((SV / "mc_validation.json").read_text())
    p = json.loads((SV / "playbook.json").read_text())
    rt = json.loads((SV / "risk_timing.json").read_text())
    ne = json.loads((SV / "natural_experiment.json").read_text())
    fig_male_ladder(rm)
    fig_strata(rm)
    fig_bytype(rm)
    fig_leakage(rm, rp)
    fig_sentiment(s)
    fig_closed_vs_mc(mc)
    fig_playbook(p)
    fig_risk_timing(rt)
    fig_natexp(ne)
    print("figures saved to", FIGS)


if __name__ == "__main__":
    main()
