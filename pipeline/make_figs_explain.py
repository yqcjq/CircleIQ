"""解释性文档(x-2-explain)的直观图:任务解剖/零膨胀/预测案例/分支直觉/诱发流向/
放大效应/自然实验案例/情绪方差分解。

用法: python make_figs_explain.py(在 pipeline/,读 experiments/pred_full/results/explain_stats.json)
产出: docs/figures/33-2-*.png, 34-2-*.png, 35-2-*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle
from figstyle import CAT, INK, INK2, MUTED

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "figures"
D = json.loads((ROOT / "experiments" / "pred_full" / "results" / "explain_stats.json").read_text())
SLIM = json.loads((ROOT / "experiments" / "strategy_v2" / "results" / "closed_form_slim.json").read_text())


def fig_task_anatomy():
    tp = D["task_profile"]
    h = np.array(tp["hourly"], float)
    hi = 720
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.fill_between(np.arange(hi), 0, h[:hi], color=CAT[0], alpha=0.25, lw=0)
    ax.plot(np.arange(hi), h[:hi], color=CAT[0], lw=1.2)
    ks = [k for k in tp["anchors"] if k < hi - 24]
    for i, k in enumerate(ks):
        ax.axvline(k, color=INK2, lw=0.7, ls=":", alpha=0.6)
    k0 = ks[min(3, len(ks) - 1)]
    y24 = tp["y24"].get(str(k0), tp["y24"].get(k0))
    ax.axvspan(max(0, k0 - 168), k0, color=CAT[2], alpha=0.12, lw=0)
    ax.axvspan(k0, k0 + 24, color=CAT[1], alpha=0.25, lw=0)
    ymax = h[:hi].max()
    ax.annotate("模型能看到的历史(168h)", xy=(k0 - 84, ymax * 0.82), ha="center",
                fontsize=9, color=INK)
    ax.annotate(f"要预测的未来24h\n(答案 y={y24}帖)", xy=(k0 + 12, ymax * 0.62),
                ha="left", fontsize=9, color=INK)
    ax.annotate("虚线 = 一个个考题(锚点)", xy=(ks[-1] - 6, ymax * 0.95), ha="right",
                fontsize=9, color=INK2)
    ax.set_xlabel(f"事件开始后的小时数 ·「{tp['event_name']}」圈3(最大稳定圈)")
    ax.set_ylabel("圈内每小时发帖量")
    ax.set_title("预测任务解剖:在事件时间轴上每 6 小时设一道\"考题\"")
    fig.savefig(FIGS / "33-2-task_anatomy.png")
    plt.close(fig)


def fig_zero_inflation():
    zi = D["zero_inflation"]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(zi["labels"]))
    vals = zi["counts"]
    cols = [CAT[1]] + [CAT[0]] * 4
    bars = ax.bar(x, vals, 0.62, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 120,
                f"{v:,}\n({v/zi['n']:.0%})", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, ["0 帖\n(圈没动静)", "1–9", "10–99", "100–999", "≥1000\n(爆发)"])
    ax.set_ylabel("考题数量")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title(f"答案的分布:{zi['n']:,} 道 test 考题里 64% 的正确答案是\"0\"")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIGS / "33-2-zero_inflation.png")
    plt.close(fig)


def fig_pred_case():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, case in zip(axes, D["pred_cases"]):
        ks = sorted(int(k) for k in case["y"])
        y = [case["y"][str(k)] for k in ks]
        ax.plot(ks, y, color=INK, lw=2.0, marker="o", ms=4, label="真实")
        for m, ci, lbl in (("naive", None, "惯性外推(naive)"),
                           ("gru", 0, "GRU"), ("gru_gated", 2, "GRU+门控")):
            v = [case[m].get(str(k), np.nan) for k in ks]
            col = figstyle.BASE if ci is None else CAT[ci]
            ax.plot(ks, v, color=col, lw=1.5, ls="--", marker=".", ms=5, label=lbl)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel(f"锚点(事件小时)·「{case['event_name']}」圈3")
        ax.set_ylabel("未来24h圈内发帖量(log刻度)")
        ax.legend(fontsize=8)
    fig.suptitle("两个 test 事件上的预测轨迹:量级和起落跟得住,峰值略保守", y=1.02)
    fig.savefig(FIGS / "33-2-pred_case.png")
    plt.close(fig)


def fig_branching():
    bi = D["branching_intuit"]
    g = bi["generations"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    gens = ["注入\n1帖", "直接转发\n(第1代)", "转发的转发\n(第2代)", "……全部后代\n(24h合计)"]
    vals = [1.0, g["gen1_sum"], g["gen2_sum"], g["total_sum"]]
    cols = [INK2, CAT[0], CAT[0], CAT[1]]
    bars = ax.bar(range(4), vals, 0.6, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + 0.3, v + 0.05, f"{v:.2f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(4), gens, fontsize=8.5)
    ax.set_ylabel("期望帖数")
    ax.set_title(f"一帖的\"回声\"逐代衰减但累积可观\n(「{bi['event_name']}」大V通道)")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    rhos = [e["rho"] for e in SLIM["events"]]
    ax.hist(rhos, bins=40, color=CAT[0], alpha=0.85)
    ax.axvline(1.0, color=CAT[7], lw=1.5)
    ax.text(0.999, ax.get_ylim()[1] * 0.9, "临界线 ρ=1\n(超过则自我增殖)", ha="right",
            fontsize=8.5, color=CAT[7])
    ax.axvline(np.median(rhos), color=INK2, ls="--", lw=1)
    ax.text(np.median(rhos) - 0.002, ax.get_ylim()[1] * 0.55,
            f"中位 {np.median(rhos):.3f}\n→ 平均1帖回声≈{1/(1-np.median(rhos)):.0f}帖",
            ha="right", fontsize=8.5, color=INK2)
    ax.set_xlabel("每个事件的\"传染力\" ρ(分支比)")
    ax.set_ylabel("事件数")
    ax.set_title("312 个事件几乎都运行在临界点附近")
    fig.savefig(FIGS / "34-2-branching.png")
    plt.close(fig)


def fig_flow():
    flow = D["induced_flow"]
    srcs = ["大V", "机构", "散户", "圈_hub", "圈_org_matrix", "圈_media_kol", "圈_organized_risk"]
    src_lbl = {"大V": "圈外大V", "机构": "圈外机构", "散户": "圈外散户",
               "圈_hub": "枢纽圈", "圈_org_matrix": "机构矩阵圈",
               "圈_media_kol": "媒体/KOL圈", "圈_organized_risk": "疑似组织化圈"}
    tgt_order = ["散户", "大V", "机构", "圈_hub", "圈_org_matrix", "圈_media_kol",
                 "圈_organized_risk", "圈_other"]
    tgt_lbl = {"散户": "散户", "大V": "大V", "机构": "机构", "圈_hub": "枢纽圈",
               "圈_org_matrix": "机构矩阵", "圈_media_kol": "媒体/KOL",
               "圈_organized_risk": "疑似组织化", "圈_other": "其他圈"}
    tgt_col = {"散户": figstyle.BASE, "大V": CAT[1], "机构": CAT[3], "圈_hub": CAT[0],
               "圈_org_matrix": CAT[2], "圈_media_kol": CAT[4],
               "圈_organized_risk": CAT[7], "圈_other": CAT[6]}
    srcs = [s for s in srcs if s in flow]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    y = np.arange(len(srcs))
    for i, s in enumerate(srcs):
        comp = flow[s]
        tot = sum(comp.get(t, 0) for t in tgt_order)
        left = 0.0
        for t in tgt_order:
            v = comp.get(t, 0) / max(tot, 1e-9)
            if v < 0.005:
                continue
            ax.barh(i, v, left=left, height=0.6, color=tgt_col[t],
                    edgecolor=figstyle.SURFACE, linewidth=1.5,
                    label=tgt_lbl[t] if i == 0 else None)
            if v > 0.07:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color="white" if t != "散户" else INK)
            left += v
    ax.set_yticks(y, [f"从{src_lbl[s]}注入" for s in srcs])
    ax.set_xlim(0, 1)
    ax.set_xlabel("诱发的后续讨论都落在谁头上(305 个事件中位构成)")
    ax.set_title("注入通道 → 回声去向:选通道=选\"回声落在哪个人群\"")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(visible=False)
    ax.invert_yaxis()
    fig.savefig(FIGS / "34-2-flow.png")
    plt.close(fig)


def fig_natexp_case():
    nc = D["natexp_case"]
    r = nc["row"]
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    hrs = np.array(nc["hours"])
    ax.fill_between(hrs, 0, nc["hourly"], color=CAT[0], alpha=0.3, lw=0,
                    step="mid")
    ax.plot(hrs, nc["hourly"], color=CAT[0], lw=1.3, drawstyle="steps-mid")
    ax.axvline(0, color=CAT[7], lw=1.8)
    ymax = max(nc["hourly"])
    ax.text(1, ymax * 0.92, f"← 大V进场(粉丝 {r['followers']/1e6:.0f}00万+)",
            fontsize=9, color=CAT[7])
    ax.annotate(f"前24h合计 {r['pre24']:,} 帖", xy=(-24, ymax * 0.55), ha="center",
                fontsize=9, color=INK2)
    ax.annotate(f"后24h合计 {r['post24']:,} 帖\n(匹配对照只涨到 {r['ctrl_post24']:,})",
                xy=(24, ymax * 0.72), ha="center", fontsize=9, color=INK)
    ax.set_xlabel(f"距大V首帖的小时数 ·「{nc['event_name']}」")
    ax.set_ylabel("全事件每小时发帖量")
    ax.set_title("一次真实大V进场:进场点前后的事件热度")
    fig.savefig(FIGS / "34-2-natexp_case.png")
    plt.close(fig)


def fig_sent_variance():
    sv = D["sent_variance"]
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    bw = np.array(sv["between_all"])
    wi = np.array(sv["within_sample"])
    bins = np.linspace(0, 1.6, 33)
    ax.hist(wi, bins=bins, alpha=0.65, color=CAT[0], density=True,
            label=f"同一事件内不同时间(中位 {sv['within_med']:.2f})")
    ax.hist(bw, bins=bins, alpha=0.65, color=CAT[1], density=True,
            label=f"不同事件之间(中位 {sv['between_med']:.2f})")
    ax.axvline(sv["within_med"], color=CAT[0], ls="--", lw=1.2)
    ax.axvline(sv["between_med"], color=CAT[1], ls="--", lw=1.2)
    ax.set_xlabel("情绪构成差异(L1 距离,0=完全相同,2=完全不同)")
    ax.set_ylabel("密度")
    ax.set_title("情绪构成的差异主要在事件之间,不在时间演化里(差 2 倍)")
    ax.legend(fontsize=9)
    fig.savefig(FIGS / "35-2-sent_variance.png")
    plt.close(fig)


def main():
    figstyle.apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_task_anatomy()
    fig_zero_inflation()
    fig_pred_case()
    fig_branching()
    fig_flow()
    fig_natexp_case()
    fig_sent_variance()
    print("explain figures saved")


if __name__ == "__main__":
    main()
