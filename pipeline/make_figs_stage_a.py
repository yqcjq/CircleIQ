"""阶段A报告图表:K网格 / 稳定圈大小分布 / 模块度对比 / 激活热力图 / 画像构成。

输入: 从服务器取回的 JSON(stable_report_K3.json, eval_stable_K3.json, leiden_per_topic.json)
输出: docs/figures/a_*.png
用法: python make_figs_stage_a.py --dir /tmp/circleiq_results
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle
from figstyle import CAT, INK, INK2, MUTED, seq_cmap

FIGS = Path(__file__).resolve().parent.parent / "docs" / "figures"


def fig_k_grid(rep, out):
    ks = sorted(int(k) for k in rep["k_grid"])
    edges = [rep["k_grid"][str(k)]["edges"] for k in ks]
    users = [rep["k_grid"][str(k)]["users"] for k in ks]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(ks, edges, "-o", color=CAT[0], label="稳定边数")
    ax.plot(ks, users, "-s", color=CAT[1], label="稳定用户数")
    for k, e, u in zip(ks, edges, users):
        ax.annotate(f"{e:,}", (k, e), textcoords="offset points", xytext=(0, 7),
                    fontsize=8, color=INK2, ha="center")
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlabel("稳定性阈值 K(≥K 个议题/事件中有互动)")
    ax.set_ylabel("数量(对数轴)")
    ax.set_title("跨议题稳定图规模随 K 的变化")
    ax.legend()
    fig.savefig(out / "a_k_grid.png")
    plt.close(fig)


def fig_size_dist(rep, out):
    sizes = [c["len"] if "len" in c else c["size"] for c in rep.get("top20_sizes", [])]
    # top20 只是头部;完整分布需要 partition 文件,此处画头部圈 + 报告 n_circles
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(1, len(sizes) + 1)
    ax.bar(x, sizes, color=CAT[0], width=0.72)
    for i, s in enumerate(sizes[:8]):
        ax.annotate(f"{s:,}", (x[i], s), textcoords="offset points", xytext=(0, 3),
                    fontsize=7.5, color=INK2, ha="center")
    ax.set_yscale("log")
    ax.set_xlabel(f"稳定圈排名(共 {rep['n_circles']:,} 圈,≥10人 {rep['n_circles_ge10']:,} 圈)")
    ax.set_ylabel("圈大小(人,对数轴)")
    ax.set_title(f"稳定互动圈大小分布(top20,K={rep['K']})")
    fig.savefig(out / "a_size_dist.png")
    plt.close(fig)


def fig_modularity(rep, per_topic, out):
    names = [r["topic"].replace("major__", "") for r in per_topic]
    mods = [r["modularity"] for r in per_topic]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(names))
    ax.bar(x, mods, color=MUTED, width=0.6, label="单议题 Leiden")
    ax.axhline(rep["modularity"], color=CAT[0], lw=1.8, label=f"跨议题稳定图 {rep['modularity']:.3f}")
    ax.set_xticks(x, [n[:6] for n in names], rotation=20, ha="right")
    for i, m in enumerate(mods):
        ax.annotate(f"{m:.2f}", (i, m), textcoords="offset points", xytext=(0, 3),
                    fontsize=8, color=INK2, ha="center")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("模块度")
    ax.set_title("稳定图 vs 单议题图的社区结构强度")
    ax.legend(loc="lower right")
    fig.savefig(out / "a_modularity.png")
    plt.close(fig)


def fig_activation(ev, out, topn=12):
    act = ev["activation"]
    topics = list(act)
    circles = sorted({c for a in act.values() for c in a["by_circle"]},
                     key=lambda c: -sum(a["by_circle"].get(c, 0) for a in act.values()))[:topn]
    M = np.zeros((len(topics), len(circles)))
    for i, t in enumerate(topics):
        for j, c in enumerate(circles):
            M[i, j] = act[t]["by_circle"].get(c, 0) / max(1, act[t]["events"]) * 100
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    im = ax.imshow(M, cmap=seq_cmap(), aspect="auto")
    ax.set_xticks(range(len(circles)), [f"圈{c}" for c in circles], rotation=0, fontsize=8)
    ax.set_yticks(range(len(topics)), [t[:6] for t in topics], fontsize=9)
    for i in range(len(topics)):
        for j in range(len(circles)):
            if M[i, j] >= 0.05:
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=7,
                        color=INK if M[i, j] < M.max() * 0.6 else "#ffffff")
    fig.colorbar(im, ax=ax, label="事件占比 %", shrink=0.85)
    ax.set_title("重大议题 × 稳定圈 激活热力图(圈成员发帖占议题总帖数 %)")
    ax.grid(False)
    fig.savefig(out / "a_activation.png")
    plt.close(fig)


def fig_stable_vs_rest(ev, out):
    rows = {r["is_stable"]: r for r in ev["stable_vs_rest"]}
    metrics = [("mean_log_followers", "log(粉丝+1) 均值"), ("mean_posts", "人均发帖"),
               ("mean_major_topics", "人均参与大议题数"), ("bigv_rate", "大V占比"), ("org_rate", "机构占比")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(9.6, 2.6))
    for ax, (k, label) in zip(axes, metrics):
        vals = [rows[False][k] or 0, rows[True][k] or 0]
        ax.bar([0, 1], vals, color=[MUTED, CAT[0]], width=0.6)
        ax.set_xticks([0, 1], ["其他", "稳定圈"], fontsize=9)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:g}", (i, v), textcoords="offset points", xytext=(0, 3),
                        fontsize=8, color=INK2, ha="center")
        ax.set_title(label, fontsize=9.5)
        ax.set_yticks([])
    fig.suptitle("稳定圈用户 vs 其他用户", y=1.04)
    fig.savefig(out / "a_stable_vs_rest.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    d = Path(args.dir)
    figstyle.apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)

    rep = json.loads((d / f"stable_report_K{args.k}.json").read_text())
    fig_k_grid(rep, FIGS)
    fig_size_dist(rep, FIGS)
    per_topic = json.loads((d / "leiden_per_topic.json").read_text())
    fig_modularity(rep, per_topic, FIGS)
    ev_p = d / f"eval_stable_K{args.k}.json"
    if ev_p.exists():
        ev = json.loads(ev_p.read_text())
        fig_activation(ev, FIGS)
        fig_stable_vs_rest(ev, FIGS)
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
