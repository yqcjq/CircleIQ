"""阶段C/D报告图表:α矩阵热力图 / 拟合增益分布 / β半衰期分布 / 验证LL对比 / 稳定占比vs增益。

输入: 服务器取回的 hawkes_summary_<cat>.json(或 hawkes/*.json 目录)
输出: docs/figures/c_*.png
用法: python make_figs_hawkes.py --summary /tmp/circleiq_results/hawkes_summary_hot.json
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


def ok_results(summary):
    return [r for r in summary if "error" not in r and "skip" not in r]


def fig_gain_dist(rs, out):
    gains = [r["ll_gain_per_event"] for r in rs]
    val_gain = [(r["ll_val"] - r["ll_val_poisson"]) / max(1, r["n_val"])
                for r in rs if r.get("ll_val") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    axes[0].hist(gains, bins=30, color=CAT[0])
    axes[0].set_xlabel("训练 LL 增益 / 事件(vs 齐次Poisson)")
    axes[0].set_ylabel("事件数")
    axes[0].set_title(f"Hawkes 拟合增益分布(n={len(gains)})")
    axes[1].hist(val_gain, bins=30, color=CAT[1])
    axes[1].axvline(0, color=INK, lw=1)
    pos = sum(g > 0 for g in val_gain) / max(1, len(val_gain))
    axes[1].set_xlabel("验证窗 LL 增益 / 事件")
    axes[1].set_title(f"留出窗验证(增益>0 占 {pos:.0%})")
    fig.savefig(out / "c_gain_dist.png")
    plt.close(fig)


def fig_beta_dist(rs, out):
    hl = [r["beta_halflife_min"] for r in rs]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    vals, counts = np.unique(hl, return_counts=True)
    x = np.arange(len(vals))
    ax.bar(x, counts, color=CAT[0], width=0.62)
    ax.set_xticks(x, [f"{v:g}" for v in vals])
    for i, c in enumerate(counts):
        ax.annotate(str(c), (x[i], c), textcoords="offset points", xytext=(0, 3),
                    fontsize=8.5, color=INK2, ha="center")
    ax.set_xlabel("最优激发核半衰期(分钟)")
    ax.set_ylabel("事件数")
    ax.set_title("传播激发的时间尺度分布")
    fig.savefig(out / "c_beta_dist.png")
    plt.close(fig)


def fig_alpha_heatmap(fit, out, name=""):
    A = np.array(fit["A"])
    K = fit["D"]
    labels = [f"圈{fit['dim_circles'][str(i)]}" if str(i) in fit["dim_circles"] else "其他"
              for i in range(K)]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(A, cmap=seq_cmap())
    ax.set_xticks(range(K), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K), labels, fontsize=8)
    ax.set_xlabel("被激发维 k")
    ax.set_ylabel("源维 j")
    for i in range(K):
        for j in range(K):
            if A[i, j] >= A.max() * 0.02:
                ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color=INK if A[i, j] < A.max() * 0.6 else "#ffffff")
    fig.colorbar(im, ax=ax, label="α (期望触发数/事件)", shrink=0.8)
    ax.set_title(f"圈层间激发矩阵 α · {name}")
    ax.grid(False)
    fig.savefig(out / f"c_alpha_{name}.png")
    plt.close(fig)


def fig_stable_share_vs_gain(rs, out):
    x = [r["stable_event_share"] * 100 for r in rs]
    y = [r["ll_gain_per_event"] for r in rs]
    n = [r["n_events"] for r in rs]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    s = np.clip(np.sqrt(n) / 8, 8, 80)
    ax.scatter(x, y, s=s, color=CAT[0], alpha=0.55, edgecolors="none")
    ax.set_xlabel("稳定圈用户事件占比 %")
    ax.set_ylabel("LL 增益 / 事件")
    ax.set_title("稳定圈参与度 vs Hawkes 结构增益(点大小=事件量)")
    fig.savefig(out / "c_share_vs_gain.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--alpha-top", type=int, default=2, help="给验证增益最高的N个事件画α热力图")
    args = ap.parse_args()
    figstyle.apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(args.summary).read_text())
    rs = ok_results(summary)
    print(f"fitted: {len(rs)} / {len(summary)}")
    fig_gain_dist(rs, FIGS)
    fig_beta_dist(rs, FIGS)
    fig_stable_share_vs_gain(rs, FIGS)
    scored = [r for r in rs if r.get("ll_val") is not None]
    scored.sort(key=lambda r: -(r["ll_val"] - r["ll_val_poisson"]))
    for r in scored[: args.alpha_top]:
        fig_alpha_heatmap(r, FIGS, name=r["topic"])
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
