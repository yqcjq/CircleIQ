"""阶段D验证图:留出窗计数误差 Hawkes vs Poisson vs naive。

用法: python make_figs_validation.py --file /tmp/circleiq_results/validation_counts.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle
from figstyle import CAT, INK2, MUTED

FIGS = Path(__file__).resolve().parent.parent / "docs" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    args = ap.parse_args()
    figstyle.apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    rs = [r for r in json.loads(Path(args.file).read_text()) if "ape_hawkes" in r]
    print(f"validated events: {len(rs)}")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    labels = [("ape_hawkes", "Hawkes 模拟", CAT[0]), ("ape_poisson", "Poisson 外推", CAT[1]),
              ("ape_naive", "Naive 持续", MUTED)]
    for key, lab, color in labels:
        v = np.sort(np.clip([r[key] for r in rs], 0, 3))
        y = np.arange(1, len(v) + 1) / len(v)
        med = float(np.median(v))
        ax.plot(v, y, color=color, lw=1.7, label=f"{lab}(中位 {med:.0%})")
    ax.set_xlim(0, 2)
    ax.set_xlabel("2小时滚动预测 APE(每事件8锚点中位,截断至 200%)")
    ax.set_ylabel("事件累积占比")
    ax.set_title(f"滚动短视界计数误差 ECDF(n={len(rs)} 事件)")
    ax.legend(loc="lower right")

    ax = axes[1]
    wins = np.mean([r["ape_hawkes"] <= min(r["ape_poisson"], r["ape_naive"]) for r in rs])
    cov = np.mean([r["coverage"] for r in rs])
    bars = [wins, cov]
    x = np.arange(2)
    ax.bar(x, bars, color=[CAT[0], CAT[2]], width=0.5)
    for i, b in enumerate(bars):
        ax.annotate(f"{b:.0%}", (i, b), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=11, color=INK2)
    ax.set_xticks(x, ["Hawkes 最优占比", "q10-q90 区间覆盖率"])
    ax.set_ylim(0, 1.1)
    ax.set_title("预测质量汇总(2h 视界)")
    fig.savefig(FIGS / "d_validation.png")
    print("saved", FIGS / "d_validation.png")


if __name__ == "__main__":
    main()
