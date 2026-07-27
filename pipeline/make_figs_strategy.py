"""阶段E策略图:ROI vs 失控概率散点(通道类型着色)。

用法: python make_figs_strategy.py --dir /tmp/circleiq_results/strategy
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle
from figstyle import CAT, INK2

FIGS = Path(__file__).resolve().parent.parent / "docs" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--exclude-major", action="store_true", default=True)
    args = ap.parse_args()
    figstyle.apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)

    groups = {"稳定圈": ([], CAT[0]), "圈外大V": ([], CAT[1]), "圈外机构": ([], CAT[2]),
              "圈外散户": ([], CAT[3])}
    for p in sorted(Path(args.dir).glob("*.json")):
        s = json.loads(p.read_text())
        if args.exclude_major and s["name"].startswith("major"):
            continue  # 大议题的长视界外推近临界,数字不可信(报告局限)
        for g in s["grid"]:
            if g["induced"] <= 0:
                continue
            lab = g["dim_label"] if g["dim_label"].startswith("圈外") else "稳定圈"
            groups[lab][0].append((g["runaway_prob"], g["roi"], g["m"]))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for lab, (pts, color) in groups.items():
        if not pts:
            continue
        x = [p[0] for p in pts]
        y = [max(p[1], 0.05) for p in pts]
        sz = [14 + 3 * p[2] for p in pts]
        ax.scatter(x, y, s=sz, color=color, alpha=0.6, edgecolors="none", label=lab)
    ax.axvline(0.2, color=INK2, lw=1, ls=":")
    ax.annotate("风险约束 0.2", (0.21, ax.get_ylim()[1] * 0.5), fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("失控概率 P(总量 > 1.5×基线q95)")
    ax.set_ylabel("ROI = 诱发增量 / 注入量(对数轴)")
    ax.set_title("策略网格:通道 × 注入量的收益-风险(热点/案例事件)")
    ax.legend()
    fig.savefig(FIGS / "e_strategy_scatter.png")
    print("saved", FIGS / "e_strategy_scatter.png")


if __name__ == "__main__":
    main()
