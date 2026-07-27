"""阶段D案例图:大V发帖的 actual / factual / counterfactual 三线对比。

用法: python make_figs_case.py --case /tmp/circleiq_results/case_hot__20260313162141.json --name 余华英执行死刑
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
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", default="")
    args = ap.parse_args()
    figstyle.apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    c = json.loads(Path(args.case).read_text())
    h = np.arange(len(c["actual_hourly"]))

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.fill_between(h, c["factual_p10"], c["factual_p90"], color=CAT[0], alpha=0.15, lw=0)
    ax.fill_between(h, c["counterfactual_p10"], c["counterfactual_p90"], color=CAT[1], alpha=0.15, lw=0)
    ax.plot(h, c["actual_hourly"], color="#0b0b0b", lw=1.8, label="实际")
    ax.plot(h, c["factual_mean"], color=CAT[0], lw=1.6, ls="--", label="模拟(含大V帖)")
    ax.plot(h, c["counterfactual_mean"], color=CAT[1], lw=1.6, ls="--", label="反事实(剔除大V帖)")
    ax.set_xlabel(f"大V发帖后小时数(粉丝 {c['kol_followers']:,},{c['kol_auth']})")
    ax.set_ylabel("每小时事件数")
    title = args.name or c["name"]
    ax.set_title(f"大V发帖的反事实影响 · {title}(Δ总量 {c['delta_total_mean']:+.0f})")
    ax.legend()
    out = FIGS / f"d_case_{c['name'].replace('__','_')}.png"
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
