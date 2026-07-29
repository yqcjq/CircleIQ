"""筛选实验报告图(复用项目 figstyle 调色板,与既有报告图同体系)。

图1 f_screen_male.png    : 双视界 MALE 排名(条形,回归/神经分色)
图2 f_screen_strata.png  : h24 按目标量级分层 MALE(top 模型折线)
图3 f_screen_cases.png   : 案例曲线 3×2(实际 vs GRU/GBDT/naive 预测,滚动锚点)
图4 f_screen_scatter.png : GRU vs naive 预测-真值散点(log)
数据: 服务器 out/{screening_results,analysis}.json + preds/*.parquet(先 scp 下来)
用法: python3 make_figs_screen.py --data-dir <本地下载目录> --out-dir docs/figures
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / ".." / "data-pipeline" / "pipeline"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--figstyle-dir", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.figstyle_dir)
    from figstyle import CAT, GRID, INK2, MUTED, apply_style
    import matplotlib.pyplot as plt
    import polars as pl
    apply_style()
    D = Path(args.data_dir)
    O = Path(args.out_dir)
    O.mkdir(parents=True, exist_ok=True)

    res = json.loads((D / "screening_results.json").read_text())
    ana = json.loads((D / "analysis.json").read_text())

    label = {"naive": "Naive 持续", "snaive": "季节 Naive", "ewma": "EWMA",
             "poisson": "Poisson 外推", "sir": "SIR", "ic": "独立级联 IC",
             "hawkes6": "Hawkes(6通道)", "gbdt": "GBDT", "gru": "GRU",
             "tgn": "TGN-lite", "evolvegcn": "EvolveGCN-O",
             "blend_gru": "GRU+Naive 混合", "blend_gbdt": "GBDT+Naive 混合"}
    neural = {"gru", "tgn", "evolvegcn"}
    C_REG, C_NEU, C_BASE = CAT[0], CAT[1], "#c3c2b7"

    # ---- 图1:双视界 MALE 排名 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, hz, ttl in zip(axes, ("h6", "h24"), ("6 小时视界", "24 小时视界")):
        tab = res["metrics"][hz]
        ms = sorted(tab, key=lambda m: -tab[m]["male"])
        vals = [tab[m]["male"] for m in ms]
        cols = [C_NEU if m in neural else (C_BASE if m in
                ("naive", "snaive", "ewma", "poisson") else C_REG) for m in ms]
        bars = ax.barh([label[m] for m in ms], vals, color=cols, height=0.62, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                    va="center", fontsize=8.5, color=INK2)
        ax.set_title(f"{ttl}(test,MALE 越低越好)")
        ax.set_xlabel("MALE = mean |Δlog1p|")
        ax.set_xlim(0, max(vals) * 1.18)
        ax.grid(axis="y", visible=False)
    import matplotlib.patches as mpatches
    fig.legend(handles=[mpatches.Patch(color=C_REG, label="经典/回归族"),
                        mpatches.Patch(color=C_NEU, label="神经族"),
                        mpatches.Patch(color=C_BASE, label="统计基线")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("圈0/圈5 活动量预测:候选算法筛选(2,146 test 锚点)", y=1.0)
    fig.tight_layout()
    fig.savefig(O / "f_screen_male.png")
    plt.close(fig)

    # ---- 图2:h24 量级分层 ----
    tab = ana["strata"]["h24"]
    strata = ["y0", "y_1_9", "y_10_99", "y_100p"]
    xlab = ["y=0", "1–9", "10–99", "≥100"]
    show = ["gru", "tgn", "gbdt", "naive", "hawkes6", "sir"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    finals = []
    for i, m in enumerate(show):
        ys = [tab[s][m] for s in strata]
        ax.plot(xlab, ys, marker="o", label=label[m],
                color=CAT[i % len(CAT)], zorder=3)
        finals.append([ys[-1], m, CAT[i % len(CAT)]])
    # 右侧直接标签:按终值排序后强制最小间距,避免重叠
    finals.sort()
    ymin, ymax = ax.get_ylim()
    min_gap = (ymax - ymin) * 0.055
    ypos = [f[0] for f in finals]
    for j in range(1, len(ypos)):
        ypos[j] = max(ypos[j], ypos[j - 1] + min_gap)
    for (yv, m, cc), yp in zip(finals, ypos):
        ax.annotate(label[m], (len(xlab) - 1 + 0.12, yp), fontsize=8.5,
                    color=cc, va="center")
    ax.set_xlim(-0.2, len(xlab) + 1.3)
    ax.set_title("24h 视界:按真实量级分层的 MALE(n=" +
                 ",".join(str(tab[s]["n"]) for s in strata) + ")")
    ax.set_ylabel("MALE")
    ax.set_xlabel("窗口真实帖数")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(O / "f_screen_strata.png")
    plt.close(fig)

    # ---- 图3:案例曲线 ----
    an = pl.read_parquet(D / "anchors.parquet")
    gru = pl.read_parquet(D / "preds_gru.parquet").filter(
        (pl.col("split") == "test") & (pl.col("horizon") == "h24"))
    gbdt = pl.read_parquet(D / "preds_gbdt.parquet").filter(
        (pl.col("split") == "test") & (pl.col("horizon") == "h24"))
    naive = pl.read_parquet(D / "preds_naive.parquet").filter(pl.col("horizon") == "h24")
    test_pairs = (gru.group_by("event", "circle").agg(pl.len(), pl.col("y").sum())
                  .filter(pl.col("len") >= 10).sort("y", descending=True))
    picks = []
    seen_ev = set()
    for r in test_pairs.iter_rows(named=True):
        if r["event"] in seen_ev:
            continue
        picks.append((r["event"], r["circle"]))
        seen_ev.add(r["event"])
        if len(picks) == 6:
            break
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    for ax, (ev, c) in zip(axes.flat, picks):
        sub = {m: df.filter((pl.col("event") == ev) & (pl.col("circle") == c)).sort("k")
               for m, df in (("gru", gru), ("gbdt", gbdt), ("naive", naive))}
        ks = sub["gru"]["k"].to_numpy()
        ax.plot(ks / 24, sub["gru"]["y"].to_numpy(), color="#0b0b0b", lw=2,
                label="实际", zorder=4)
        for i, (m, ls) in enumerate((("gru", "-"), ("gbdt", "-"), ("naive", "--"))):
            ax.plot(sub[m]["k"].to_numpy() / 24, sub[m]["yhat"].to_numpy(),
                    ls, color=CAT[i], lw=1.5, label=label[m], zorder=3, alpha=0.9)
        ax.set_yscale("symlog", linthresh=10)
        nm = ev.replace("major__", "").replace("hot__", "热点 ").replace("case__", "案例 ")
        ax.set_title(f"{nm[:24]} · 圈{c}", fontsize=9.5)
        ax.set_xlabel("事件时间(天)", fontsize=9)
        ax.set_ylabel("未来24h帖数", fontsize=9)
    axes.flat[0].legend(fontsize=8.5, ncol=2)
    fig.suptitle("test 事件案例:滚动锚点的 24h 预测轨迹(symlog 轴)", y=1.0)
    fig.tight_layout()
    fig.savefig(O / "f_screen_cases.png")
    plt.close(fig)

    # ---- 图4:预测-真值散点 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (m, df) in zip(axes, (("gru", gru), ("naive", naive))):
        if m == "naive":
            keys = set(zip(gru["event"], gru["circle"], gru["k"]))
            df = df.filter(pl.struct(["event", "circle", "k"]).map_elements(
                lambda s: (s["event"], s["circle"], s["k"]) in keys,
                return_dtype=pl.Boolean))
        y = df["y"].to_numpy().astype(float)
        yh = np.clip(df["yhat"].to_numpy().astype(float), 0, None)
        c0m = (df["circle"] == 0).to_numpy()
        for cm, cc, nm2 in ((c0m, CAT[0], "圈0"), (~c0m, CAT[1], "圈5")):
            ax.scatter(np.log1p(y[cm]), np.log1p(yh[cm]), s=9, alpha=0.45,
                       color=cc, label=nm2, edgecolors="none", zorder=3)
        lim = max(np.log1p(y).max(), np.log1p(yh).max()) * 1.05
        ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", zorder=2)
        male = float(np.mean(np.abs(np.log1p(yh) - np.log1p(y))))
        ax.set_title(f"{label[m]}(h24,MALE={male:.3f})")
        ax.set_xlabel("log1p(实际)")
        ax.set_ylabel("log1p(预测)")
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(O / "f_screen_scatter.png")
    plt.close(fig)
    print("figures ->", O)


if __name__ == "__main__":
    main()
