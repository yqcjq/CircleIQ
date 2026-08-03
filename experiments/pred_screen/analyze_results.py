"""深挖筛选结果:非零分层、量级分层、bootstrap 置信区间、log 空间混合预测器。
产物: out/analysis.json(供报告与图)
用法: python3 analyze_results.py
"""
import json

import numpy as np
import polars as pl

from common import OUT

MODELS = ["naive", "snaive", "ewma", "poisson", "sir", "ic", "hawkes6",
          "gbdt", "gru", "tgn", "evolvegcn"]


def male(y, yh):
    return float(np.mean(np.abs(np.log1p(np.clip(yh, 0, None)) - np.log1p(y))))


def main():
    dfs = {}
    for m in MODELS:
        d = pl.read_parquet(OUT / "preds" / f"{m}.parquet")
        if "split" in d.columns:
            d = d.filter(pl.col("split") == "test")
        dfs[m] = d.select("event", "circle", "k", "horizon", "y", "yhat")

    # 只保留全模型共有键;宽表:每行一个 (event,circle,k,horizon),列=各模型 yhat
    base = dfs["naive"].rename({"yhat": "yhat_naive"})
    wide = base
    for m in MODELS:
        if m == "naive":
            continue
        wide = wide.join(dfs[m].rename({"yhat": f"yhat_{m}"}).drop("y"),
                         on=["event", "circle", "k", "horizon"], how="inner")
    # 混合:log 空间均值(持续性锚定)
    for m in ("gru", "gbdt"):
        wide = wide.with_columns(
            (((pl.col(f"yhat_{m}").clip(0) + 1).log() * 0.5 +
              (pl.col("yhat_naive").clip(0) + 1).log() * 0.5).exp() - 1)
            .clip(0).alias(f"yhat_blend_{m}"))
    all_models = MODELS + ["blend_gru", "blend_gbdt"]

    out = {"n_rows": wide.height, "strata": {}, "bootstrap": {}}
    rng = np.random.default_rng(7)

    for hz in ("h6", "h24"):
        sub = wide.filter(pl.col("horizon") == hz)
        y = sub["y"].to_numpy().astype(float)
        strata = {
            "all": np.ones(len(y), bool),
            "y0": y == 0,
            "y_1_9": (y > 0) & (y < 10),
            "y_10_99": (y >= 10) & (y < 100),
            "y_100p": y >= 100,
        }
        for c in (0, 5):
            strata[f"c{c}"] = (sub["circle"] == c).to_numpy()
        tab = {}
        for sname, mask in strata.items():
            if mask.sum() == 0:
                continue
            tab[sname] = {"n": int(mask.sum())}
            for m in all_models:
                yh = sub[f"yhat_{m}"].to_numpy().astype(float)
                tab[sname][m] = round(male(y[mask], yh[mask]), 4)
        out["strata"][hz] = tab

        # bootstrap(按事件重抽,尊重事件内相关):MALE 差值 CI
        events = sub["event"].to_numpy()
        uev = np.unique(events)
        idx_by_ev = {e: np.where(events == e)[0] for e in uev}
        pairs = [("gru", "naive"), ("gbdt", "naive"), ("gru", "gbdt"),
                 ("tgn", "gru"), ("blend_gru", "gru"), ("blend_gbdt", "gbdt")]
        boots = {f"{a}_vs_{b}": [] for a, b in pairs}
        yhs = {m: sub[f"yhat_{m}"].to_numpy().astype(float) for m in all_models}
        for _ in range(400):
            pick = rng.choice(uev, len(uev), replace=True)
            idx = np.concatenate([idx_by_ev[e] for e in pick])
            for a, b in pairs:
                boots[f"{a}_vs_{b}"].append(male(y[idx], yhs[a][idx]) - male(y[idx], yhs[b][idx]))
        out["bootstrap"][hz] = {
            k: {"mean": round(float(np.mean(v)), 4),
                "ci95": [round(float(np.percentile(v, 2.5)), 4),
                         round(float(np.percentile(v, 97.5)), 4)]}
            for k, v in boots.items()}

    (OUT / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for hz in ("h6", "h24"):
        print(f"\n===== {hz} strata MALE =====")
        tab = out["strata"][hz]
        cols = ["all", "y0", "y_1_9", "y_10_99", "y_100p", "c0", "c5"]
        hdr = "model".ljust(12) + "".join(c.rjust(9) for c in cols)
        print(hdr)
        order = sorted(all_models, key=lambda m: tab["all"][m])
        for m in order:
            print(m.ljust(12) + "".join(
                (f"{tab[c][m]:.3f}" if c in tab else "-").rjust(9) for c in cols))
        print("bootstrap ΔMALE (95% CI):")
        for k, v in out["bootstrap"][hz].items():
            sig = "SIG" if v["ci95"][0] * v["ci95"][1] > 0 else "   "
            print(f"  {k:<22} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] {sig}")


if __name__ == "__main__":
    main()
