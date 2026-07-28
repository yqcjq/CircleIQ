"""内容调制分析(修正3):事件内容特征如何解释 Hawkes 拟合参数。

被解释量(每事件):
  - branching = α 矩阵总质量的加权平均(每事件的平均期望触发数)
  - cross_excite_share = 跨维激发占比(非对角 α 质量 / 总质量)
  - beta(激发时间尺度)
解释变量: 情绪分布/敏感占比/主行业/大V占比等(event_features.parquet)
方法: OLS + 按主导情绪分组的参数均值对比
输出: ~/data/content_modulation.json
用法: python3 analyze_content_modulation.py --pattern "hot__*"
"""
import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path.home() / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*")
    args = ap.parse_args()
    import polars as pl

    feats = pl.read_parquet(DATA / "event_features.parquet")
    rows = []
    for f in sorted((DATA / "hawkes").glob(f"{args.pattern}.json")):
        r = json.loads(f.read_text())
        if "A" not in r:
            continue
        A = np.array(r["A"])
        nbd = np.array(r["n_by_dim"], dtype=float)
        w = nbd / max(1, nbd.sum())
        branching = float((A.sum(axis=1) * w).sum())  # 每事件平均期望触发数
        total = A.sum()
        cross = float((total - np.trace(A)) / total) if total > 0 else None
        rows.append({"category": r["category"], "topic": r["topic"],
                     "branching": branching, "cross_excite_share": cross,
                     "beta_halflife_min": r["beta_halflife_min"],
                     "ll_gain": r["ll_gain_per_event"],
                     "stable_share": r["stable_event_share"]})
    fits = pl.DataFrame(rows)
    df = fits.join(feats, on=["category", "topic"], how="inner")
    print(f"joined events: {df.height}")

    SENTS = ["愤怒", "悲伤", "恐惧", "惊奇", "喜悦"]
    xcols = [f"sent_{s}" for s in SENTS] + ["sensitive_share", "bigv_share", "org_share", "stable_share"]
    out = {"n": df.height}

    # 1. 主导情绪分组对比
    df = df.with_columns(
        pl.concat_list([pl.col(f"sent_{s}") for s in SENTS]).alias("_sl"))
    dom = []
    for r in df.to_dicts():
        vals = {s: r[f"sent_{s}"] for s in SENTS}
        top, tv = max(vals.items(), key=lambda kv: kv[1])
        dom.append(top if tv >= 0.15 and r["sentiment_coverage"] > 0.5 else "中性主导")
    df = df.with_columns(pl.Series("dominant_sent", dom))
    grp = df.group_by("dominant_sent").agg(
        pl.len().alias("n"),
        pl.col("branching").median().round(4).alias("branching_med"),
        pl.col("cross_excite_share").median().round(4).alias("cross_med"),
        pl.col("beta_halflife_min").median().round(1).alias("halflife_med"),
    ).sort("n", descending=True)
    out["by_dominant_sentiment"] = grp.to_dicts()

    # 2. OLS: log(branching) ~ 内容特征
    from sklearn.linear_model import LinearRegression
    X = df.select(xcols).fill_null(0).to_numpy()
    ok = np.isfinite(X).all(axis=1)
    for target in ["branching", "cross_excite_share"]:
        y = df[target].fill_null(0).to_numpy()
        yl = np.log(np.clip(y, 1e-4, None)) if target == "branching" else y
        m = LinearRegression().fit(X[ok], yl[ok])
        r2 = float(m.score(X[ok], yl[ok]))
        out[f"ols_{target}"] = {
            "r2": round(r2, 4),
            "coef": {c: round(float(b), 4) for c, b in zip(xcols, m.coef_)},
            "intercept": round(float(m.intercept_), 4),
        }

    with open(DATA / "content_modulation.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("MODULATION DONE")


if __name__ == "__main__":
    main()
