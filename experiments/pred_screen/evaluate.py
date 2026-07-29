"""汇总所有模型的 test 预测 -> 指标表 + 分层分析 + 排名。
用法: python3 evaluate.py [--split test]
"""
import argparse
import json

import numpy as np
import polars as pl

from common import OUT, load_anchors, metrics

MODELS_STAT = ["naive", "snaive", "ewma", "poisson"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    an = load_anchors()
    split_of = {(r["event"], r["circle"], r["k"]): r["split"] for r in an.iter_rows(named=True)}

    preds = {}
    for f in sorted((OUT / "preds").glob("*.parquet")):
        df = pl.read_parquet(f)
        if "split" not in df.columns:
            df = df.with_columns(
                pl.struct(["event", "circle", "k"]).map_elements(
                    lambda s: split_of.get((s["event"], s["circle"], s["k"]), "?"),
                    return_dtype=pl.String).alias("split"))
        preds[f.stem] = df

    # 统一到 split 子集,并对齐每个模型都覆盖的 (event,circle,k,horizon) 键集
    keysets = {}
    for m, df in preds.items():
        sub = df.filter(pl.col("split") == args.split)
        keysets[m] = set(zip(sub["event"], sub["circle"], sub["k"], sub["horizon"]))
    common_keys = set.intersection(*keysets.values())
    print(f"models: {list(preds)}; common {args.split} keys: {len(common_keys)}")

    results = {}
    for m, df in preds.items():
        sub = df.filter(pl.col("split") == args.split).with_columns(
            pl.struct(["event", "circle", "k", "horizon"]).map_elements(
                lambda s: (s["event"], s["circle"], s["k"], s["horizon"]) in common_keys,
                return_dtype=pl.Boolean).alias("keep")).filter(pl.col("keep"))
        for (hz,), g in sub.group_by(["horizon"]):
            res = metrics(g["y"].to_numpy(), g["yhat"].to_numpy())
            results.setdefault(hz, {})[m] = res
        for (hz, c), g in sub.group_by(["horizon", "circle"]):
            res = metrics(g["y"].to_numpy(), g["yhat"].to_numpy())
            results.setdefault(f"{hz}_c{c}", {})[m] = res

    # 与 naive 配对胜率(MALE per 样本)
    naive = preds["naive"].filter(pl.col("split") == args.split)
    nv = {(r["event"], r["circle"], r["k"], r["horizon"]): r["yhat"]
          for r in naive.iter_rows(named=True)}
    winrates = {}
    for m, df in preds.items():
        if m == "naive":
            continue
        sub = df.filter(pl.col("split") == args.split)
        for (hz,), g in sub.group_by(["horizon"]):
            wins = tot = 0
            for r in g.iter_rows(named=True):
                key = (r["event"], r["circle"], r["k"], r["horizon"])
                if key not in nv or key not in common_keys:
                    continue
                e_m = abs(np.log1p(max(0, r["yhat"])) - np.log1p(r["y"]))
                e_n = abs(np.log1p(max(0, nv[key])) - np.log1p(r["y"]))
                wins += e_m < e_n
                tot += 1
            winrates.setdefault(hz, {})[m] = round(wins / max(1, tot), 4)

    out = {"split": args.split, "n_common": len(common_keys),
           "metrics": results, "win_vs_naive": winrates}
    (OUT / "screening_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    for hz in ("h6", "h24"):
        print(f"\n=== {hz} (test, n={results[hz][list(results[hz])[0]]['n']}) ===")
        print(f"{'model':<12}{'MALE':>8}{'RMSLE':>8}{'medAPE':>9}{'bias':>8}{'AUC':>7}{'win%':>7}")
        for m in sorted(results[hz], key=lambda m: results[hz][m]["male"]):
            r = results[hz][m]
            w = winrates.get(hz, {}).get(m, float("nan"))
            print(f"{m:<12}{r['male']:>8.4f}{r['rmsle']:>8.4f}{r['med_ape']:>9.3f}"
                  f"{r['bias_log']:>8.3f}{r.get('auc_nonzero', float('nan')):>7.3f}"
                  f"{w if isinstance(w, float) else float('nan'):>7.3f}")
    print("\nby circle (h24):")
    for c in (0, 5):
        key = f"h24_c{c}"
        if key in results:
            best = sorted(results[key], key=lambda m: results[key][m]["male"])[:5]
            print(f"  c{c}: " + "  ".join(f"{m}={results[key][m]['male']:.4f}" for m in best))
    print("\nsaved -> out/screening_results.json")


if __name__ == "__main__":
    main()
