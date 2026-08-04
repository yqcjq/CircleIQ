"""汇总所有模型 test 预测 -> 总榜 + 量级分层 + 圈规模/类型分组 + 按事件 bootstrap。
用法: python3 evaluate.py [--partition main|pre] [--split test]
"""
import argparse
import json

import numpy as np
import polars as pl

from common import metrics, out_dir


def male_arr(y, yhat):
    return np.abs(np.log1p(np.clip(yhat, 0, None)) - np.log1p(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="main", choices=["main", "pre"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()
    OUT = out_dir(args.partition)

    members = json.loads((OUT / "members.json").read_text())
    size_of = {c["circle_id"]: c["size"] for c in members["circles"]}
    type_of = {c["circle_id"]: c["type"] for c in members["circles"]}

    preds = {}
    for f in sorted((OUT / "preds").glob("*.parquet")):
        preds[f.stem] = pl.read_parquet(f).filter(pl.col("split") == args.split)

    # 共同键集
    keysets = {m: set(zip(df["event"], df["circle"], df["k"], df["horizon"]))
               for m, df in preds.items()}
    common = set.intersection(*keysets.values())
    print(f"models: {list(preds)}; common {args.split} keys: {len(common)}")
    for m in preds:
        df = preds[m]
        keep = pl.Series([k in common for k in
                          zip(df["event"], df["circle"], df["k"], df["horizon"])])
        preds[m] = df.filter(keep).sort(["event", "circle", "k", "horizon"])

    ref = next(iter(preds.values()))
    ev = ref["event"].to_numpy()
    circ = ref["circle"].to_numpy()
    hz = ref["horizon"].to_numpy()
    y = ref["y"].to_numpy()
    yh = {m: df["yhat"].to_numpy() for m, df in preds.items()}
    for m, df in preds.items():
        assert np.array_equal(df["y"].to_numpy(), y), f"{m} y mismatch"

    results = {"split": args.split, "n_common": len(common), "metrics": {},
               "strata": {}, "by_size": {}, "by_type": {}, "bootstrap": {},
               "win_vs_naive": {}}
    size_tier = np.array([("5k+" if size_of[c] >= 5000 else
                           "1k-5k" if size_of[c] >= 1000 else
                           "300-1k" if size_of[c] >= 300 else "100-300")
                          for c in circ])
    ctype = np.array([type_of[c] for c in circ])

    for h in ("h6", "h24"):
        hm = hz == h
        results["metrics"][h] = {m: metrics(y[hm], yh[m][hm]) for m in preds}
        # 量级分层
        edges = [(0, 0), (1, 9), (10, 99), (100, 10 ** 9)]
        results["strata"][h] = {}
        for lo, hi in edges:
            sm = hm & (y >= lo) & (y <= hi)
            results["strata"][h][f"{lo}-{hi if hi < 1e9 else 'inf'}"] = {
                "n": int(sm.sum()),
                **{m: round(float(male_arr(y[sm], yh[m][sm]).mean()), 4)
                   for m in preds}}
        # 圈规模 / 类型
        for gname, garr, store in (("size", size_tier, "by_size"),
                                   ("type", ctype, "by_type")):
            results[store][h] = {}
            for gval in sorted(set(garr)):
                gm = hm & (garr == gval)
                results[store][h][gval] = {
                    "n": int(gm.sum()),
                    **{m: round(float(male_arr(y[gm], yh[m][gm]).mean()), 4)
                       for m in preds}}
        # 与 naive 的配对胜率
        e_n = male_arr(y[hm], yh["naive"][hm])
        results["win_vs_naive"][h] = {}
        for m in preds:
            if m == "naive":
                continue
            e_m = male_arr(y[hm], yh[m][hm])
            results["win_vs_naive"][h][m] = round(float((e_m < e_n).mean()), 4)
        # 按事件 bootstrap:model - naive 的 MALE 差
        uev = np.unique(ev[hm])
        ev_idx = {e: np.where(hm & (ev == e))[0] for e in uev}
        rng = np.random.default_rng(7)
        results["bootstrap"][h] = {}
        for m in preds:
            if m == "naive":
                continue
            e_m_full = male_arr(y, yh[m])
            e_n_full = male_arr(y, yh["naive"])
            per_ev = {e: (float(e_m_full[ix].sum()), float(e_n_full[ix].sum()), len(ix))
                      for e, ix in ev_idx.items()}
            diffs = []
            for _ in range(args.boot):
                samp = rng.choice(uev, size=len(uev), replace=True)
                sm_ = sum(per_ev[e][0] for e in samp)
                sn_ = sum(per_ev[e][1] for e in samp)
                nn_ = sum(per_ev[e][2] for e in samp)
                diffs.append((sm_ - sn_) / nn_)
            lo_, hi_ = np.percentile(diffs, [2.5, 97.5])
            results["bootstrap"][h][m] = {
                "diff": round(float(np.mean(diffs)), 4),
                "ci95": [round(float(lo_), 4), round(float(hi_), 4)],
                "sig": bool(hi_ < 0 or lo_ > 0)}

    (OUT / "full_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    for h in ("h6", "h24"):
        r = results["metrics"][h]
        print(f"\n=== {h} ({args.split}, n={r[list(r)[0]]['n']}) ===")
        print(f"{'model':<12}{'MALE':>8}{'RMSLE':>8}{'medAPE':>9}{'bias':>8}{'AUC':>7}{'win%':>7}")
        for m in sorted(r, key=lambda m: r[m]["male"]):
            x = r[m]
            w = results["win_vs_naive"][h].get(m, float("nan"))
            print(f"{m:<12}{x['male']:>8.4f}{x['rmsle']:>8.4f}{x['med_ape']:>9.3f}"
                  f"{x['bias_log']:>8.3f}{x.get('auc_nonzero', float('nan')):>7.3f}{w:>7.3f}")
        print("bootstrap vs naive:",
              {m: f"{v['diff']} {v['ci95']} {'SIG' if v['sig'] else 'ns'}"
               for m, v in results["bootstrap"][h].items()})
    print("\nby_type h24 (MALE):")
    for t, v in results["by_type"]["h24"].items():
        best = sorted((k for k in v if k != "n"), key=lambda m: v[m])[:3]
        print(f"  {t:<15} n={v['n']:<7}" + "  ".join(f"{m}={v[m]}" for m in best))
    print("\nsaved -> full_results.json")


if __name__ == "__main__":
    main()
