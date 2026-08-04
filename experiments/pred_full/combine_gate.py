"""侧路:分类门控 x GRU 回归的组合(gru_gated)。

动机:两段式 GBDT 赢在 y=0 层(硬阈值全零),但 AUC 从 0.92 掉到 0.84、1-9 层大幅
变差;GRU 是非零层最强回归器。第一性原理:零膨胀应由分类器处理,量级应由最强
回归器处理——两者正交,直接组合。
门控:P(y>0) 由 GBDT 分类器出,tau 在 val 上选;yhat = gru_yhat if p>tau else 0。
用法: python3 combine_gate.py [--partition main|pre]
"""
import argparse
import json
import time

import numpy as np
import polars as pl

from common import HORIZONS, out_dir, save_preds
from train_gbdt import build_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="main", choices=["main", "pre"])
    args = ap.parse_args()
    t0 = time.time()
    X, meta = build_features(args.partition, cascade=False)
    split = np.array([m[3] for m in meta])
    key_of = {(m[0], m[1], m[2]): i for i, m in enumerate(meta)}
    print(f"features {X.shape} {time.time()-t0:.0f}s", flush=True)

    gru = pl.read_parquet(out_dir(args.partition) / "preds" / "gru.parquet")
    from sklearn.ensemble import HistGradientBoostingClassifier
    kw = dict(max_iter=500, learning_rate=0.06, max_depth=None, max_leaf_nodes=63,
              min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
              random_state=7)
    tr = split == "train"
    rows, chosen = [], {}
    for hname, H in HORIZONS.items():
        y = np.array([m[4] if H == 6 else m[5] for m in meta], np.float32)
        clf = HistGradientBoostingClassifier(**kw)
        clf.fit(X[tr], (y[tr] > 0).astype(int))
        sub = gru.filter(pl.col("horizon") == hname)
        idx = np.array([key_of[(e, c, k)] for e, c, k in
                        zip(sub["event"], sub["circle"], sub["k"])])
        p = clf.predict_proba(X[idx])[:, 1]
        gy = sub["yhat"].to_numpy()
        sv = sub["split"].to_numpy()
        yv = sub["y"].to_numpy()
        # val 上选 tau(含 soft p*yhat)
        vm = sv == "val"
        best = ("soft", None, float(np.abs(np.log1p(p[vm] * gy[vm]) - np.log1p(yv[vm])).mean()))
        for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
            yh = np.where(p[vm] > tau, gy[vm], 0.0)
            male = float(np.abs(np.log1p(yh) - np.log1p(yv[vm])).mean())
            if male < best[2]:
                best = ("hard", tau, male)
        chosen[hname] = {"combine": best[0], "tau": best[1], "val_male": round(best[2], 4)}
        yh_all = p * gy if best[0] == "soft" else np.where(p > best[1], gy, 0.0)
        for i in range(len(sub)):
            rows.append({"event": sub["event"][i], "circle": int(sub["circle"][i]),
                         "k": int(sub["k"][i]), "horizon": hname,
                         "y": float(yv[i]), "yhat": float(yh_all[i]), "split": str(sv[i])})
    save_preds("gru_gated", rows, args.partition)
    (out_dir(args.partition) / "preds" / "gru_gated_meta.json").write_text(
        json.dumps(chosen))
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
