"""GBDT 特征回归(HistGradientBoosting,log1p 目标)。pred_screen 38 维特征的全圈版:
圈指示 -> log 圈规模;六通道 -> 六视图(目标圈/其余稳定/大V/机构/散户/总量)。

可选:
  --cascade    +级联集中度特征(近6/24h 目标圈内最大单根级联规模与份额)——攻爆发层
  --two-stage  零膨胀两段式:P(y>0) 分类器 x 条件回归,val 上选组合方式
用法: python3 train_gbdt.py [--partition main|pre] [--cascade] [--two-stage]
"""
import argparse
import json
import time

import numpy as np

from common import HORIZONS, load_anchors, load_panel, out_dir, save_preds, six_view


def build_features(variant, cascade):
    import polars as pl
    an = load_anchors(variant)
    panel = load_panel(variant)
    C = panel["C"]
    members = json.loads((out_dir(variant) / "members.json").read_text())
    size_of = {c["circle_id"]: c["size"] for c in members["circles"]}
    OUT = out_dir(variant)
    WINS = [1, 3, 6, 12, 24, 48, 72, 168]
    feats, meta = [], []
    for (event,), g in an.group_by(["event"], maintain_order=True):
        z = np.load(OUT / "events" / f"{event}.npz")
        counts = z["counts"].astype(float)
        b0 = int(z["b0"])
        p_ts, p_uid, p_cid = z["p_ts"], z["p_uid"], z["p_cid"]
        rt_ts, rt_cid, rt_root = z["rt_ts"], z["rt_cid"], z["rt_root"]
        nb = counts.shape[1]
        views = {}
        for cidx in g["cidx"].unique().to_list():
            v = six_view(counts, cidx, C)
            views[cidx] = np.concatenate([np.zeros((6, 1)), np.cumsum(v, 1)], 1)
        for row in g.iter_rows(named=True):
            c, cidx, k = row["circle"], row["cidx"], row["k"]
            cv = views[cidx]
            f = []
            for w in WINS:
                lo = max(0, k - w)
                f.append(np.log1p(cv[0, k] - cv[0, lo]))
            for ch in range(6):
                for w in (6, 24, 72):
                    lo = max(0, k - w)
                    f.append(np.log1p(cv[ch, k] - cv[ch, lo]))
            for w in (6, 24):
                a = cv[0, k] - cv[0, max(0, k - w)]
                b = cv[0, max(0, k - w)] - cv[0, max(0, k - 2 * w)]
                f.append(np.log1p(a) - np.log1p(b))
            tot = cv[5, k] - cv[5, max(0, k - 24)]
            tot_prev = cv[5, max(0, k - 24)] - cv[5, max(0, k - 48)]
            f.append(np.log1p(tot) - np.log1p(tot_prev))
            t_anchor = (b0 + k) * 3600
            hod = (t_anchor / 3600 + 8) % 24
            dow = ((t_anchor + 8 * 3600) // 86400 + 4) % 7
            f += [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
                  np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]
            f += [k / 24.0, np.log1p(cv[5, k])]
            for w in (24, 72):
                a, b = np.searchsorted(p_ts, [t_anchor - w * 3600, t_anchor])
                seg = p_uid[a:b][p_cid[a:b] == cidx]
                f.append(np.log1p(len(np.unique(seg))))
            f.append(np.log1p(size_of[c]))
            if cascade:
                for w in (6, 24):
                    a, b = np.searchsorted(rt_ts, [t_anchor - w * 3600, t_anchor])
                    roots = rt_root[a:b][rt_cid[a:b] == cidx]
                    if len(roots):
                        top = int(np.bincount(roots).max())
                        f += [np.log1p(top), top / len(roots)]
                    else:
                        f += [0.0, 0.0]
            feats.append(f)
            meta.append((event, c, k, row["split"], row["y6"], row["y24"]))
    return np.array(feats, np.float32), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="main", choices=["main", "pre"])
    ap.add_argument("--cascade", action="store_true")
    ap.add_argument("--two-stage", action="store_true")
    ap.add_argument("--loss", default="squared_error",
                    choices=["squared_error", "absolute_error", "poisson"])
    args = ap.parse_args()
    name = "gbdt" + ("_casc" if args.cascade else "") + ("_2s" if args.two_stage else "")
    if args.loss != "squared_error":
        name += "_" + {"absolute_error": "l1", "poisson": "pois"}[args.loss]
    t0 = time.time()
    X, meta = build_features(args.partition, args.cascade)
    split = np.array([m[3] for m in meta])
    print(f"features {X.shape} built {time.time()-t0:.0f}s "
          f"(train={int((split=='train').sum())} test={int((split=='test').sum())})", flush=True)

    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    kw = dict(max_iter=500, learning_rate=0.06, max_depth=None, max_leaf_nodes=63,
              min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
              random_state=7)
    rows, cost, extra = [], {}, {}
    tr, va, te = split == "train", split == "val", split == "test"
    for hname, H in HORIZONS.items():
        y = np.array([m[4] if H == 6 else m[5] for m in meta], np.float32)
        t1 = time.time()
        if args.two_stage:
            clf = HistGradientBoostingClassifier(**kw)
            clf.fit(X[tr], (y[tr] > 0).astype(int))
            pos = tr & (y > 0)
            reg = HistGradientBoostingRegressor(loss="squared_error", **kw)
            reg.fit(X[pos], np.log1p(y[pos]))

            def predict(mask):
                p = clf.predict_proba(X[mask])[:, 1]
                mu = np.expm1(reg.predict(X[mask]))
                return p, np.clip(mu, 0, None)
            # val 上选组合:硬阈值(网格) vs 软乘积
            pv, muv = predict(va)
            yv = y[va]
            best = ("soft", None,
                    float(np.abs(np.log1p(pv * muv) - np.log1p(yv)).mean()))
            for tau in (0.3, 0.4, 0.5, 0.6):
                yh = np.where(pv > tau, muv, 0.0)
                male = float(np.abs(np.log1p(yh) - np.log1p(yv)).mean())
                if male < best[2]:
                    best = ("hard", tau, male)
            extra[hname] = {"combine": best[0], "tau": best[1],
                            "val_male": round(best[2], 4)}
            outs = {}
            for mask, sp in ((va, "val"), (te, "test")):
                p, mu = predict(mask)
                outs[sp] = p * mu if best[0] == "soft" else np.where(p > best[1], mu, 0.0)
        else:
            if args.loss == "poisson":
                model = HistGradientBoostingRegressor(loss="poisson", **kw)
                model.fit(X[tr], y[tr])
                outs = {sp: model.predict(X[mask])
                        for mask, sp in ((va, "val"), (te, "test"))}
            else:
                model = HistGradientBoostingRegressor(loss=args.loss, **kw)
                model.fit(X[tr], np.log1p(y[tr]))
                outs = {sp: np.expm1(model.predict(X[mask]))
                        for mask, sp in ((va, "val"), (te, "test"))}
        cost[hname] = round(time.time() - t1, 1)
        for mask, sp in ((va, "val"), (te, "test")):
            yh = outs[sp]
            for (event, c, k, _, y6, y24), p in zip(
                    [m for m, mm in zip(meta, mask) if mm], yh):
                rows.append({"event": event, "circle": c, "k": k, "horizon": hname,
                             "y": float(y6 if H == 6 else y24),
                             "yhat": float(max(0, p)), "split": sp})
    save_preds(name, rows, args.partition)
    (out_dir(args.partition) / "preds" / f"{name}_meta.json").write_text(json.dumps(
        {"train_cost_s": cost, "n_features": X.shape[1], "two_stage": extra}))
    print("cost:", cost, extra)


if __name__ == "__main__":
    main()
