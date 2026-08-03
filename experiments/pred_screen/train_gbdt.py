"""GBDT 特征回归(sklearn HistGradientBoosting,log1p 目标)。

特征全部来自锚点前:目标圈多尺度窗口计数、六通道 6/24/72h、趋势比、日周期、
事件年龄、活跃成员数、圈指示。train 锚点训练,val 早停参考,test 出预测。
用法: python3 train_gbdt.py
"""
import json
import time

import numpy as np

from common import HORIZONS, OUT, load_anchors, save_preds

WINS = [1, 3, 6, 12, 24, 48, 72, 168]


def build_features():
    import polars as pl
    an = load_anchors()
    feats, meta = [], []
    for (event,), g in an.group_by(["event"], maintain_order=True):
        z = np.load(OUT / "events" / f"{event}.npz")
        counts = z["counts"].astype(float)
        b0 = int(z["b0"])
        p_ts, p_gid = z["p_ts"], z["p_gid"]
        n0 = N0
        cums = np.concatenate([np.zeros((6, 1)), np.cumsum(counts, 1)], 1)
        nb = counts.shape[1]
        for row in g.iter_rows(named=True):
            c, k = row["circle"], row["k"]
            ci = 0 if c == 0 else 1
            f = []
            # 目标圈多尺度
            for w in WINS:
                lo = max(0, k - w)
                f.append(np.log1p(cums[ci, k] - cums[ci, lo]))
            # 六通道 6/24/72
            for ch in range(6):
                for w in (6, 24, 72):
                    lo = max(0, k - w)
                    f.append(np.log1p(cums[ch, k] - cums[ch, lo]))
            # 趋势:近24 vs 前24;近6 vs 前6
            for w in (6, 24):
                a = cums[ci, k] - cums[ci, max(0, k - w)]
                b = cums[ci, max(0, k - w)] - cums[ci, max(0, k - 2 * w)]
                f.append(np.log1p(a) - np.log1p(b))
            # 全通道趋势
            tot = cums[:, k].sum() - cums[:, max(0, k - 24)].sum()
            tot_prev = cums[:, max(0, k - 24)].sum() - cums[:, max(0, k - 48)].sum()
            f.append(np.log1p(tot) - np.log1p(tot_prev))
            # 日周期(北京时)与星期
            t_anchor = (b0 + k) * 3600
            hod = (t_anchor / 3600 + 8) % 24
            dow = ((t_anchor + 8 * 3600) // 86400 + 4) % 7
            f += [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
                  np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]
            # 事件年龄 / 全事件累计量
            f += [k / 24.0, np.log1p(cums[:, k].sum())]
            # 圈活跃成员数(近24/72h)
            in_c = (p_gid < n0) if c == 0 else (p_gid >= n0)
            for w in (24, 72):
                m = (p_ts < t_anchor) & (p_ts >= t_anchor - w * 3600) & in_c
                f.append(np.log1p(len(np.unique(p_gid[m]))))
            # 圈指示
            f.append(1.0 if c == 5 else 0.0)
            feats.append(f)
            meta.append((event, c, k, row["split"], row["y6"], row["y24"]))
    X = np.array(feats, np.float32)
    return X, meta


N0 = None


def main():
    global N0
    t0 = time.time()
    N0 = int(np.load(OUT / "members.npz")["n0"])
    X, meta = build_features()
    split = np.array([m[3] for m in meta])
    print(f"features {X.shape} built {time.time()-t0:.0f}s "
          f"(train={int((split=='train').sum())} test={int((split=='test').sum())})", flush=True)

    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = []
    cost = {}
    for hname, H in HORIZONS.items():
        y = np.array([m[4] if H == 6 else m[5] for m in meta], np.float32)
        tr, va, te = split == "train", split == "val", split == "test"
        t1 = time.time()
        model = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=500, learning_rate=0.06,
            max_depth=None, max_leaf_nodes=63, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=False, random_state=7)
        model.fit(X[tr], np.log1p(y[tr]))
        cost[hname] = round(time.time() - t1, 1)
        for mask, sp in ((va, "val"), (te, "test")):
            yh = np.expm1(model.predict(X[mask]))
            for (event, c, k, _, y6, y24), p in zip(
                    [m for m, mm in zip(meta, mask) if mm], yh):
                rows.append({"event": event, "circle": c, "k": k, "horizon": hname,
                             "y": y6 if H == 6 else y24, "yhat": float(max(0, p)),
                             "split": sp})
    save_preds("gbdt", rows)
    (OUT / "preds" / "gbdt_meta.json").write_text(json.dumps(
        {"train_cost_s": cost, "n_features": X.shape[1]}))
    print("cost:", cost)


if __name__ == "__main__":
    main()
