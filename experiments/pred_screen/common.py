"""pred_screen 公共定义:路径、锚点网格、切分、指标。通道定义在 panel.json(builder 产出)。"""
import json
from pathlib import Path

import numpy as np

DATA = Path.home() / "data"
WORK = Path.home() / "circleiq" / "pred_screen"
OUT = WORK / "out"

CIRCLES = (0, 5)
BIN = 3600             # 1h 分箱,对齐 Unix 整点(时间特征真实)
STEP_BINS = 6          # 锚点步长 6h
CTX_MIN_BINS = 24      # 锚点前最少 24h 历史
HORIZONS = {"h6": 6, "h24": 24}
PAIR_MIN_TOTAL = 100   # (事件,圈)对可用门槛:该圈总帖量
CAPS = {"train": 64, "val": 16, "test": 16}   # 每事件锚点数上限(均匀抽)


def event_split(names):
    """按事件确定性切分 60/10/30(md5 排序后交错,类别自然混合)。"""
    import hashlib
    order = sorted(names, key=lambda n: hashlib.md5(n.encode()).hexdigest())
    return {n: ("train" if i % 10 < 6 else ("val" if i % 10 == 6 else "test"))
            for i, n in enumerate(order)}


def subsample(arr, cap):
    arr = np.asarray(arr)
    if len(arr) <= cap:
        return arr
    return arr[np.unique(np.linspace(0, len(arr) - 1, cap).round().astype(int))]


def metrics(y, yhat):
    y = np.asarray(y, float)
    yhat = np.clip(np.asarray(yhat, float), 0, None)
    dlog = np.log1p(yhat) - np.log1p(y)
    ape = np.abs(yhat - y) / np.maximum(1, y)
    out = {"n": int(len(y)),
           "male": float(np.mean(np.abs(dlog))),
           "rmsle": float(np.sqrt(np.mean(dlog ** 2))),
           "med_ape": float(np.median(ape)),
           "bias_log": float(np.mean(dlog))}
    if (y > 0).any() and (y == 0).any():
        from sklearn.metrics import roc_auc_score
        out["auc_nonzero"] = float(roc_auc_score((y > 0).astype(int), np.log1p(yhat)))
    return out


def load_panel():
    return json.loads((OUT / "panel.json").read_text())


def load_event(name):
    z = np.load(OUT / "events" / f"{name}.npz")
    return {k: z[k] for k in z.files}


def load_anchors():
    import polars as pl
    return pl.read_parquet(OUT / "anchors.parquet")


def save_preds(model, rows):
    """rows: dict(event, circle, k, horizon, y, yhat)"""
    import polars as pl
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(OUT / "preds" / f"{model}.parquet")
    print(f"saved {len(rows)} preds -> preds/{model}.parquet", flush=True)
