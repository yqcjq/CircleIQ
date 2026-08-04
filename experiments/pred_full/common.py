"""pred_full 公共定义:路径、常量、切分、指标、圈层类型映射、通道视图。

与 pred_screen 的差异:
- 圈范围从 {0,5} 扩到全部 >=100 人稳定圈(C 个,构建时确定),通道 = C+4
  (0..C-1 top圈按规模降序 / C rest_stable / C+1 bigv / C+2 org / C+3 other)
- 支持 --partition pre(仅 2025-07-01 前互动定义的圈)防泄漏对照,产物在 out_pre/
- 六通道"视图"由 six_view() 从全通道矩阵动态组装,模型间共享
"""
import json
from pathlib import Path

import numpy as np

DATA = Path.home() / "data"
WORK = Path.home() / "circleiq" / "pred_full"

MIN_CIRCLE = 100       # 圈成员数门槛(top 圈定义)
BIN = 3600
STEP_BINS = 6
CTX_MIN_BINS = 24
HORIZONS = {"h6": 6, "h24": 24}
PAIR_MIN_TOTAL = 100
CAPS = {"train": 64, "val": 16, "test": 16}

# 圈层类型学(docs/12,top20 圈;其余归 other_small)——仅主 partition 有意义
CIRCLE_TYPE = {}
for _c in (3, 5, 11, 0, 14, 1, 10):
    CIRCLE_TYPE[_c] = "hub"            # 时政枢纽群
for _c in (18, 12):
    CIRCLE_TYPE[_c] = "org_matrix"     # 机构矩阵
CIRCLE_TYPE[9] = "media_kol"           # 媒体/KOL
for _c in (28, 20):
    CIRCLE_TYPE[_c] = "interest"       # 兴趣圈
for _c in (30, 41, 8):
    CIRCLE_TYPE[_c] = "regional"       # 地域圈
for _c in (26, 2):
    CIRCLE_TYPE[_c] = "organized_risk" # 疑似组织化

SENTS = ["中性", "愤怒", "惊奇", "喜悦", "悲伤", "恐惧"]


def out_dir(variant="main"):
    return WORK / ("out" if variant == "main" else f"out_{variant}")


def event_split(names):
    """与 pred_screen 完全一致:md5 排序交错 60/10/30。"""
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


def load_panel(variant="main"):
    return json.loads((out_dir(variant) / "panel.json").read_text())


def load_event(name, variant="main"):
    z = np.load(out_dir(variant) / "events" / f"{name}.npz")
    return {k: z[k] for k in z.files}


def load_anchors(variant="main"):
    import polars as pl
    return pl.read_parquet(out_dir(variant) / "anchors.parquet")


def six_view(counts, cidx, C):
    """[target, stable_other, bigv, org, other, total] 六视图(float64)。"""
    stable = counts[: C + 1].sum(0)          # top圈 + rest_stable
    target = counts[cidx].astype(np.float64)
    return np.stack([target, stable - target,
                     counts[C + 1], counts[C + 2], counts[C + 3],
                     counts.sum(0)]).astype(np.float64)


def save_preds(model, rows, variant="main"):
    import polars as pl
    d = out_dir(variant) / "preds"
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(d / f"{model}.parquet")
    print(f"saved {len(rows)} preds -> {d.name}/{model}.parquet", flush=True)
