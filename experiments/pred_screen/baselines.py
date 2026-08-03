"""无参/统计基线:naive-24h、季节naive(24h前同长窗)、EWMA、Poisson外推。
全部只用锚点前历史。对 h6/h24 双视界出预测。
用法: python3 baselines.py
"""
import time

import numpy as np

from common import HORIZONS, OUT, load_anchors, load_event, save_preds


def main():
    an = load_anchors()
    t0 = time.time()
    preds = {m: [] for m in ("naive", "snaive", "ewma", "poisson")}
    for (event,), g in an.group_by(["event"], maintain_order=True):
        z = load_event(event)
        counts = z["counts"]
        ci_map = {0: 0, 5: 1}
        cums = {c: np.concatenate([[0], np.cumsum(counts[i])]) for c, i in ci_map.items()}
        nb = counts.shape[1]
        for row in g.iter_rows(named=True):
            c, k = row["circle"], row["k"]
            cc = cums[c]
            for hname, H in HORIZONS.items():
                y = row[f"y{H}"]
                base = {"event": event, "circle": c, "k": k, "horizon": hname, "y": y}
                # naive: 上一个等长窗
                lo = max(0, k - H)
                naive = (cc[k] - cc[lo]) * (H / max(1, k - lo))
                preds["naive"].append({**base, "yhat": float(naive)})
                # seasonal naive: 24h 前同长窗(捕捉日周期)
                s_lo, s_hi = k - 24, k - 24 + H
                if s_lo >= 0 and s_hi <= k:
                    sn = float(cc[s_hi] - cc[s_lo])
                else:
                    sn = float(naive)
                preds["snaive"].append({**base, "yhat": sn})
                # EWMA 速率(半衰 12h)外推
                w = 0.5 ** (1 / 12)
                hist = np.diff(cc[: k + 1])
                ww = w ** np.arange(len(hist) - 1, -1, -1)
                rate = float((hist * ww).sum() / ww.sum())
                preds["ewma"].append({**base, "yhat": rate * H})
                # Poisson: 全历史平均速率
                preds["poisson"].append({**base, "yhat": float(cc[k] / k * H)})
    for m, rows in preds.items():
        save_preds(m, rows)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
