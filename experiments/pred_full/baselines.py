"""统计基线:naive、季节naive、EWMA、Poisson外推(与 pred_screen 同口径,扩到全圈)。
用法: python3 baselines.py [--partition main|pre]
"""
import argparse
import time

import numpy as np

from common import HORIZONS, load_anchors, load_event, save_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="main", choices=["main", "pre"])
    args = ap.parse_args()
    an = load_anchors(args.partition)
    t0 = time.time()
    preds = {m: [] for m in ("naive", "snaive", "ewma", "poisson")}
    w = 0.5 ** (1 / 12)
    for (event,), g in an.group_by(["event"], maintain_order=True):
        z = load_event(event, args.partition)
        counts = z["counts"]
        nb = counts.shape[1]
        cums = {}
        for cidx in g["cidx"].unique().to_list():
            cums[cidx] = np.concatenate([[0], np.cumsum(counts[cidx])])
        for row in g.iter_rows(named=True):
            c, cidx, k = row["circle"], row["cidx"], row["k"]
            cc = cums[cidx]
            for hname, H in HORIZONS.items():
                y = row[f"y{H}"]
                base = {"event": event, "circle": c, "k": k, "horizon": hname,
                        "y": y, "split": row["split"]}
                lo = max(0, k - H)
                naive = (cc[k] - cc[lo]) * (H / max(1, k - lo))
                preds["naive"].append({**base, "yhat": float(naive)})
                s_lo, s_hi = k - 24, k - 24 + H
                sn = float(cc[s_hi] - cc[s_lo]) if s_lo >= 0 and s_hi <= k else float(naive)
                preds["snaive"].append({**base, "yhat": sn})
                hist = np.diff(cc[: k + 1])
                ww = w ** np.arange(len(hist) - 1, -1, -1)
                rate = float((hist * ww).sum() / ww.sum())
                preds["ewma"].append({**base, "yhat": rate * H})
                preds["poisson"].append({**base, "yhat": float(cc[k] / k * H)})
    for m, rows in preds.items():
        save_preds(m, rows, args.partition)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
