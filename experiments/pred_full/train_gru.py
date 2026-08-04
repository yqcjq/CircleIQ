"""GRU 序列回归(全圈版):输入锚点前 168h 六视图 log1p + 时间特征,静态特征
[log圈规模/10, 事件年龄],双头输出 log1p(y6)/log1p(y24)。

内存控制:X 用 float16 存(log1p 后动态范围小,精度足够),训练时转 float32。
用法: python3 train_gru.py [--partition main|pre] [--epochs 30]
"""
import argparse
import json
import time

import numpy as np

from common import load_anchors, load_panel, out_dir, save_preds, six_view

CTX = 168


def build_tensors(variant):
    import polars as pl
    an = load_anchors(variant)
    panel = load_panel(variant)
    C = panel["C"]
    members = json.loads((out_dir(variant) / "members.json").read_text())
    size_of = {c["circle_id"]: c["size"] for c in members["circles"]}
    OUT = out_dir(variant)
    X, S, Y6, Y24, meta = [], [], [], [], []
    for (event,), g in an.group_by(["event"], maintain_order=True):
        z = np.load(OUT / "events" / f"{event}.npz")
        counts = z["counts"]
        b0 = int(z["b0"])
        views = {}
        for cidx in g["cidx"].unique().to_list():
            views[cidx] = six_view(counts, cidx, C)
        for row in g.iter_rows(named=True):
            c, cidx, k = row["circle"], row["cidx"], row["k"]
            v = views[cidx]
            lo = k - CTX
            if lo >= 0:
                seq = v[:, lo:k]
            else:
                seq = np.concatenate([np.zeros((6, -lo)), v[:, :k]], 1)
            seq = np.log1p(seq)
            hrs = (b0 + np.arange(k - CTX, k)) % 24
            tfeat = np.stack([np.sin(2 * np.pi * hrs / 24),
                              np.cos(2 * np.pi * hrs / 24)])
            X.append(np.concatenate([seq, tfeat], 0).T.astype(np.float16))
            S.append([np.log1p(size_of[c]) / 10.0, k / (24 * 90)])
            Y6.append(row["y6"])
            Y24.append(row["y24"])
            meta.append((event, c, k, row["split"]))
    return (np.stack(X), np.array(S, np.float32),
            np.array(Y6, np.float32), np.array(Y24, np.float32), meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="main", choices=["main", "pre"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--weight-loss", action="store_true",
                    help="样本权重 1+log1p(y24):对抗零膨胀主导,攻爆发层")
    args = ap.parse_args()
    mname = "gru_wl" if args.weight_loss else "gru"
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    X, S, Y6, Y24, meta = build_tensors(args.partition)
    split = np.array([m[3] for m in meta])
    print(f"tensors {X.shape} ({X.nbytes/1e9:.1f}GB fp16) built {time.time()-t0:.0f}s", flush=True)

    class GRUReg(nn.Module):
        def __init__(self, nin=8, nh=64, nstatic=2):
            super().__init__()
            self.gru = nn.GRU(nin, nh, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(nh + nstatic, nh), nn.ReLU(), nn.Linear(nh, 2))

        def forward(self, x, s):
            _, h = self.gru(x)
            return self.head(torch.cat([h[-1], s], 1))

    torch.manual_seed(7)
    model = GRUReg(nh=args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xt = torch.tensor(X)          # fp16, CPU 常驻
    St = torch.tensor(S)
    Yt = torch.tensor(np.stack([np.log1p(Y6), np.log1p(Y24)], 1))
    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    te = np.where(split == "test")[0]
    Wt = torch.tensor(1.0 + np.log1p(Y24), dtype=torch.float32) \
        if args.weight_loss else None

    def run_eval(idx):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 8192):
                b = idx[i:i + 8192]
                outs.append(model(Xt[b].to(dev).float(), St[b].to(dev)).cpu())
        return torch.cat(outs).numpy()

    best_val, best_state, t_train = 9e9, None, time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            out = model(Xt[b].to(dev).float(), St[b].to(dev))
            if Wt is None:
                loss = nn.functional.huber_loss(out, Yt[b].to(dev))
            else:
                l = nn.functional.huber_loss(out, Yt[b].to(dev), reduction="none")
                w = Wt[b].to(dev)
                loss = (l.mean(1) * w).sum() / w.sum()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        pv = run_eval(va)
        vmale = float(np.abs(pv - Yt[va].numpy()).mean())
        if vmale < best_val:
            best_val = vmale
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep{ep} train_loss={tot/len(perm):.4f} val_MALE(log)={vmale:.4f} "
                  f"[{time.time()-t_train:.0f}s]", flush=True)
    train_cost = time.time() - t_train
    model.load_state_dict(best_state)

    rows = []
    for sp, idx in (("val", va), ("test", te)):
        p = run_eval(idx)
        for j, i in enumerate(idx):
            event, c, k, _ = meta[i]
            rows.append({"event": event, "circle": c, "k": k, "horizon": "h6",
                         "y": float(Y6[i]), "yhat": float(max(0, np.expm1(p[j, 0]))), "split": sp})
            rows.append({"event": event, "circle": c, "k": k, "horizon": "h24",
                         "y": float(Y24[i]), "yhat": float(max(0, np.expm1(p[j, 1]))), "split": sp})
    save_preds(mname, rows, args.partition)
    (out_dir(args.partition) / "preds" / f"{mname}_meta.json").write_text(json.dumps(
        {"train_cost_s": round(train_cost, 1), "best_val_male_log": round(best_val, 4),
         "params": sum(p.numel() for p in model.parameters())}))
    print(f"train {train_cost:.0f}s best_val {best_val:.4f}")


if __name__ == "__main__":
    main()
