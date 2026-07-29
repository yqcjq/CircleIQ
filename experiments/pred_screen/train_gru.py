"""GRU 序列回归(神经族基线):输入锚点前 168h 六通道 log1p 计数 + 时间特征,
GRU 编码,双头输出 log1p(y6)/log1p(y24)。GPU 训练。
用法: python3 train_gru.py [--epochs 30]
"""
import argparse
import json
import time

import numpy as np

from common import OUT, load_anchors, save_preds

CTX = 168  # 输入窗口(小时)


def build_tensors():
    an = load_anchors()
    X, T, Y6, Y24, meta = [], [], [], [], []
    for (event,), g in an.group_by(["event"], maintain_order=True):
        z = np.load(OUT / "events" / f"{event}.npz")
        counts = z["counts"].astype(np.float32)
        b0 = int(z["b0"])
        nb = counts.shape[1]
        for row in g.iter_rows(named=True):
            c, k = row["circle"], row["k"]
            lo = k - CTX
            if lo >= 0:
                seq = counts[:, lo:k]
            else:
                seq = np.concatenate([np.zeros((6, -lo), np.float32), counts[:, :k]], 1)
            ci = 0 if c == 0 else 1
            # 通道重排:目标圈在第0通道,另一圈在第1 -> 模型共享圈0/圈5样本
            order = [ci, 1 - ci, 2, 3, 4, 5]
            seq = np.log1p(seq[order])
            hrs = (b0 + np.arange(k - CTX, k)) % 24
            tfeat = np.stack([np.sin(2 * np.pi * hrs / 24),
                              np.cos(2 * np.pi * hrs / 24)]).astype(np.float32)
            X.append(np.concatenate([seq, tfeat], 0).T)  # [CTX, 8]
            T.append([1.0 if c == 5 else 0.0, k / (24 * 90)])
            Y6.append(row["y6"])
            Y24.append(row["y24"])
            meta.append((event, c, k, row["split"]))
    return (np.stack(X), np.array(T, np.float32),
            np.array(Y6, np.float32), np.array(Y24, np.float32), meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    X, T, Y6, Y24, meta = build_tensors()
    split = np.array([m[3] for m in meta])
    print(f"tensors {X.shape} built {time.time()-t0:.0f}s", flush=True)

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
    Xt = torch.tensor(X)
    Tt = torch.tensor(T)
    Yt = torch.tensor(np.stack([np.log1p(Y6), np.log1p(Y24)], 1))
    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    te = np.where(split == "test")[0]

    def run_eval(idx):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 4096):
                b = idx[i:i + 4096]
                outs.append(model(Xt[b].to(dev), Tt[b].to(dev)).cpu())
        return torch.cat(outs).numpy()

    best_val, best_state, t_train = 9e9, None, time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            out = model(Xt[b].to(dev), Tt[b].to(dev))
            loss = nn.functional.huber_loss(out, Yt[b].to(dev))
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
            print(f"ep{ep} train_loss={tot/len(perm):.4f} val_MALE(log)={vmale:.4f}", flush=True)
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
    save_preds("gru", rows)
    (OUT / "preds" / "gru_meta.json").write_text(json.dumps(
        {"train_cost_s": round(train_cost, 1), "best_val_male_log": round(best_val, 4),
         "params": sum(p.numel() for p in model.parameters())}))
    print(f"train {train_cost:.0f}s best_val {best_val:.4f}")


if __name__ == "__main__":
    main()
