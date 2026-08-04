"""舆论线(P1-1):圈层情绪构成的预测与基线对比。

任务:给定锚点前历史,预测目标圈在 (t, t+24h] 的情绪构成(6 维份额向量)。
样本:pred_full anchors 中 y24>=5 的锚点(份额在小计数下无意义)。
模型:
  persist  : 上一 24h 窗的圈内情绪份额(平滑 +1 伪计数)
  event    : 锚点前全事件(全通道)累计情绪份额——"圈随事件走"假设
  markov   : 全局 6x6 情绪转移矩阵(train 事件上按小时聚合估计)乘 persist
  gru_sent : GRU 情绪头(输入 168h x [目标圈6情绪+全通道6情绪+2时间],
             softmax 输出 24h 份额,KL 损失)
指标:平均 L1 距离(总变差 x2)、平均 KL、主导情绪命中率。按事件 bootstrap。
用法: python3 sentiment_line.py [--partition main] [--epochs 25]
输出: out/sentiment_results.json + preds_sent/
"""
import argparse
import json
import time

import numpy as np

from common import load_anchors, load_panel, out_dir

EPS = 1.0  # 伪计数


def build_data(variant):
    import polars as pl
    an = load_anchors(variant)
    an = an.filter(pl.col("y24") >= 5)
    panel = load_panel(variant)
    C = panel["C"]
    OUT = out_dir(variant)
    X, Ytar, Yprev, Yevt, meta = [], [], [], [], []
    for (event,), g in an.group_by(["event"], maintain_order=True):
        z = np.load(OUT / "events" / f"{event}.npz")
        scnt = z["scnt"].astype(np.float64)     # [C+4, 6, nb]
        b0 = int(z["b0"])
        nb = scnt.shape[2]
        cums = {}
        all_c = scnt.sum(0)                     # [6, nb] 全通道
        call = np.concatenate([np.zeros((6, 1)), np.cumsum(all_c, 1)], 1)
        for row in g.iter_rows(named=True):
            cidx, k = row["cidx"], row["k"]
            if cidx not in cums:
                cums[cidx] = np.concatenate(
                    [np.zeros((6, 1)), np.cumsum(scnt[cidx], 1)], 1)
            cc = cums[cidx]
            tar = cc[:, min(k + 24, nb)] - cc[:, k]
            if tar.sum() < 5:
                continue
            prev = cc[:, k] - cc[:, max(0, k - 24)]
            evt = call[:, k]                    # 全事件历史累计
            lo = k - 168
            seq_c = (cc[:, max(0, lo) + 1: k + 1] - cc[:, max(0, lo): k]) \
                if lo >= 0 else np.concatenate(
                    [np.zeros((6, -lo)), cc[:, 1: k + 1] - cc[:, : k]], 1)
            seq_a = (call[:, max(0, lo) + 1: k + 1] - call[:, max(0, lo): k]) \
                if lo >= 0 else np.concatenate(
                    [np.zeros((6, -lo)), call[:, 1: k + 1] - call[:, : k]], 1)
            hrs = (b0 + np.arange(k - 168, k)) % 24
            tfeat = np.stack([np.sin(2 * np.pi * hrs / 24),
                              np.cos(2 * np.pi * hrs / 24)])
            X.append(np.concatenate([np.log1p(seq_c), np.log1p(seq_a), tfeat],
                                    0).T.astype(np.float16))
            Ytar.append(tar)
            Yprev.append(prev)
            Yevt.append(evt)
            meta.append((event, row["circle"], k, row["split"]))
    return (np.stack(X), np.array(Ytar), np.array(Yprev), np.array(Yevt), meta)


def norm(v, eps=EPS):
    v = v + eps
    return v / v.sum(-1, keepdims=True)


def estimate_markov(variant, split_events):
    """全局小时级情绪转移:P[i,j] ~ sum_t n_i(t) n_j(t+1) / 归一。train 事件。"""
    OUT = out_dir(variant)
    M = np.zeros((6, 6))
    for event in split_events:
        z = np.load(OUT / "events" / f"{event}.npz")
        a = z["scnt"].sum(0).astype(np.float64)   # [6, nb]
        sh = norm(a.T, 0.0)  # 不能用伪计数——空小时会拉平
        w = a.sum(0)
        ok = (w[:-1] > 0) & (w[1:] > 0)
        M += (sh[:-1][ok] * w[:-1][ok, None]).T @ sh[1:][ok]
    M = M / np.maximum(M.sum(1, keepdims=True), 1e-12)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="main", choices=["main", "pre"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    t0 = time.time()
    X, Ytar, Yprev, Yevt, meta = build_data(args.partition)
    split = np.array([m[3] for m in meta])
    events = np.array([m[0] for m in meta])
    print(f"samples {X.shape} built {time.time()-t0:.0f}s "
          f"(train={int((split=='train').sum())} test={int((split=='test').sum())})",
          flush=True)

    Ptar = norm(Ytar, 0.0)   # 真实份额(y24>=5 保证非空)
    preds = {"persist": norm(Yprev), "event": norm(Yevt)}
    M = estimate_markov(args.partition,
                        sorted(set(events[split == "train"].tolist())))
    # markov 24 步(小时) ~ M^24
    M24 = np.linalg.matrix_power(M, 24)
    preds["markov"] = norm(Yprev) @ M24
    print("markov M24 diag:", np.round(np.diag(M24), 3).tolist(), flush=True)

    # blend 侧路:alpha*event + (1-alpha)*persist,alpha 在 val 上选
    vm = split == "val"
    best_a, best_l1 = 0.5, 9e9
    for a in np.arange(0, 1.01, 0.1):
        P = a * preds["event"] + (1 - a) * preds["persist"]
        l1 = float(np.abs(P - Ptar)[vm].sum(1).mean())
        if l1 < best_l1:
            best_a, best_l1 = float(a), l1
    preds["blend"] = best_a * preds["event"] + (1 - best_a) * preds["persist"]
    print(f"blend alpha={best_a} val_L1={best_l1:.4f}", flush=True)

    # GRU 情绪头
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class GRUSent(nn.Module):
        def __init__(self, nin=14, nh=64):
            super().__init__()
            self.gru = nn.GRU(nin, nh, batch_first=True)
            self.head = nn.Linear(nh, 6)

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1])

    torch.manual_seed(7)
    model = GRUSent().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xt = torch.tensor(X)
    Tt = torch.tensor(norm(Ytar).astype(np.float32))   # 平滑目标
    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    te = np.where(split == "test")[0]

    def run_eval(idx):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 8192):
                b = idx[i:i + 8192]
                outs.append(torch.softmax(model(Xt[b].to(dev).float()), -1).cpu())
        return torch.cat(outs).numpy()

    best_val, best_state = 9e9, None
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            logp = torch.log_softmax(model(Xt[b].to(dev).float()), -1)
            loss = -(Tt[b].to(dev) * logp).sum(1).mean()
            loss.backward()
            opt.step()
        sched.step()
        pv = run_eval(va)
        l1 = float(np.abs(pv - Ptar[va]).sum(1).mean())
        if l1 < best_val:
            best_val, best_state = l1, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep{ep} val_L1={l1:.4f}", flush=True)
    model.load_state_dict(best_state)
    preds["gru_sent"] = run_eval(np.arange(len(X)))

    # 评估(test)
    def kl(p, q):
        q = np.clip(q, 1e-9, None)
        p = np.clip(p, 1e-9, None)
        return (p * np.log(p / q)).sum(1)

    res = {"n_test": int((split == "test").sum()), "models": {}}
    tm = split == "test"
    uev = np.unique(events[tm])
    rng = np.random.default_rng(7)
    for m, P in preds.items():
        l1 = np.abs(P - Ptar).sum(1)
        klv = kl(Ptar, P)
        hit = (P.argmax(1) == Ptar.argmax(1)).astype(float)
        # 按事件 bootstrap L1 差 vs persist
        res["models"][m] = {
            "L1": round(float(l1[tm].mean()), 4),
            "KL": round(float(klv[tm].mean()), 4),
            "dom_hit": round(float(hit[tm].mean()), 4)}
    l1_all = {m: np.abs(P - Ptar).sum(1) for m, P in preds.items()}
    for m in preds:
        if m == "persist":
            continue
        diffs = []
        per_ev = {e: (float((l1_all[m][tm & (events == e)]).sum()),
                      float((l1_all["persist"][tm & (events == e)]).sum()),
                      int((tm & (events == e)).sum())) for e in uev}
        for _ in range(400):
            samp = rng.choice(uev, size=len(uev), replace=True)
            sm_ = sum(per_ev[e][0] for e in samp)
            sn_ = sum(per_ev[e][1] for e in samp)
            nn_ = sum(per_ev[e][2] for e in samp)
            diffs.append((sm_ - sn_) / nn_)
        lo_, hi_ = np.percentile(diffs, [2.5, 97.5])
        res["models"][m]["boot_vs_persist"] = [round(float(lo_), 4), round(float(hi_), 4)]
    (out_dir(args.partition) / "sentiment_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("SENTIMENT DONE", flush=True)


if __name__ == "__main__":
    main()
