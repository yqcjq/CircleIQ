"""EvolveGCN-O(自实现):6h 快照序列上用 GRU 演化 GCN 权重,预测圈层未来计数。

按事件构建:节点 = 该事件活跃圈成员(p_gid 出现过)+ 4 聚合外部节点;
每锚点取前 S=28 个 6h 快照(7 天),边 = 快照内转发交互(对称归一);
节点特征 = [是否圈0, 是否圈5, 快照内发帖 log1p, 快照内出/入度 log1p];
EvolveGCN-O:W_t = GRU(W_{t-1}, W_{t-1}),H = ReLU(Â H W_t) 两层;
读出 = 目标圈节点 mean+max 池化 + 六通道计数特征 -> log1p(y6/y24)。
用法: python3 train_evolvegcn.py [--epochs 6]
"""
import argparse
import json
import time

import numpy as np

from common import BIN, OUT, load_anchors, save_preds

SNAP_H = 6      # 快照粒度(小时)
N_SNAP = 28     # 每锚点回看快照数(=7天)
HID = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-cap", type=int, default=6000, help="每 epoch 训练锚点抽样上限")
    args = ap.parse_args()
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)

    mem = np.load(OUT / "members.npz")
    n0, n5 = int(mem["n0"]), int(mem["n5"])
    an = load_anchors()

    # ---------- 事件预处理:活跃节点集 + 交互按快照分箱 ----------
    events = {}
    for (event,), g in an.group_by(["event"], maintain_order=True):
        z = np.load(OUT / "events" / f"{event}.npz")
        b0 = int(z["b0"])
        counts = z["counts"].astype(np.float32)
        p_ts, p_gid = z["p_ts"], z["p_gid"].astype(np.int64)
        r_ts = z["r_ts"]
        r_src = z["r_src"].astype(np.int64)
        r_dst = z["r_dst"].astype(np.int64)
        act = np.unique(p_gid)
        # 本地编号:活跃成员 0..m-1,聚合节点 m..m+3
        loc = {int(gg): i for i, gg in enumerate(act)}
        m = len(act)
        A = m + 4

        def code(x):
            out = np.empty(len(x), np.int64)
            for i, v in enumerate(x):
                if v >= 0:
                    out[i] = loc.get(int(v), m + 3)  # 圈成员但未发帖(仅被转)->归 other 也行
                else:
                    out[i] = m + (-int(v)) - 1
            return out

        e_src, e_dst = code(r_src), code(r_dst)
        e_bin = ((r_ts // BIN).astype(np.int64) - b0) // SNAP_H
        p_bin = ((p_ts // BIN).astype(np.int64) - b0) // SNAP_H
        is0 = (act < n0).astype(np.float32)
        events[event] = {"b0": b0, "counts": counts, "m": m, "A": A,
                         "act": act, "is0": is0,
                         "e_src": e_src, "e_dst": e_dst, "e_bin": e_bin,
                         "p_gid_loc": np.array([loc[int(v)] for v in p_gid], np.int64),
                         "p_bin": p_bin,
                         "rows": [dict(r) for r in g.iter_rows(named=True)]}
    print(f"events: {len(events)}", flush=True)

    class Model(nn.Module):
        def __init__(self, nin=4, nh=HID):
            super().__init__()
            self.nin, self.nh = nin, nh
            self.W1_0 = nn.Parameter(torch.randn(nin, nh) * 0.2)
            self.W2_0 = nn.Parameter(torch.randn(nh, nh) * 0.2)
            self.gru1 = nn.GRUCell(nin * nh, nin * nh)
            self.gru2 = nn.GRUCell(nh * nh, nh * nh)
            self.head = nn.Sequential(
                nn.Linear(2 * nh + 18 + 1, 96), nn.ReLU(), nn.Linear(96, 2))

        def evolve(self, W, gru):
            flat = W.reshape(1, -1)
            return gru(flat, flat).reshape(W.shape)

        def forward(self, snap_list, tgt_idx, cfeat):
            W1, W2 = self.W1_0, self.W2_0
            H2 = None
            for (idx, w, X, A) in snap_list:
                W1 = self.evolve(W1, self.gru1)
                W2 = self.evolve(W2, self.gru2)
                adj = torch.sparse_coo_tensor(idx, w, (A, A))
                H1 = torch.relu(torch.sparse.mm(adj, X @ W1))
                H2 = torch.relu(torch.sparse.mm(adj, H1 @ W2))
            pooled = torch.cat([H2[tgt_idx].mean(0), H2[tgt_idx].max(0).values])
            return self.head(torch.cat([pooled, cfeat])[None])

    model = Model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def build_snaps(evd, k):
        """锚点 k(小时)前 N_SNAP 个 6h 快照。返回 snap_list 与目标圈索引。"""
        kb = k // SNAP_H  # 锚点所在快照(不含)
        A = evd["A"]
        m = evd["m"]
        snaps = []
        for b in range(kb - N_SNAP, kb):
            if b < 0:
                idx = torch.zeros(2, 0, dtype=torch.long, device=dev)
                w = torch.zeros(0, device=dev)
                X = torch.zeros(A, 4, device=dev)
            else:
                em = evd["e_bin"] == b
                s, d = evd["e_src"][em], evd["e_dst"][em]
                # 对称化 + 自环
                si = np.concatenate([s, d, np.arange(A)])
                di = np.concatenate([d, s, np.arange(A)])
                vals = np.ones(len(si), np.float32)
                deg = np.bincount(si, weights=vals, minlength=A)
                norm = 1.0 / np.sqrt(np.maximum(deg, 1))
                wv = (vals * norm[si] * norm[di]).astype(np.float32)
                idx = torch.tensor(np.stack([si, di]), dtype=torch.long, device=dev)
                w = torch.tensor(wv, device=dev)
                pm = evd["p_bin"] == b
                posts = np.bincount(evd["p_gid_loc"][pm], minlength=A).astype(np.float32)
                outd = np.bincount(s, minlength=A).astype(np.float32)
                X = torch.tensor(np.stack([
                    np.concatenate([evd["is0"], np.zeros(4, np.float32)]),
                    np.concatenate([1 - evd["is0"], np.zeros(4, np.float32)]),
                    np.log1p(posts), np.log1p(outd)], 1), device=dev)
            snaps.append((idx, w, X, A))
        return snaps

    def cfeats(evd, k, c):
        counts = evd["counts"]
        ci = 0 if c == 0 else 1
        order = [ci, 1 - ci, 2, 3, 4, 5]
        cums = np.concatenate([np.zeros((6, 1), np.float32), np.cumsum(counts, 1)], 1)
        f = []
        for w in (6, 24, 72):
            lo = max(0, k - w)
            f += [float(np.log1p(cums[ch, k] - cums[ch, lo])) for ch in order]
        f.append(1.0 if c == 5 else 0.0)
        return torch.tensor(f, device=dev)

    def tgt_index(evd, c):
        mask = evd["is0"] > 0.5 if c == 0 else evd["is0"] < 0.5
        idx = np.where(mask)[0]
        if len(idx) == 0:
            idx = np.array([evd["m"] + 3])
        return torch.tensor(idx, dtype=torch.long, device=dev)

    # 样本表
    samples = {"train": [], "val": [], "test": []}
    for e, evd in events.items():
        for r in evd["rows"]:
            samples[r["split"]].append((e, r["circle"], r["k"], r["y6"], r["y24"]))
    for sp, ss in samples.items():
        print(f"{sp}: {len(ss)} samples", flush=True)

    def predict(e, c, k):
        evd = events[e]
        snaps = build_snaps(evd, k)
        return model(snaps, tgt_index(evd, c), cfeats(evd, k, c))

    t_train = time.time()
    best_val, best_state = 9e9, None
    rng = np.random.default_rng(7)
    for ep in range(args.epochs):
        model.train()
        idxs = rng.permutation(len(samples["train"]))[: args.train_cap]
        tot = 0.0
        for step, si in enumerate(idxs):
            e, c, k, y6, y24 = samples["train"][si]
            opt.zero_grad()
            out = predict(e, c, k)
            target = torch.tensor([[np.log1p(y6), np.log1p(y24)]],
                                  dtype=torch.float32, device=dev)
            loss = nn.functional.huber_loss(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            if step % 1000 == 0:
                print(f"  ep{ep} step{step}/{len(idxs)} loss={tot/max(1,step+1):.4f} "
                      f"({time.time()-t_train:.0f}s)", flush=True)
        model.eval()
        errs = []
        with torch.no_grad():
            for e, c, k, y6, y24 in samples["val"]:
                out = predict(e, c, k)
                errs.append(abs(float(out[0, 1]) - np.log1p(y24)))
        vmale = float(np.mean(errs))
        print(f"ep{ep} val_MALE24(log)={vmale:.4f} ({time.time()-t_train:.0f}s)", flush=True)
        if vmale < best_val:
            best_val = vmale
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_cost = time.time() - t_train

    model.eval()
    rows = []
    with torch.no_grad():
        for sp in ("val", "test"):
            for e, c, k, y6, y24 in samples[sp]:
                out = predict(e, c, k)
                rows.append({"event": e, "circle": c, "k": k, "horizon": "h6",
                             "y": y6, "yhat": float(max(0, np.expm1(float(out[0, 0])))),
                             "split": sp})
                rows.append({"event": e, "circle": c, "k": k, "horizon": "h24",
                             "y": y24, "yhat": float(max(0, np.expm1(float(out[0, 1])))),
                             "split": sp})
    save_preds("evolvegcn", rows)
    (OUT / "preds" / "evolvegcn_meta.json").write_text(json.dumps(
        {"train_cost_s": round(train_cost, 1), "best_val_male24_log": round(best_val, 4),
         "params": sum(p.numel() for p in model.parameters())}))
    print(f"done, train {train_cost:.0f}s")


if __name__ == "__main__":
    main()
