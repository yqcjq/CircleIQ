"""TGN-lite:记忆式时间图网络(自实现,单文件)。

设计(忠于 TGN 思想、按本任务裁剪):
- 节点 = 圈0∪圈5 成员(14,602)+ 4 个聚合外部节点(stab/bigv/org/other)
- 事件流 = 转发交互(src 转发 dst);消息 = [src_mem, dst_mem, Δt编码, 方向]
- 记忆更新 = GRUCell;时间编码 = Time2Vec 余弦
- 预测头:锚点时刻聚合每圈成员记忆(mean+max)+ 六通道近期计数特征 -> log1p(y6),log1p(y24)
- 训练:按事件时序回放交互流,在锚点处取快照算损失;test 事件只回放不更新梯度
成本注记:全成员记忆矩阵 14606×d,单事件回放 O(n_interactions)。
用法: python3 train_tgn.py [--epochs 8]
"""
import argparse
import json
import time

import numpy as np

from common import BIN, OUT, load_anchors, save_preds

MEM_DIM = 64
MSG_DIM = 64
N_AGG = 4  # -1..-4 聚合节点


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)

    mem = np.load(OUT / "members.npz")
    n0, n5 = int(mem["n0"]), int(mem["n5"])
    N = n0 + n5 + N_AGG  # 聚合节点排在成员后
    an = load_anchors()

    # 预载事件(交互流 + 锚点 + 计数特征)
    import polars as pl
    events = {}
    for (event,), g in an.group_by(["event"], maintain_order=True):
        z = np.load(OUT / "events" / f"{event}.npz")
        r_ts, r_src, r_dst = z["r_ts"], z["r_src"].astype(np.int64), z["r_dst"].astype(np.int64)
        # 编码变换:负数聚合节点 -> n0+n5 + (|x|-1)
        r_src = np.where(r_src >= 0, r_src, n0 + n5 - r_src - 1)
        r_dst = np.where(r_dst >= 0, r_dst, n0 + n5 - r_dst - 1)
        anchors = sorted(set(zip(g["k"].to_list(), g["split"].to_list())))
        rows = {(r["circle"], r["k"]): (r["y6"], r["y24"], r["split"])
                for r in g.iter_rows(named=True)}
        circles = sorted(set(r["circle"] for r in g.iter_rows(named=True)))
        events[event] = {"r_ts": r_ts, "r_src": r_src, "r_dst": r_dst,
                         "b0": int(z["b0"]), "counts": z["counts"].astype(np.float32),
                         "anchors": anchors, "rows": rows, "circles": circles,
                         "split": g["split"][0]}
    ev_names = sorted(events)
    print(f"events loaded: {len(ev_names)}, nodes={N}", flush=True)

    class Time2Vec(nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.w = nn.Parameter(torch.randn(d))
            self.b = nn.Parameter(torch.zeros(d))

        def forward(self, dt):  # dt [B] 秒,log 压缩
            x = torch.log1p(dt.clamp(min=0)) / 12.0
            return torch.cos(x[:, None] * self.w + self.b)

    class TGN(nn.Module):
        def __init__(self):
            super().__init__()
            self.t2v = Time2Vec(8)
            self.msg = nn.Sequential(nn.Linear(2 * MEM_DIM + 8 + 2, MSG_DIM), nn.ReLU())
            self.upd = nn.GRUCell(MSG_DIM, MEM_DIM)
            self.head = nn.Sequential(
                nn.Linear(2 * MEM_DIM + 18 + 1, 96), nn.ReLU(), nn.Linear(96, 2))

        def forward_batch(self, memory, last_t, src, dst, t):
            """处理同一小批交互:src 转发 dst(消息双向)。"""
            dt_s = t - last_t[src]
            dt_d = t - last_t[dst]
            m_s = self.msg(torch.cat([memory[src], memory[dst], self.t2v(dt_s),
                                      torch.ones(len(src), 2, device=memory.device)], 1))
            m_d = self.msg(torch.cat([memory[dst], memory[src], self.t2v(dt_d),
                                      torch.zeros(len(dst), 2, device=memory.device)], 1))
            nodes = torch.cat([src, dst])
            msgs = torch.cat([m_s, m_d])
            # 同节点多消息取平均(scatter mean)
            uniq, inv = torch.unique(nodes, return_inverse=True)
            agg = torch.zeros(len(uniq), MSG_DIM, device=memory.device)
            cnt = torch.zeros(len(uniq), 1, device=memory.device)
            agg.index_add_(0, inv, msgs)
            cnt.index_add_(0, inv, torch.ones(len(nodes), 1, device=memory.device))
            agg = agg / cnt
            memory = memory.clone()
            memory[uniq] = self.upd(agg, memory[uniq])
            last_t = last_t.clone()
            last_t[uniq] = t
            return memory, last_t

    model = TGN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def circle_slice(c):
        return (0, n0) if c == 0 else (n0, n0 + n5)

    def count_feats(evd, k, c):
        """18 维:六通道 6/24/72h log1p 计数,目标圈通道换到首位。"""
        counts = evd["counts"]
        ci = 0 if c == 0 else 1
        order = [ci, 1 - ci, 2, 3, 4, 5]
        cums = np.concatenate([np.zeros((6, 1), np.float32), np.cumsum(counts, 1)], 1)
        f = []
        for w in (6, 24, 72):
            lo = max(0, k - w)
            f += [float(np.log1p(cums[ch, k] - cums[ch, lo])) for ch in order]
        return f

    def run_event(event, train_mode):
        """回放一个事件的交互流,在锚点处出预测;返回损失项与预测行。
        时间用事件内相对秒(float32 下 Unix 绝对秒粒度 ~128s,会毁掉 Δt)。"""
        evd = events[event]
        memory = torch.zeros(N, MEM_DIM, device=dev)
        last_t = torch.zeros(N, device=dev)
        t_base = float(evd["b0"]) * BIN
        r_ts, r_src, r_dst = evd["r_ts"], evd["r_src"], evd["r_dst"]
        b0 = evd["b0"]
        losses, preds = [], []
        ai = 0
        anchors = evd["anchors"]
        n = len(r_ts)
        i = 0
        BATCH = 512
        detach_every = 2048
        seen = 0
        while ai < len(anchors):
            t_anchor = (b0 + anchors[ai][0]) * BIN
            # 回放到锚点前
            j = i
            while j < min(i + BATCH, n) and r_ts[j] < t_anchor:
                j += 1
            if j > i:
                t_mid = float(r_ts[j - 1]) - t_base
                src = torch.tensor(r_src[i:j], device=dev)
                dst = torch.tensor(r_dst[i:j], device=dev)
                memory, last_t = model.forward_batch(memory, last_t,
                                                     src, dst, torch.tensor(t_mid, device=dev))
                seen += j - i
                if seen >= detach_every:
                    memory = memory.detach()
                    seen = 0
                i = j
                continue
            # 到达锚点:所有相关圈出预测
            k, sp = anchors[ai]
            for c in evd["circles"]:
                key = (c, k)
                if key not in evd["rows"]:
                    continue
                y6, y24, _ = evd["rows"][key]
                lo, hi = circle_slice(c)
                mseg = memory[lo:hi]
                pooled = torch.cat([mseg.mean(0), mseg.max(0).values])
                cf = count_feats(evd, k, c)
                feat = torch.cat([pooled,
                                  torch.tensor(cf, device=dev, dtype=torch.float32),
                                  torch.tensor([1.0 if c == 5 else 0.0], device=dev)])
                out = model.head(feat[None])
                target = torch.tensor([[np.log1p(y6), np.log1p(y24)]],
                                      dtype=torch.float32, device=dev)
                if train_mode:
                    losses.append(nn.functional.huber_loss(out, target))
                else:
                    preds.append({"event": event, "circle": c, "k": int(k),
                                  "p6": float(out[0, 0]), "p24": float(out[0, 1]),
                                  "y6": float(y6), "y24": float(y24), "split": sp})
            ai += 1
        return losses, preds

    tr_events = [e for e in ev_names if events[e]["split"] == "train"]
    ev_eval = [e for e in ev_names if events[e]["split"] in ("val", "test")]
    print(f"train events {len(tr_events)}, eval events {len(ev_eval)}", flush=True)

    t_train = time.time()
    best_val, best_state = 9e9, None
    for ep in range(args.epochs):
        model.train()
        np.random.shuffle(tr_events)
        tot, nl = 0.0, 0
        for e in tr_events:
            opt.zero_grad()
            losses, _ = run_event(e, True)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(losses)
            nl += len(losses)
        # val
        model.eval()
        vp = []
        with torch.no_grad():
            for e in ev_eval:
                _, p = run_event(e, False)
                vp += [x for x in p if x["split"] == "val"]
        if vp:
            vmale = float(np.mean([abs(x["p24"] - np.log1p(x["y24"])) for x in vp]))
        else:
            vmale = 9e9
        print(f"ep{ep} loss={tot/max(1,nl):.4f} val_MALE24(log)={vmale:.4f} "
              f"({time.time()-t_train:.0f}s)", flush=True)
        if vmale < best_val:
            best_val = vmale
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_cost = time.time() - t_train

    model.eval()
    rows = []
    with torch.no_grad():
        for e in ev_eval:
            _, p = run_event(e, False)
            for x in p:
                rows.append({"event": x["event"], "circle": x["circle"], "k": x["k"],
                             "horizon": "h6", "y": x["y6"],
                             "yhat": float(max(0, np.expm1(x["p6"]))), "split": x["split"]})
                rows.append({"event": x["event"], "circle": x["circle"], "k": x["k"],
                             "horizon": "h24", "y": x["y24"],
                             "yhat": float(max(0, np.expm1(x["p24"]))), "split": x["split"]})
    save_preds("tgn", rows)
    (OUT / "preds" / "tgn_meta.json").write_text(json.dumps(
        {"train_cost_s": round(train_cost, 1), "best_val_male24_log": round(best_val, 4),
         "params": sum(p.numel() for p in model.parameters()), "n_nodes": N}))
    print(f"done, train {train_cost:.0f}s")


if __name__ == "__main__":
    main()
