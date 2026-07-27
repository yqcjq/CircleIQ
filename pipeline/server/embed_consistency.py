"""服务器:BGE 编码稳定用户文本 + 圈内内容一致性评估(4090 GPU)。

一致性口径与单议题实验(b2_consistency)一致:
  圈内一致性 = 圈内随机抽 pair 的平均余弦;基线 = 全体随机 pair 平均余弦
输出: ~/data/content_consistency_stable.json
用法: python3 embed_consistency.py --partition partition_stable_K3.parquet [--model BAAI/bge-small-zh-v1.5]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

DATA = Path.home() / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="partition_stable_K3.parquet")
    ap.add_argument("--text", default="user_text_stable.parquet")
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--min-circle", type=int, default=30)
    ap.add_argument("--pairs-per-circle", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    t0 = time.time()

    import polars as pl
    import torch
    from transformers import AutoModel, AutoTokenizer

    txt = pl.read_parquet(DATA / args.text)
    part = pl.read_parquet(DATA / args.partition)
    df = txt.join(part, on="md5_author", how="inner").filter(pl.col("text").str.len_chars() >= 10)
    print(f"users with text: {df.height:,}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).cuda().eval().half()
    texts = df["text"].to_list()
    embs = np.zeros((len(texts), 512), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            b = tok(texts[i:i + args.batch], padding=True, truncation=True, max_length=256,
                    return_tensors="pt").to("cuda")
            out = model(**b)[0][:, 0]  # CLS
            out = torch.nn.functional.normalize(out, dim=-1)
            embs[i:i + args.batch] = out.float().cpu().numpy()
            if i % (args.batch * 40) == 0:
                print(f"  encoded {i}/{len(texts)} [{time.time()-t0:.0f}s]", flush=True)
    np.save(DATA / "user_text_emb_stable.npy", embs)

    rng = np.random.default_rng(0)
    cids = df["circle_id"].to_numpy()
    report = {"n_users": len(texts), "model": args.model}

    # 基线:全体随机 pair
    idx = rng.choice(len(texts), size=(20000, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    report["random_baseline"] = float((embs[idx[:, 0]] * embs[idx[:, 1]]).sum(1).mean())

    per_circle = []
    for cid in np.unique(cids):
        members = np.where(cids == cid)[0]
        if len(members) < args.min_circle:
            continue
        p = rng.choice(members, size=(args.pairs_per_circle, 2))
        p = p[p[:, 0] != p[:, 1]]
        per_circle.append({"circle_id": int(cid), "size": int(len(members)),
                           "avg_cos": float((embs[p[:, 0]] * embs[p[:, 1]]).sum(1).mean())})
    per_circle.sort(key=lambda r: -r["size"])
    sizes = np.array([r["size"] for r in per_circle])
    cons = np.array([r["avg_cos"] for r in per_circle])
    report["n_circles_evaluated"] = len(per_circle)
    report["weighted_avg_cos"] = float((cons * sizes).sum() / sizes.sum()) if len(sizes) else None
    report["unweighted_avg_cos"] = float(cons.mean()) if len(sizes) else None
    report["per_circle_top30"] = per_circle[:30]
    report["secs"] = round(time.time() - t0, 1)
    with open(DATA / "content_consistency_stable.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "per_circle_top30"}, indent=1))
    print("CONSISTENCY DONE", flush=True)


if __name__ == "__main__":
    main()
