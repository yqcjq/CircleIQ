"""本地:为稳定圈用户抽取聚合文本(供服务器 BGE 编码)。

输入: 服务器下载的 partition_stable_K{K}.parquet(先 scp 到本地)
     + 本地 text/major/*.parquet
输出: user_text_stable.parquet (md5_author, text[<=1500字], n_posts_text) -> 上传服务器
用法: python extract_user_text.py --partition /tmp/partition_stable_K3.parquet
"""
import argparse
from pathlib import Path

import pandas as pd

DATA = Path("/Users/ppio/Desktop/CircleIQ-data")
CAP = 1500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--out", default=str(DATA / "user_text_stable.parquet"))
    args = ap.parse_args()

    users = set(pd.read_parquet(args.partition)["md5_author"])
    print(f"stable users: {len(users):,}")

    acc = {}  # md5_author -> [texts]
    files = sorted((DATA / "text" / "major").glob("*.parquet"))
    for f in files:
        df = pd.read_parquet(f, columns=["md5_author", "title", "content_full"])
        df = df[df.md5_author.isin(users)]
        txt = df.content_full.fillna(df.title)
        for a, t in zip(df.md5_author.values, txt.values):
            if t is None or not t:
                continue
            cur = acc.setdefault(a, ["", 0])
            if len(cur[0]) < CAP:
                cur[0] = (cur[0] + " " + str(t))[:CAP]
            cur[1] += 1
        print(f"{f.name}: 累计 {len(acc):,} 用户有文本")
    out = pd.DataFrame(
        {"md5_author": list(acc), "text": [v[0].strip() for v in acc.values()],
         "n_posts_text": [v[1] for v in acc.values()]})
    out.to_parquet(args.out, compression="zstd", index=False)
    print(f"saved {args.out}: {len(out):,} users")


if __name__ == "__main__":
    main()
