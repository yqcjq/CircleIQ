"""服务器端:全量 core parquet -> 用户级画像表(polars 多核)。

每用户: 主省份/主认证/主行业/主情绪/log粉丝/发帖数/参与议题数(major)/参与事件数。
输出: ~/data/user_profiles.parquet
"""
import time
from pathlib import Path

import polars as pl

DATA = Path.home() / "data"
t0 = time.time()

files = sorted((DATA / "core").rglob("*.parquet"))
print(f"core files: {len(files)}", flush=True)

cols = ["md5_author", "province", "auth_tier", "sentiment", "industry",
        "n_followers", "topic", "category"]
lf = pl.scan_parquet([str(f) for f in files]).select(cols)

prof = lf.group_by("md5_author").agg(
    pl.col("province").drop_nulls().mode().first().alias("province"),
    pl.col("auth_tier").filter(pl.col("auth_tier") != "unknown").mode().first().alias("auth_tier"),
    pl.col("sentiment").drop_nulls().mode().first().alias("dominant_sentiment"),
    pl.col("industry").drop_nulls().mode().first().alias("industry_top"),
    pl.col("n_followers").max().alias("n_followers_max"),
    pl.len().alias("n_posts"),
    pl.col("topic").filter(pl.col("category") == "major").n_unique().alias("n_major_topics"),
    pl.col("topic").n_unique().alias("n_topics"),
).with_columns(
    pl.col("auth_tier").fill_null("unknown"),
    (pl.col("n_followers_max").fill_null(0) + 1).log().alias("log_followers"),
)

prof.collect(engine="streaming").write_parquet(DATA / "user_profiles.parquet", compression="zstd")
out = pl.read_parquet(DATA / "user_profiles.parquet")
print(f"users: {out.height:,}  [{time.time()-t0:.0f}s]")
print(out.group_by("auth_tier").len().sort("len", descending=True))
print("n_major_topics dist:", out["n_major_topics"].value_counts().sort("n_major_topics").to_dicts()[:9])
print("PROFILES DONE", flush=True)
