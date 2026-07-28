"""全量预处理驱动:385 zip -> core/text parquet(每 zip 一对)+ 每 zip 统计 JSON。

用法:
  python preprocess.py [--only major|hot|case] [--limit N] [--workers 9]

输出:
  /Users/ppio/Desktop/CircleIQ-data/core/<cat>/<topic>__<zipstem>.parquet
  /Users/ppio/Desktop/CircleIQ-data/text/<cat>/<topic>__<zipstem>.parquet
  /Users/ppio/Desktop/CircleIQ-data/stats/<cat>__<zipstem>.json
幂等:已有 stats json 的 zip 跳过。
"""
import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from common import iter_zip_csvs, process_frame, NULL_MD5  # noqa: E402

import pandas as pd  # noqa: E402

RAW = Path("/Users/ppio/Desktop/第八届传播数据挖掘竞赛")
CATEGORIES = {
    "major": RAW / "重大议题与话题数据",
    "hot": RAW / "2025年热点事件",
    "case": RAW / "2025年传播案例数据",
}
OUT = Path("/Users/ppio/Desktop/CircleIQ-data")
CHUNK = 200_000

S = pa.string()
CORE_SCHEMA = pa.schema([
    ("md5_author", S), ("md5_mid", S), ("md5_parent_mid", S), ("md5_root_mid", S), ("md5_root_uid", S),
    ("ts", pa.int64()), ("is_original", pa.bool_()),
    ("sentiment", S), ("info_attr", S), ("media_type", S), ("source_site", S), ("source_level", S),
    ("auth_tier", S), ("auth_type", S), ("user_type", S), ("actor_l1", S), ("actor_l2", S),
    ("gender", S), ("province", S), ("region_valid", pa.bool_()), ("city", S),
    ("industry", S), ("keywords", S), ("topic_tag", S), ("content_type", S), ("post_type", S),
    ("n_followers", pa.int64()), ("n_follows", pa.int64()), ("n_posts_acct", pa.int64()),
    ("n_repost", pa.int64()), ("n_comment", pa.int64()), ("n_like", pa.int64()), ("n_read", pa.int64()),
    ("reg_time", S), ("birth_year", S), ("interest_tags", S),
    ("root_ts", pa.int64()), ("root_author_name", S), ("author_name", S),
    ("id_source", S), ("csv_file", S), ("topic", S), ("category", S),
])
TEXT_SCHEMA = pa.schema([
    ("md5_mid", S), ("md5_author", S), ("title", S), ("content_full", S), ("origin_content", S),
    ("root_title", S), ("user_bio", S), ("image_text", S), ("verified_info", S), ("url", S),
    ("csv_file", S), ("topic", S), ("category", S),
])


def list_jobs(only=None):
    jobs = []
    for cat, root in CATEGORIES.items():
        if only and cat != only:
            continue
        for p in sorted(root.iterdir()):
            if p.suffix == ".zip":
                jobs.append((cat, p.stem, str(p)))
            elif p.is_dir():
                for q in sorted(p.glob("*.zip")):
                    jobs.append((cat, p.name, str(q)))
    # LPT:大 zip 先跑;major 类别优先
    jobs.sort(key=lambda j: (j[0] != "major", -Path(j[2]).stat().st_size))
    return jobs


def process_zip(job):
    cat, topic, zp = job
    stem = Path(zp).stem
    core_path = OUT / "core" / cat / f"{topic}__{stem}.parquet"
    text_path = OUT / "text" / cat / f"{topic}__{stem}.parquet"
    stat_path = OUT / "stats" / f"{cat}__{topic}__{stem}.json"
    if stat_path.exists():
        return f"skip {topic}/{stem}"
    core_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    stat_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    stats = {
        "category": cat, "topic": topic, "zip": stem, "n_csv": 0, "rows": 0, "bad_ts": 0,
        "dup_mid_dropped": 0, "sentinel_root_mid": 0, "null_parent": 0, "reposts": 0,
        "ts_min": None, "ts_max": None, "sentiment_filled": 0, "md5_computed": False,
    }
    seen_mids = set()
    users = set()
    cw = pq.ParquetWriter(core_path, CORE_SCHEMA, compression="zstd", compression_level=3)
    tw = pq.ParquetWriter(text_path, TEXT_SCHEMA, compression="zstd", compression_level=7)
    try:
        for csv_name, stream in iter_zip_csvs(zp):
            stats["n_csv"] += 1
            for raw in pd.read_csv(stream, dtype=str, engine="c", keep_default_na=False,
                                   na_values=[], on_bad_lines="skip", chunksize=CHUNK):
                core, text = process_frame(raw, csv_name)
                del raw
                # zip 内去重(重复抓取):md5_mid 已见过的丢弃
                mids = core["md5_mid"]
                dup = mids.notna() & (mids.isin(seen_mids) | mids.duplicated())
                if dup.any():
                    stats["dup_mid_dropped"] += int(dup.sum())
                    core, text = core[~dup.values], text[~dup.values]
                seen_mids.update(mids.dropna())
                core = core.assign(topic=topic, category=cat)
                text = text.assign(topic=topic, category=cat)

                stats["rows"] += len(core)
                stats["bad_ts"] += int(core["ts"].isna().sum())
                stats["sentinel_root_mid"] += int(core["md5_root_mid"].isna().sum())
                stats["reposts"] += int((~core["is_original"]).sum())
                stats["null_parent"] += int(core["md5_parent_mid"].isna().sum())
                stats["sentiment_filled"] += int(core["sentiment"].notna().sum())
                if (core["id_source"] != "given").any():
                    stats["md5_computed"] = True
                users.update(core["md5_author"].dropna())
                tsx = core["ts"].dropna()
                if len(tsx):
                    mn, mx = int(tsx.min()), int(tsx.max())
                    stats["ts_min"] = mn if stats["ts_min"] is None else min(stats["ts_min"], mn)
                    stats["ts_max"] = mx if stats["ts_max"] is None else max(stats["ts_max"], mx)

                cw.write_table(pa.Table.from_pandas(core, schema=CORE_SCHEMA, preserve_index=False))
                tw.write_table(pa.Table.from_pandas(text, schema=TEXT_SCHEMA, preserve_index=False))
    finally:
        cw.close()
        tw.close()
    stats["n_users"] = len(users)
    stats["secs"] = round(time.time() - t0, 1)
    stats["core_mb"] = round(core_path.stat().st_size / 2**20, 1)
    stats["text_mb"] = round(text_path.stat().st_size / 2**20, 1)
    with open(stat_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False)
    return (f"done {cat}/{topic}/{stem}: rows={stats['rows']:,} users={stats['n_users']:,} "
            f"core={stats['core_mb']}MB text={stats['text_mb']}MB {stats['secs']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["major", "hot", "case"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    jobs = list_jobs(args.only)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"jobs: {len(jobs)}", flush=True)
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_zip, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            n_done += 1
            try:
                msg = fut.result()
            except Exception:
                msg = f"ERROR {j[1]}/{Path(j[2]).stem}\n{traceback.format_exc()}"
            print(f"[{n_done}/{len(jobs)} {time.time()-t0:.0f}s] {msg}", flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
