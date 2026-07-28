"""扫描全部原始 zip 的 CSV 表头,盘点 schema 变体。

产出 pipeline/out/schema_scan.json:
  每个 zip -> [{csv, size, n_cols, header_sig}], 以及全局 header 变体去重表。
只读每个 CSV 的第一行(GB18030),不解压整个文件。
"""
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RAW = Path("/Users/ppio/Desktop/第八届传播数据挖掘竞赛")
CATEGORIES = {
    "major": RAW / "重大议题与话题数据",
    "hot": RAW / "2025年热点事件",
    "case": RAW / "2025年传播案例数据",
}
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def list_zips():
    jobs = []  # (category, topic_name, zip_path)
    for cat, root in CATEGORIES.items():
        for p in sorted(root.iterdir()):
            if p.suffix == ".zip":
                jobs.append((cat, p.stem, str(p)))
            elif p.is_dir():  # 生育话题:目录内多 zip
                for q in sorted(p.glob("*.zip")):
                    jobs.append((cat, p.name, str(q)))
    return jobs


def read_header(zf, member):
    with zf.open(member) as f:
        raw = f.read(65536)
    line = raw.split(b"\n", 1)[0].decode("gb18030", errors="replace").strip("\r﻿")
    return next(csv.reader(io.StringIO(line)))


def scan_zip(job):
    cat, topic, zp = job
    rows = []
    try:
        with zipfile.ZipFile(zp) as zf:
            for m in zf.infolist():
                if not m.filename.lower().endswith(".csv") or m.file_size == 0:
                    continue
                try:
                    hdr = read_header(zf, m.filename)
                except Exception as e:  # noqa: BLE001
                    rows.append({"csv": m.filename, "size": m.file_size, "error": str(e)})
                    continue
                rows.append({
                    "csv": Path(m.filename).name,
                    "size": m.file_size,
                    "header": hdr,
                })
    except Exception as e:  # noqa: BLE001
        return {"category": cat, "topic": topic, "zip": zp, "error": str(e), "csvs": []}
    return {"category": cat, "topic": topic, "zip": zp, "csvs": rows}


def main():
    jobs = list_zips()
    print(f"zips to scan: {len(jobs)}", flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(scan_zip, jobs, chunksize=4))

    # header 变体去重
    variants = {}  # sig -> {"header": [...], "n_csvs": int, "example": str}
    total_csv, total_bytes = 0, 0
    for r in results:
        for c in r["csvs"]:
            if "header" not in c:
                continue
            total_csv += 1
            total_bytes += c["size"]
            sig = "|".join(c["header"])
            v = variants.setdefault(sig, {"header": c["header"], "n_csvs": 0,
                                          "example": f'{r["topic"]}/{c["csv"]}',
                                          "categories": Counter()})
            v["n_csvs"] += 1
            v["categories"][r["category"]] += 1
            c["header_sig"] = f"v{list(variants).index(sig)}"
            del c["header"]

    for v in variants.values():
        v["categories"] = dict(v["categories"])

    out = {
        "n_zips": len(jobs),
        "n_csvs": total_csv,
        "total_csv_bytes": total_bytes,
        "variants": [{"sig": f"v{i}", **v} for i, v in enumerate(variants.values())],
        "zips": results,
    }
    with open(OUT / "schema_scan.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"CSVs: {total_csv}, raw bytes: {total_bytes/2**30:.1f} GiB")
    print(f"header variants: {len(variants)}")
    for i, v in enumerate(variants.values()):
        print(f"  v{i}: {v['n_csvs']} csvs, {len(v['header'])} cols, cats={v['categories']}, e.g. {v['example']}")
    errs = [(r["topic"], c) for r in results for c in r["csvs"] if "error" in c]
    if errs:
        print(f"errors: {len(errs)}", errs[:5])


if __name__ == "__main__":
    sys.exit(main())
