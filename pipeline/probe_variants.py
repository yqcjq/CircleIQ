"""每个 schema 变体抽一个 CSV 的前 N 行,验证:
1. MD5-作者ID == md5(作者ID)?  (跨议题用户匹配的关键)
2. 关键列的取值分布/填充率
"""
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize)

SCAN = json.load(open(Path(__file__).parent / "out" / "schema_scan.json"))
N = 500

# 每变体挑一个代表 csv
examples = {}
for z in SCAN["zips"]:
    for c in z.get("csvs", []):
        sig = c.get("header_sig")
        if sig and sig not in examples:
            examples[sig] = (z["zip"], c["csv"], z["topic"])

def sample_rows(zp, csv_name, n):
    with zipfile.ZipFile(zp) as zf:
        member = next(m for m in zf.namelist() if m.endswith(csv_name))
        with zf.open(member) as f:
            text = io.TextIOWrapper(f, encoding="gb18030", errors="replace")
            reader = csv.reader(text)
            header = next(reader)
            rows = []
            for i, row in enumerate(reader):
                if i >= n: break
                rows.append(row)
    return header, rows

def md5hex(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()

for sig in sorted(examples, key=lambda s: int(s[1:])):
    zp, cname, topic = examples[sig]
    header, rows = sample_rows(zp, cname, N)
    idx = {c: i for i, c in enumerate(header)}  # 后者覆盖前者(重复列)
    print(f"\n===== {sig} {topic}/{cname} ({len(rows)} rows) =====")

    def col(row, name):
        i = idx.get(name)
        return row[i].strip() if i is not None and i < len(row) else None

    # 1. md5 验证
    if "MD5-作者ID" in idx and "作者ID" in idx:
        ok = bad = 0
        for r in rows[:100]:
            raw, given = col(r, "作者ID"), col(r, "MD5-作者ID")
            if raw and given:
                if md5hex(raw) == given.lower(): ok += 1
                else: bad += 1
        print(f"md5(作者ID)==MD5-作者ID: ok={ok} bad={bad}")
    else:
        print("no MD5-作者ID column" if "MD5-作者ID" not in idx else "no raw 作者ID")

    # 2. 关键列
    for key in ["日期", "原创/转发", "微博情绪", "媒体类型", "认证类型", "信息属性",
                "用户注册地", "精准地域", "涉及行业", "粉丝数", "根微博发布时间", "平台传播主体一级", "用户类型"]:
        if key not in idx: continue
        vals = Counter(col(r, key) for r in rows)
        top = ", ".join(f"{v!r}×{n}" for v, n in vals.most_common(4))
        print(f"  {key}: {top}")
