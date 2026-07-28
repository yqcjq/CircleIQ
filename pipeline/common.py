"""CircleIQ 预处理公共逻辑:canonical schema、字段派生、单 CSV 处理。

设计要点(依据 pipeline/out/schema_scan.json 的 8 种表头变体):
- 所有变体统一映射到 canonical 列;缺失列填 null
- MD5 空值哨兵 d41d8cd98f00b204e9800998ecf8427e -> null
- v1(体重管理)无 MD5 列:md5_author = md5(作者ID), md5_mid = md5(原文/评论链接)
- 时间戳:秒级 int64(北京时间 naive epoch)
- core / text 两族列,text 仅长文本
"""
import hashlib
import io
import zipfile

import numpy as np
import pandas as pd

NULL_MD5 = "d41d8cd98f00b204e9800998ecf8427e"   # md5(""),主要出现在 根微博用户UID
NULL_MD5_2 = "e6fda0f0d3e0adfff69e334462d1ef6a"  # 原像未知的第二哨兵,父/根mid 列的空值主力(原创帖 parent 100% 是它)
SENTINELS = {NULL_MD5, NULL_MD5_2}

# canonical -> 原始列名候选(按优先级 coalesce;pandas 对重复列自动加 .1)
CORE_MAP = {
    "md5_author": ["MD5-作者ID"],
    "md5_mid": ["MD5-mid"],
    "md5_parent_mid": ["MD5-父微博ID"],
    "md5_root_mid": ["MD5-根微博mid"],
    "md5_root_uid": ["MD5-根微博用户UID"],
    "raw_author_id": ["作者ID"],
    "raw_parent_id": ["父微博ID"],
    "author_name": ["原文作者"],
    "root_author_name": ["根微博作者"],
    "root_ts_str": ["根微博发布时间"],
    "date_str": ["日期"],
    "orig_flag": ["原创/转发"],
    "sentiment": ["微博情绪"],
    "info_attr": ["信息属性"],
    "media_type": ["媒体类型"],
    "source_site": ["来源网站", "来源网站.1"],
    "source_level": ["信源级别"],
    "auth_type": ["认证类型"],
    "user_type": ["用户类型"],
    "account_type": ["账号类型"],
    "actor_l1": ["平台传播主体一级"],
    "actor_l2": ["平台传播主体二级"],
    "gender": ["性别"],
    "reg_province": ["用户注册地"],
    "precise_region": ["精准地域"],
    "source_region": ["信源地域"],
    "city": ["城市"],
    "industry": ["涉及行业"],
    "keywords": ["涉及词"],
    "topic_tag": ["话题", "话题.1"],
    "content_type": ["微博内容类型"],
    "post_type": ["微博类型"],
    "n_followers": ["粉丝数"],
    "n_follows": ["用户关注数"],
    "n_posts_acct": ["微博数"],
    "n_repost": ["转发数"],
    "n_comment": ["评论数"],
    "n_like": ["点赞数"],
    "n_read": ["阅读数/浏览热度"],
    "reg_time": ["注册时间"],
    "birth_year": ["用户画像出生年份", "用户画像出生月份"],
    "verified_info": ["用户画像用户认证信息"],
    "interest_tags": ["用户画像用户兴趣标签"],
    "url": ["原文/评论链接"],
}

TEXT_MAP = {
    "title": ["标题／微博内容"],
    "content_full": ["全文内容"],
    "origin_content": ["原微博内容"],
    "root_title": ["根微博标题"],
    "user_bio": ["用户简介"],
    "image_text": ["图文识别"],
}

# 明确丢弃(报告里说明):视频链接/视频封面图/视频封面/网页快照/根微博链接/原微博作者/
# 原文作者ID/id/MD5-id/平台传播主体三级/用户画像@内容/@标签/公司/感情/学校/关注话题/抖音认证信息/末尾空列

MD5_COLS = ["md5_author", "md5_mid", "md5_parent_mid", "md5_root_mid", "md5_root_uid"]
NUM_COLS = ["n_followers", "n_follows", "n_posts_acct", "n_repost", "n_comment", "n_like", "n_read"]

# 认证类型 -> auth_tier
def derive_auth_tier(auth: pd.Series, user_type: pd.Series, actor_l1: pd.Series) -> pd.Series:
    a = auth.fillna("")
    tier = pd.Series("unknown", index=a.index, dtype="object")
    tier[a.str.contains("金V|黄V", na=False)] = "bigv"
    tier[a.str.contains("机构|企业|组织|媒体|政务|网站认证", na=False)] = "org"
    tier[a.str.contains("橙V|达人|个人认证|蓝V", na=False) & (tier == "unknown")] = "verified"
    tier[(a == "普通用户") & (tier == "unknown")] = "normal"
    # v0/v4/v5 的补充信号
    ut, al = user_type.fillna(""), actor_l1.fillna("")
    tier[(tier == "unknown") & (ut.str.contains("企业|媒体|政府", na=False) | al.str.contains("机构|媒体|政务", na=False))] = "org"
    tier[(tier == "unknown") & (ut == "个人认证")] = "verified"
    return tier

_PROV_STRIP = ("省", "市", "自治区", "壮族", "回族", "维吾尔", "特别行政区")
VALID_PROV = set("北京 天津 上海 重庆 河北 山西 辽宁 吉林 黑龙江 江苏 浙江 安徽 福建 江西 山东 河南 湖北 湖南 广东 海南 四川 贵州 云南 陕西 甘肃 青海 台湾 内蒙古 广西 西藏 宁夏 新疆 香港 澳门".split())

def _norm_prov(s: str) -> str:
    if not s:
        return ""
    for suf in _PROV_STRIP:
        s = s.replace(suf, "")
    return s if s in VALID_PROV else ""

def derive_province(reg: pd.Series, precise: pd.Series) -> pd.Series:
    reg = reg.fillna("").str.strip()
    prov = reg.map(_norm_prov)
    # fallback: 精准地域第一段
    need = prov == ""
    first_seg = precise.fillna("").str.split(",", n=1).str[0].map(_norm_prov)
    prov[need] = first_seg[need]
    return prov.replace("", None)


def md5hex_series(s: pd.Series) -> pd.Series:
    return s.map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest() if x else None)


def process_frame(raw: pd.DataFrame, csv_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """raw: 全 str dataframe(keep_default_na=False)。返回 (core, text)。"""
    cols = {}
    for canon, cands in {**CORE_MAP, **TEXT_MAP}.items():
        s = None
        for c in cands:
            if c in raw.columns:
                cur = raw[c]
                if isinstance(cur, pd.DataFrame):  # 同名列未被 pandas 重命名时兜底
                    cur = cur.iloc[:, 0]
                s = cur if s is None else s.where(s.str.len() > 0, cur)
        if s is None:
            s = pd.Series("", index=raw.index, dtype="object")
        cols[canon] = s.str.strip()

    df = pd.DataFrame(cols)

    # MD5 归一化:哨兵/空 -> None;缺 MD5 列时自算
    if (df["md5_author"] == "").all() and (df["raw_author_id"] != "").any():
        df["md5_author"] = md5hex_series(df["raw_author_id"])
        df["id_source"] = "computed"
    else:
        df["id_source"] = "given"
    if (df["md5_mid"] == "").all() and (df["url"] != "").any():
        df["md5_mid"] = md5hex_series(df["url"])
        df.loc[:, "id_source"] = df["id_source"] + "+url_mid"
    for c in MD5_COLS:
        s = df[c].str.lower().replace("", None)
        df[c] = s.where(~s.isin(SENTINELS), None)

    # 时间(pandas 3.0 to_datetime 返回 datetime64[ms/us],统一转秒)
    ts = pd.to_datetime(df["date_str"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["ts"] = pd.Series(ts.astype("datetime64[s]").astype("int64"), index=df.index).where(ts.notna(), None).astype("Int64")
    rts = pd.to_datetime(df["root_ts_str"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["root_ts"] = pd.Series(rts.astype("datetime64[s]").astype("int64"), index=df.index).where(rts.notna(), None).astype("Int64")

    df["is_original"] = df["orig_flag"] == "原创"
    for c in ["author_name", "root_author_name"]:
        df[c] = df[c].str.strip("'\" ").replace("", None)

    # 数值列(脏数据可能出现超 int64 范围的浮点,先夹取再转)
    for c in NUM_COLS:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.where(s.abs() < 2**62, other=pd.NA)
        df[c] = s.round().astype("Int64")

    # 派生
    df["auth_tier"] = derive_auth_tier(df["auth_type"], df["user_type"], df["actor_l1"])
    df["province"] = derive_province(df["reg_province"], df["precise_region"])
    df["region_valid"] = df["province"].notna()
    for c in ["sentiment", "info_attr", "media_type", "gender", "industry", "keywords"]:
        df[c] = df[c].replace({"": None, "--": None, "null": None, " ": None})

    df["csv_file"] = csv_name

    core_cols = (MD5_COLS + ["ts", "is_original", "sentiment", "info_attr", "media_type",
                 "source_site", "source_level", "auth_tier", "auth_type", "user_type", "actor_l1", "actor_l2",
                 "gender", "province", "region_valid", "city", "industry", "keywords", "topic_tag",
                 "content_type", "post_type"] + NUM_COLS +
                 ["reg_time", "birth_year", "interest_tags", "root_ts", "root_author_name",
                  "author_name", "id_source", "csv_file"])
    text_cols = ["md5_mid", "md5_author", "title", "content_full", "origin_content",
                 "root_title", "user_bio", "image_text", "verified_info", "url", "csv_file"]
    text = df[text_cols].copy()
    for c in ["title", "content_full", "origin_content", "root_title", "user_bio", "image_text", "verified_info", "url"]:
        text[c] = text[c].replace({"": None, " ": None})
    return df[core_cols], text


def iter_zip_csvs(zip_path: str):
    """yield (csv_name, text_stream)"""
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.infolist():
            if not m.filename.lower().endswith(".csv") or m.file_size == 0:
                continue
            with zf.open(m.filename) as f:
                yield m.filename.rsplit("/", 1)[-1], io.TextIOWrapper(f, encoding="gb18030", errors="replace", newline="")


def read_csv_all(stream) -> pd.DataFrame:
    return pd.read_csv(
        stream, dtype=str, engine="c", keep_default_na=False, na_values=[],
        on_bad_lines="skip", quoting=0,
    )
