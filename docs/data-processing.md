# 数据处理说明:从原始数据到建模数据集

**日期**: 2026-07-27 · **状态**: 管线已实现并全量运行中,统计数字待回填
**代码**: `pipeline/`(本地预处理)+ `pipeline/server/`(服务器端构图/建模)

---

## 1. 原始数据盘点

来源:第八届传播数据挖掘竞赛,三批数据(热搜榜单数据经确认为无用数据,不进管线)。

| 批次 | zip 数 | CSV 数 | 原始体积(解压) | 说明 |
|---|---|---|---|---|
| 重大议题与话题数据 | 12(7 议题,生育话题含 4 zip) | 552 | ~117 GiB | 跨月-跨年长周期 |
| 2025年热点事件 | 263 | ~3359 | ~150 GiB | 单事件 ~10 天窗口 |
| 2025年传播案例数据 | 112 | ~732 | ~50 GiB | ~120 天窗口 |
| 合计 | 385(有效) | 4648 | **317.1 GiB** | 压缩后 92 GB |

- 单 CSV 30MB~350MB,GB18030 编码,内含超 128KB 的长字段(需解除 csv field limit)
- zip→事件名映射来自《数据集目录、示例与字段说明.json》,共 383 个命名条目

## 2. Schema 变体与统一

全量表头扫描(`pipeline/scan_schemas.py`)发现 **8 种表头变体**(v0-v7),列数 49-58:

| 变体 | 覆盖 | 关键差异 |
|---|---|---|
| v0 | AI就业(34 csv) | 无"微博情绪";多"平台传播主体/用户类型"列 |
| v1 | 体重管理(37 csv) | **无任何 MD5 列**,无 id/父微博ID |
| v2 | 假期调休(37 csv) | 标准 54 列 |
| v3 | 城市养犬/婚俗改革/延迟退休(366 csv) | 标准 53 列(无"城市") |
| v4/v5 | 生育话题(83 csv) | v4 无"用户注册地/根微博作者";含 dy/bz/xhs/db 多平台 |
| v6/v7 | 全部热点+案例(4091 csv) | 55 列,v6 多"出生月份",v7 多"出生年份" |

统一策略(`pipeline/common.py`):
- 每个 canonical 列给出原始列名候选表,按优先级 coalesce;缺失填 null
- 重复列名("来源网站"×2、"话题"×2)取首个非空
- 末尾空列名丢弃

## 3. 关键清洗规则(重要发现)

### 3.1 MD5 空值哨兵有两个
| 哨兵 | 出现位置 | 占比 |
|---|---|---|
| `d41d8cd98f00b204e9800998ecf8427e` = MD5("") | MD5-根微博用户UID | 87-95% |
| `e6fda0f0d3e0adfff69e334462d1ef6a`(原像未知,**本次新发现**) | MD5-父微博ID / MD5-根微博mid | 73-86% |

原创帖的父/根 mid 100% 为第二哨兵。**若不过滤,groupby(根mid) 会把所有独立原创帖聚成一棵几万帖的伪"超大传播树"**——此前基于单哨兵过滤的传播树规模结论(top1 树占 90%)已被本发现推翻并修正。两哨兵在全部 5 个 MD5 列统一置 null。

### 3.2 MD5 无盐,可自算
抽样验证 `md5(作者ID明文) == MD5-作者ID`(4 变体 × 100 行,全对)。因此:
- 体重管理(v1,无 MD5 列):`md5_author = md5(作者ID)`,与其他议题的用户身份**跨议题可匹配**
- v1 无帖子 ID:`md5_mid = md5(原文/评论链接)`;传播链用"根微博作者名"回连(标记 `id_source`)

### 3.3 重复数据 zip(新发现)

《社交媒体-性别与婚育观(wb)2025年7月-2026年1月.zip》与《生育话题(wb)2025年7月-2026年1月.zip》(各 12GB)**高度疑似同一份数据两个名字**——两者经确定性管线处理的中间产物逐字节一致(完整重验进行中,确认后建模数据集只保留一份并在目录标记 `duplicate_of`,避免双重计数)。

### 3.4 数值列脏数据

生育话题(wb)存在超出 int64 安全范围的数值(如阅读数字段的异常浮点),导致严格整型转换失败。处理:`to_numeric` 后夹取 `|x| < 2^62` 再转 Int64,超界置 null。

### 3.5 其他规则
- 编码 GB18030 + errors=replace;`on_bad_lines=skip`
- 议题内按 md5_mid 去重(重复抓取,假期调休单 zip 去掉 39.7 万重复)
- 时间戳→秒级 int64(北京时间 naive);极少量坏值置 null(假期调休 34/220万)
- 认证类型→`auth_tier` 五档:bigv(金V/黄V)/org(机构/企业/媒体/政务)/verified(橙V/达人/个人认证)/normal(普通用户)/unknown(含混入的数字代码);v0/v4/v5 用"用户类型/平台传播主体"补充判档
- 省份:用户注册地→精准地域第一段 fallback,规范到 34 省级;`region_valid` 标志
- '--'/'null'/空串/单空格 → null

## 4. 产出数据集(两层结构)

**分层动机**: 上行带宽 2.3MB/s,建模列(core)与长文本(text)体积比 ≈ 1:16,分层后 core 可在 1 小时内全量上云,text 留本地母本按需取用(BGE embedding 只需稳定圈用户的聚合文本)。

```
CircleIQ-data/                          # 本地母本(Mac)
├── core/{major,hot,case}/<topic>__<zip>.parquet   # 建模列,43 列(见下)
├── text/{major,hot,case}/<topic>__<zip>.parquet   # md5_mid/md5_author + 6 个长文本列
├── stats/<cat>__<topic>__<zip>.json               # 每 zip 处理统计
└── catalog.parquet                     # zip→事件名→统计 总目录

ppio-gpu:~/data/                        # 服务器(建模用)
├── core/...(与本地同构,全量)
├── edges/<cat>__<topic>.parquet        # 每议题互动边表(服务器端构建)
├── user_profiles.parquet               # 用户级画像聚合
├── stable_pairs.parquet / stable_edges_K3.parquet / partition_stable_K3.parquet
└── hawkes/<cat>__<topic>.json          # 每议题 Hawkes 参数
```

**core 列清单(43)**: md5_author/md5_mid/md5_parent_mid/md5_root_mid/md5_root_uid, ts, is_original, sentiment, info_attr, media_type, source_site, source_level, auth_tier, auth_type, user_type, actor_l1/l2, gender, province, region_valid, city, industry, keywords, topic_tag, content_type, post_type, n_followers/n_follows/n_posts_acct/n_repost/n_comment/n_like/n_read, reg_time, birth_year, interest_tags, root_ts, root_author_name, author_name, id_source, csv_file, topic, category

**text 列清单**: title, content_full, origin_content, root_title, user_bio, image_text, verified_info, url(+ 关联键)

**明确丢弃列**(建模无用,报告说明): 视频链接/视频封面图/视频封面/网页快照/根微博链接/原微博作者/原文作者ID/id/MD5-id/平台传播主体三级/用户画像@内容/@标签/公司名称/感情状况/学校名称/关注的话题/抖音认证信息

## 5. 互动边表定义(服务器端 `build_edges.py`)

- 边:转发者 → 父帖作者(md5_parent_mid → mid→author 映射),去自转发
- 体重管理 fallback:转发者 → 根微博作者名→md5 映射(`method=root_name`)
- 聚合:(src, dst) 求 weight(互动次数)、weight_pre(2025-07-01 前次数,防泄漏用)、ts_min/ts_max
- 双哨兵已在预处理置 null,边表构建天然免疫伪边

## 6. 全量处理统计(2026-07-27 实测)

- **总量**: 385 zip / 4648 CSV → 处理耗时 **56.6 分钟**(Mac 10 核,7 worker 并行)+ 2 个含脏数值的 zip 修复重跑
- **产出**: core 3.1GB(43 列建模数据)/ text 41GB(长文本,本地母本)
- **行数**(不含重跑中的 2 zip): 重大议题 21.9M(含疑似重复双计)+ 热点事件 9.76M + 传播案例 5.03M ≈ **36.7M 行**
- **zip 内去重**: 丢弃重复抓取 5.1M 行
- **坏时间戳**: 5,619 行(0.020%)
- **情绪填充率**: 84.7%(缺情绪列的 AI就业/生育话题拉低;有该列的议题接近 100%)
- **转发占比**: 传播案例 45.5% / 热点事件 43.5% / 重大议题约 16-24%(因议题而异)
- **异常**: 无 0 行 zip;31 个 zip 行数 <1000(小事件,正常)

上传:core 全量 + stats + catalog 已在服务器 `~/data`(2.3MB/s 上行,~35 分钟)。

## 7. 复现步骤

```bash
# 本地(Mac,10 核):约 1.5-2.5 小时
python pipeline/preprocess.py            # 385 zip → core/text parquet
python pipeline/catalog.py               # 目录+统计汇总
bash   pipeline/uploader.sh              # 增量上传 core 到服务器 ~/data

# 服务器(24 vCPU):
python3 build_edges.py --workers 4       # 每议题边表(polars 内部再并行)
python3 user_profiles.py                 # 用户画像聚合
python3 stable_circles.py --k 3          # 跨议题稳定图 + Leiden
python3 leiden_per_topic.py              # 单议题 Leiden(对比基线)
python3 eval_stable.py --k 3             # A5 评估
python3 run_hawkes.py --category hot     # 传播预测拟合
```
| zip | 行数 | 用户数 | 转发数 | 时间范围 | core MB |
|---|---|---|---|---|---|
| AI就业（数智员工） | 2,247,196 | 1,002,280 | 1,830,287 | 2025-01-01~2025-11-30 | 164.5 |
| 体重管理 | 1,133,918 | 303,160 | 145,595 | 2012-05-11~2026-01-01 | 57.4 |
| 假期调休 | 2,200,777 | 1,199,663 | 475,631 | 2025-01-01~2026-01-01 | 140.2 |
| 城市养犬 | 1,669,726 | 861,318 | 388,983 | 2025-01-01~2026-01-01 | 107.1 |
| 婚俗改革 | 2,000,222 | 979,308 | 392,274 | 2025-01-01~2026-01-01 | 115.3 |
| 延迟退休 | 2,626,936 | 1,315,121 | 261,764 | 2025-01-01~2026-01-01 | 152.2 |
| 生育话题 （dy、bz、xhs、db）2025年12月-2026年1月 | 2,043,910 | 1,045,552 | 8,303 | 2025-12-01~2026-01-29 | 109.3 |
| 生育话题（wb）2025年1-6月 | 7,951,935 | 1,879,578 | 3,668,060 | 2025-01-01~2025-07-01 | 497.6 |

| 热点事件(top10) | 行数 | 用户数 | 转发占比 |
|---|---|---|---|
| 青岛大学一宿管人员离世 | 302,059 | 222,739 | 74% |
| 人贩子余华英被执行死刑 | 280,342 | 190,387 | 27% |
| 防城港：女司机亮证逼迫让路事件 | 254,484 | 149,835 | 65% |
| 家长称上海多校午餐虾仁鸡蛋发臭 | 222,407 | 133,346 | 53% |
| 广西一老师被曝性侵女生致其自杀 | 195,689 | 152,042 | 84% |
| 北京首例宠物中毒刑事公诉案一审宣判 | 179,556 | 137,768 | 70% |
| 百度副总裁未成年女儿传播他人隐私信息 | 176,075 | 123,952 | 61% |
| 19岁女大学生景区遇害 | 174,419 | 139,763 | 90% |
| 福耀科技大学举行开学典礼 | 162,972 | 110,971 | 24% |
| 福建8岁男童随家人爬山后失联 | 156,266 | 82,284 | 12% |

(附表:重大议题各 zip 统计与 top10 热点事件;完整 383 条目见 catalog.parquet)
