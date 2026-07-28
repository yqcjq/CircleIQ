# CircleIQ 建模数据集(ppio-gpu:~/data)

第八届传播数据挖掘竞赛"社交媒体圈层用户引导策略与效果预测"的建模就绪数据。
从 317GiB 原始 CSV(GB18030, 8 种表头变体)经统一预处理产出;完整处理规则见项目仓库 `docs/data-processing.md`。

## 目录

| 路径 | 内容 | 生成方 |
|---|---|---|
| `core/{major,hot,case}/<topic>__<zip>.parquet` | 建模列(43 列,无长文本),每 zip 一件 | 本地管线上传 |
| `stats/<cat>__<topic>__<zip>.json` | 每 zip 处理统计(行数/用户/去重/时间范围) | 本地管线上传 |
| `catalog.parquet` | zip→事件名→统计总目录 | 本地管线上传 |
| `edges/<cat>__<topic>.parquet` | 每议题用户互动边表(src,dst,weight,weight_pre,ts_min/max) | `build_edges.py` |
| `user_profiles.parquet` | 用户级画像聚合(全量) | `user_profiles.py` |
| `stable_pairs*.parquet` | 跨议题用户互动对(n_topics/n_major/total_weight) | `stable_circles.py` |
| `stable_edges_K{K}*.parquet` / `partition_stable_K{K}*.parquet` | K 阈值稳定图 + 稳定圈划分 | `stable_circles.py` |
| `partition_topic/major__<topic>.parquet` | 单议题 Leiden(对比基线) | `leiden_per_topic.py` |
| `hawkes/<cat>__<topic>.json` + `hawkes/events/*.npz` | 每议题 Hawkes 参数与事件流 | `run_hawkes.py` |
| `strategy/<cat>__<topic>.json` | 策略搜索结果(ROI/风险网格) | `strategy_optimizer.py` |

## 关键字段约定

- 所有 MD5 空值哨兵(`d41d8cd98...` 与 `e6fda0f0d...`)已置 null,可直接聚合
- `ts`:秒级 int64,北京时间 naive epoch;`is_original`:原创/转发
- `auth_tier`:bigv/org/verified/normal/unknown(派生规则见处理文档)
- `province`:34 省级规范化,`region_valid` 标志有效性
- 议题内 `md5_mid` 已去重(跨 zip 重复在 edges/hawkes 构建时再去)
- 体重管理议题:`id_source='computed+url_mid'`,md5_author 自算(与其他议题可匹配),帖子 ID 为链接哈希

## 环境

`source ~/circleiq/venv/bin/activate`(Python 3.10, polars/pandas/pyarrow/igraph/leidenalg/numba/torch cu121)
