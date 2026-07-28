# CircleIQ 项目总览与对照文档(快速上手指南)

**用途**: 后续阅读者(或未来会话)从零理解项目架构、运行逻辑、每一步的决策依据与结果落点。
**日期**: 2026-07-27(全量落地日)· 配套:[服务器指南](server-guide.md) · [运行日志](run-log-2026-07-27.md)

---

## 1. 一页架构

```
研究课题: 社交媒体圈层用户引导策略与效果预测(第八届传播数据挖掘竞赛)
三层建模: 圈层识别 → 传播预测 → 策略优化

原始数据(本地 Mac, 92GB zip / 317GiB CSV, GB18030, 8种表头变体)
  /Users/ppio/Desktop/第八届传播数据挖掘竞赛/{重大议题7个, 热点事件263个, 传播案例112个}
        │  pipeline/preprocess.py(10核并行, 57分钟)
        ▼
本地母本 /Users/ppio/Desktop/CircleIQ-data/
  ├─ core/  2.8GB 43列建模数据 ──rsync──▶ 服务器 ~/data/core(全量)
  ├─ text/ 48GB 长文本(BGE用) ──rsync──▶ 服务器 ~/data/text(后台补传)
  └─ stats/ + catalog.parquet(zip→事件名→统计)
        │
        ▼  GPU服务器 ppio-gpu(RTX4090 + 24 vCPU),脚本全在 /root/data/
[阶段A 圈层识别] build_edges → stable_circles(K=3跨议题稳定图+Leiden) → eval_stable
        │           产出: partition_stable_K3.parquet(用户→圈)
        ▼
[阶段C 传播预测] run_hawkes: 每事件 双时间尺度核+分段μ 多维Hawkes
        │           维度 = top12稳定圈(门槛过滤) + 圈外大V/机构/散户
        │           产出: hawkes/<事件>.json(μ,A1,A2,β1,β2) + events/*.npz
        ▼
[阶段D 验证]     validate_rolling(2h滚动预测 vs Poisson/naive) + case_study(大V反事实)
[阶段E 策略]     strategy_optimizer(通道×注入量×时机 网格 + 失控风险)
[内容调制]       content_features + analyze_content_modulation(情绪分组的参数差异)
```

## 2. 运行 Timeline(决策依据 → 执行 → 结果 → 详细文档)

| # | 步骤 | 前提/决策依据 | 执行 | 结果 | 详见 |
|---|---|---|---|---|---|
| 1 | 表头扫描+抽样验证 | 8种schema变体传闻;需统一映射 | `pipeline/scan_schemas.py` + `probe_variants.py` | 385zip/4648csv/317GiB;**发现第二MD5哨兵 e6fda0f0**(推翻旧传播树结论);**MD5无盐可自算** | data-processing.md §2-3 |
| 2 | 带宽实测定架构 | 上行仅2.3MB/s(双流不提速) | scp 测速 | **core/text 分层**:建模列先上云,长文本本地母本 | data-processing.md §4 |
| 3 | 全量预处理 | schema映射+清洗规则定稿 | `preprocess.py` 7worker | 4385万行(去重后);**两个12GB zip逐字节重复**(剔一);2个zip数值溢出修复重跑 | data-processing.md §3.3-3.4, §6 |
| 4 | 阶段A稳定圈 | 主线=跨议题稳定互动圈(单议题圈是"结果"非"结构");**修正:议题计数扩到382单元**(热点事件放大配对观测) | `stage_a.sh`(24核3.5分钟) | K=3: 9.9万用户/3188圈/模块度0.845;稳定圈=传播骨干(大V率36×);激活谱差异1.7%-18% | stage-a-report.md;plan-revision §1 |
| 5 | 内容一致性 | 沿用单议题实验口径(BGE) | 本地抽文本→4090编码(27秒) | 稳定圈 0.586 vs 随机 0.498,与单议题持平(假设"更高"未获支持,诚实记录) | stage-a-report.md §8 |
| 6 | Hawkes小规模 | 固定β网格+每维凸MLE(自测参数还原通过);**修正:维度改混合通道**(稳定圈事件占比仅1-17%,纯圈维度other吞83-99%) | `run_hawkes.py --limit 12` | 12/12验证LL增益为正;α矩阵可解释 | stage-c-methodology.md;plan-revision §2 |
| 7 | Hawkes全量 v1→v3 | v1长窗验证APE 900% → 诊断:常数μ遇事件冷却+近临界放大14×;v2分段μ无效 → 诊断:快核状态2h前衰减光;**v3双核(快1.4-6.9分+慢1.4-16.6时)+中位数点预测** | `chain_v3.sh` | 312事件拟合(3.5分钟);滚动2h APE 225%→**74%**,39%事件最优,覆盖87.5% | stage-c-methodology.md §1(版本演化) |
| 8 | α通道池化 | 228热点事件跨事件可比(全局统一圈池) | 分析 hawkes_summary_all | **权威→稳定圈→散户单向链**(大V→圈0.069,回流0.010);圈自激0.402,散户自激0.935 | stage-cde-report.md §2 |
| 9 | 内容调制 | **修正:弃XGBoost-α,改分组拟合+参数回归**(情绪列部分议题缺失;可解释性) | `analyze_content_modulation.py` | 愤怒/惊奇事件更自维持(分支比+4-6pp)、更穿圈(+8-11pp)、更持久 | stage-cde-report.md §5;plan-revision §3 |
| 10 | 大V单帖反事实 | 上升期锚点+共享μ差值分离激发贡献 | `case_study.py` ×4事件 | **负结果有价值**:单帖Δ≈分支比量级;低潮期Δ(+320)≫爆发期(≈0)→时机>咖位;引爆是帖级级联现象 | stage-cde-report.md §4 |
| 11 | 策略网格 | 通道×注入量{1,5,20}×时机{0,+6h},失控概率≤0.2约束;大议题长视界外推剔除(近临界不可信) | `strategy_optimizer.py` top8 | 稳定圈注入=高ROI低溢出(定向);散户=大总量全溢出(广度);高ROI伴随失控>0.4 | stage-cde-report.md §6 |

## 3. 文档索引

| 文档 | 内容 | 论文对应 |
|---|---|---|
| `docs/data-processing.md` | 原始数据盘点/8变体/清洗规则/双哨兵/重复zip/复现步骤/附录统计表 | 数据处理说明(交付物) |
| `docs/stage-a-report.md` | 稳定圈全量结果+5图(K网格/大小分布/模块度/激活/画像) | 圈层识别章节 |
| `docs/stage-c-methodology.md` | Hawkes数学定义/估计/防泄漏/评估/**v1→v3版本演化**/局限 | 方法章节 |
| `docs/stage-cde-report.md` | α通道表/滚动验证/案例/内容调制/策略网格+9图 | 实验与结果章节 |
| `docs/plan-revision-2026-07-27.md` | 对旧计划的4项修正与依据 | 方法动机 |
| `docs/run-log-2026-07-27.md` | 逐小时执行日志+算力踩坑+现场决策 | 实验过程 |
| `docs/persistent-community-framework.md` | 稳定圈方法论原始论证(2026-07-26) | 圈层定义动机 |
| `plans/master-todo.md` | 项目计划(顶部有状态快照) | - |
| `results/*.json` | 全部结果数字的版本化留存(与服务器一致) | 所有表格数据源 |
| `docs/figures/*.png` | 16张报告图(validated palette,中文,200dpi) | 所有插图 |

## 4. 代码索引(本地 `pipeline/`)

| 脚本 | 作用 | 输入→输出 | 并行方式 |
|---|---|---|---|
| `scan_schemas.py` | 全量表头扫描 | zip→out/schema_scan.json | 8进程 |
| `probe_variants.py` | 变体抽样验证 | zip→stdout | - |
| `common.py` | canonical映射/清洗规则(双哨兵/派生列) | 被preprocess调用 | - |
| `preprocess.py` | zip→core/text parquet | 原始zip→CircleIQ-data/ | 7进程×chunk流式 |
| `catalog.py` | 目录+统计汇总(标重复zip) | stats/*→catalog.parquet | - |
| `uploader.sh` | 增量上传core(major优先) | →服务器~/data/core | 循环rsync |
| `extract_user_text.py` | 稳定用户聚合文本 | text/*→user_text_stable.parquet | - |
| `figstyle.py` | 图表样式(CVD校验过的调色板+中文字体) | 被make_figs_*调用 | - |
| `make_figs_{stage_a,hawkes,validation,strategy,case}.py` | 报告图渲染 | results JSON→docs/figures/*.png | - |

服务器端脚本(`pipeline/server/`,部署于 ppio-gpu:/root/data/)见 [server-guide.md](server-guide.md)。

## 5. 关键前提与约束(读代码前必知)

1. **双MD5空值哨兵** `d41d8cd98...`(根UID列)与 `e6fda0f0d...`(父/根mid列)——core parquet 已置null,直接用
2. **MD5无盐**:`md5(明文ID)==MD5列`;体重管理的 md5_author 是自算的(`id_source='computed+url_mid'`)
3. **重复zip**:《社交媒体-性别与婚育观(wb)7月-1月》=《生育话题(wb)7月-1月》,catalog 里 `duplicate_of` 标记,服务器只留后者
4. **上行带宽2.3MB/s**:任何"传大文件到服务器"的想法先算时间
5. **多核纪律**(用户明确要求):polars+多进程必须 spawn(fork死锁);8+进程时设 `POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 NUMBA_NUM_THREADS=2`;跑前 `top` 验证
6. **时间戳**:秒级int64,北京时间naive;`ts` 全体、`root_ts` 仅v1/v2/v3/v6/v7变体有
7. **防泄漏产物成对存在**:`*_pre` 后缀 = 仅用2025-07-01前互动(传播层严格模式)
8. **策略/模拟的信任边界**:热点/案例事件24h视界内可信;大议题(年跨度)近临界外推不可信
