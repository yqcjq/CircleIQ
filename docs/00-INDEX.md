# 文档索引

## 阶段编号定义

| 编号 | 阶段 | 说明 |
|---|---|---|
| 0 | 元信息/基础设施 | 项目总览、决策记录、服务器指南、数据处理流程 |
| 1 | 圈层识别 | 稳定互动圈的定义、算法、实验、结果解读 |
| 2 | 内容维度 | 内容特征（情绪/行业）对传播的调节效应分析 |
| 3 | 传播预测+策略 | Hawkes 建模、滚动验证、反事实、策略优化、算法筛选 |
| 4 | （预留） | 后续新阶段使用 |

---

## 0x 元信息/基础设施

| 编号 | 文件 | 做了什么 | 方法 | 结论/产出 |
|---|---|---|---|---|
| 01 | 01-project-guide.md | 项目总览与快速上手 | — | 三层架构图+11步Timeline+文档代码索引+关键约束(MD5哨兵/带宽/多核纪律) |
| 02 | 02-decision-timeline.md | 记录所有技术决策转向 | — | Leiden→稳定圈/GNN弃用/SIR否决等决策链+否决清单 |
| 03 | 03-server-guide.md | GPU服务器使用指南 | — | SSH别名/目录结构/产物路径/scp命令/vLLM部署步骤 |
| 04 | 04-data-processing.md | 全管线数据处理逻辑 | 表头扫描+Schema统一+MD5哨兵+边表构建 | 8种变体→统一schema; 385zip/4648csv/317GB→core 2.8GB parquet |

## 1x 圈层识别

| 编号 | 文件 | 做了什么 | 方法 | 结论/产出 |
|---|---|---|---|---|
| 10 | 10-persistent-community-framework.md | 定义"跨议题稳定互动圈" | 多议题边合并→n_topics≥K过滤→稳定图上Leiden | 核心定义文档：单议题互动是结果不是结构，K≥3为稳定阈值 |
| 11 | 11-stage-a-report.md | 阶段A全量实验 | K网格搜索+modularity评估+种子稳定性 | 382单元/906万边/K=3: 99,337用户/3,203圈/mod=0.8448/ARI≥0.91 |
| 12 | 12-circle-identification-results.md | 圈层结果解读(论文素材) | Top20圈定性分析+激活谱+风险识别 | 五类圈层类型学; 稳定圈=传播骨干(大V率36x); 疑似组织化传播风险 |

## 2x 内容维度

| 编号 | 文件 | 做了什么 | 方法 | 结论/产出 |
|---|---|---|---|---|
| 20 | 20-content-dimension-experiment.md | 内容维度实验 | Leiden/Hierarchical/GNN三组对比(婚俗改革3csv) | GNN三组全败(mod 0.013-0.085); Leiden稳定性100%胜出 |
| 21 | 21-content-dimension-conclusion.md | 内容维度结论 | 对比分析+收敛判断 | 结构与内容是不同信号应分层处理; GNN弃用; 内容作评估维度非融合 |

## 3x 传播预测+策略

| 编号 | 文件 | 做了什么 | 方法 | 结论/产出 |
|---|---|---|---|---|
| 30 | 30-stage-c-methodology.md | Hawkes方法论(论文素材) | 双核(快β1分钟级+慢β2小时级)+混合通道+固定β网格+每维凸MLE+Ogata thinning模拟器 | v1→v3演化; 分段mu+内容调制改分组拟合+OLS; 防泄漏设计 |
| 31 | 31-stage-cde-report.md | CDE终版报告 | 312事件全量拟合+滚动2h验证+反事实+策略网格 | α通道链(大V→稳定圈→散户); APE74%(39%事件最优); 时机>咖位; 圈注入=高ROI/散户=广度 |
| 32 | 32-prediction-screening-report.md | 传播预测算法筛选 | 圈0/圈5上GRU/GBDT/Hawkes/SIR/IC+naive对比 | GRU+GBDT入选; Hawkes/SIR/IC全败给naive; 图结构无增益; 混合方向关闭 |

## archive/（已废弃/归档）

| 文件 | 原因 | 备注 |
|---|---|---|
| circle-identification-report.md | 早期单议题Leiden POC，被10+11+12取代 | 五算法对比结论(ARI=0.04)仍有论文引用价值 |
| hierarchical-circle-framework.md | 分层框架方案未实施，被稳定圈天然解决 | 降级为"可选补充" |
| content-dimension-plan.md | 文件内自标"未推进,存档"，被20+21取代 | 路径A/B设计稿 |
| plan-revision-2026-07-27.md | 计划修正日志，历史参考 | 议题计数/Hawkes维度/内容调制/反事实4项修正 |
| run-log-2026-07-27.md | 实验运行日志，历史参考 | 全量执行时间线+算力踩坑+现场决策 |

## figures/

| 图片文件 | 引用于 | 内容 |
|---|---|---|
| 11+12-k_grid.png | 11, 12 | K值网格搜索结果 |
| 11+12-size_dist.png | 11, 12 | 圈层规模分布 |
| 11+12-modularity.png | 11, 12 | 模块度对比 |
| 11+12-stable_vs_rest.png | 11, 12 | 稳定圈 vs 其他用户统计对比 |
| 11+12-activation.png | 11, 12 | 议题×圈层激活热力图 |
| 31-beta_dist.png | 31 | β衰减参数分布 |
| 31-alpha_20260313162141.png | 31 | α矩阵示例(人贩子余华英被执行死刑) |
| 31-alpha_20260313162103.png | 31 | α矩阵示例 |
| 31-alpha_假期调休.png | 31 | α矩阵示例(假期调休) |
| 31-gain_dist.png | 31 | 拟合增益分布 |
| 31-share_vs_gain.png | 31 | 稳定占比 vs 增益散点 |
| 31-validation.png | 31 | 2h滚动验证结果 |
| 31-case_hot_20260319181009.png | 31 | 热点案例图 |
| 31-case_hot_20260319181003.png | 31 | 热点案例图 |
| 31-case_hot_20260313162141.png | 31 | 热点案例图 |
| 31-case_case_20260317191701.png | 31 | 传播案例图 |
| 31-strategy_scatter.png | 31 | 策略散点图 |
| 32-screen_male.png | 32 | 筛选总榜 |
| 32-screen_strata.png | 32 | 量级分层 |
| 32-screen_cases.png | 32 | 案例轨迹 |
| 32-screen_scatter.png | 32 | 散点图 |
