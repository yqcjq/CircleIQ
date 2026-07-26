---
name: project-algorithm-decisions
description: "2026-07-25 基于真实数据验证后确定的算法选择——Leiden(POC 已验)主线圈层,Hawkes(未验)主线传播"
metadata: 
  node_type: memory
  type: project
  originSessionId: 15487124-fe45-437e-9258-0dbe69b7b9e6
  modified: 2026-07-26T09:35:31.601Z
---

**Why**: 2026-07-25 用真实索引 parquet 跑通了 Leiden POC,数据说话,推翻了此前"图太稀疏"的错误判断,并确定了不需要 GPU 的技术路线。**注意区分已 POC 验证的决策 vs 仍是理论选型的决策。**

**How to apply**: 后续算法讨论以下面这些决策为准。落地时按"已验证 → 待验证 → 加分项"顺序推进。

---

## 圈层识别算法【已 POC 验证】

**评估标准**(用户 2026-07-25 明确):
- 圈层内传播一致性(圈内互动 > 圈间)
- 圈层稳定性(不同种子下结果一致)
- 圈层可解释性(能起清晰名字)
- **不加内容相似度**——用户原话:"你无法要求一个用户发的帖子始终在这一个话题下面,但如果他们之间能够产生转发或者互动的联系,实际上他们本身可以理解为在一个圈层"

**主线算法**: **Leiden** (通过 `igraph.community_leiden` 或 `leidenalg`)
- **不用 Louvain**:Leiden 是 Louvain 的直接改进版,修复了"内部不连通社区"缺陷,速度快 20-50%,稳定性更高

**图构造**:
- 节点:用户(MD5-作者ID)
- 边:子帖作者 → 父帖作者(通过 MD5-父微博ID 查作者),无向
- 权重:互动次数
- 过滤:去自转发、去NULL哨兵、只保留最大连通分量(可选)

**POC 实测结果**(2026-07-25):
| 议题 | 帖数 | 图节点 | 图边 | 社区数 | 模块度 | 稳定性 | 圈内占比 | 用时 |
|---|---|---|---|---|---|---|---|---|
| 余华英 | 35万 | 5.5万 | 6.1万 | 702 | 0.890 | 99.9% | 95.3% | 4s |
| 广西教师 | 21万 | 13万 | 15.7万 | 224 | 0.819 | 100% | 85.5% | 12s |
| 婚俗改革 | 248万 | 26万 | 30万 | 1920 | 0.767 | 100% | 99.0% | 27s |

**结论**: 模块度均 >0.7 是极强社区结构,3 次不同种子稳定性 99.9-100%,所有评估标准通过。

**已识别的优化方向**:
- 头部社区过大(婚俗改革 top 社区占 50%),可通过 resolution>1 或递归 Leiden 细分
- 目前边只用了"转发关系",可增加"同一时间窗内讨论同一话题"作为软边

**2026-07-26 方法论转向**:

用户提出关键反思:"单议题下的互动图是**结果**不是**结构**,不能通过一次事件的观测判断圈层归属。真正的圈层应当是在**多议题下都稳定存在互动**的一批人。"

**因此圈层主线定义修正为**:
- **主线** = 跨议题稳定互动圈(persistent community): 在 K 个以上议题都有互动的用户对,构成稳定图,在稳定图上跑 Leiden
- **单议题 Leiden**: 从主线降级为中间产物,提供每议题的边表,合并后做稳定过滤
- **元圈层类型学**: 从"跨议题方法"改为"稳定圈的类型学分析补充"
- 详见 [docs/persistent-community-framework.md](../../docs/persistent-community-framework.md)

**转向理由**:
1. 竞赛课题是"引导策略",策略需要针对**持续存在的群体**才有意义
2. 单议题圈层可能是水军/事件参与者的偶然聚集,不是真社交关系
3. 跨议题稳定互动才是"真圈层"的物理定义

**K(稳定性阈值)**: 起点 K=3,根据数据调

**下面记录的是原(单议题) POC 过程,作为背景保留**:
用户提出 Leiden 只用图结构、可能忽略用户自带的画像/内容,需要 POC 对比多种算法验证是否值得升级。因此在婚俗改革 3csv 数据(3.7 万节点、5 万边)上跑 5 种圈层算法对比:

1. **Leiden**(baseline, C++ 多核,0.6 秒完成)
2. **node2vec + HDBSCAN**(pecanpy,Cython/numba 多核,node2vec 9.6 秒 + HDBSCAN 32 秒)
3. **LINE + HDBSCAN**(手写 PyTorch GPU 版,训练 0.2 秒 + HDBSCAN 31 秒)
4. **纯画像 K-means**(sklearn 多核 24 核 = 2187% CPU,16 秒)
5. **结构 emb ‖ 画像 → HDBSCAN**(拼接融合,77 秒 HDBSCAN 主导)

## POC 5 算法对比结论(2026-07-26 实测)

**Modularity 对比**:
| 算法 | Modularity |
|---|---|
| **Leiden** | **0.752** |
| node2vec | 0.441 |
| LINE | -0.013 |
| profile_kmeans | -0.011 |
| fused | -0.035 |

**共识矩阵 ARI**(关键诊断):
- Leiden ↔ node2vec ↔ LINE **高度一致 (ARI 0.75-0.91)** → 话题内圈层是**数据里客观存在的结构**,不是算法伪影
- Leiden ↔ profile_kmeans **几乎正交 (ARI 0.04)** → 图上圈层结构与"地域+行业+认证"切分是**两码完全不同的东西**
- 融合方案 ARI 全都 <0.31 → 简单拼接不是好方案

**画像熵**:
- profile_kmeans 圈内熵极低是同义反复(它本来就是按画像切的),不是它好
- Leiden 圈内省份熵 1.48 → 圈层**跨地域、跨行业**,是纯粹的互动集群

**结论(直接回答用户最初担心)**:
Leiden 忽略画像/内容不是缺陷,是特性——图上的圈层结构和画像切分是两码不同的东西,ARI 只有 0.04。策略引导的对象应该是"有实际互动关系的社群",不是"同省同行业的一堆陌生人"。

**主线算法确认: Leiden**。node2vec / LINE 作为论文对比方法保留(证明客观存在性),但不作主线。

**给策略层的重要修正**:
- 圈层不是"同一地域",不是"同一行业"
- 圈层是"因互动关系形成的社群",跨地域、跨行业
- 引导"某圈"和引导"某地域"是不同操作,评委会问要说清

**用户对话题内建圈的疑问**:担心"话题内所有用户天然是一个大圈,再划分是伪问题"。数据反驳:模块度 0.77-0.89 太高不是均匀图,圈内互动占比 99% 不可能来自随机划分,且三种独立图算法一致性 0.75-0.91 证明结构客观存在。**结论**:圈层含义要精准表述为"**该话题下由某某类用户组成的互动子群**",不是"泛社交网络圈层"。跨话题不合并(用户 Jaccard ≈ 0.01)。

---

---

## 内容维度实验结论(2026-07-26 完成)

**动机**:用户在 Leiden POC 完成后提出评估标准应加内容维度,要求两个方向都做:
- 路径 A:内容作评估指标(不改主算法)
- 路径 B:内容作主算法输入(改主算法)

**执行**:在婚俗改革 3 csv(94k 用户,37k 图内,49k 边)上用 BGE-small-zh 编码用户文本 embedding,然后:
- 路径 A → 算 Leiden 圈内文本一致性
- 路径 B → 跑 Hierarchical(Leiden + 大圈内容子分)+ GraphSAGE 3 组调参

**路径 A 结论(内容作为评估)**:
- 随机对相似度 baseline 0.49
- Leiden 圈内平均 0.58(加权)/ 0.64(未加权)——高于随机但不"高一致"
- **头部大圈内容散乱**:CID=0(7281 用户)相似度 0.49(等于随机!),CID=1(0.55),CID=2(0.52)
- 中等圈相对纯净(size 1500 左右,相似度 0.65-0.71)
- **含义**:Leiden 抓住了互动关系但没抓住内容,大圈是"多话题子群混合"

**路径 B 结论(内容作为算法)**:
| 方案 | Modularity | 内容一致性 | 说明 |
|---|---|---|---|
| Leiden(baseline) | 0.752 | 0.614 | 结构强,内容中 |
| Hierarchical(Leiden顶层+大圈子分) | 0.227 | 0.728 | 结构衰减但顶层还在,内容提升 |
| GNN 原版 | 0.085 | 0.867 | 结构崩溃 |
| GNN-A(增容量) | 0.031 | 0.891 | 更差 |
| GNN-B(拼接结构特征) | 0.013 | 0.877 | 最差 |
| GNN-C(内容降维) | 0.019 | 0.888 | 未改善 |

**结构 vs 内容 trade-off 是数据本身的性质,不是算法缺陷**。三方案两两 ARI 都在 0.12-0.29,说明它们看到的是**根本不同的信号**——与上一轮"结构与画像 ARI=0.04"的发现一致。

**GNN 三组调参一致性**:全部使结构信号变差(Modularity 从原版 0.085 降到 0.013-0.031),证明**不是调参问题,是 GraphSAGE + 无监督对比学习 的方法结构问题**。BGE 512 维内容信号在直连节点间本就相近,任何"给 GNN 更多容量"的方向都被用来记住内容而非结构。

**最终决定**:
- **主线**:Leiden(策略作用域)+ Hierarchical(策略精细定位) 两级架构
- **内容维度**:作为**评估指标**(路径 A)进论文,展示"Leiden 圈内容一致性显著高于随机基线"
- **GNN**:弃用,论文里作对比方法说明"简单融合不适合本任务"
- **不再折腾内容作为算法输入**,继续推进阶段 C 传播预测

**未来可能改进方向**(不在本竞赛范围内):
- DGI(Deep Graph Infomax)—— 强制利用结构信号
- BGRL(Bootstrap Your Own Latent for Graphs)—— 避免负采样退化
- 监督式 GNN —— 用 Leiden 结果作监督标签

参见 [docs/content-dimension-experiment.md](../../docs/content-dimension-experiment.md) 详细实验报告。

---

## 传播预测算法【尚未 POC,是理论选型】

**主推**: **多维 Hawkes 过程**(圈层维度由 Leiden 结果决定,预计 5-8 维)
- 参数:圈层自激系数 α_kk、圈层间激发系数 α_kj、衰减速度 β
- 用极大似然估计参数
- 内容影响:用 XGBoost 学 α 关于内容特征(情绪、涉及行业、涉及词)的调制

**Why 选 Hawkes**:
- 事件时间序列可从数据提取(秒级精度)
- 天然支持 What-if(策略=在时刻 t 注入事件)
- 参数少、可解释性高
- 多维形式可对应多圈层

**放弃/降级的方法**:
- **SIR/SEIZ**:降级为宏观热度曲线拟合的辅助方法,不作主传播模型
  - 原因:数据没有 S(易感)人群的直接对应,S(0) 只能拍脑袋
  - 可用于:总热度曲线的 baseline 对比
- **纯 LSTM/XGBoost 预测器**:不作主线,不支持 What-if 反事实

**加分项(时间允许时做)**:
- **Neural Hawkes**:替换 Hawkes 内部为 RNN,精度更高
- **时序 GNN(TGN/EvolveGCN)**:论文对比方法

**尚未落地/待 POC 验证**:
- Hawkes 参数拟合的实际效果
- 情绪演化用什么模型(马尔可夫链 or RNN)
- 反事实验证的具体方案(historical strategy validation 的执行细节)

---

## 计算资源使用原则(2026-07-26 更新)

**当前服务器**:PPIO ppio-gpu = RTX 4090 24GB + 24 vCPU + 62GB RAM(见 [[project-gpu-server]])

**运行前必做的 dry-run 检查**(见 [[project-perf-lessons]]):
每个长跑任务启动 5-10 秒后,SSH 上服务器 `top -bn1` 看 %CPU 是否接近"核数×100";`nvidia-smi` 看 GPU-Util。**不到就 kill 换库**,不要盲等。

**按算法分层的算力配置**:
- **Leiden(圈层识别)**:leidenalg + igraph,C++ 底层,多核。**用 CPU**
- **node2vec**:用 **pecanpy**(Cython/numba 多核)。**不要用 karateclub**(纯 Python 单核锁 GIL,3.7 万节点跑 22 分钟无进展)
- **LINE**:手写 **PyTorch GPU** 版(4090 加速)或 GraphVite
- **GNN(GraphSAGE/GAT/TGN)**:必须 GPU(PyTorch Geometric)
- **HDBSCAN**:参数 `core_dist_n_jobs=-1` 才用多核,默认单核
- **KMeans (sklearn)**:默认多核
- **文本 embedding(BGE/BERT)**:必须 GPU
- **Hawkes 极大似然**:CPU 多核并行
- **XGBoost**:CPU 多核(GPU 加速可选)
- **Neural Hawkes**:GPU
- **索引扫描(CSV → Parquet)**:CPU IO-bound

**推荐云环境**:
- 主机:高核数 CPU + 显存足够的 GPU(4090 或 H100)
- 存储:云对象存储(OSS/S3)存原始 zip 和索引后 Parquet
- 数据集不大时(<3.7 万节点),4090 + 24 vCPU 完全够用

**竞赛级图算法库速查**:详见 [[project-perf-lessons]]。核心规则:选库时先看是否**能跑满硬件**,而不是"能跑就行"。

---

## 云端处理策略(2026-07-26 更新)
- 数据放云存储,用云端 GPU/CPU 处理
- 本地只做 pipeline 验证和踩坑
- **接入方式**:通过 SSH ControlMaster socket 复用免密(见 [[project-gpu-server]]),Claude 可直接 `ssh ppio-gpu` 操作服务器
- 云端脚本(已产出):
  - `build_index.py`:CSV → Parquet 索引
  - `poc_leiden.py`:Leiden 圈层识别
  - `prepare_data.py`:抽取用户互动图和用户级特征
  - `run_algorithms_v2.py`:5 种圈层算法对比(Leiden + pecanpy node2vec + PyTorch LINE + KMeans + fused)
  - `evaluate.py`:多维评估(modularity/画像熵/共识 ARI)
- 待产出:Hawkes 拟合、情绪演化、反事实验证

## 索引层字段清单(build_index.py 产出的 Parquet)

55 列:
- **标识**:md5_id, md5_mid, md5_author, md5_parent_mid, md5_root_mid, md5_root_uid
- **时间**:ts (timestamp[us]), date_str
- **原创/转发**:is_original (bool)
- **核心圈层信号**:sentiment, province (派生), region_valid (bool), industry (list<string>), auth_tier (派生: bigv/org/verified/normal/unknown)
- **辅助信号**:media_type, source, source_level, interest_tags, birth_month, topic_tag, keywords, user_bio
- **数值**:n_followers(唯一可靠), n_likes(废), n_repost(废,非0率<1%), n_comment(废), n_read(废)
- **内容**:title, content_full(用户决定保留全文), image_text
- **溯源**:topic_zip, topic_category, csv_file
- **原始备份**:region_raw, auth_type, industry_raw 等

## 已识别的踩坑清单(供云端复用)
1. **pyarrow 需单独装**——venv 默认没有
2. **编码用 gb18030**——比 GBK 兼容性更好,加 `errors='replace'` 兜底
3. **NULL 哨兵** `d41d8cd98f00b204e9800998ecf8427e` 必须过滤——见 [[project-md5-null-sentinel]]
4. **CSV 异常行**需 `on_bad_lines='skip'`
5. **含长文本**(全文内容),单 CSV 可达 300MB,pandas 读入约 500MB 内存,注意分片处理
6. **时间戳字段少量空**(约 0.02%),要允许 ts=NaT
7. **来源网站字段在原始有两列同名**,pandas 会自动重命名为 `.1`
8. **pandas 3.0 + pyarrow 25**:`dtype=str` 后一些空值是 `'nan'` 字符串
9. **多列名义 100% 填充但含哨兵值**('其他'/'境外'/'--'),各字段需单独规范

参见 [[project-circle-iq]] 整体方案、[[project-data-reality]] 数据情况、[[project-md5-null-sentinel]] MD5 处理注意事项。
