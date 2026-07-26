# 内容维度引入方案(未推进,存档)

**创建**: 2026-07-26
**状态**: 只记录方案,未实施。等用户决定要看 Leiden 圈内的内容一致性数据时启动阶梯 1。
**前置**: 已完成关系网络 Leiden POC,详见 [circle-identification-report.md](circle-identification-report.md)

---

## 1. 起因和讨论

**用户观察**: 最初评估标准里没有"内容文本相似度"这一维度,现在觉得应该加。担心当前 Leiden 圈层没有考虑用户发的帖子内容,可能存在"同圈但讨论内容各异"的情况。

**两种可能的路径**:

- **路径 A**: 内容文本**作为评估指标**(不改主算法)
  - Leiden 依然是主线
  - 只在评估阶段多算一个"圈内内容一致性",看看数据
  - **不改变圈层划分**

- **路径 B**: 内容文本**作为划分依据**(改主算法)
  - 主算法从纯图算法变成"图+内容"融合
  - **会改变圈层划分**

**结论(2026-07-26)**:
用户选择先只记录方案,等看到 Leiden 圈内内容一致性数据之后再决定要不要走路径 B。

**动机链**: `看数据 → 决定`。如果 Leiden 圈内内容相似度已经很高(比如 >0.7),意味着"有互动的人自然讨论相近话题",路径 B 就没必要走。如果内容相似度很低,再考虑路径 B。

---

## 2. 路径 A:内容作为评估指标(推荐最先做)

### 2.1 步骤

**第 1 步:算每个用户的文本 embedding**
- 每个用户把 TA 所有帖子的文本(title + content_full)拼接或分句 embedding
- 用 `BAAI/bge-small-zh-v1.5`(中文 SBERT,512 维)编码
- 池化方式:mean pooling(平均池化)或对 TA 影响力大的帖子加权
- 输出:37022 × 512 的用户文本 embedding 矩阵

**第 2 步:算圈内内容一致性**
- 对每个 Leiden 圈,计算圈内两两用户 embedding 的**平均余弦相似度**
- 大社区(size >= 50)才算,小社区噪声太多
- 或者:算圈内 embedding 的方差(越低越一致)

**第 3 步:出对比表**
- 每个圈:size / avg_cosine_sim / std / top3 主题词
- 全局:平均圈内相似度 vs 随机情况的对比
- 类似 4.3 节的"画像一致性熵",但换成"内容一致性"

### 2.2 判据

| 平均圈内 avg_cosine_sim | 含义 | 决策 |
|---|---|---|
| > 0.7 | Leiden 圈内内容已经很一致 | 路径 B 不需要 |
| 0.5 - 0.7 | 部分圈一致,部分散乱 | 只对散乱圈做后处理细分(阶梯 2) |
| < 0.5 | Leiden 圈内容散乱 | 应该改主算法(阶梯 3) |

### 2.3 成本估算

- BGE embedding 编码 37022 个用户,每人取 5-10 条代表性帖子拼接 = ~20 万条句子 embedding
- 4090 上 BGE-small 大约 500 条/秒 → **约 5-8 分钟**
- 圈内相似度计算 = O(社区内用户对数),Leiden top10 社区总用户 26k,pairs ~40M,GPU 上 1 分钟
- **总计:半小时以内**

### 2.4 脚本骨架(未实现)

```python
# Step 1: encode
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

model = SentenceTransformer('BAAI/bge-small-zh-v1.5', device='cuda')

df = pd.read_parquet('hunsu_first3.parquet',
                    columns=['md5_author','title','content_full'])
# 每个用户拼接前 5 条帖子
grp = df.groupby('md5_author').agg({
    'title': lambda x: ' '.join(list(x.dropna())[:5]),
    'content_full': lambda x: ' '.join(list(x.dropna())[:5])
})
texts = (grp['title'] + ' ' + grp['content_full']).tolist()
users = grp.index.tolist()

# 每人 mean-pool 或直接拼接后编码
embs = model.encode(texts, batch_size=64, show_progress_bar=True,
                    normalize_embeddings=True)  # (37022, 512)

pd.DataFrame({'md5_author': users, 'emb': list(embs)}).to_parquet('user_text_emb.parquet')

# Step 2: 圈内一致性
from sklearn.metrics.pairwise import cosine_similarity
partition = pd.read_parquet('partition_leiden.parquet')
merged = partition.merge(pd.DataFrame({'md5_author':users, 'idx': range(len(users))}))

results = []
for cid, sub in merged.groupby('community_id'):
    if len(sub) < 50: continue
    sub_embs = embs[sub['idx'].values]
    # 对大社区,采样计算(避免 pairs 太多)
    if len(sub_embs) > 500:
        idx = np.random.choice(len(sub_embs), 500, replace=False)
        sub_embs = sub_embs[idx]
    sim_mat = cosine_similarity(sub_embs)
    upper = sim_mat[np.triu_indices_from(sim_mat, k=1)]
    results.append({'community_id': cid, 'size': len(sub),
                    'avg_cosine': float(upper.mean()),
                    'p50_cosine': float(np.percentile(upper, 50))})

pd.DataFrame(results).to_csv('leiden_content_consistency.csv', index=False)
```

---

## 3. 路径 B:内容进入圈层划分(不推荐立即做)

**只在阶梯 1 显示"Leiden 圈内内容散乱"时启动**。

### 3.1 五大类算法家族

按"内容和结构如何融合",从简单到复杂:

#### 3.1.1 类别 1:后处理融合(层次化圈层)

**思路**:先跑 Leiden 得顶层,再在每个大社区内部按内容再细分。

**做法**:
- Leiden 顶层圈层保留
- 对头部大社区(size >= 200,预计前 20-30 个),取内部用户的文本 embedding
- 对这些 embedding 做 KMeans 或 HDBSCAN → 得到"社区.子内容圈"
- 输出层次结构:CID=1 → 1.a、1.b、1.c

**优点**:
- 保留 Leiden 优势(结构模块度 0.75)
- 增加内容细分维度
- 论文上可以写成"层次化圈层"
- 直接解决"头部社区过大"的问题(POC 里 CID=0 有 7281 用户)

**缺点**:
- 内容不影响顶层划分——两个内容相似但没互动的用户依然分不到一个圈
- 是"加法"不是"融合"

**成本**:一天工作量

#### 3.1.2 类别 2:多视角/联合嵌入

**2a. Concat + 加权**(最简单,POC 里 fused 用画像失败过)
- 结构 embedding(node2vec 64 维)+ 内容 embedding(BGE 512 维)
- 拼接或加权,进 HDBSCAN
- 注意:POC 里 fused_struct_profile 的 ARI 全都 <0.31,说明简单拼接效果差
- 换成内容 emb 可能好些(内容比画像信号更强),但不保证

**2b. Multi-View Clustering(MVC)**
- 分别在结构和内容上做聚类
- 通过共识优化对齐两个划分
- 库:`multiview` 或论文级实现,不够成熟

**2c. CCA / Deep CCA(典范相关分析)**
- 找结构空间和内容空间的公共子空间
- 在公共空间聚类

**评价**:中等复杂,`fused` POC 已证明简单拼接不行,MVC/CCA 需要具体实验才知道效果。

**成本**:3-5 天

#### 3.1.3 类别 3:图神经网络(GNN,论文最能加分)

**核心思想**:让神经网络**同时处理**图结构和节点属性(内容 embedding)。

**3a. GraphSAGE 无监督**
- 输入:每个节点的内容 embedding(BGE 512 维)作为节点特征
- GNN 消息传递:每个节点向量融合"自己的内容"+"邻居的内容"+"邻居的邻居的内容"
- 训练目标:让直连节点向量相近,非直连节点相远(无监督对比学习)
- 最后节点向量做 HDBSCAN 得社区

**3b. DGI (Deep Graph Infomax)**
- 通过"节点向量应该和图全局摘要一致"的对比目标训练
- 无监督 GNN 的经典方法

**3c. GAT(注意力机制)**
- 类似 GraphSAGE,但每个邻居的贡献权重是学出来的
- 学到的注意力权重可以做可解释性分析("哪些邻居对我的圈层归属更重要")

**优点**:
- 天然融合结构+内容,不是拼接
- 论文亮点强,评委喜欢
- 学习到的向量下游可以做别的(链接预测、传播预测)

**缺点**:
- 训练重(GNN 相比 CPU 算法慢),需要 GPU
- 调参多(层数、隐藏维度、聚合方式、学习率)
- 可解释性差
- 需要 PyTorch Geometric 或 DGL

**成本**:3-5 天(实现+调参)

#### 3.1.4 类别 4:属性图社区发现(经典算法)

**思路**:在传统社区发现算法的目标函数里同时包含"结构模块度"和"属性一致性"。

**4a. Attributed Network Embedding (ANE)** - DANE / AANE
- 联合优化结构损失和属性损失

**4b. CESNA(Communities from Edge Structure and Node Attributes)**
- 概率生成模型:每个用户有若干"社区隶属度",观测到的边和属性都由此生成
- EM 或变分推断学社区

**4c. Attributed SBM(属性化随机块模型)**
- 传统 SBM 只用图,Attributed SBM 加入节点属性
- 贝叶斯框架,可解释性好

**评价**:
- 理论完美,可解释性比 GNN 好
- 但实现库不成熟(基本都是论文代码,不像 PyG/DGL 那样工业级)
- 踩坑多

**成本**:高(实现难度大)

#### 3.1.5 类别 5:文本聚类为主 + 图后过滤

**思路**:反着来——先按内容聚类,再用图关系验证。

**做法**:
- BGE 编码 → HDBSCAN 得内容圈
- 对每个内容圈,验证内部图连通率
- 连通率高保留,连通率低拆分

**评价**:内容主导,不符合"以关系为主"的原始哲学。快但不契合项目定位。

### 3.2 方法家族对比表

| 类别 | 代表算法 | 融合方式 | 可解释性 | 论文加分 | 工作量 |
|---|---|---|---|---|---|
| 1 后处理 | Leiden + KMeans(subclust) | 层次(加法) | ★★★★★ | ★★ | 1 天 |
| 2a Concat | node2vec ‖ BGE + HDBSCAN | 向量拼接 | ★★★ | ★★ | 1 天 |
| 2b MVC | Multi-View Consensus | 划分对齐 | ★★★ | ★★★ | 3 天 |
| 3a GNN | GraphSAGE 无监督 + HDBSCAN | 神经消息传递 | ★★ | ★★★★★ | 5 天 |
| 3b DGI | Deep Graph Infomax | 对比学习 | ★★ | ★★★★★ | 5 天 |
| 4 属性图 | CESNA / Attributed SBM | 联合概率模型 | ★★★★ | ★★★★ | 5-7 天 |
| 5 内容主导 | BGE + 图过滤 | 反向 | ★★★ | ★ | 2 天 |

### 3.3 如果真要做,推荐路径

**首选**: 类别 1(后处理),因为:
- 保留 Leiden 主线,不破坏已有结论
- 直接解决"头部社区过大"问题
- 论文可写成"两级层次化圈层",结构清晰
- 一天完成

**次选**: 类别 3a(GraphSAGE),因为:
- 论文加分最大
- GPU 有实质工作(前面 POC 里 4090 只用了 0.2 秒)
- 学到的向量可以复用到传播预测层的用户特征
- 但工作量大,时间紧张时不推荐

---

## 4. 现阶段决定(2026-07-26)

**状态**: 只记录方案文档,不启动任何实施。

**触发条件**:
- 用户后续想看 Leiden 圈内的内容一致性数据 → 启动阶梯 1(路径 A)
- 阶梯 1 显示内容散乱 → 启动阶梯 2(类别 1)或阶梯 3(类别 3a)
- 论文时间紧张 → 只做阶梯 1(评估维度),不改主算法

**为什么现在不做**:
- Leiden 已经跑出模块度 0.75、稳定性 100% 的好结果
- 内容一致性数据没看到之前,加内容维度是猜测性优化
- 加了不一定更好,反而稀释信号(POC 里 fused 就是先例:结构+画像拼接,五个指标都变差)
- 时间紧,优先推进传播预测层(Hawkes)

---

## 5. 备用脚本清单(未执行)

若后续启动阶梯 1,需在服务器上做:

```bash
# 1. 装依赖
pip install sentence-transformers

# 2. 下载模型(下载后可缓存复用)
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# 3. 跑内容 embedding + 一致性评估(见 2.4 节脚本骨架)
python encode_users_text.py
python leiden_content_consistency.py
```

---

## 6. 论文写作提示

如果最后决定"不做路径 B,只做路径 A",论文里"网络关系提取"部分应包含以下段落:

> 在评估指标层,除了传统的模块度、稳定性、可解释性,我们额外加入了"圈内内容一致性"(cosine similarity of user text embeddings)作为辅助验证。实验显示,Leiden 得到的圈层内容一致性为 X.XX,表明基于关系互动形成的社群同时也在讨论主题上高度相近——这一发现进一步支持了"选择关系网络作为圈层主定义"的合理性。

若做路径 B(类别 1),额外加:

> 我们进一步在每个大社区内部按内容 embedding 做二级细分,得到"社区.子内容圈"的层次化圈层结构。相比单纯的 Leiden 划分,层次化圈层解决了头部社区过大的问题(如 CID=0 从 7281 用户细分为 X 个内容子圈),并为后续策略层提供了更精细的引导对象。

参见 [circle-identification-report.md](circle-identification-report.md)。
