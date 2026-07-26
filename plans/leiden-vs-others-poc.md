# Leiden vs 其他圈层算法 POC 计划

**创建**: 2026-07-26
**执行环境**: `ppio-gpu`(RTX 4090 单卡 24GB / 24 vCPU / 62GB RAM / Ubuntu 22.04 / Python 3.10.12,空环境)
**预估总时**: 2-3 小时(含环境搭建、上传数据、跑算法、评估)

## 一、目标

回答两个问题:

1. **Leiden 之外的算法(node2vec、LINE、纯画像切分、结构+画像融合)在同一数据集上效果如何?**
2. **Leiden 得到的社区是"真实圈层"还是"算法伪影"?**
   - 具体诊断:Leiden 结果和纯画像切分的一致性(ARI)。高 → Leiden 没多加信息;低 → Leiden 抓到了画像之外的结构信号

## 二、数据

- **数据集**: 婚俗改革 3 csv 样本(2.4 万独立用户 / 7.4 万帖 / 6.9 万转发边)
- **本地路径**: `/Users/ppio/.claude/jobs/15487124/tmp/index_output/major_topics/婚俗改革.parquet` (只用前 3 csv 的行,即已存的完整索引后的 head)
- **实际做法**: 直接用完整索引 parquet(3.09GB),POC 阶段先只提取 3 个 csv 对应的行,这样和先前的样本统计一致

**选择理由**:
- 内容和画像多样性最好(横跨山东/台湾/河南/安徽等地)
- 图密度和用户数适中,不太可能撑破 24GB 显存
- 前面已经跑过 Leiden POC(模块度 0.767),有 baseline

## 三、要跑的 5 种算法

| # | 方法 | 信号来源 | 库 |
|---|------|---------|---|
| 1 | Leiden(baseline) | 只有图结构 | `leidenalg` + `igraph` |
| 2 | node2vec + HDBSCAN | 只有图结构(嵌入路径) | `pecanpy` 或 `karateclub` |
| 3 | LINE + HDBSCAN | 只有图结构(嵌入路径) | `karateclub` |
| 4 | 纯画像 K-means | 只有用户属性字段 | `sklearn` |
| 5 | 结构 emb ‖ 画像 → HDBSCAN | 两者融合 | 拼接向量 + `hdbscan` |

**内容 embedding 暂时不加**——用户明确评估标准不包含内容相似度。如果 POC 结果诡异再补。

### 各算法关键参数

- **Leiden**: modularity partition,3 个不同种子取共识 baseline
- **node2vec**: 128 维,walk_length=40,num_walks=10,p=1,q=1(平衡 BFS/DFS)。之后 HDBSCAN min_cluster_size=50
- **LINE**: 128 维,一阶+二阶各半
- **画像特征**: province 独热(34 维)+ industry 多标签(~10 维)+ auth_tier 独热(5 维)+ log(n_followers+1) 单一数值
- **K-means k**: 用 Leiden 得到的社区数附近搜(k = 500, 1000, 2000)

## 四、评估指标

**结构维度**:
- Modularity(圈内边密度 vs 期望)—— Leiden 天然满分,其他算法能到多少?
- Conductance(圈边界的"割")

**画像一致性维度**:
- 圈内地域熵(越低越集中)
- 圈内行业熵
- 圈内认证类型熵

**稳定性维度**:
- 同算法 3 个不同种子的 ARI(调整兰德指数)

**共识度维度** (关键诊断):
- 5 种算法两两 ARI
- **Leiden vs 纯画像 K-means 的 ARI** ← 直接回答"Leiden 是否值得"

**可解释性维度** (人工):
- 对每个算法的 top 10 社区,自动提取"top 省份/top 行业/top 大V/情绪top1",人工判断能不能起清晰名字

## 五、执行步骤

### Step 1: 服务器搭环境(20-30 分钟)

```bash
ssh ppio-gpu
mkdir -p ~/circleiq && cd ~/circleiq
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# 核心
pip install numpy pandas pyarrow scipy scikit-learn
# 图/社区
pip install python-igraph leidenalg
# 图嵌入
pip install karateclub gensim
# 聚类
pip install hdbscan
# GPU torch(可选,先不装,POC 用不上)
```

### Step 2: 上传数据(2-5 分钟)

```bash
scp /Users/ppio/.claude/jobs/15487124/tmp/index_output/major_topics/婚俗改革.parquet \
    ppio-gpu:~/circleiq/
```

只传 parquet,不传原始 CSV。3GB,家用宽带上行 5-10 分钟。

### Step 3: 数据准备(5 分钟)

写 `prepare_data.py`:
- 从 parquet 抽出前 3 csv 对应的行
- 构造用户-用户互动图(child_author → parent_author,通过 parent_mid 查作者)
- 输出:
  - `edges.parquet`(child_author, parent_author, weight)
  - `user_features.parquet`(md5_author, province, industries, auth_tier, log_followers, dominant_sentiment, n_posts)

### Step 4: 跑 5 种算法(30-60 分钟)

写 5 个独立脚本或一个统一 driver,每种算法产出:
- `partition_<method>.parquet`(md5_author, community_id)
- 用于共识对比

### Step 5: 评估(15-30 分钟)

写 `evaluate.py`:
- 对每个 partition 计算所有指标
- 输出 `results.csv` 和 `human_check_top10.md`(每方法的前10社区自动画像)

### Step 6: 拉结果回本地(2 分钟)

```bash
scp ppio-gpu:~/circleiq/results.csv ./poc_results/
scp ppio-gpu:~/circleiq/human_check_top10.md ./poc_results/
```

## 六、成功标准

- 5 种算法都跑通,得到 5 个 partition
- 各维度指标都出得来
- **关键回答**: Leiden vs 纯画像 K-means 的 ARI 是多少?
  - < 0.3: Leiden 抓到了画像之外的信号,值得作为主线
  - 0.3–0.6: 部分重合,可以考虑"画像 + 图微调"的混合方案
  - > 0.6: 画像 K-means 就够了,Leiden 加值不明显 → 简化方案

## 七、可能出问题的地方

- **karateclub 依赖 networkx 老版本**:装不上就换 pecanpy(专门做 node2vec 的 CPU 加速版)
- **7 万边图 node2vec**:每节点 10 walks × 40 步 = 24 万节点 × 400 步... 应该几分钟能出。真出问题就降到 5 walks × 20 步
- **HDBSCAN 在 128 维直接跑**:通常没问题,如果太慢先 UMAP 到 30 维
- **服务器主连接掉**:重连需要用户输密码,任务开始前先 `ssh ppio-gpu 'true'` 确认

## 八、POC 完成后的收敛

结论合并到 `memory/project_algorithm_decisions.md`:
- 明确"Leiden 是否值得"的答案
- 如果换方案,更新主线算法决策

不合并到 memory 的内容:
- 5 种算法各自跑通的具体脚本(留在服务器 ~/circleiq/)
- 详细指标数字(留在本地 poc_results/)

## 九、下一步(POC 之外)

- 传播预测 POC(Hawkes 参数拟合)—— 与本 POC 独立,可以并行准备
- 反事实验证方案设计
