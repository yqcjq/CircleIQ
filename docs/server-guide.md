# GPU 服务器指南(ppio-gpu:/root/data 目录结构与结果获取)

**用途**: 快速定位服务器上的脚本、数据、结果与日志;不重跑就能取到所有结论数字。
**接入**: `ssh ppio-gpu`(ControlMaster 免密复用,主连接断需用户重输一次密码;详见 memory: project-gpu-server)
**环境**: `source ~/circleiq/venv/bin/activate`(Python 3.10 + polars/numba/igraph/leidenalg/torch cu121)
**硬件**: RTX 4090 24GB + 24 vCPU + 62GB RAM;磁盘 200G(当前用约 30G)

---

## 1. 目录结构(2026-07-27 实况)

```
/root/data/
│
├── 数据层(本地管线上传)
│   ├── core/major/   9 个 parquet      # 7大议题建模数据(重复zip已剔),43列
│   ├── core/hot/     263 个 parquet    # 热点事件
│   ├── core/case/    112 个 parquet    # 传播案例          [共2.8GB,全量]
│   ├── text/{major,hot,case}/          # 长文本层(48GB后台补传中)
│   ├── stats/        385 个 json       # 每zip处理统计(行数/用户/去重/时间范围)
│   ├── catalog.parquet                 # zip→事件名→统计总目录(duplicate_of列标重复)
│   └── user_text_stable.parquet        # 稳定圈用户聚合文本(9.6万人,BGE输入)
│
├── 阶段A产物(圈层识别)
│   ├── edges/        382 个 parquet    # 每议题互动边表(src,dst,weight,weight_pre,ts_min/max)
│   ├── edges_summary.json              # 各议题边数/方法汇总
│   ├── stable_pairs[_pre].parquet      # 用户对×议题数聚合(n_topics/n_major/total_weight)
│   ├── stable_edges_K3[_pre].parquet   # K=3 稳定图边
│   ├── partition_stable_K3[_pre].parquet  # ★ 用户→稳定圈ID(_pre=仅2025上半年互动,防泄漏版)
│   ├── partition_topic/  7 个 parquet  # 单议题Leiden(对比基线)
│   ├── stable_report_K3[_pre].json     # ★ K网格/模块度/种子ARI/top圈
│   ├── leiden_per_topic.json           # 单议题模块度汇总
│   ├── user_profiles.parquet           # 用户级画像聚合(1223万人)
│   ├── eval_stable_K3.json             # ★ A5评估:画像构成/激活谱/ARI/稳定vs其他
│   ├── content_consistency_stable.json # ★ BGE内容一致性(vs随机基线)
│   └── user_text_emb_stable.npy        # BGE embedding(9.1万×512)
│
├── 阶段C/D/E产物(传播预测与策略)
│   ├── hawkes/       312 个 json       # ★ 每事件拟合参数:mu[K,6],mu_edges,A1,A2,beta1,beta2,
│   │                                   #   dim_labels,LL增益,验证LL(v3双核版)
│   ├── hawkes/events/ 312 个 npz       # 每事件事件流(times,dims)——模拟/验证复用
│   ├── hawkes_summary_all.json         # ★ 全量拟合汇总(报告图数据源)
│   ├── event_features.parquet          # 事件级内容特征(情绪分布/敏感/行业/大V占比)
│   ├── content_modulation.json         # ★ 情绪分组参数对比 + OLS系数
│   ├── validation_rolling.json         # ★ 滚动2h验证(312事件×8锚点,APE三基线对比)
│   ├── validation_counts.json          # (v1长窗验证,已被rolling取代,留作方法演化记录)
│   ├── strategy/     8 个 json         # ★ 每事件策略网格(通道×注入量×时机→ROI/失控/溢出)
│   ├── strategy_summary.json           # top策略汇总
│   └── case_studies/ 4 个 json         # ★ 大V反事实三线曲线(actual/factual/counterfactual)
│
├── 脚本(与仓库 pipeline/server/ 一一对应,仓库为准)
│   ├── build_edges.py  stable_circles.py  leiden_per_topic.py  user_profiles.py  eval_stable.py
│   ├── hawkes_fit.py(模型核心)  run_hawkes.py(批量拟合)  simulate.py(反事实模拟器)
│   ├── validate_rolling.py  analyze_content_modulation.py  content_features.py
│   ├── strategy_optimizer.py  case_study.py  embed_consistency.py
│   ├── stage_a.sh(阶段A串联)  stage_cde.sh(验证/调制/策略/案例串联)
│   ├── chain_all.sh / chain_v2.sh / chain_v3.sh(全链重跑,v3=当前版)
│   ├── kill_chain.sh(安全清杀)  README.md(数据集说明)
│   └── (t_edge.py=冒烟测试残留;validate_counts.py=v1已弃用)
│
└── 日志(每次运行的完整记录)
    stage_a_final.log hawkes_v3.log stage_cde_v3.log chain_v3.log  # 当前版
    hawkes_small*.log hawkes_full*.log chain_all.log ...           # 历史版(演化记录)
```

## 2. 快速获取结果(不重跑)

```bash
# 一把抓全部结论JSON到本地(小文件,秒级)
scp 'ppio-gpu:/root/data/{stable_report_K3.json,eval_stable_K3.json,hawkes_summary_all.json,validation_rolling.json,content_modulation.json,strategy_summary.json,content_consistency_stable.json}' ./

# 圈层:某用户属于哪个稳定圈
ssh ppio-gpu "source ~/circleiq/venv/bin/activate && python3 -c \"
import polars as pl
p = pl.read_parquet('/root/data/partition_stable_K3.parquet')
print(p.filter(pl.col('md5_author')=='<某md5>'))\""

# 某事件的Hawkes参数(α矩阵/维度标签)
scp 'ppio-gpu:/root/data/hawkes/hot__20260313162141.json' ./

# 某事件的策略网格
scp 'ppio-gpu:/root/data/strategy/hot__20260313162142.json' ./
```

**注**: 仓库 `results/` 已版本化留存上述所有 JSON(与服务器同源),优先看仓库副本。

## 3. 重跑指南

```bash
ssh ppio-gpu
cd /root/data && source ~/circleiq/venv/bin/activate
# 线程纪律(必须,否则24核load会冲到130):
export POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMBA_NUM_THREADS=2

nohup bash chain_v3.sh > chain_v3.log 2>&1 &   # 全链:自测→Hawkes全量→验证/调制/策略/案例(约12分钟)
nohup bash stage_a.sh  > stage_a.log  2>&1 &   # 仅阶段A:边表→画像→单议题Leiden→稳定圈→评估(约5分钟)

# 单步示例
python3 run_hawkes.py --category hot --limit 12 --pool 12 --workers 8   # 小规模
python3 stable_circles.py --k 4                                         # 换K值
python3 case_study.py --name hot__20260319181009                        # 指定事件案例
python3 strategy_optimizer.py --targets hot__20260313162142 --n-runs 200
```

**踩坑速查**(详见 memory: project-perf-lessons):polars 子进程必须 spawn;pkill 模式别匹配到自己的 ssh 命令行;远程 python 超一行就写文件 scp 过去执行,别拼引号。

## 4. 产物↔脚本↔报告 对照

| 想要的结论 | 看哪个文件 | 产出脚本 | 报告章节 |
|---|---|---|---|
| 稳定圈规模/模块度/K网格 | stable_report_K3.json | stable_circles.py | stage-a §3-4 |
| 圈的画像/激活谱/ARI | eval_stable_K3.json | eval_stable.py | stage-a §5-7 |
| 内容一致性 | content_consistency_stable.json | embed_consistency.py | stage-a §8 |
| 某事件α矩阵/β/增益 | hawkes/<事件>.json | run_hawkes.py | stage-cde §1-2 |
| 预测精度三基线对比 | validation_rolling.json | validate_rolling.py | stage-cde §3 |
| 情绪×传播参数 | content_modulation.json | analyze_content_modulation.py | stage-cde §5 |
| 策略ROI/失控/溢出 | strategy/<事件>.json | strategy_optimizer.py | stage-cde §6 |
| 大V反事实曲线 | case_studies/<事件>.json | case_study.py | stage-cde §4 |
