# pred_screen — 圈0/圈5 传播预测算法筛选实验

**目标**: 在圈 0(高产政务转发层, 5,678 人)与圈 5(大V枢纽, 8,924 人)上,统一协议对比
经典/回归类与神经类传播预测算法,各选出 1 个进入后续大规模验证。

## 任务定义

给定事件 e 在锚点 t 之前的**全部真实历史**(帖子流、互动流),预测圈 c ∈ {0, 5} 的成员
在 (t, t+H] 内于该事件中的发帖量(原创+转发,按 md5_mid 去重)。

- 主视界 H=24h,副视界 H=6h(圈粒度 2h 窗口中位计数为 0,不可用——见 screening_scan)
- 锚点: 事件窗 [T0+24h, T1-24h] 每 6h 滚动
- 切分: **按事件** 60/10/30 = train/val/test(类别×体量分层交错),模型不得见 test 事件任何数据
- 锚点过滤(仅依据历史,无标签泄漏): 全通道近 7 天计数 > 0(事件休眠期不预测)
- (事件, 圈) 对可用条件: 该圈在该事件总帖量 ≥ 100

## 通道定义(优先级从上到下)

| 通道 | 定义 |
|---|---|
| c0 / c5 | 稳定圈 0 / 5 成员(K=3 分区, partition_stable_K3) |
| stab | 其他稳定圈成员 |
| bigv / org | 圈外大V / 圈外机构(auth_tier) |
| other | 其余散户 |

## 候选算法

经典/回归族: naive-24h 持续、季节 naive(7 天前同窗)、EWMA、Poisson 外推(基线);
SIR 滑窗 ODE 拟合;独立级联 IC(圈内稳定图 MC);4 通道 Hawkes(锚点前重拟合,双时间尺度核,
vendored `hawkes_lib.py` = 已验证的 data-pipeline 拟合器);LightGBM 特征回归。

神经族: GRU 多通道序列;TGN-lite(记忆式时间图网络,交互流驱动);EvolveGCN-O(6h 快照,
GRU 演化 GCN 权重,自实现)。

## 指标(test 事件 × ≤16 锚点统一网格)

MALE(mean |Δlog1p|,主指标)、RMSLE、中位 APE(分母 max(1,y),与阶段D口径一致)、
胜过 naive 占比、AUC(y>0 判别)、每模型训练/推理成本。

## 运行(服务器 ~/circleiq/pred_screen)

```bash
source ~/circleiq/venv/bin/activate
export POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 NUMBA_NUM_THREADS=2
python3 build_dataset.py --workers 10        # core+partition → out/{events,streams}/*.npz + anchors.parquet
python3 baselines.py                         # naive/seasonal/ewma/poisson
python3 fit_sir.py / fit_ic.py / fit_hawkes4.py --workers 12
python3 train_gbdt.py / train_gru.py / train_tgn.py / train_evolvegcn.py   # GPU
python3 evaluate.py                          # → out/results/screening_results.json + 排名表
```
