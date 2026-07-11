# 下游应用评估任务与实验协议

日期：2026-07-10  
状态：主实验任务、划分、下游算法与判据的锁定规格

## 1. 研究问题

本协议不再问“哪一种融合向量在无监督几何指标上最好”，而是回答：

1. 在相同原始双流、相同训练/测试划分和相同下游算法下，哪种方法输出的表示更适合应用任务？
2. 双流表示是否超过两个单流中的较优者？
3. 当采样不规则、时钟漂移、缺失和跨流响应时延增强时，Chronaris 是否下降更慢？
4. 连续时间、物理一致性和因果滞后是否能解释任务差异？

## 2. 稳定任务标识

| 机器任务 ID | 读者可见名称 | 数据层 | 主类型 |
| --- | --- | --- | --- |
| `dingxin_maneuver_intensity_weak_classification_v2` | 机动强度弱监督分类 | 鼎新现有真实双流 | 三分类 |
| `dingxin_future_physiology_response_v2` | 机动诱发生理响应预测 | 鼎新现有真实双流 | 回归 + 高响应分类 |
| `sim_workload_forecast_v1` | 仿真负荷提前评估 | 半物理仿真 | 回归 + 三分类 |
| `sim_maneuver_segmentation_v1` | 仿真机动状态分段 | 半物理仿真 | 序列分段 |
| `sim_clock_lag_recovery_v1` | 时间偏移与响应时延恢复 | 半物理仿真 | 机制指标 |

历史 `T1/T2/T3` 只保留在旧代码字段和历史 artifact；新报告、图表和论文表格使用上表名称。

## 3. 鼎新样本组织

### 3.1 原始范围

固定使用：

- 两个 sortie。
- 三个 view。
- 每个 view 37 个连续 5 秒窗口，共 111 个窗口。
- 不追加后续架次，不从其他目录拼接样本。

### 3.2 上下文与预测时域

- 基础窗口：5 秒。
- 历史上下文：连续 6 个窗口，即 30 秒。
- 上下文步长：5 秒。
- 融合表示查询点：每个基础窗口 16 点，30 秒共 96 点。
- 分类标签位置：第六个窗口，即上下文末端。
- 生理响应目标：紧随上下文之后的第七个窗口。

在三个 view 都完整时：

- 机动强度分类每个 view 32 个样本，共 96 个。
- 生理响应预测每个 view 31 个样本，共 93 个。

实际数量以连续性检查为准；窗口时间不连续、目标字段不足或原始点损坏的样本写入 exclusion manifest。

### 3.3 连续性检查

相邻窗口必须满足：

- `next.window_index = current.window_index + 1`。
- `next.start_offset_ms - current.start_offset_ms = 5000 ± 10 ms`。
- 6/7 个窗口属于相同 sortie、view 和 pilot。

不满足时不跨缺口拼接。

## 4. 机动强度弱监督分类

### 4.1 标签目的

该任务衡量当前 5 秒是否呈现低、中、高机动强度，不等价于人工标注的具体机动科目。

### 4.2 标签字段

字段必须先通过 MySQL 元数据映射到以下物理类别：

- 俯仰、滚转、偏航角速度。
- 法向/纵向/侧向过载或加速度。
- 操纵杆、脚蹬或控制面变化。
- 空速、垂直速度或航向变化率。

不允许使用“字段名无法解释时选全部数值字段”的 fallback。每个入选字段必须在 `label_field_manifest.json` 记录 code、中文含义、单位、物理类别和来源。

### 4.3 标签公式

对训练折中的每个标签字段 `k`，计算窗口标准差 `std(k,t)` 和起止变化量 `delta(k,t)`，并拟合稳健尺度：

```text
z_std(k,t)   = clip((std(k,t) - median_train(std_k)) / (IQR_train(std_k) + eps), -5, 5)
z_delta(k,t) = clip((abs(delta(k,t)) - median_train(abs(delta_k))) /
                    (IQR_train(abs(delta_k)) + eps), -5, 5)
field_score(k,t) = max(0, z_std(k,t)) + max(0, z_delta(k,t))
maneuver_score(t) = mean_k(field_score(k,t))
```

其中 `eps=1e-6`。测试折只使用训练折 median/IQR。

若某个物理语义组在训练折中的 `std` 与 `abs(delta)` 两个 IQR 都不大于 `eps`，该组标记为 `both_train_iqrs_are_zero` 并从该 fold 的 `maneuver_score` 分母中删除；不得用 `eps` 把常量字段放大成有效机动信号。每个 fold 至少保留 4 个动态语义组，否则结构化标记为不可用。

训练折 `maneuver_score` 的 33% 和 67% 分位数定义低、中、高三类；阈值写入 fold manifest 后应用于测试折。

### 4.4 标签字段隔离

进入编码器前，从六种方法的航电输入中删除：

- 标签公式使用的原始字段。
- 同字段的窗口统计、差分、变化率和标准化副本。
- 可以确定性重建标签分数的聚合字段。

删除只针对任务输入，不删除原始审计副本。若删除后航电有效字段少于 4 个，该 fold 标记 `insufficient_non_label_vehicle_features`。

### 4.5 主指标

- 主指标：macro-F1。
- 次指标：balanced accuracy、one-vs-rest macro AUPRC、per-class recall、混淆矩阵。
- 稳定性：最差 view fold、三折方向一致性。

## 5. 机动诱发生理响应预测

### 5.1 任务目的

使用截至当前的 30 秒双流历史，预测下一 5 秒生理状态相对于当前状态的变化幅度。该任务描述生理响应，不描述人工工作负荷真值。

### 5.2 目标字段

优先使用在三个 view 均有覆盖且能通过元数据解释的：

- EEG 派生幅值、频带或稳健统计。
- SpO₂。
- 其他已在既有数据合同中确认的生理指标。

字段需要满足：外层训练折有效样本率至少 80%，且训练折 IQR 大于 `1e-6`。不得因为测试折表现重新选择字段。

### 5.3 连续目标

对每个目标字段 `k`：

```text
past(k,t)   = median(上下文最后 5 秒内的有效观测)
future(k,t) = median(未来 5 秒内的有效观测)
delta(k,t)  = abs(future(k,t) - past(k,t))
scaled_delta(k,t) = delta(k,t) / (IQR_train(delta_k) + eps)
response(t) = mean_k(clip(scaled_delta(k,t), 0, 10))
```

训练折拟合 `IQR_train`；测试折只应用。至少两个目标字段有效才生成目标。

G1 固定数据审计只能读取既有特征导出的窗口汇总，因此以窗口均值完成字段覆盖、IQR 和 fold 合同可行性验证，并在 manifest 中强制标记 `window_mean_fallback`。这一结果不进入正式主指标；G2a 原始点冻结完成后，真实任务主实验必须按上述窗口中位数公式重新生成目标，二者不得混算。

### 5.4 高响应分类

训练折 `response` 的 75% 分位数定义高响应阈值。输出：

- 连续响应幅度回归。
- 高响应/非高响应二分类。

未来 5 秒任何数值不得进入输入或 encoder pretext reconstruction。

### 5.5 主指标

- 回归主指标：RMSE。
- 回归次指标：MAE、Spearman 相关系数、每个目标字段的误差。
- 高响应主指标：AUPRC。
- 高响应次指标：macro-F1、recall、Brier score。

## 6. 仿真负荷提前评估

### 6.1 输入与目标

- 使用过去 30 秒原始仿真双流。
- 预测查询时刻后 10 秒的潜在负荷 `w(t+10)`。
- `w` 被生成器限制在 `[0,1]`。
- 负荷等级固定为：低 `<0.35`、中 `[0.35,0.70)`、高 `>=0.70`，不按模型或 fold 重算。

### 6.2 指标

- 连续负荷：RMSE、MAE、Spearman。
- 三分类：macro-F1、balanced accuracy、macro AUPRC。
- 高负荷事件：事件召回、每小时误报数、首次预警提前量、Brier score。

事件定义为 `w>=0.70` 连续至少 5 秒；相邻间隔小于 3 秒的区间合并。

## 7. 仿真机动状态分段

### 7.1 状态集合

固定五类：

1. 稳态飞行。
2. 机动进入。
3. 持续机动。
4. 机动退出。
5. 恢复。

生成器提供逐时间点标签和精确边界；模型只消费双流表示。

### 7.2 指标

- frame-level macro-F1。
- segmental F1@10/25/50。
- normalized edit score。
- 边界 F1，容差 ±2 秒。
- 进入/退出检测延迟。

主指标为 segmental F1@25；边界 F1 是连续对齐贡献的关键指标。

## 8. 方法与表示族

主表固定六种方法：

- 生理单流。
- 航电单流。
- 朴素时间同步。
- MulT。
- ContiFormer。
- Chronaris。

主结果使用 `frozen_task_agnostic_v1` 表示族。另设两个辅助表示族：

- `end_to_end_finetuned_v1`：使用任务标签微调，必须独立成表。
- `synthetic_pretrain_real_adapt_v1`：六方法接受相同仿真预训练预算后适配鼎新，不能替代 real-only 主表。

## 9. 划分协议

### 9.1 鼎新主协议

`leave_one_view_out`：三个 view 各作为一次外层测试组。

对每个外层 fold：

- 剩余 view 用于训练和内部选择。
- encoder 配置只按公共 pretext validation loss 选择，不读取外层测试标签。
- downstream 超参数在训练 view 内部轮换选择。

### 9.2 鼎新辅助协议

`leave_one_sortie_out`：两个 sortie 各作为一次测试组。

由于某个外层训练集合可能只有一个 view，该协议不重新搜索 encoder 或 downstream 配置，直接使用主协议锁定配置；只用于检查跨架次方向，不作显著性结论。

### 9.3 仿真协议

- G1 train：96 架次。
- G1 validation：24 架次，用于任务无关 encoder 选择和 downstream 调参。
- G2 locked test：48 架次，不参与任何选择。
- pilot profile、latent trajectory seed 和 observation seed 在三组间均不重叠。

## 10. 编码器候选选择

每个深度方法使用相同四候选预算：

| 候选 | hidden dim | learning rate | dropout |
| --- | ---: | ---: | ---: |
| A | 64 | `1e-3` | 0.10 |
| B | 64 | `3e-4` | 0.10 |
| C | 32 | `1e-3` | 0.10 |
| D | 64 | `1e-3` | 0.20 |

共同配置：2 层、4 heads、AdamW、weight decay `1e-5`、batch size 128、最多 50 epoch、patience 8、grad clip 1.0。

候选选择分数只使用验证集公共 pretext loss：

```text
selection_loss = 0.50 * normalized_masked_reconstruction
               + 0.25 * normalized_short_horizon
               + 0.25 * normalized_lag_discrimination
```

各 loss 先按同一方法四候选的验证分布归一化。Chronaris 的方法专属正则不进入候选排序，避免用自己定义的损失偏置选择。

归一化固定为同一方法、同一损失项内的 min-max：

```text
normalized_loss = (loss - min_candidate_loss)
                / (max_candidate_loss - min_candidate_loss)
```

若四候选该项损失完全相同，则该项四个归一化值均记为 0。总分相同时先选择参数量更小的候选，再按候选 A、B、C、D 的顺序确定唯一结果。候选 C 只把编码器内部隐层缩小到 32 维；所有方法仍通过合同投影导出 64 维时序表示，下游接口不随候选改变。

G1 正式开发划分使用 96 个训练 profile；24 个 validation profile 中前 23 个用于候选排序，最后 1 个仅用于选定配置的开发确认。`locked_test` 在候选排序期间保持封存。训练增强按 epoch 变化，验证增强固定为 seed 17、epoch 0；每个候选保存独立 `best.pt` 与 `last.pt`，中断后从 `last.pt` 的下一 epoch 恢复优化器、patience 和最佳验证状态。

开发 seed 固定 17；配置锁定后使用 17、29、43 三个 seed 正式确认。

## 11. 冻结表示下游算法

### 11.1 线性探针

分类：

```text
StandardScaler -> LogisticRegression
C in {0.1, 1, 10}
class_weight = balanced
max_iter = 5000
```

回归：

```text
StandardScaler -> Ridge
alpha in {0.1, 1, 10, 100}
```

### 11.2 MiniRocket

使用当前环境已验证的 `aeon==1.5.0`：

```text
MiniRocket(n_kernels=10000, max_dilations_per_kernel=32,
           random_state=seed)
-> StandardScaler(with_mean=False)
-> 与线性探针相同的 LogisticRegression 或 Ridge 网格
```

所有方法的 `n_kernels`、随机种子和后端 estimator 网格一致；不为 Chronaris 单独增加搜索。

### 11.3 状态分段模型

固定浅层 TCN：

- 输入 64 维表示。
- 两个 residual block，channel 64。
- kernel size 5，dilation 1/2。
- dropout 0.1。
- 线性五分类发射头。
- AdamW `1e-3`、最多 40 epoch、patience 6。

后处理使用仓库内实现的 duration-constrained Viterbi：

- 转移概率和状态持续时间只从训练架次估计。
- 不新增 `hmmlearn`。
- 所有方法共享同一平滑配置。

### 11.4 端到端微调

作为辅助实验：

- 从任务无关 checkpoint 初始化。
- 六方法统一 learning rate `1e-4`、最多 20 epoch、patience 5。
- 下游 head 与冻结探针相同容量。
- 输出使用独立表示族，不能覆盖冻结表示结果。

## 12. 融合增益

每个任务和指标都计算：

```text
fusion_gain = score(fusion_method)
            - max(score(physiology_only), score(vehicle_only))
```

对于误差指标，先转换为 `-error` 或直接定义：

```text
fusion_gain_error = min(error(single_streams)) - error(fusion_method)
```

正值表示双流优于最佳单流。

## 13. 统计与汇总

### 13.1 鼎新

- 报告每个 seed × view fold 原始值、均值、标准差和范围。
- 统计单位只有三个 view，不报告窗口级显著性 p-value。
- 结论强调三折方向和最差折，不用 111 个窗口伪装独立样本。

### 13.2 仿真

- 以 locked test 架次为配对单位。
- 对方法差值做 2000 次 paired bootstrap，报告 95% CI。
- 对主指标做 10000 次 paired sign permutation，作为辅助检验。
- 同时报告 clean、单因素 stress 和 mixed-severe，不只报告总体均值。

## 14. 主实验矩阵

| 数据 | 训练轨道 | 任务 | 方法 | 下游 consumer |
| --- | --- | --- | --- | --- |
| 鼎新 | real-only | 机动强度弱监督分类 | 六方法 | 线性 + MiniRocket |
| 鼎新 | real-only | 机动诱发生理响应 | 六方法 | 线性 + MiniRocket |
| G1 -> G2 | synthetic | 仿真负荷提前评估 | 六方法 | 线性 + MiniRocket |
| G1 -> G2 | synthetic | 仿真机动状态分段 | 六方法 | TCN + Viterbi |
| G1 -> G2 | synthetic | 时间偏移/时延恢复 | 四种双流方法 | 机制指标 |
| G1 -> 鼎新 | transfer | 两个鼎新任务 | 六方法 | 与 real-only 相同 |
| UAB/NASA | existing public route | 既有公开任务 | 既有矩阵 | 只刷新汇总 |
| 鼎新/G2 | appendix | 结构诊断 | 锁定方法 | ClaSP/STUMPY |

## 15. 模型选择和论文判据

E3 不参与选择。锁定 Chronaris 需要：

1. encoder OOF、合同和三 seed 完整。
2. 鼎新生理响应 RMSE 不劣于最佳深度基线，且至少两个 view fold 方向为正；或仿真负荷主指标达到最佳/统计并列最佳。
3. 在 mixed-severe 或至少两个单因素高强度场景中，Chronaris 的退化斜率优于最佳深度基线。
4. 至少一项双流任务的 fusion gain 为正。
5. 关键消融与方法设计方向一致。

若条件未满足，保留完整负结果并将结论写成 mixed；不得继续无边界搜索直到出现胜出表格。
