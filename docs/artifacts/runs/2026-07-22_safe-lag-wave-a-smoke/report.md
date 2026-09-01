# 波次 A 工程冒烟：安全滞后感知融合是否保留航电信息

> 状态：`superseded_due_to_protocol_defects`。本结果受重复归一化和训练确定性不足影响，不进入毕业论文主结果；原数字和运行说明仅用于追溯研究过程，等待评价协议 v3 重跑。

状态：completed（机制工程冒烟，非锁定确认）。分支：`research/safe-lag-aware-fusion-20260718`。日期：2026-07-22。

## 目的

验证新主线 `SafeLagAwareFusion`（`z_out=[phys_private, vehicle_private, gate·z_cross]`，门控初始化近 0）相对原 `MultiScaleCausalLagFusion`（单一 256→64 瓶颈、无单流旁路）是否在真实训练后改善审计确认的两项缺陷：航电单流信息丢失（Q1/Q2）与维度塌缩（Q8）。

## 设置

- 数据：G1 仿真固定集（`artifacts/application_evaluation/2026-07-10_aviation-simulation-formal`），只读。取 48 训练 + 12 验证 + 24 留出观测上下文（每上下文 30 秒）。
- 训练：`train_common_pretext_method`，15 epoch、batch 8、seed 17，公共遮挡重构 + 短期预测 + 时延判别。三组：`chronaris_safe_lag`、`chronaris_multiscale`、`vehicle_only`（航电单流参考）。
- 诊断（reeval 脚本，从 checkpoint 重算，特征经 StandardScaler 标准化）：
  - 有效秩：在全量 84 上下文表示上的参与比 `sum(s)²/sum(s²)`。
  - 航电信息线性恢复 R²：Ridge 从冻结 64 维窗口表示预测 12 个航电字段的窗口均值；为消除训练→留出分布漂移，采用**直推式**（在同一留出集上拟合并评价，度量“航电信息是否线性可恢复”，跨方法公平）。

## 结果

| 方法 | 有效秩 | 航电恢复 R²（直推） | 安全门控均值 |
| --- | --- | --- | --- |
| Chronaris safe_lag | **5.13** | **0.944** | 0.022 |
| Chronaris multiscale（旧） | 2.73 | 0.914 | n/a |
| vehicle_only（航电单流参考） | 6.56 | 0.933 | n/a |

## 判断

- **航电信息保留改善**：safe_lag 的航电恢复 R²（0.944）高于旧 multiscale（0.914），甚至略高于航电单流参考（0.933）。这与单元测试 `test_safe_lag_preserves_vehicle_info_via_bypass` 一致——航电信息经专用 `vehicle_private` 旁路进入输出，不再被迫穿过单一瓶颈。
- **维度塌缩减轻**：safe_lag 有效秩 5.13 约为旧 multiscale 2.73 的 1.88 倍，私有/交叉解耦降低了同刻共线与瓶颈造成的塌缩压力。
- **安全门控保守**：safe_lag 门控均值 0.022，15 epoch 后仍接近安全回退（近 0），符合“先无损、再渐进加入跨模态增量”的设计；模型默认不破坏单流。

## 边界与下一步

- 本轮为**机制工程冒烟**：15 epoch、单 seed、仿真数据、直推式恢复探针。它支持“安全旁路改善航电保真与有效秩”的机制判断，**不构成鼎新锁定确认**。
- 真实下游负迁移是否消除需在波次 A 完整预算（50 epoch、多随机种子）下的鼎新 inner-validation 未来机动任务上验证，并按评价协议 v2 的预冻结门禁裁定。
- 跨流增量（波次 B）、公开预训练迁移（波次 C）与任务感知优化（波次 D）尚未运行。
- 重型 checkpoint 位于被忽略目录 `artifacts/application_evaluation/2026-07-22_safe-lag-wave-a-smoke/`；本目录只提交紧凑指标与报告。

## 下游机动代理预测（负迁移机制检验）

为检验“安全旁路是否在车辆主导任务上消除负迁移”，用波次 A 的 checkpoint（不重训）在车辆主导的下游目标上比较：用 30 秒上下文表示预测其后 5 秒（[30,35]s）的机动强度（角速率+加速度通道的 L2 范数均值）。该目标结构与鼎新未来机动任务同构（航电单流领先、融合方法可能负迁移）。

| 方法 | 直推 R² | 直推 Spearman | 泛化 R²（train→heldout） | 泛化 Spearman |
| --- | --- | --- | --- | --- |
| Chronaris safe_lag | **0.474** | **0.332** | −1.919 | 0.287 |
| Chronaris multiscale（旧） | 0.372 | 0.232 | −1.644 | −0.029 |
| vehicle_only（航电单流参考） | 0.363 | 0.270 | −0.782 | 0.503 |

判断：在车辆主导的机动代理任务上，旧 multiscale 融合最弱（直推 R² 0.372、泛化 Spearman −0.029），与“旧融合破坏航电信息”一致；safe_lag 直推 R² 与 Spearman 均最高（0.474/0.332），甚至略超航电单流，表明安全旁路在该任务上消除了负迁移并保住了航电信号。泛化 R² 三组均为负（不同仿真轨迹的分布漂移，15 epoch 表示弱）；但泛化 Spearman 显示 safe_lag（0.287）为正、multiscale（−0.029）为零/负，方向一致。该结果为机制层面的证据，非锁定确认；鼎新真实未来机动任务上的确认仍为决定性下一步。
