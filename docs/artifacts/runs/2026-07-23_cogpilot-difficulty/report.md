# CogPilot 飞行难度分类（公开数据双流增量证据，gate 3/5）

状态：completed（公开数据真实下游结果，单 LOSO 折）。分支：`research/safe-lag-aware-fusion-20260718`。日期：2026-07-23。

## 目的

在公开 CogPilot/PhysioNet 数据上检验融合是否在“真正需要双流”的任务上超过最佳单流（晋级门禁 3 公开数据、门禁 5 双流增量超最佳单流）。飞行难度同时驱动飞行员生理唤醒（EDA/PPG/呼吸）与飞机操纵/状态活动（空速/姿态/ILS 偏差/上升率），是该假设的合适任务。

## 设置

- 数据：CogPilot task-ils，10 名参与者、109 个 30 秒窗口（窗口起点 60s，10 Hz 重采样）。生理流：PPG、EDA、呼吸（慢变自中枢/外周信号）；飞机流：X-Plane 状态 8 通道（指示空速、俯仰、滚转、AGL 高度、上升率、ILS 航向/下滑偏差、地速）。
- 任务：飞行难度四分类（level-01B..04B → 0..3），标签分布 28/27/27/27。
- 划分：留一受试者组（LOSO），3 名参与者作测试（test=36），训练 73；按参与者分组，无窗口跨组泄漏。
- 训练：`train_common_pretext_method`，12 epoch、batch 8、seed 17，公共遮挡重构+短期预测+时延判别；四组 chronaris_safe_lag、chronaris_multiscale、vehicle_only、physiology_only。
- 消费者：冻结窗口表示（64 维）→ StandardScaler + class-balanced Logistic。

## 结果（LOSO，seed 17，12 epoch）

| 方法 | macro-F1 | balanced accuracy |
| --- | --- | --- |
| **Chronaris safe_lag** | **0.4122** | **0.4167** |
| Chronaris multiscale（旧） | 0.4017 | 0.4167 |
| vehicle_only（飞机单流） | 0.3396 | 0.3611 |
| physiology_only（生理单流） | 0.1521 | 0.2222 |

## 判断

- **公开数据双流增量成立（gate 3/5 方向性满足）**：safe_lag macro-F1 `0.4122` 同时高于旧融合 `0.4017`、最佳单流 vehicle_only `0.3396` 与 physiology_only `0.1521`。在“难度同时驱动生理与飞机状态”的任务上，融合真正超过最佳单流，与鼎新车辆主导机动任务（融合难超航电单流）形成对照——**任务语义决定融合是否有增量**。
- safe_lag 略优于旧 multiscale，安全旁路在公开数据上同样不损失并略增。
- 相对最佳单流 vehicle_only 的增量：`+0.0726` macro-F1（相对 +21%）。

## 边界与下一步

- 单 LOSO 折（3 测试受试者）、12 epoch、10 参与者、单 seed——**非锁定确认**，但为公开数据上的真实正向证据。
- 下一步：扩展至全部 35 参与者、GroupKFold-5/全 LOSO、增种子、延长预算，并按评价协议 v2 在“难度分类 + 累计误差回归 + 事件后生理响应”上形成公开数据主结果，作为鼎新车辆主导任务之外的融合优势证据。
- 重型 checkpoint 位于被忽略目录 `artifacts/application_evaluation/2026-07-23_cogpilot-difficulty/`。

## 队列规模修正（重要，诚实修订）

将队列从 10 参与者扩到 20 参与者（5 测试受试者）后，结论**改变**：

| 方法 | 10subj seed17 | 10subj seed29 | 20subj seed17 |
| --- | --- | --- | --- |
| Chronaris safe_lag | 0.4122 | 0.3353 | 0.4654 |
| Chronaris multiscale | 0.4017 | 0.3045 | 0.3543 |
| vehicle_only | 0.3396 | 0.2060 | **0.5045** |
| physiology_only | 0.1521 | 0.2103 | 0.1833 |

**诚实修订**：在 10 参与者（3 测试受试者）时 safe_lag 超过 vehicle_only，但扩到 20 参与者后 **vehicle_only（0.5045）反超 safe_lag（0.4654）**。说明“safe_lag 超最佳单流”在小队列上不稳健——难度分类在更大队列上也变得可由飞机状态单独预测（ILS 偏差/空速等与难度强相关）。

**稳健成立的结论**：safe_lag **始终优于旧 multiscale 融合**（10subj 两种子与 20subj 均成立，0.465 vs 0.354、0.374 vs 0.353），即安全旁路相对旧融合是稳健改进。但**融合未稳健超过最佳单流**——车辆/飞机状态在难度与机动任务上都过于强势。

（20subj seed29 待运行以确认 vehicle 占优是否跨种子；将补充。）
