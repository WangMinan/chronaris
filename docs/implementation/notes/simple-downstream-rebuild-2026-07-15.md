# 鼎新简化下游评价执行说明

更新时间：2026-07-16
状态：正式确认、论文证据和独立复核全部完成
执行分支：`codex/simple-downstream-rebuild-20260715`
起点：`719c47c58bcea9149bb7d16b33e2980d6d1cda25`

## 1. 决策记录

本轮不把仓库回退到 `0a38171a8310b5f9888d978cf4b7d89b89ca9a98`。当前代码已经包含原始点冻结、时间边界检查、统一表示、断点恢复、CPU/GPU 稳定性修复和大量测试，旧比较入口也仍可追溯。重新从早期提交开发会增加人工迁移和遗漏风险。

本轮实际执行的是：

> 保留成熟工程基础，停止目标重构门禁研究线的自动后继任务，建立一条只比较融合表示下游价值的简化主线。

完整研究合同见 [鼎新简化下游评价协议 v1](../../requirements/simple-downstream-evaluation-v1.md)。若聊天记录、旧任务说明与该协议冲突，以该协议和当前任务队列为准。

## 2. 已批准的研究口径

- 主要评价使用留一架次，留一视图仅作同一飞行过程中的视图适配诊断。
- 未来机动以连续分数预测为主，低/中/高分类为辅助。
- 未来生理状态按 12 个字段逐字段预测，并以持久性技能为首要判断。
- 两项任务均使用未来 5 秒完整目标；预期 90 个 view-context。
- 机动任务按约 60 个唯一 `vehicle_context_id` 去重或加权，避免同一架次双视图重复计权。
- 主结果为冻结的 64 维统一表示加相同 Logistic/Ridge 消费者。
- 不重跑端到端联合训练，不根据正式留一架次结果调参。
- 既有仿真、时间机制和消融不重训，只做配置一致性审计和论文图表整理。

## 3. 代码复用原则

优先复用当前实现，而不是复制一套并行框架：

- 原始点与上下文：`src/chronaris/evaluation/application_tasks/` 和现有鼎新数据入口；
- 统一表示与模型适配器：`src/chronaris/modeling/fusion_encoders/`；
- 旧对比入口：`src/chronaris/evaluation/dingxin/pipelines/thirdparty_comparison.py` 只作历史冒烟和可复用函数来源；
- 新任务合同和轻量消费者进入 `src/chronaris/evaluation/dingxin/` 下的小模块；
- `scripts/` 只负责命令行编排，不承载目标计算、去重、指标或兼容性判断。

不得修改确认指标表、实验注册表和 claim boundary 历史快照。新任务使用新 ID 和新 run 根，不覆盖旧产物。

## 4. 实施阶段

| 阶段 | 工作 | 通过条件 | 状态 |
|---|---|---|---|
| S0 协议固化 | 建分支、写协议、同步状态和任务入口 | 文档一致，旧路线明确收口 | 已完成 |
| S1 兼容性审计 | 审查 checkpoint、表示、字段和池化合同 | 生成三态兼容性报告 | 已完成 |
| S2 新任务合同 | 构造三个新任务、统计单位和持久性基线 | 90 个完整目标、60 个唯一机动上下文、零未来输入 | 已完成 |
| S3 轻量下游算法 | 接入统一 Ridge/Logistic、聚合和指标 | 单元测试覆盖去重、权重、阈值和技能 | 已完成 |
| S4 工程冒烟 | 一个留一架次折、seed 17、六方法 | 全部输出可恢复，合同检查通过 | 已完成 |
| S5 正式确认 | 两个留一架次折、三随机种子、六方法 | 一次性完成，不再调参 | 已完成 |
| S6 论文证据 | 复用既有仿真和消融并重绘 | 正负结果均保留，关键图抽查 | 已完成 |

### 4.1 已完成实跑

- [协议与兼容性审计](../../artifacts/runs/2026-07-16_simple-downstream-protocol/report.md)：实跑确认 90 个完整视图上下文、60 个独立机动上下文；旧表示因排除 20 个合法历史运动学字段且任务清单覆盖不完整，判定为 `retraining_required`。
- [选定配置工程重训](../../artifacts/runs/2026-07-16_simple-downstream-pretraining-smoke/report.md)：留一架次第一折、seed 17 的五个可训练方法全部完成，8/8 验收通过；任务目标和留出架次始终关闭。
- [六方法完整表示导出](../../artifacts/runs/2026-07-16_simple-downstream-representations-smoke/report.md)：六种方法均覆盖 30 个训练视图上下文和 60 个留出视图上下文，12 份表示目录全部完成，5/5 验收通过。
- [固定下游算法工程冒烟](../../artifacts/runs/2026-07-16_simple-downstream-consumer-smoke/report.md)：六个方法单元和 84 条指标均成功生成，4/4 验收通过。该运行只证明工程闭环，其分数不进入论文主表，也不用于修改正式配置。
- [正式任务无关预训练](../../artifacts/runs/2026-07-16_simple-downstream-pretraining-confirmation/report.md)：两个留一架次折、三个随机种子、五个可训练方法共 30/30 单元完成，8/8 验收通过。Chronaris 第一折 seed 43 曾发生一次 CUDA 启动故障，随后按原 checkpoint、原设备和相同配置恢复完成，没有修改预算或切换 CPU。
- [正式完整表示导出](../../artifacts/runs/2026-07-16_simple-downstream-representations-confirmation/report.md)：六种方法共 72/72 份训练与留出表示完成，5/5 验收通过；表示阶段没有打开任务目标或外层指标。
- [正式固定下游评价](../../artifacts/runs/2026-07-16_simple-downstream-confirmation/report.md)：六种方法在两个折和三个随机种子上形成 36/36 个完整评价单元与 504 条有限指标，4/4 验收通过。
- [论文证据与独立复核](../../artifacts/runs/2026-07-16_simple-downstream-thesis-evidence/report.md)：复用既有仿真时间机制、压力和消融证据，形成四张已抽查中文图和五张结果表；108 项逐样本文件哈希和 504 个原始预测指标独立复算通过，7/7 验收通过。

### 4.2 已冻结的正式预算

- 编码器候选固定为生理单流 A、航电单流 C、MulT A、ContiFormer A 和 Chronaris A。
- 两个留一架次折、随机种子 17/29/43，最多 50 epoch、batch size 32、patience 8。
- 编码器早停只使用训练架次内部的公共自监督 validation；正式任务目标和留出架次在表示完成前保持关闭。
- 五个可训练方法采用单 CUDA 进程串行运行；朴素时间同步只在训练折拟合无监督变换。
- 工程冒烟的 3 epoch 分数和最佳 epoch 不参与正式配置选择。

### 4.3 正式结果与退出判断

- 航电单流在未来机动上取得宏平均 F1 `0.8084`、平衡准确率 `0.8111`、Spearman `0.6165` 和相对当前状态技能 `0.6685`，是明确领先方法。
- Chronaris 的未来机动宏平均 F1 为 `0.1948`、Spearman 为 `0.0685`、相对当前状态技能为 `-123.5859`，没有形成真实未来机动优势。
- 未来生理字段任务中，六种方法在两个折上的正技能字段比例均为 `0`；航电单流的字段级标准化 RMSE 宏平均最低，为 `7.0838`，Chronaris 为 `8.8901`。正确判断是所有学习方法都未超过持久性基线。
- 既有仿真中 Chronaris 的时钟偏移恢复误差 `0.888` 秒、生理响应时延恢复误差 `7.486` 秒，均为四种融合方法中最低；但随机缺失与连续缺失退化斜率为 `-0.073/-0.126`，部分组件消融结果也不利。
- 本轮最终结论为“时间机制局部有效，真实下游整体优势尚未成立”。正式结果打开后不再修改模型、任务或消费者。
- 留一视图只能诊断同一飞行轨迹下的视图适配，不能改变跨架次主结论。为避免正式结果打开后的范围扩张，本轮不新增该训练线，并在报告中明确记录这一退出决定。

## 5. 计划文件

预计新增或调整：

```text
src/chronaris/evaluation/dingxin/simple_downstream_protocol.py
src/chronaris/evaluation/dingxin/simple_downstream_consumers.py
src/chronaris/evaluation/dingxin/representation_compatibility.py
scripts/evaluation/dingxin/audit_simple_downstream.py
scripts/evaluation/dingxin/run_simple_downstream.py
tests/evaluation/dingxin/test_simple_downstream_protocol.py
tests/evaluation/dingxin/test_simple_downstream_consumers.py
tests/evaluation/dingxin/test_representation_compatibility.py
```

若现有模块已经完整提供某项能力，应直接复用并只补薄适配层，不机械创建上述全部文件。

## 6. 产物合同

协议与兼容性审计首先写入：

```text
docs/artifacts/runs/2026-07-16_simple-downstream-protocol/
├── report.md
├── protocol.json
├── representation_compatibility_report.json
├── task_manifest_summary.json
└── resume_command.txt
```

重型任务清单、逐样本预测、checkpoint 和稠密表示继续保存在被忽略目录。正式紧凑证据只提交配置、哈希、统计、指标表、图和结论边界。

## 7. 验证要求

最低验证包括：

- 输入最大时间严格小于目标开始时间；
- 未来目标覆盖完整 5 秒；
- 三个新任务 ID 不与历史任务混用；
- 训练折之外的数据不参与字段尺度、阈值和消费者标准化；
- 机动训练权重按唯一 vehicle-context 归一；
- 机动评价先聚合同一 vehicle-context；
- 生理持久性技能与字段级宏平均指标可复核；
- 表示兼容性总判定只能取协议规定的三种状态；
- 恢复运行不会改变已完成文件哈希；
- `git diff --check` 通过。

旧中期数值只作历史入口冒烟，不作为新协议的分数门槛。任何模型分数都不能阻止 S2–S4 的工程闭环。

## 8. 停止和变更规则

正式留一架次结果打开前，可以修复合同实现错误；打开后只允许修复由测试证明的程序错误。以下变化必须另起协议版本，不能原地修改 v1：

- 更换目标公式或时间窗口；
- 改变历史运动学字段政策；
- 改变正式划分或统计单位；
- 为某个方法单独更换消费者；
- 根据结果修改模型结构、表示池化或超参数。

本轮不承诺 Chronaris 全面领先。报告继续采用“局部有效、分项领先、整体优势尚未成立”的有界表述，除非新确认结果提供足够证据改变这一结论。
