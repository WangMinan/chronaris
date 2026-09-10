# Chronaris 复核入口

更新时间：2026-09-10

本目录保存代码与协议复核的计划、发现、修复及验证记录。当前研究和执行入口为[融合表示下游评价方案](../requirements/thesis-downstream-representation-plan-20260909.md)、[任务队列](../implementation/TASKS.md)和[状态页](../STATE.md)。

## 当前复核

[阶段 1 修复与复用核验](stage/thesis-v4/stage1-repair-20260910/README.md)完成导出池化、压力失败终态修复及原计算的兼容核验；真实失败单元八条件重放完成，逐类保留与重建范围已记录。

[文档同步与过时入口清理](stage/thesis-v4/document-sync-20260910/README.md)核对两段历史会话、用户最新确认和当前代码，统一共同下游评价主线、标签含义、近期方法接入与论文节点。该次工作仅修改文档，后续实现见阶段 1。

[导师要求与后台盘点](stage/thesis-v4/advisor-status-20260909/README.md)记录 26 个新初筛配置、14,300 次更新及进程 82623 的失败终态。该次工作定位池化设备差异；修复已由阶段 1 承接，旧队列未重启，近期方法实跑仍待执行。

## v4 实现证据

下表导航已完成的工程模块及其来源，具体日期状态以对应报告为准；当前后台状态统一读取状态页和实际运行记录。

| 范围 | 复核入口 |
| --- | --- |
| 当前实现与历史验收索引 | [v4 实施复核](stage/thesis-v4/README.md) |
| 单入口、恢复和报告 | [代码交付](stage/thesis-v4/code-delivery-20260908/README.md) |
| 初筛、三种子、采用及压力队列 | [选型与复核](stage/thesis-v4/selection-20260908/README.md) |
| 公开开发与结果排名 | [公开首折](stage/thesis-v4/public-screen-20260908/README.md)、[结果重放与排名](stage/thesis-v4/public-result-ranking-20260908/README.md) |
| 正式训练与鼎新去重 | [确认接续](stage/thesis-v4/confirmation-handoff-20260908/README.md)、[鼎新去重结果](../artifacts/runs/2026-09-08_v4-dingxin-deduplicated/report.md) |
| 输入、目标和冻结后评价 | [确认合同](stage/thesis-v4/confirmation-contracts-20260908/README.md)、[原生留出评价](stage/thesis-v4/native-frozen-evaluation-20260908/README.md) |
| 核心消融 | [消融合同](../artifacts/runs/2026-09-08_v4-core-ablation-contract/report.md) |
| 运行性能 | [性能复核](stage/thesis-v4/performance-20260908/README.md) |

## 历史研究与维护

- [远程研究建议复核](stage/thesis-mainline/remote-pro-review-2026-09-05.md)：旧权重学习规模、注意力、物理解码与负面结果诊断。
- [v3.2.3 实验与安全门](stage/thesis-mainline/frozen-simulation-review-2026-09-04.md)：旧版本四项机制门通过、两项失败；不以 v4 工程修复改写这些结果。
- [主分支集成](stage/thesis-mainline/main-integration-2026-09-05.md)与[冗余清理](stage/thesis-mainline/redundancy-and-docs-review-2026-09-05.md)：已完成代码整合、路径与文档清理。
- [旧论文主线复核](stage/thesis-mainline/README.md)：v3 阶段范围和历史记录。

## 记录要求

每次复核说明范围与计划、实际发现、修复状态、剩余工作和验证结果。历史指标、图表和检查点保留来源；当前状态只保留最新可核验事实。新产物从本入口及[产物索引](../artifacts/ARTIFACTS.md)或任务页可达，研究效果与工程检查分别报告。
