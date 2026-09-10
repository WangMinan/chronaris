# v4 连续对齐与语义融合实施复核

更新时间：2026-09-10

当前按[融合表示下游评价与近期模型接入方案](../../../requirements/thesis-downstream-representation-plan-20260909.md)继续现有分支。[阶段 1](stage1-repair-20260910/README.md)已修复导出池化与初筛压力失败终态，并完成已有计算复用核验；原后台保持停止，近期模型尚未接入。实际进度见[状态页](../../../STATE.md)。

## 当前事实与后续工作

[代码交付](code-delivery-20260908/README.md)完成原六方法的单入口和工程验收；[9 月 9 日盘点](advisor-status-20260909/README.md)确认新初筛 26/26、52 份双路线结果、14,300 次更新，压力仅完成 9/52 个路线单元。原任务引导导出与压力路径的池化设备差异已修复，真实失败窗口逐值重放通过；原队列完成量保持历史记录。公开复核、正式确认和新方法评价尚无本轮完整结果。

[文档同步复核](document-sync-20260910/README.md)记录本次目标、任务解释、导航与历史执行指令清理。原 432 单元范围属于六方法计划，不包含新模型；后续先核验表示接口与监督，再确定完整比较矩阵。

## 已有实现与证据索引

下表按职责保留已有报告，说明可以复用的工程基础；测试数字与当时状态回到原报告读取，不将历史启动状态展示为当前运行状态。

| 范围 | 报告与复核 |
| --- | --- |
| 历史可见域与有效掩码 | [因果修复](../../../artifacts/runs/2026-09-05_v4-causality-repair/report.md) |
| 物理观测锚定 | [参考冒烟](../../../artifacts/runs/2026-09-05_v4-reference-smoke/report.md) |
| 更新计数、任务头与恢复 | [更新闭环](../../../artifacts/runs/2026-09-05_v4-update-smoke/report.md)、[公共双路线](../../../artifacts/runs/2026-09-06_v4-dual-route-smoke/report.md)、[四域实跑](../../../artifacts/runs/2026-09-06_v4-real-domain-smoke/report.md) |
| 运行性能与连续计算 | [初始性能](../../../artifacts/runs/2026-09-05_v4-performance/report.md)、[原生状态执行](../../../artifacts/runs/2026-09-06_v4-native-recurrence/report.md)、[调度复核](performance-20260908/README.md) |
| 仿真与公开数据准备 | [初始仿真](../../../artifacts/runs/2026-09-06_v4-simulation-data/report.md)、[开发数据](../../../artifacts/runs/2026-09-06_v4-development-data/report.md)、[扩展至 512 条](../../../artifacts/runs/2026-09-07_v4-training-expansion/report.md) |
| 鼎新目标与内容隔离 | [原内部拟合](../../../artifacts/runs/2026-09-05_v4-dingxin-inner-targets/report.md)、[重复发现](vehicle-isolation-20260908/README.md)、[单记录修订](../../../artifacts/runs/2026-09-08_v4-dingxin-deduplicated/report.md) |
| 下游算法与留出评价 | [分组拟合](../../../artifacts/runs/2026-09-06_v4-grouped-consumers/report.md)、[目标合同](confirmation-contracts-20260908/README.md)、[冻结后评价](native-frozen-evaluation-20260908/README.md) |
| 初始学习诊断与压力 | [学习曲线入口](../../../artifacts/runs/2026-09-06_v4-learning-curves/report.md)、[初始结果](../../../artifacts/runs/2026-09-07_v4-initial-diagnostics/report.md)、[压力输入](../../../artifacts/runs/2026-09-06_v4-development-conditions/report.md)、[八条件实跑](../../../artifacts/runs/2026-09-07_v4-development-pressure/report.md) |
| 候选模块 | [注意力与配对](../../../artifacts/runs/2026-09-07_v4-candidate-components/report.md)、[增强与预测](../../../artifacts/runs/2026-09-07_v4-common-candidates/report.md)、[保真与质量门](../../../artifacts/runs/2026-09-08_v4-branch-candidates/report.md) |
| 选型与结果重放 | [选型](selection-20260908/README.md)、[公开首折](public-screen-20260908/README.md)、[排名](public-result-ranking-20260908/README.md)、[原生归档](native-results-20260908/README.md) |
| 固定网格与分组统计 | [固定网格基线](naive-baseline-20260908/README.md)、[重采样统计](grouped-statistics-20260908/README.md) |
| 正式训练、消融与报告 | [确认接续](confirmation-handoff-20260908/README.md)、[主表入口](../../../artifacts/runs/2026-09-08_v4-formal-mainline/report.md)、[核心消融](../../../artifacts/runs/2026-09-08_v4-core-ablation-contract/report.md)、[单入口交付](code-delivery-20260908/README.md) |
| 中文图表 | [渲染合同](diagnostic-figures-20260908/chart_contract.md)、[初始七张图](../../../artifacts/runs/2026-09-08_v4-diagnostic-figures/report.md) |

## 继承与结果边界

原[开发合同](../../../requirements/thesis-v4-development-plan.md)保留已有训练规则的来源；目标与后续执行以新方案为准。初始 256 条仿真诊断、扩展后的候选结果及最终确认分别归档，不能混为一条学习收益曲线。

鼎新两折 18/6/6 的旧内部目标记录用于追溯，当前单记录时间块按 12/3/3 及边界隔离执行。原始重复文件和失败结果保留，不恢复跨架次推断。旧 v3 的门失败也不因 v4 工程检查而改变。

连续缺失下的误差和状态增长仍是待复核结果。工程模块已实现、测试通过和训练完成各有不同含义；当前尚未建立近期模型共同下游比较或真实鼎新业务任务优势。
