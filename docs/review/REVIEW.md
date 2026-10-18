# Chronaris 复核入口

更新时间：2026-10-18

[原始数据、简单参照与同维度读出](stage/chapter5-controlled-readouts-20261018/README.md)：原始 CSV 与元数据审计、严格事件端点单因素控制、三折嵌套读出和全候选表；33 条冻结导出、零编码器更新，原结果保留，给出具体修复顺序。

[首轮退化诊断复核与第二轮方案](stage/chapter5-degradation-20261018/REVISION.md)：同验证集训练均值复算确认加强正则尚未超过简单参照；补同维度控制、导师目标验收矩阵、目标构造审计与资源上限，新公开独立确认因缺少已核验新来源保持关闭。

[完整方法退化诊断与统计审查](stage/chapter5-degradation-20261018/README.md)：公开原始预测复算、受试者配对统计、训练预算、冻结分支读出、简单参照、正则范围与无更新梯度实跑；保留冻结结果，给出分阶段修复和补实验方案。

[第六章工作区合并验收](stage/chapter6-merge-20261018/README.md)：检查当前分支与未提交改动，保存完整方法 GPU 证据；验证服务、工作流、物理与语义功能，在主分支复核真实浏览器链路。

本目录保存代码与协议复核的计划、发现、修复及验证记录。当前研究和执行入口为[融合表示下游评价方案](../requirements/thesis-downstream-representation-plan-20260909.md)、[任务队列](../implementation/TASKS.md)和[状态页](../STATE.md)。

## 当前复核

[传感器语言模型原设备重放恢复](stage/thesis-v4/sensor-reference-recovery-20261016/README.md)：原 RTX 4090 逐层对照、固定原阈值、四窗口真实链路零差异及 42 项测试；总控 `10896` 已启动，完整五折资格通过后自动接续正式实验。

[第五章首批写作交付](stage/thesis-v4/chapter5-writing-delivery-20261015/README.md)：独立冻结门、四项新增训练、108 个原生评价单元、近期模型显存适配和开放格式素材；历史队列已因传感器模型重放失败退出，当前接续见上方原设备恢复记录。

[历史信息隔离的数值容差](stage/thesis-v4/causality-tolerance-20261015/README.md)：逐层定位GPU单精度形状相关舍入，双精度与反例对照、48组扩展核验及29项回归通过；审计策略独立绑定，冻结训练源码与旧失败保留。

[第六章完整方法与GPU独立验收](stage/chapter6-independent-20261015/README.md)：独立验证原训练物理梯度、训练标定及真实CUDA计算，两类CPU/GPU案例和63项测试通过；仿真完整方法与实测应用分别记述。该次检查时实现未推送、新证据未跟踪；10 月 18 日整合状态见上方合并记录。

[第六章最终应用验收](stage/chapter6-ready-20261015/README.md)：实测语义配置完成训练与真实推理；59项测试、20项浏览器检查、9次性能测量及真实中断恢复通过，写作交接包已齐备。下方目录审阅及10月14日意见保留为收口前记录。

[第六章目录与管线完备性审查](stage/chapter6-outline-20261015/README.md)：基础管线覆盖新目录，36项定向测试和真实CPU推理复核通过；完整事件语义权重接入及最终配置测时待收口。

[融合系统复核修复与验收](stage/fusion-workbench-20261014/repair.md)：七项应用修复、归档和文件接入、任务恢复、增量处理及视觉验收；36 项测试与 20 项浏览器检查通过，完整事件语义模型应用验证待兼容权重。原始意见保留在[独立复核](stage/fusion-workbench-20261014/README.md)。

[本机跨设备验证入口](stage/thesis-v4/local-validation-20261012/README.md)：来源可追溯的设备兼容副本、独立恢复、完整内部重放及首批门；36项定向回归通过，研究稳定性和工期等待实跑。

[本机主分支整合](../artifacts/runs/2026-10-12_local-main-integration/README.md)：核对调用链、旧协议兼容、机制配置与冻结门；57项定向回归和真实资产准备通过，历史恢复的设备条件单独保留。

[论文方法对应与语义执行优化](stage/thesis-v4/thesis-alignment-20261012/README.md)：实际组件、原队列预算、12 时段决定、语义分段归约的 CPU 输出与梯度资格；显卡与独立恢复验证待执行。

[第五章数据与核心实验接续](stage/thesis-v4/chapter5-continuation-20261011/README.md)：冻结主模型复用、核心矩阵、外层门、详细数据和论文侧独立绘图验收。

[阶段 5 显卡异常恢复](stage/thesis-v4/stage5-cuda-retry-20261004/README.md)：同秒宿主机显卡事件、原始失败保留、来源核验与同源码重试检查。

[阶段 5 执行性能优化与恢复](stage/thesis-v4/stage5-performance-20261002/README.md)：共享未变化生理递推、逐值输入整理、执行资格及 140 个完成单元保留。

[阶段 5 传感器语言对齐修复](stage/thesis-v4/stage5-sensor-recovery-20260930/README.md)：真实趋势监督支持、独立恢复、已完成八单元来源核验与后台恢复。

[阶段 5 收口与正式训练入口](stage/thesis-v4/stage5-formal-20260928/README.md)：独立绘图包、完整方法覆盖、划分与种子、训练角色隔离及冻结启动检查。

[阶段 5 补充评价修复与恢复](stage/thesis-v4/stage5-recovery-20260926/README.md)：源记录与模型重放、统一消费者协议、显式继承、冻结源码验证及后台启动。

[阶段 5 综合评估与接续建议](stage/thesis-v4/stage5-integrated-assessment-20260925/README.md)：核对两个会话，实测固定库可处理失败表示，匹配准备输入与缺失分支，确认正式入口配置差异；恢复与协议修订尚未实施。

[9 月 25 日阶段 5 停机与全分支复核](stage/thesis-v4/stage5-status-review-20260925/README.md)：来源与分组结果核验、失败诊断、文档读者检查与远端交付；本轮不修改或重启实验。

[阶段 5 复核](stage/thesis-v4/stage5-20260923/README.md)：固定参考、历史来源、三折三种子矩阵、执行资格及后台恢复。

[阶段 4.5-C 复核](stage/thesis-v4/stage45c-20260922/README.md)：六次单因素微调完成、未通过采用门；八份完成收据与 77 份来源核验通过，全部不利结果保留。

[阶段 4.5-B 复核](stage/thesis-v4/stage45b-20260921/README.md)：训练内分组选模、归一化诊断、条件结构候选及独立恢复；十二次首轮已完成，无候选通过原研究门。

[阶段 4.5 结果收口与下一轮复核](../artifacts/runs/2026-09-21_v4-stage45-closeout/report.md)：核验全部来源、五种候选和八个对照尝试、补充下游与梯度记录；18/18 完成、零入围，22 个后续槽位按规则跳过。计划与结果均见报告；远端问题见[GPT PRO 复核任务](../artifacts/runs/2026-09-21_v4-stage45-closeout/pro-review.md)。

[阶段 4.5 运行负载与优化决策](stage/thesis-v4/stage45-runtime-assessment-20260918/README.md)：全流程执行修订及首次联合更新稳定性修复已验收，完整显卡回归 697 通过、22 跳过；恢复队列现已完成，原失败保留。见[最新恢复报告](../artifacts/runs/2026-09-19_v4-stage45-accelerated/report.md)。

[阶段 4.5 暂停复核与恢复](stage/thesis-v4/stage45-pause-review-20260914/README.md)：保留第 75 次状态，进行能力和执行性能复核，取消 72 小时硬上限后恢复。

[阶段 4.5 模型与性能优化](stage/thesis-v4/stage45-20260914/README.md)核对远端建议、参考配方、分支诊断及有界队列；本轮先完成代码与验证，再启动后台开发实验。

[完整开发比较收口](stage/thesis-v4/stage4-closeout-20260914/README.md)已完成结果、学习信号、恢复成本和后续计算预算核验。

[完整开发比较执行迁移](stage/thesis-v4/stage4-execution-migration-20260913/README.md)确认第 200 次状态保持、九个单元继承、成本来源和第 225 次新检查点。

[完整开发比较性能试验](stage/thesis-v4/stage4-performance-20260913/README.md)检查第 200 次更新的停止与保留、显卡运算图重放的数值等价性及恢复。

[阶段 4 修复与恢复](stage/thesis-v4/stage4-recovery-20260912/README.md)核验基线参数分派、实际入口、旧成果复用与成本继承。

[阶段 4 完整开发比较](stage/thesis-v4/stage4-comparison-20260911/README.md)检查完整角色、任务覆盖、恢复、失败停止和成本记录，启动后按用户要求结束会话。

[阶段 3 近期模型接入](stage/thesis-v4/stage3-models-20260911/README.md)核验官方表示、真实开发下游、训练与缓存恢复，以及资源成本；适配合同见[近期模型开发接入](../requirements/recent-model-adapters-v1.md)。

[阶段 2 共同下游合同](stage/thesis-v4/stage2-contract-20260910/README.md)已完成任务、角色、监督和表示能力约束；三域真实开发闭环及完整显卡回归通过。

[阶段 1 修复与复用核验](stage/thesis-v4/stage1-repair-20260910/README.md)完成导出池化、压力失败终态修复及原计算的兼容核验；真实失败单元八条件重放完成，逐类保留与重建范围已记录。

[文档同步与过时入口清理](stage/thesis-v4/document-sync-20260910/README.md)核对两段历史会话、用户最新确认和当前代码，统一共同下游评价主线、标签含义、近期方法接入与论文节点。该次工作仅修改文档，后续实现见阶段 1。

[导师要求与后台盘点](stage/thesis-v4/advisor-status-20260909/README.md)记录 26 个新初筛配置、14,300 次更新及进程 82623 的失败终态。该次工作定位池化设备差异；修复已由阶段 1 承接，旧队列未重启，近期方法后续实跑由阶段 3 单独归档。

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
