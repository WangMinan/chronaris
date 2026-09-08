# Chronaris 产物索引

更新时间：2026-09-06

## 目录定位

当前可引用产物入口统一放在 `docs/artifacts/runs/`。目录命名采用 `YYYY-MM-DD_intent`，避免把当前入口继续绑定到历史阶段编号。

历史阶段编号报告、旧资产目录和兼容入口已经移入 `docs/artifacts/archive/`。清理和迁移记录放在 `docs/artifacts/cleanup/` 与 `docs/maintenance/`。

## 当前九月研究证据

- [鼎新保留单记录](runs/2026-09-08_v4-dingxin-deduplicated/report.md)：30 个唯一航电时段、两个生理视图、11 个有效字段；完整支撑隔离通过，按用户授权替代重复两架次评价。

- [v4 固定基线开发归档](runs/2026-09-08_v4-fixed-native-development/report.md)：24 单元数值重放通过；鼎新另发现跨架次航电内容重复，正式隔离未通过。

- [v4 分组统计复核](runs/2026-09-08_v4-grouped-statistics/report.md)：初始 48 项档案配对统计、每项 2,000 次，公开接口不把折或种子当独立受试者。

- [v4 初始诊断图表](runs/2026-09-08_v4-diagnostic-figures/report.md)：七张中文图及矢量版本，核验 243 个来源，保留梯度尖峰和连续缺失退化。

- [v4 单流保真与质量门](runs/2026-09-08_v4-branch-candidates/report.md)：九项候选入口全部可用，570 项完整测试通过；有效状态异常拦截已补齐。

- [v4 公共增强与预测候选](runs/2026-09-07_v4-common-candidates/report.md)：固定缺失混合、实际秒数预测及双路线继承通过，完整测试 559 项通过、8 项跳过。

- [v4 候选组件与捕获修复](runs/2026-09-07_v4-candidate-components/report.md)：四项单因素实现接入共享训练；配对独立性、日程与恢复通过，完整测试 554 项通过。

- [v4 八条件开发压力](runs/2026-09-07_v4-development-pressure/report.md)：80 个评价单元与 2,560 条总体指标，保留连续缺失退化及单种子状态增长信号。
- [v4 一次训练规模扩展](runs/2026-09-07_v4-training-expansion/report.md)：512 条训练轨迹、64 个参数档案，原有文件与输入张量完全一致。

- [v4 五方法双路线初始诊断](runs/2026-09-07_v4-initial-diagnostics/report.md)：30 组阶段消费者与 960 条开发指标，全部文件核验通过；压力接口完整测试 536 项通过。

- [v4 八条件开发数据](runs/2026-09-06_v4-development-conditions/report.md)：2,048 个配对窗口完成数据检查，时钟偏移与响应时延独立；首次 500 更新触发预定的一次训练规模扩展，尚未激活。

- [v4 公开与鼎新分组消费者](runs/2026-09-06_v4-grouped-consumers/report.md)：六个真实数据工程组合完成消费者读写，部分字段标签与无观测窗口保留，公开按受试者统计。

- [v4 完整开发学习曲线](runs/2026-09-06_v4-learning-curves/report.md)：共享训练、阶段快照与三类消费者入口通过集成测试，先执行仿真干净曲线。

- [v4 真实数据双路线冒烟](runs/2026-09-06_v4-real-domain-smoke/report.md)：四域两路训练、冻结消费者及历史隔离通过，350 次新增更新含保留失败；完整测试 524 项通过。

- [v4 原生连续状态执行对照](runs/2026-09-06_v4-native-recurrence/report.md)：同参数原生 CogPilot 批次前后向约提速 12.32 倍；输出、梯度和历史隔离通过，失败探查保留。

- [v4 开发数据与分组](runs/2026-09-06_v4-development-data/report.md)：CogPilot 2,524 个、CLARE 794 个原生窗口；新仿真 1,280 个训练与开发上下文，确认观测未打开。

- [v4 双路线工程闭环](runs/2026-09-06_v4-dual-route-smoke/report.md)：十次自监督更新、50 次头预热与十次联合更新，训练后两路线历史隔离通过。
- [v4 新仿真开发数据](runs/2026-09-06_v4-simulation-data/report.md)：256 条训练与 64 条开发轨迹生成完成；确认参数清单已冻结，确认观测未生成。

- [v4 更新训练与任务头](runs/2026-09-05_v4-update-smoke/report.md)：十个 CUDA 完整更新、80 个实际批次及中断重放验证，任务头支持多字段有效掩码。
- [v4 鼎新内部拟合](runs/2026-09-05_v4-dingxin-inner-targets/report.md)：实际两折均形成 18/6/6 航电上下文划分，目标尺度与阈值按显式内部训练部分拟合。

- [v4 性能与时间边界](runs/2026-09-05_v4-performance/report.md)：事件提取输出与梯度对照通过，四窗口 CUDA 前向实测提速约 8.27 倍；未据此推断整轮训练成本。

- [v4 因果与掩码修复](runs/2026-09-05_v4-causality-repair/report.md)：旧权重在新实现上的三个 CUDA 种子及 CPU 诊断通过历史隔离，检查点不变，属于工程修复证据。
- [v4 观测锚定冒烟](runs/2026-09-05_v4-reference-smoke/report.md)：显式单位与训练尺度的物理关系、同头观测锚定完成十次 CUDA 更新，训练后历史隔离通过；不包含新应用成绩。

本轮主分支清理的发现与新验证见[清理复核](../review/stage/thesis-mainline/redundancy-and-docs-review-2026-09-05.md)。下列运行记录保持原始日期与源码来源。

- [常规仿真六方法对比](runs/2026-09-03_thesis-simulation-consumers-v3p2p1/report.md)：三随机种子的冻结下游结果。

- `runs/2026-09-04_thesis-simulation-v3p2p3/`：**批准修复后的受控仿真完成状态**。提交 `6a2393e4` 上十个编排环节完成或核验复用；27 个既有训练检查点未改，公开数据与鼎新外层授权始终关闭。
- `runs/2026-09-04_thesis-simulation-stress-representations-v3p2p3/`、`runs/2026-09-04_thesis-simulation-stress-consumers-v3p2p3/`：**完整压力评价**。630 份表示、20,160 条可计算指标、4,032 条退化斜率和 630 条配对统计；验收 5/5、6/6。三个生理单流无观测窗口保留在总体中，单独预测保存在被忽略目录。连续缺失与随机缺失下的不利退化全部保留。
- `runs/2026-09-04_thesis-simulation-mechanism-representations-v3p2p3/`、`runs/2026-09-04_thesis-simulation-mechanism-consumers-v3p2p3/`：**冻结时间机制评价**。144 份训练/验证表示、24 个诊断消费者、840 次评价、3,360 条指标与 630 条配对统计，验收 5/5、6/6。420 项相关系数因常量目标不可用；偏移目标为幅值、响应时延限主通道，不推广为全部时钟或响应恢复能力。
- `runs/2026-09-04_thesis-simulation-gates-v3p2p3/`：**最终硬门与未来依赖定位**。显式时移、事件配对、连续演化和物理一致性四项通过；未来信息隔离和安全旁路失败，整体未通过。三个种子历史扰动均超过 10⁻⁶，事件分数全窗归一化的只读干预定位有 CPU/CUDA 证据；模型未修。恢复重放六份核心结果哈希一致、141 份缓存字节及修改时间不变。详见[完整分析与待确认边界](../review/stage/thesis-mainline/frozen-simulation-review-2026-09-04.md)，不能作为外层授权。

- `runs/2026-09-04_no-observation-safety-repair/`：**无观测表示与严格安全门修复**。用户确认 v3.2.3 后完成工程验收 6/6，CPU 完整测试 481 项通过、15 项跳过，实际失败窗口 CPU/CUDA 确定性推理与导出恢复通过。严格安全门 9/12 达标、3 个失败，全部原始指标保留；复用原训练与已完成表示，不调模型、不开放外层。

- `runs/2026-09-03_thesis-simulation-ablation-pretraining-v3p2p1/`、`runs/2026-09-03_thesis-simulation-ablation-representations-v3p2p1/`、`runs/2026-09-03_thesis-simulation-ablation-consumers-v3p2p1/`：**四项结构消融三随机种子正式结果**。12 个完整训练检查点、36 份表示、768 条指标和 72 条配对统计已完成，验收分别为 6/6、5/5、6/6；完整性通过不等价于所有研究硬门成立。压力导出随后因无观测样本合同冲突退出，最新分析与待确认项见[冻结仿真实跑与安全门复核](../review/stage/thesis-mainline/frozen-simulation-review-2026-09-04.md)。

- `runs/2026-09-01_thesis-mainline-preflight/`：**论文主线现场保护与基线预检**。从 `eab2c0b` 建立唯一论文开发分支和历史证据标签，完整测试 `429 passed, 8 skipped`，RTX 4090、CogPilot、CLARE、鼎新、仿真和磁盘预检通过；当前用户未提交改动保持原样。该 run 不训练模型、不产生论文指标。
- `runs/2026-09-01_training-trust-repair/`：**论文主线可信训练第一批修复**。公共预训练统一复用 validation-backed candidate trainer，三条训练路径封闭随机状态并使用 v2 checkpoint；表示导出只归一化一次，CogPilot/CLARE 下游缩放只拟合训练折并增加受试者分组 inner-validation。聚焦测试 `16 passed`、完整测试 `433 passed, 8 skipped`、CUDA 短重放 `2 passed`。该 run 只形成工程可信度证据，不产生论文任务指标；原生时间公开数据输入仍待下一子阶段完成。
- `runs/2026-09-01_native-time-euler-repair/`：**原生时间输入与连续演化子步修复**。CogPilot/CLARE 双流保留各自原始时间戳、采样密度和缺失 mask，多传感器按时间戳并集组织；lazy batch provider 和有界缓存避免一次性读取完整公共数据。ODE-RNN 增加兼容默认关闭的批量 mask Euler 子步。真实数据抽样可直接产生有限的 96 点、64 维表示；聚焦测试 `26 passed`、完整测试 `439 passed, 8 skipped`，RTX 4090 前后向通过。该 run 不选择 ODE 方案、不产生任务指标。
- `runs/2026-09-01_learnable-semantic-objectives/`：**可学习事件语义与新训练目标实现**。safe-lag 主干新增飞行事件、生理响应和人机协调三个可学习残差查询，语义上下文仅进入 16 维门控跨模态支路，并按查询时刻构造严格因果前缀。五分类显式时移和跨 group 事件—响应配对已接入统一 candidate trainer，三种训练入口不再维护重复核心循环。聚焦测试 `85 passed`、完整测试 `447 passed, 8 skipped`，RTX 4090 组合目标单 epoch 通过。该 run 只形成实现证据，多随机种子机制门尚未运行。
- `runs/2026-09-01_ode-solver-validation/`：**连续演化数值方案训练内验证**。RTX 4090 上使用 12 个仿真训练样本和 5 个训练内验证样本，同预算比较单步 Euler、最大 0.5 秒 Euler 子步和四阶 Runge-Kutta；预留样本与应用结果未打开。三者对齐损失/用时分别为 `0.278590/21.54` 秒、`0.282455/24.70` 秒和 `0.282015/149.02` 秒，梯度均有限；四阶 Runge-Kutta 超过 2 倍时间门，冻结选择单步 Euler。原 ODE 候选因此与基础配置重合，人工已确认在后续矩阵中去重。
- `runs/2026-09-01_seed17-clean-baseline/`：**seed 17 协议修复干净基线矩阵**。完成鼎新滞后损失 2 单元、仿真波次 A 3 单元、CogPilot 4 单元和 CLARE 五折 20 单元，共 29 个有效 CUDA 训练单元；公开数据外层结果和鼎新分组确认结果保持关闭。CogPilot safe-lag 训练内宏平均 F1 `0.3708` 居四方法首位；CLARE safe-lag 五折均值 `0.4680` 高于两条单流但低于旧融合 `0.5251`，最差折 `0.0769`。5/54 个 CLARE 窗口因一个模态完全缺失而不进入同样本比较，未进行填充；3 个首次尝试结果保留并标为 superseded。该矩阵不进入论文主表。
- `runs/2026-09-01_native-ode-runtime-repair/`：**原生高频 ODE-RNN 等价执行与协议漂移修复**。GRU 观测更新改为同时间点批量调用，重构与投影改为整段张量调用；逐样本/逐点实现的输出和梯度对照通过。真实 CogPilot 四受试者单轮由旧记录 `1438.6` 秒降至 batch 4 的 `573.2` 秒，batch 13 为 `190.4` 秒、峰值显存 `3392.5 MiB`，未删除或重采样原生时间点。首次四候选运行因源码快照漂移退出排序；v3.1 新根冻结 commit、训练源码与 runner SHA-256，并以 30 秒墙钟心跳重跑全部训练内候选。本 run 不含外层结果或论文排名。
- `runs/2026-09-02_physics-objective-repair/`：**运动学一致性目标收窄与训练内重跑**。v3.1 完成 144/144 个训练内单元且显式时移、事件配对门通过，但 CogPilot/CLARE 的高采样率生理二阶导使物理项达到 `10^16` 至 `10^17`，整批结果退出排序。v3.2 只保留字段可支持的平移、垂向和旋转航电运动学残差；真实 CogPilot 原生窗口确认垂向残差计数 `268`、值 `8.17e-4`，生理伪物理项明确不可用。聚焦测试 `20 passed`、完整测试 `451 passed, 8 skipped`；外层结果未打开。
- `runs/2026-09-02_candidate-screen-v3p2/`：**论文模型四候选三随机种子训练内冻结**。评价协议 v3.2 在提交 `41683e98` 上完成仿真 12、鼎新 60、CogPilot 12、CLARE 60，共 144/144 个不重复单元；外层结果始终关闭。双目标候选在 seeds `17/29/43` 的均衡五分类时移准确率为 `0.2000/0.2909/0.3091`，正确—错误事件配对相似度差为 `0.2016/0.1447/0.1950`，以两个新机制门独占第一并冻结。目录保留原始紧凑结果、门禁审计、候选摘要和训练内应用指标；重型 checkpoint 继续位于被忽略目录。
- `runs/2026-09-03_thesis-simulation-pretraining-v3p2/`：**论文级受控仿真六方法三随机种子预训练**。在外层打开前冻结提交 `58db4458` 上完成 15/15 个训练 checkpoint、9/9 验收；全部方法使用 CUDA，Chronaris 启用安全滞后感知融合、可学习事件语义、显式时移和事件配对目标。训练期间 G2、任务真值和下游指标保持关闭；重型 checkpoint 与逐步辅助损失留在被忽略目录。
- `runs/2026-09-03_simulation-mask-alignment-repair/`：**跨方法有效查询 mask 审计修订**。clean 表示首次导出在旧 mask 相等假设处 fail-fast；实际 train/validation/held-out 的样本、查询时间和来源哈希一致，不同单流、朴素同步与双流方法的真实有效点数按各自观测支持不同。v3.2.1 保留各方法自身 mask 与池化合同，并把方法名和 mask 纳入对齐哈希；未生成任务指标，冻结 checkpoint 不重训。
- `runs/2026-09-03_native-physiology-integrity/`：**原生生理预处理完整性修复**。CogPilot 皮电输入只保留正电阻并换算为皮肤电导，事件响应改为前后窗皮肤电导中位数差；CogPilot 与 CLARE 删除使用窗外和后续观测的派生心率，改为原生心电图信号。数据结构定义升级后，CogPilot 难度 77 个样本、事件响应 225 个样本和 CLARE 认知负荷 60 个样本通过 6/6 项验收，公开与鼎新外层结果仍关闭。

> 2026 年 7 月的安全滞后感知融合、CogPilot 和 CLARE 研究结果现仅作历史追溯；受训练确定性、重复归一化、测试折缩放或上下文时间轴缺陷影响的报告均已退出毕业论文主结果，等待评价协议 v3 重跑。下列原始条目和数字不覆盖、不删除。

## 历史研究证据

以下条目描述六月至七月原始协议下的结果和执行状态，用于追溯；不能替代上方九月冻结结果，也不作为当前设备或任务指令。

- `runs/2026-07-23_cogpilot-difficulty/`：**CogPilot 飞行难度四分类（公开数据双流增量正向证据）**。LOSO（10 参与者训练/3 测试），12 epoch、seed 17：safe_lag macro-F1 `0.4122` 同时高于旧融合 `0.4017`、最佳单流 vehicle_only `0.3396`、physiology_only `0.1521`。难度同时驱动生理唤醒与飞机操纵/状态，融合真正超过最佳单流（晋级门禁 3/5 方向性满足）；与鼎新车辆主导机动任务形成对照。单 LOSO 折非锁定确认；重型 checkpoint 位于被忽略目录。

- `runs/2026-07-22_dingxin-safe-lag-maneuver/`：**鼎新未来机动安全滞后感知融合单折确认（真实下游）**。留一架次 fold01、seed 17、30 epoch，三组同口径同消费者比较：safe_lag 机动 macro-F1 `0.3387` 是旧 multiscale 融合 `0.1667` 的约 2 倍（同口径超全部融合基线），但仍低于航电单流 `0.4821`。证明安全旁路在真实任务上显著减小负迁移；非锁定确认（单折单种子）。重型 checkpoint 位于被忽略目录。

- `runs/2026-07-22_safe-lag-wave-a-smoke/`：**安全滞后感知融合波次 A 工程冒烟（机制验证）**。在 G1 仿真固定集（48 训练+12 验证+24 留出，15 epoch、seed 17）训练 chronaris safe_lag / multiscale / vehicle_only 三组；从 checkpoint 重算诊断（标准化特征 + 直推式恢复探针）。safe_lag 航电恢复 R² `0.944` 高于旧 multiscale `0.914`（亦高于航电单流 `0.933`），有效秩 `5.13` 为旧 `2.73` 的 1.88 倍，安全门控均值 `0.022` 近安全回退。支持“安全旁路改善航电保真与抗塌缩”的机制判断，非锁定确认；重型 checkpoint 位于被忽略目录。

- `runs/2026-07-22_safe-lag-aware-fusion-audit/`：**安全滞后感知融合：当前 Chronaris 与新数据审计**。逐项代码证据确认当前 Chronaris 九项缺陷（单流旁路缺失、256→64 瓶颈、96 点均值池化、同刻相似与滞后冲突、辅助损失前 10 epoch 为零、早停只看公共重构、尺度门控可塌缩、无有效秩诊断、预训练偏原值重构），物理约束健康；盘点 CogPilot/PhysioNet（主公开双流，35 名参与者，生理 + X-Plane 飞机状态，LSL 公共时钟）与 CLARE（辅助跨受试者，中枢 EEG vs 外周，LOSO）。为新主线研究计划与评价协议 v2 提供事实依据；本 run 不训练模型、不修改确认指标。

- `runs/2026-07-16_simple-downstream-thesis-evidence/`：**鼎新简化下游论文证据与独立复核**。汇总正式真实任务与既有仿真机制、压力和消融证据，生成技术报告、四张已抽查中文图、五张结果表和可渲染报告 artifact。报告阶段复核 108 项逐样本文件哈希，并从原始预测独立重算 504 个核心单元指标；与正式汇总的最大绝对差为浮点舍入量级，7/7 验收通过。结论为 Chronaris 时间机制局部有效，但真实下游整体优势尚未成立；本 run 未训练模型或修改确认指标。
- `runs/2026-07-16_simple-downstream-confirmation/`：**鼎新未来下游固定算法正式确认**。六种冻结表示在两个留一架次折、三个随机种子上完成 36/36 个方法—折—随机种子单元，形成 504 条有限指标。航电单流在未来机动上取得宏平均 F1 `0.8084`、Spearman `0.6165` 和相对当前状态技能 `0.6685`；Chronaris 对应为 `0.1948/0.0685/-123.5859`。未来生理字段中六种方法的正技能字段比例均为 `0`，全部未超过持久性基线。正式结果打开后禁止结果驱动调参。
- `runs/2026-07-16_simple-downstream-representations-confirmation/`：**鼎新简化下游正式表示导出**。两个留一架次折、三个随机种子和六种方法共 72/72 份表示导出完成，覆盖完整训练与留出任务清单，5/5 验收通过。表示阶段未打开任务目标或外层指标；稠密表示保存在被忽略重型目录。
- `runs/2026-07-16_simple-downstream-pretraining-confirmation/`：**鼎新简化下游正式任务无关预训练**。五个可训练方法在两个留一架次折和三个随机种子上完成 30/30 个串行训练单元，8/8 验收通过。一次 Chronaris CUDA 启动故障按原 checkpoint 和相同配置原地恢复成功，未切换设备或修改预算；任务目标和留出架次始终关闭。
- `runs/2026-07-16_simple-downstream-protocol/`：**鼎新简化下游任务协议与表示兼容性审计**。冻结原始点实跑形成 90 个完整未来视图上下文和 60 个独立机动上下文；未来生理状态按训练架次分别选择 12/11 个有效字段。旧冻结表示因排除 20 个在未来预测中合法的历史运动学字段，且两个留一架次折均未覆盖完整新任务清单，兼容性判定为 `retraining_required`。目录包含协议、任务统计、三态兼容性报告和 7/7 验收；本 run 未训练模型、未打开指标或修改历史确认结果。
- `runs/2026-07-16_simple-downstream-pretraining-smoke/`：**选定配置任务无关预训练工程冒烟**。留一架次第一折、seed 17 的生理单流、航电单流、MulT、ContiFormer 和 Chronaris 均按包含过去运动学字段的新输入合同完成 3 epoch 串行训练，5/5 checkpoint、8/8 验收通过；任务目标和留出架次始终关闭。该 run 只验证字段合同、设备、恢复和资源路径，不形成任务排名，也不改变正式 50 epoch、batch size 32、patience 8 预算。
- `runs/2026-07-16_simple-downstream-representations-smoke/`：**六方法完整任务清单表示导出工程冒烟**。五个任务无关 checkpoint 与训练折拟合的朴素时间同步共同导出 30 个训练视图上下文和 60 个留出视图上下文，六方法共 12 份 96 点、64 维表示目录，5/5 验收通过。旧内部划分遗漏的边界上下文已在冻结表示导出阶段恢复到完整任务清单；本 run 未读取任务目标或指标。
- `runs/2026-07-16_simple-downstream-consumer-smoke/`：**固定 Ridge/Logistic 下游算法工程冒烟**。六种方法在同一留一架次折完成未来机动连续预测、辅助强度分类和未来生理字段预测，生成 6 个方法单元、84 条指标，4/4 验收通过。机动结果先聚合为 30 个独立飞机上下文，生理结果覆盖 12 个字段；所有分数均明确标记为工程输出，不进入论文主表、不用于模型或消费者调参。
- `runs/2026-07-15_dingxin-residual-activation-task-decoupled/`：**鼎新教师辅助残差激活与任务解耦筛选**。在 6 个唯一训练内验证支持上完成 8 个预注册候选共 48 个候选—支持运行；候选覆盖固定、全局与样本条件门控，完整目标与显式基座误差目标，共享与任务独立适配器，以及选择性交叉拟合教师辅助。任务独立检查点组合后的最佳候选取得机动平均宏平均 F1 `0.9460`、连续响应中位 RMSE 比率 `0.8626` 和高响应平均归一化平均精确率 `0.4042`，但连续响应平均技能仍为 `-0.0171`，中位残差贡献比仅 `0.0005`，低于预注册的 `0.02` 激活下限。安全门禁通过，激活与研究门禁失败，三随机种子稳定性确认未启动。目录包含协议、架构与教师清单、候选级和逐支持指标、训练轨迹、门控与残差诊断、访问审计、恢复命令、2 幅已抽查中文图和差距报告；约 18 MB checkpoint 与逐单元状态位于被忽略的重型目录。外层观测、标签、预测和指标均未打开，既有确认指标不变。
- `runs/2026-07-15_dingxin-task-aware-safe-residual/`：**鼎新任务感知安全残差与部分解冻筛选**。复用 6 个唯一训练内验证支持和统一移除 30 个时间类航电通道的输入合同，完成六方法同预算任务基座、5 个冻结安全残差候选的 30 个候选—验证单元训练，以及 2 个主干学习率比例的 12 个部分解冻单元。冻结阶段选中的航电机动基座与直接观测响应基座取得机动平均/中位/最差宏平均 F1 `0.9164/0.9630/0.7500`、连续响应中位 RMSE 比率 `1.0071`、高响应平均归一化平均精确率 `0.3773`，三任务无损支持数均为 `6/6`。部分解冻比例 `0.02` 只改善高响应，比例 `0.05` 只轻微改善连续响应，均只改善 `1/3` 个任务，因此研究门禁失败，训练外教师蒸馏与配置锁定未启动。目录包含协议、架构、候选与训练记录、同预算基座、逐支持指标、门控和可训练参数审计、恢复命令、2 幅已抽查中文图及结构化差距报告；约 20 MB checkpoint 与训练状态保存在被忽略的重型目录。外层观测、标签、预测和指标均未打开，既有确认指标不变。
- `runs/2026-07-15_dingxin-target-reconstruction-confirmation/`：**鼎新目标重构、时间捷径抑制与统一条件完整预算确认**。六种方法在 6 个主选模验证单元和第三训练池压力单元中统一删除 30 个时间类航电通道，采用最多 50 epoch、patience 8 的相同无标签预算及固定任务头，完成 42 个方法—验证单元训练和 84 份训练内表示导出。未来机动连续分数最佳中位斯皮尔曼秩相关系数为 `0.5979`，未来机动趋势最佳平均/最差宏平均 F1 为 `0.6598/0.3535`；生理剩余响应最佳中位 RMSE 比率与平均技能为 `1.1614/-0.2529`，高剩余响应最佳平均/中位归一化平均精确率为 `0.0833/0.0000`。只有第三训练池连续目标评价通过，其余五项门禁均失败；高剩余响应阈值在 `5/6` 个主验证单元产生零正样本，说明标尺仍不稳定。目录包含预注册、协议、42 个训练记录、84 个表示导出记录、分单元指标与预测、访问审计、结构化差距报告和 5 幅已抽查中文图；约 512 MB checkpoint 与表示保存在被忽略的重型目录。本轮未读取外层测试，未修改或使用任务标签训练 Chronaris 主干，既有确认指标不变。
- `runs/2026-07-15_dingxin-target-reconstruction/`：**鼎新目标重构短预算筛选**。在与确认运行相同的输入、任务、划分和方法合同下，使用最多 12 epoch、patience 4 完成全链路筛选。35 个可训练单元中有 24 个最佳 epoch 落在预算上限，因此该根只用于发现未收敛风险和验证运行链路，不承担最终退出结论；其紧凑证据和约 512 MB 被忽略重型产物保留，不与完整预算确认混算。
- `runs/2026-07-14_dingxin-task-stability/`：**鼎新任务协议修复与跨视图稳定化审计**。新协议在两个可用外层训练池中形成 4+2 个唯一主选模验证单元，全部覆盖低、中、高三类，完整 35 秒支持区间互不重叠且额外保留至少 35 秒间隔；第三个训练池在不降低约束时无法形成合法单元，只保留为分布压力诊断。22 个预注册轻量候选完成 132 个候选—验证单元运行；机动强度分类最佳平均/中位/最差 Macro-F1 为 `0.8286/0.9111/0.6627`，连续生理响应最佳中位 RMSE 比率为 `1.0000`，高生理响应最佳平均/中位归一化平均精确率为 `0.2089/0.0687`。仅用时间块位置的诊断模型最高 Macro-F1 达 `0.9153`，因此正式候选统一移除 30 个实际存在的时间类航电通道；标签源直接与确定性派生重叠均为 0。三项相对门禁均未完整通过，两个后续放行标志均为 false。目录包含协议、划分、目标稳定性、候选结果、访问与输入审计、结构化差距报告和 7 幅已抽查中文图；重型特征缓存与候选状态位于被忽略目录。本轮未读取外层测试，未修改或训练 Chronaris 主干，既有确认指标不变。
- `runs/2026-07-14_dingxin-core-feasibility/`：**鼎新核心任务可行性与安全融合审计**。只在净化后的 inner-train/validation 上比较 12 个有界候选，外层测试保持关闭。机动分类 Macro-F1、未来生理响应 RMSE 和高响应识别 AUPRC 的最佳训练内均值为 `0.8241/0.8368/0.7875`，均未越过 `0.950/0.285/0.900` 门槛；冻结专家安全融合按协议未启动。开发清单剔除第三折两个与完整 35 秒支持区间相交的训练上下文并移除全部外层标识，标签与阈值只用净化后的 inner-train 重拟合。目录包含协议、哈希证据、636 条 consumer 比较、102 条折级指标、任务诊断、访问审计、结构化 gap 和三幅已抽查中文图；重型因果查询缓存与 33 个候选状态位于被忽略目录，历史确认指标不变。
- `runs/2026-07-12_downstream-evidence-pack/`：**固定数据下游评估论文证据包**。锁定汇总鼎新真实弱监督、仿真 clean、七因素压力、时间机制恢复、Chronaris 四机制消融、端到端辅助、仿真预训练到鼎新适配和 UAB/NASA 公开数据适配共 8 层证据，生成 7 幅中文图、预声明主指标表、54 项迁移增量、证据矩阵和结论边界，9/9 门禁通过。七图已逐图抽查并从渲染源码移除机器折 ID、轨迹 ID 和未解释英文；真实与仿真、冻结与标签微调继续独立表述。
- `runs/2026-07-12_simulation-chronaris-ablation-consumers/`：**Chronaris 四机制锁定下游消融**。四变体 × 三随机种子完成 12/12 个冻结 consumer、768 条指标、384 条完整模型方向归一优势和 72 条 48 轨迹配对统计，6/6 门禁通过。完整模型相对去连续演化在负荷分类/回归/机动分段为 +0.0691/+0.0278/+0.0142，相对去因果掩码为 +0.0060/+0.0082/+0.0662，相对单尺度时延为 +0.0056/+0.0118/+0.0623；去物理约束为 -0.0193/+0.0118/+0.0044。结果支持分机制、分任务解释，不支持四机制全任务一致正向。
- `runs/2026-07-12_dingxin-synthetic-pretrain-adapt-consumers-coalesced/`：**仿真预训练到鼎新适配的冻结下游评估**。90/90 个方法—折—seed consumer、5,040 条指标、3,360 条融合增益和 504 条主折汇总完成，8/8 门禁通过；表示族唯一为 `synthetic_pretrain_real_adapt_v1`。与 real-only 相同 seed、三个主视图折、消费者和指标配对后，Chronaris 的机动分类 Macro-F1 变化 -0.0048，高生理响应 AUPRC 变化 -0.0336，生理响应 RMSE 方向归一改善 +0.0062。迁移后绝对值为 0.7346、0.8503 和 0.3303；结果不支持仿真预训练带来一致真实任务收益。
- `runs/2026-07-12_dingxin-synthetic-pretrain-adapt-representations-coalesced/`：**仿真预训练到鼎新适配的统一表示**。15/15 个 seed—折单元导出生理单流、航电单流、朴素时间同步、MulT、ContiFormer 和 Chronaris 的 train/validation/outer-test 三角色表示，共 270/270 份 `[N,96,64]`，6/6 门禁通过。表示族固定为 `synthetic_pretrain_real_adapt_v1`，outer-test 只导出表示而不在本阶段计算任务指标。
- `runs/2026-07-12_dingxin-synthetic-pretrain-adapt-pretraining-coalesced/`：**仿真预训练到鼎新无标签适配的三随机种子五折重训**。五个可训练方法共完成 75/75 个“方法—折—seed”单元，8/8 门禁通过，全部实际使用 RTX 4090；早停只读取公共自监督 validation，任务目标与 outer-test 访问均为零。六方法使用相同仿真预训练数据预算；跨 schema 只复制同名同形任务无关参数，字段相关输入与重构层重新初始化。复制目标编码器元素比例按方法为 97.93%、24.61%、84.47%、36.96% 和 39.05%，源 checkpoint 哈希逐单元记录。重型 checkpoint 留在被忽略目录，本 run 只完成无标签适配，不形成迁移效果结论。
- `runs/2026-07-12_dingxin-locked-consumers-coalesced/`：**鼎新三随机种子五折冻结表示正式下游评估**。90/90 个方法—折—种子 consumer、5,040 条指标、3,360 条双流增益和 504 条主折汇总全部完成，8/8 门禁通过。主表只汇总三个留一视图折，两个留一架次折只作辅助，不报告窗口级显著性。Chronaris 在 MiniRocket 高生理响应识别上以 AUPRC 0.8839 居首，最差视图折平均 0.6516 也居首，相对最佳单流增益 0.0094。机动强度 Macro-F1 0.7394 低于 ContiFormer 0.9409，生理响应 RMSE 0.3365 低于 MulT 0.2912；鼎新证据支持高响应识别的分项优势，不支持三任务全面领先。
- `runs/2026-07-12_dingxin-locked-representations-coalesced/`：**鼎新三随机种子五折六方法统一表示**。75 个训练 checkpoint 全部完整后，15/15 个种子—折导出生理单流、航电单流、朴素时间同步、MulT、ContiFormer 和 Chronaris 的 train/validation/outer-test 三角色表示，共 270/270 份 `[N,96,64]`，6/6 门禁通过。表示族唯一为 `frozen_task_agnostic_v1`；outer-test 只导出表示，该阶段不计算任务指标。
- `runs/2026-07-12_dingxin-locked-pretraining-coalesced/`：**鼎新三随机种子五外层折锁定重训**。五个选定配置共完成 75/75 个“方法—折—种子” checkpoint，8/8 门禁通过。六方法公共输入使用 100 ms 因果时间箱，归一化仅拟合 inner-train；五个可训练方法均实际使用 RTX 4090。Chronaris 方法专属损失参与反向传播，但早停只读取公共自监督 validation；15 个 Chronaris 单元最佳 epoch 范围为 8–46。任务目标与 outer-test 访问数全程为零；约 948 MB checkpoint 只位于被忽略目录，本 run 不形成任务排名。
- `runs/2026-07-12_simulation-locked-stress-consumers/`：**G2 七因素锁定压力下游评估**。三随机种子、35 场景和六方法完成 630/630 个冻结 consumer 评估，生成 20,160 条指标、4,032 条退化斜率和 630 条 48 轨迹配对统计，6/6 门禁通过；压力场景不重训、不调参。Chronaris 在 21 个预声明“任务—因素”组合的最高单因素压力下有 15 个绝对表现位于前二，但随机/连续缺失的主指标平均退化斜率为 -0.0733/-0.1264，均为六方法第六。该结果支持单因素重压下的绝对任务竞争力，不支持全因素退化最慢；混合重压也不写成统一优势。
- `runs/2026-07-12_simulation-chronaris-ablation-pretraining/`：**Chronaris 四项机制消融锁定重训**。无连续演化、无物理一致性、无因果掩码和单尺度时延四个变体在 seeds 17/29/43 上共完成 12/12 个 checkpoint，6/6 门禁通过。训练和保留确认只读取 G1 公共自监督目标，G2 锁定测试与任务真值全程关闭；重型 checkpoint 约 38 MB，位于被忽略目录。本 run 只冻结消融 encoder，不形成任务性能结论。
- `runs/2026-07-12_simulation-chronaris-ablation-representations/`：**Chronaris 机制消融 G1→G2 统一表示**。12 个锁定消融 checkpoint 全部完整后，每个变体—种子均导出 G1 train、G1 validation 和 G2 held-out 三个角色，共 36/36 份 `[N,96,64]` 表示，5/5 门禁通过。表示阶段不读取任务真值；后续只允许与完整 Chronaris 共用同一冻结 consumer 协议做 48 轨迹配对消融。
- `runs/2026-07-12_simulation-end-to-end-finetuning/`：**仿真端到端微调辅助表**。三随机种子六方法共 18/18 单元、54 份 `end_to_end_finetuned_v1` 表示和 864 条指标完成，7/7 门禁通过；五个可训练方法更新完整编码器，朴素同步只更新同容量任务头，held-out 从未参与梯度或早停。MulT 在微调负荷分类和机动分段领先，Chronaris 只在回归 RMSE 0.2666 略居首。Chronaris 微调 macro-F1/RMSE/frame macro-F1 为 0.3145/0.2666/0.3638，弱于冻结主表 0.5110/0.1928/0.4687；该 run 作为小样本过拟合与适配能力诊断，不替代冻结任务无关表示主结论。
- `runs/2026-07-12_simulation-mechanism-consumers/`：**时间偏移与生理响应时延恢复锁定评价**。四方法在三个随机种子上完成 24 个 G1 Ridge 探针和 840 个 G2 压力场景评价，输出 3,360 条正式指标、630 条以 48 条潜在轨迹为单位的配对统计，6/6 门禁通过。Chronaris 在 35 场景与三 seed 汇总的时钟偏移 MAE/RMSE/±0.5 秒命中率为 0.8883/0.9932/0.5210，响应时延 MAE/RMSE/±2 秒命中率为 7.4859/8.7352/0.2073，均为四方法最佳；MulT 的响应时延 Spearman 0.3584 高于 Chronaris 0.1766。该证据支持绝对误差与容差命中优势，不支持所有机制指标全面领先。
- `runs/2026-07-12_simulation-mechanism-representations/`：**时间偏移与生理响应时延恢复表示**。三个随机种子的朴素同步、MulT、ContiFormer 和 Chronaris 在 G1 train/validation 六个观测场景上导出 144/144 份统一表示，36 个随机种子—角色—场景单元均含四方法，5/5 门禁通过。时间偏移与响应时延真值在该阶段保持关闭；下游只允许 G1 train 拟合、G1 validation 选择 Ridge 强度，再到 G2 压力输入锁定评价。
- `runs/2026-07-12_batched-viterbi-runtime-benchmark/`：**持续时间约束批量解码运行时基准**。在 seed 17 生理单流、192 条真实压力场景 TCN logits 上，batch 维向量化将逐样本外推 77.55 秒降到 1.76 秒，约 43.95 倍；时间、类别、持续时间候选顺序和回溯规则均不变，真实前四样本与随机单元测试逐位一致。旧标量 run 未完成任何 seed—场景单元并已隔离，不进入正式压力指标。
- `runs/2026-07-12_simulation-locked-consumers/`：**G1 到 G2 三随机种子六方法锁定下游主表**。18/18 方法—随机种子单元完成 Logistic/Ridge、MiniRocket 10,000 kernels、GPU 因果 TCN 和持续时间约束解码，生成 1,152 条指标、768 条双流增益、90 条以 48 条 G2 轨迹为独立单位的配对统计，7/7 门禁通过。Chronaris 在线性探针负荷分类/回归（macro-F1 0.5110、RMSE 0.1928）和持续时间约束机动分段（frame macro-F1 0.4687、segmental F1@0.25 0.5319）居首；MulT 在 MiniRocket 负荷分类及边界 F1/延迟上居首。Chronaris 相对最佳单流的分段 frame/segmental 平均增益为 0.1194/0.1109，边界 F1@1s 增益为 -0.0053；本结果支持分项机制优势，不支持所有指标全面领先。
- `runs/2026-07-12_simulation-locked-stress-representations/`：**G2 三随机种子 35 场景压力表示**。seeds 17/29/43、六方法和 35 个严格成对压力场景共导出 630/630 份 `[192,96,64]` 表示，105 个随机种子—场景单元均含六方法，每条 48 轨迹保留四个上下文，5/5 门禁通过。该阶段只读取原始异步双流观测，不打开任务真值或指标；后续只复用 clean 主表锁定的 consumer，不能按压力结果重新拟合。
- `runs/2026-07-12_minirocket-solver-runtime-benchmark/`：**MiniRocket 高维逻辑回归求解器运行时基准**。只使用 seed 17 生理单流的 G1 train/validation 冻结表示，MiniRocket 输出为 `[384,9996]`；默认 `lbfgs` 所在完整组件约 1743.68 秒，显式 OvR `liblinear` 三个 C 分别为 2.20、2.40、2.37 秒。求解器只按运行时与高维小样本形态锁定，不按验证分数选择；规则对六方法完全一致，G2 锁定测试比较未参与。旧模型只留在被忽略 runtime 目录，不进入正式主表。
- `runs/2026-07-12_simulation-locked-representations/`：**仿真三随机种子六方法统一表示**。三个随机种子的生理单流、航电单流、朴素时间同步、MulT、ContiFormer 和 Chronaris 均导出 G1 train、G1 validation、G2 held-out 表示，共 54/54 份 `[N,96,64]`、7/7 门禁通过。五个任务无关 encoder 全部来自锁定 checkpoint，朴素时间同步只在训练折拟合；所有 checkpoint 完整后才读取 G2 原始观测，任务真值与指标在本 run 中保持关闭。
- `runs/2026-07-12_simulation-locked-pretraining/`：**仿真选定配置三随机种子锁定重训**。seeds 17/29/43 的生理单流、航电单流、MulT、ContiFormer 和 Chronaris 共 15 个训练单元均完成 50 epoch，15/15 checkpoint 和 9/9 门禁通过。训练只读取 G1 原始异步双流，Chronaris 专属机制损失参与反向传播但早停只读取公共自监督 validation；任务真值和 G2 锁定测试始终关闭。本 run 冻结正式表示导出的任务无关 encoder，不构成任务性能排名。
- `runs/2026-07-12_dingxin-coalesced-pretraining-smoke/`：**鼎新公共因果时间箱五方法运行时验证**。六方法统一在归一化和增强前使用 100 ms 固定因果时间箱，箱内同字段取均值、时间戳取最后真实观测；首个上下文航电/生理事件由 1779/150 点变为 300/120 点且末端时间不变。留一视图第一折 seed 17 的五方法均完成 1 epoch、8/8 门禁通过，四个 GPU 深度基线为 8.59–16.46 秒、Chronaris CPU 为 91.86 秒。该 run 只锁定输入与运行时合同，不形成任务排名。
- `runs/2026-07-12_dingxin-coalesced-chronaris-gpu-smoke/`：**鼎新 Chronaris 合并输入 GPU 运行时验证**。与公共时间箱 run 使用相同折、seed、配置和输入，Chronaris 在 RTX 4090 完成 1 epoch，训练耗时 40.07 秒、8/8 门禁通过；相对同批 CPU 91.86 秒明显更快。因此正式鼎新队列采用单 GPU 串行，重复驱动故障时才按 checkpoint 设备历史迁移 CPU。本 run 不形成任务排名。
- `runs/2026-07-12_simulation-locked-representations-gpu-smoke/`：**锁定表示混合设备导出验证**。使用 seed 17 五个已完成任务无关 checkpoint 与训练折朴素同步变换，四个深度/单流基线在 RTX 4090、Chronaris 在 CPU、朴素同步在 CPU/PCA 上导出 G1 train、G1 validation、G2 held-out 三角色共 18 份 `[N,96,64]` 表示；18/18 输出、7/7 门禁通过，耗时约 2 分 41 秒。任务 oracle 和指标保持关闭；本 run 只验证正式三 seed 导出的混合设备与 lineage，不形成方法排名。
- `runs/2026-07-12_dingxin-synthetic-pretrain-adapt-smoke/`：**仿真预训练到鼎新无标签适配协议验证**。使用已完成的 G1 生理单流 checkpoint 初始化鼎新留一视图第一折；只复制同名且形状一致的任务无关参数，字段相关输入层重新初始化。复制目标编码器元素比例为 97.93%，源 checkpoint SHA-256、复制清单和 schema 差异进入新 checkpoint；任务目标与 outer-test 均保持关闭，8/8 验收通过。本 run 只验证跨 schema 迁移协议，不构成迁移效果结论。
- `runs/2026-07-12_simulation-chronaris-ablation-pretraining-smoke/`：**Chronaris 机制消融锁定重训协议验证**。seed 17 的无物理约束变体完成 1 epoch，checkpoint 明确记录 `variant=no_physics`，物理一致性项有效计数为 0；恢复复用后 6/6 验收通过。首次训练约 3 分 03 秒，重型 checkpoint 位于被忽略目录；本 run 不形成消融性能结论。
- `runs/2026-07-12_simulation-chronaris-ablation-representations-smoke/`：**Chronaris 机制消融表示协议验证**。seed 17 无物理约束 checkpoint 回载后，对 G1 train、G1 validation 和 G2 held-out 的 384/96/192 个上下文分别导出 `[N,96,64]` 表示，共 3/3 输出、5/5 验收通过。当前并行条件下耗时约 22 秒、峰值约 0.89 GB；任务 oracle 和指标保持关闭，本 run 不形成消融性能结论。
- `runs/2026-07-12_dingxin-locked-pretraining-smoke/`：**鼎新锁定重训单折协议验证**。留一视图第一折、seed 17 的五个选定配置各完成 1 epoch；生理单流、航电单流、MulT、ContiFormer 实际使用 RTX 4090，Chronaris 使用同批实测更快的 CPU 连续演化路径。inner-train 归一化、公共 validation 早停、方法专属损失、outer-test 禁止访问和任务目标关闭均写入 manifest，5/5 checkpoint、8/8 验收通过。首次耗时 15 分 52 秒、峰值约 8.9 GB；本 run 只验证正式 75 单元队列协议，不形成任务排名。
- `runs/2026-07-12_aviation-simulation-locked-stress-audit/`：**G2 锁定压力场景生成与审计**。48 条 G2 潜在轨迹各生成 35 个观测版本，覆盖时间戳抖动、时钟偏移、时钟漂移、随机缺失、连续缺失、生理响应额外时延、观测信噪比的全部固定等级和 mixed-severe，共 1,680 个场景。同一轨迹跨等级共享 latent、事件、负荷真值和 observation seed，只改变观测配置；方法无关源码与成对随机性 7/7 验收通过。约 1.7 GB 原始双流和真值位于被忽略目录，本 run 只形成压力输入，不含模型任务指标。
- `runs/2026-07-11_encoder-candidate-screen-seed17/`：**seed 17 编码器候选正式筛选**。五个可训练方法各完成 A–D 四候选、最多 50 epoch 的公共自监督训练，共 20/20 checkpoint；排序只使用 23 个 G1 validation profile，另 1 个预留 profile 仅确认五个选定配置。Chronaris、MulT、ContiFormer、生理单流选择 A，航电单流选择 C；预留确认损失均有限。任务标签、仿真真值和封存测试保持关闭，7/7 验收通过。Chronaris 参考采样向量化与旧算法输出/梯度一致；CPU 17.07 秒/epoch，RTX 4090 为 49.87 秒/epoch，因此该主干正式筛选使用 CPU。约 117 MB checkpoint 位于被忽略目录；本 run 冻结配置但不构成锁定测试结论。
- `runs/2026-07-11_encoder-candidate-screen-smoke/`：**seed 17 编码器候选筛选全链路冒烟验证**。G1 原始异步双流按 96 个训练 profile、23 个候选排序 validation profile 和 1 个开发确认 profile 组织；五个可训练方法各运行 A–D 四候选单 epoch，共 20 个 checkpoint。验证增强固定，排序只使用三项公共自监督损失并做方法内 min-max；任务标签、仿真真值和封存测试均未打开。全链路耗时 5 分 03 秒、峰值内存 4.34 GB、重型 checkpoint 约 114 MB，6/6 验收通过。该 run 只证明正式 50 epoch/patience 8 筛选可运行，单 epoch 排名不作为候选结论。
- `runs/2026-07-11_dingxin-nested-validation/`：**鼎新嵌套目标 validation-only consumer**。五折六方法使用 inner-train 嵌套目标拟合固定线性与 MiniROCKET，形成 30 个方法—折 bundle；组件恢复 60/60，预测哈希 30/30 一致。只评价 validation，生成 840 条全部可计算指标和 560 条双流增益；指标 role 唯一为 validation、threshold scope 唯一为 inner-train nested，outer-test 零预测/指标。三个 validation 缺低机动类时 macro-F1 固定三类集合，不补类。约 35 MB 模型/预测位于被忽略目录，12/12 通过；本 run 是正式 screen 前协议确认，不形成排名。
- `runs/2026-07-11_dingxin-nested-targets/`：**鼎新 inner-train 嵌套目标**。五折机动语义尺度、分位阈值、生理字段有效性/IQR 与高响应阈值均只用 inner-train 重拟合，validation/outer-test 仅应用参数。生成 10 个确定性 archive，机动分类 440 个角色上下文、生理响应 425 个可用角色上下文；相对 outer-train 工程冒烟口径，75 个机动类别和 51 个高响应标签变化，连续响应 Spearman 为 0.9787–0.9971。三个时间块 validation 只覆盖中/高机动类，保持真实分布不补类。snapshot 哈希不变，10/10 通过；本 run 不训练模型或生成指标。
- `runs/2026-07-11_dingxin-consumer-smoke/`：**鼎新五折冻结表示下游 consumer 工程冒烟**。五个固定外层折和六种表示方法共享同一线性模型与 MiniROCKET 配置，形成 30 个方法—折 bundle、60 个消费者组件；首次拟合累计 144.34 秒，恢复 60/60，预测哈希 30/30 一致。机动强度分类、生理响应回归和高生理响应识别共生成 1680 条全部可计算的 smoke-only 指标，方向归一双流增益 1120 条；15/15 验收通过。约 36 MB 模型与逐样本预测位于被忽略目录，鼎新与仿真指标保持分层。当前仍使用 outer-train 阈值，只证明消费链路，不形成排名；正式 screen 前必须重建 inner-train 嵌套目标。
- `runs/2026-07-11_dingxin-five-fold-pretraining/`：**鼎新五折六方法公共预训练与统一表示聚合审计**。三个留一视图主协议折与两个留一架次辅助折均完成，每折 6 个 checkpoint、18 份 train/validation/outer-test 表示，合计 30 个 checkpoint 和 90 份 `[N,96,64]` 表示。聚合层逐项重验 archive/manifest、样本顺序、source hash、checkpoint 文件与 inner-train fit hash；每折恢复 18/18，五个子 run 60/60、聚合 13/13。五折五方法累计训练 1066.45 秒，最高峰值 2047.1 MB；约 393 MB 重型产物位于被忽略目录。任务目标与 outer-test 指标全程关闭，本 run 不形成模型排名。
- `runs/2026-07-11_dingxin-fold-pretraining-smoke/`：**鼎新主协议首折六方法公共预训练与统一表示导出**。留一视图第一折的 inner-train/validation/outer-test 各 31 个完整上下文；五个可训练方法共享 inner-train 归一化、增强和三个公共目标，各完成 1 epoch、31 step，朴素同步仅拟合因果 forward-fill 与随机化主成分分析。六个 checkpoint 导出三角色共 18 份 `[N,96,64]` 表示，恢复 18/18 复用并通过同角色对齐。累计训练 213.63 秒、完整成功链路峰值 1967.4 MB，约 79 MB 重型产物位于被忽略目录；紧凑证据约 292 KB，12/12 验收通过。本 run 未打开任务目标、未计算 outer-test 指标或形成排名。
- `runs/2026-07-11_dingxin-inner-splits/`：**鼎新外层折训练内验证划分**。五个外层折均形成 inner-train、validation、overlap embargo 和 outer-test 四种互斥角色；训练组含两个架次时完整留出一个架次，只含同一架次时按末端七个时间块验证并删除重叠上下文。五折 inner-train/validation 为 31/31、31/31、38/14、19/7、38/14，共享航电原始时间区间重叠数为 0；分类各角色覆盖三类，生理响应各角色均覆盖连续值和高/非高两类，11/11 验收通过。现有 outer-train 阈值只供固定配置 smoke，正式候选筛选前必须嵌套重拟合。本 run 未训练模型、未读取 outer-test 指标或形成排名。
- `runs/2026-07-11_dingxin-context-bindings/`：**鼎新原始双流上下文与弱监督目标绑定**。96 个标签上下文中 93 个具备完整 30 秒输入，三个部分末窗只有 25.991 秒并结构化不可用；五折机动分类绑定 93 个、生理响应绑定 90 个唯一可用上下文。公共 schema 为 12 个生理字段、955 个航电字段，20 个机动标签源在 raw 映射层删除，最大输入相对时间 29.999 秒。允许字段采用 78.6 MB CSR 缓存而非 GB 级稠密 bundle，完整审计 12.34 秒、峰值 767 MB；目标、阈值、snapshot 哈希和外层 group 隔离均通过，12/12 验收完成。本 run 未训练模型或生成任务指标。
- `runs/2026-07-11_dingxin-application-targets/`：**鼎新两项弱监督任务独立目标归档**。机动强度分类保留 G1 五折训练阈值和 96 个上下文标签，每折 train/test 均覆盖低、中、高三类，20 个标签源字段继续全部禁止进入输入。生理响应从冻结原始点重算当前/未来 5 秒窗口中位数，字段筛选、IQR 与高响应阈值只用训练上下文；因原始点在 181 秒结束，3 个末端候选不足完整 5 秒未来区间，正式目标保留 90/93。五折两个任务共 10 个 archive，恢复 10/10 复用，原 snapshot 哈希不变，12/12 验收通过。约 248 KB 目标与阈值文件留在被忽略目录，本 run 未训练模型或生成任务指标。
- `runs/2026-07-11_application-consumer-smoke/`：**六方法应用型下游消费者闭环冒烟验证**。从 16 条 G1 仿真训练轨迹构造 64 个跨状态上下文和 32/16/16 profile 隔离划分；六方法共导出 18 份 96 点、64 维表示。固定线性探针、MiniROCKET 10,000 kernels、两层因果 TCN 与训练折持续时间解码生成 384 条全部可计算的 smoke-only 指标、256 条方向归一融合增益和 30 条以 4 条留出轨迹为独立单位的配对统计。删除 Chronaris 表示、MiniROCKET 和 TCN 后均只重建缺失组件，未删除文件哈希不变，预测哈希一致；12/12 验收通过。约 15 MB 表示、消费者模型和逐样本预测留在被忽略目录，本 run 不形成模型排名。
- `runs/2026-07-11_shallow-baseline-adapter-smoke/`：**两个单流与朴素时间同步生产适配器冒烟验证**。仿真三划分与鼎新三个不同视图分别完成生理单流、航电单流和朴素时间同步留出折导出，共 6 个输出；恢复复核 6/6 复用，14/14 验收通过。未来观测扰动与非激活模态扰动对历史输出的最大变化均为 0。该 run 只验证生产适配器、因果边界和训练折隔离，未运行公共自监督训练或下游指标；约 4.3 MB 检查点和稠密表示留在被忽略目录。
- `runs/2026-07-11_deep-baseline-adapter-smoke/`：**MulT 与 ContiFormer 任务头前生产适配器冒烟验证**。仿真与鼎新各完成两个方法的留出折导出，共 4 个 96 点、64 维输出；恢复复核 4/4 复用，15/15 验收通过。严格未来扰动对历史输出的最大变化为 0，生理与航电历史扰动均产生非零表示变化。该 run 只验证因果深度基线、双流计算路径和训练折隔离，未运行公共自监督训练或下游指标；约 11 MB 检查点和稠密表示留在被忽略目录。
- `runs/2026-07-11_chronaris-continuous-adapter-smoke/`：**Chronaris 连续融合生产主干机制验证**。原始异步双流经独立 ODE-RNN、96 点连续查询、逐项物理可用性和 0–5/5–15/15–30 秒因果融合形成 64 维表示；仿真与鼎新各完成一个留出折输出，恢复 2/2 复用，21/21 验收通过。未来扰动不改变历史完整模型输出，而无因果掩码消融在两套数据上均暴露未来敏感性；四项固定消融共 8 次有限值前向通过。该 run 未执行公共自监督训练或下游指标；约 2.0 MB 检查点和稠密表示留在被忽略目录。
- `runs/2026-07-11_common-pretraining-loop-smoke/`：**六方法公共预训练—折外表示—线性下游闭环验证**。从仿真 train split 的 16 个不同 G1 profile 构造 8/4/4 折，五个方法共享增强、遮挡重构、短期预测和时延判别训练 1 epoch，朴素时间同步只拟合训练折无监督变换。六方法三种 role 共 18 个输出恢复 18/18 复用；自动删除 Chronaris 留出折后单项重建哈希一致。workload 真值只在五个 checkpoint 完成后打开，固定 Logistic/Ridge 产生 72 条 smoke-only 指标；20/20 验收通过。约 37 MB checkpoint、表示和逐样本预测留在被忽略目录。
- `runs/2026-07-11_representation-contract-smoke/`：**统一双流输入与融合表示合同冒烟验证**。鼎新 12 个生理字段、955 个跨架次同序航电字段和 20 个标签源排除项通过同一批次合同；仿真只读取 7/12 个观测字段。六个方法接口均生成 96 点、64 维合同探针表示，样本标识、查询时间、有效掩码和留出折检查点来源一致，14/14 验收通过。该 run 只证明基础设施和恢复机制贯通，不是六种模型结果；探针检查点与稠密表示位于被忽略目录。
- `runs/2026-07-10_aviation-simulation-audit/`：**方法无关航空人机异步双流仿真正式审计**。被忽略的重型 bundle 约 1,018 MB，包含训练/验证/锁定测试 96/24/48 条潜在架次和每条六个观测版本，共 1,008 个 scenario；本目录保留 oracle validation、paired latent 审计、19 项验收、split/profile/seed 隔离和 4 张已抽查中文图。低/中/高负荷全局占比 23.3%/43.0%/33.6%，G1 残差中位数最坏 0.0114，G2 残差 95% 分位最坏 0.0421，干净场景时延 ±1 秒命中率 100%。该证据只承担仿真真值与压力机制验证，不等价于新增鼎新真实数据。
- `runs/2026-07-10_dingxin-input-snapshot/`：**鼎新原始异步点冻结紧凑证据**。实际 5.3 MB 高频值保存在被忽略的 `artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot/`；本目录只提交文件 SHA-256、点数、时间范围、measurement 分布和字段排除合同。2 个 sortie 的航电点均为 28,824，3 个 view 的生理点均为 905，6/6 与既有窗口 manifest 一致，20/20 机动标签源字段明确禁止进入新分类输入。本 run 未训练模型、未生成下游指标。
- `runs/2026-07-10_fixed-data-audit/`：**固定数据、字段语义与下游任务合同审计**。确认现有 2 个 sortie、3 个 view、111 个窗口可构造 96 个机动分类上下文和 93 个未来生理响应上下文；5 个外层折均完成，MySQL 字段语义解析无错误。该 run 不含模型训练结果；窗口均值只用于 G1 目标可行性验证，正式原始点实验将使用窗口中位数。历史对齐后投影因可能包含机动标签源信息，不具备新的防泄漏分类主结果资格。
- `runs/2026-07-06_fusion-stream-structure-plan/`：**融合表示流结构评价开发计划（E3）**，是 planning 文档，不是实验结果。包含 `report.md`、`input_contract.md`、`metric_contract.md`、`implementation_plan.md`、`acceptance_checklist.md`、`evidence_manifest.json`。本轮未编码、未训练、未改 confirmed metrics；后续编码任务需等待人工 review 本计划后再执行。
- `runs/2026-07-03_thesis-protocol-snapshot/`：论文协议快照，包含 `experiment_registry.csv`、`result_matrix_long.csv`、`result_matrix_summary.csv`、`claim_boundary_table.csv`、`thesis_protocol_summary.json` 和 `report.md`。
- `runs/2026-07-02_metric-calibration/`：指标校准，保留分类任务校准、公开路线校准、检索任务沿用边界和 GPU/runtime 记录。
- `runs/2026-07-02_selected-model-summary/`：选定模型汇总。
- `runs/2026-07-02_selected-model-reevaluation/`：选定模型再评估。
- `runs/2026-07-02_stream-role-fusion/`：流角色融合。
- `runs/2026-07-02_task-head-calibration/`：任务头校准。
- `runs/2026-07-02_cross-evidence-matrix/`：跨证据矩阵。
- `runs/2026-07-02_dingxin-thirdparty-comparison/`：鼎新真实数据第三方模型对比。
- `runs/2026-07-02_public-fusion-ablation/`：公开融合消融。
- `runs/2026-07-01_public-model-comparison/`：公开模型对比。
- `runs/2026-07-01_public-fusion-calibration/`：公开融合校准和 GPU profiling 子目录。
- `runs/2026-06-21_thesis-materials-report-figures/`：论文图表材料。
- `runs/2026-06-19_dingxin-leakage-safe-ablation/`：鼎新防泄漏组件消融。
- `runs/2026-06-19_rotation-audit-figure-refresh/`：rotation audit 图件刷新。
- `runs/2026-06-14_llm-preprocessing-context/`：LLM preprocessing context。
- `runs/2026-06-14_llm-preprocessing-comparison/`：LLM preprocessing 对比。
- `runs/2026-06-13_runtime-schema-contract/`：runtime schema contract。
- `runs/2026-06-13_dingxin-weak-label-sweep-resume/`：鼎新弱监督任务扫描 stable resume。
- `runs/2026-06-13_dingxin-weak-label-sweep-partial/`：鼎新弱监督任务扫描 partial。
- `runs/2026-06-07_evidence-closure/`：证据闭环。
- `runs/2026-06-07_dingxin-multitask-real-closure/`：鼎新 weak-label multitask real closure。
- `runs/2026-06-07_dingxin-weak-label-multitask-sweep/`：鼎新弱监督任务 multitask sweep。
- `runs/2026-06-07_dingxin-opt-package/`：鼎新优化包。
- `runs/2026-06-07_dingxin-component-ablation/`：鼎新组件消融。
- `runs/2026-06-07_public-adapter-calibration/`：公开数据适配校准。
- `runs/2026-06-07_public-transfer-boundary/`：公开数据 transfer boundary。
- `runs/2026-06-07_rigid-body-diagnostics/`：刚体约束诊断。
- `runs/2026-06-07_rotation-audit-closure/`：rotation audit closure。
- `runs/2026-06-07_semantic-support/`：semantic support。
- `runs/2026-06-07_semantic-event-support/`：semantic event support。
- `runs/2026-06-07_runtime-inference-service/`：runtime inference replay。
- `runs/2026-06-07_midterm-evidence-pack/`：中期证据包。
- `runs/2026-05-09_public-fusion-nasa-full-confirm/`：NASA 公开融合确认。
- `runs/2026-05-08_public-mainline-uab-robust-prior-r1/`：公开主线汇总。
- `runs/2026-05-08_public-opt-nasa-prepared-v2/`、`runs/2026-05-08_public-opt-uab-robust-prior-r1/` 与 `runs/2026-05-08_public-opt-uab-heat-specialist-r1/`：公开数据准备、稳健 prior 和 heat specialist 优化证据。
- `runs/2026-05-06_semantic-support-baseline/` 与 `runs/2026-05-06_anchor-windows/`：semantic support baseline 与 anchor windows。
- `runs/2026-05-06_public-fusion-screen-round2/`：公开融合 screen round2。
- `runs/2026-05-06_public-fusion-nasa-confirm/`：NASA 公开融合确认。
- `runs/2026-05-06_public-fusion-uab-confirm/`：UAB 公开融合确认。
- `runs/2026-05-06_public-opt-nasa-round1/`：NASA 公开优化结果。
- `runs/2026-05-06_public-opt-nasa-prepared/`、`runs/2026-05-06_public-opt-uab/` 与 `runs/2026-05-06_public-opt-uab-torch/`：公开数据准备和优化输入。
- `runs/2026-05-04_public-opt-uab-prepared/` 与 `runs/2026-05-04_dingxin-opt-package/`：UAB 公开准备根和鼎新优化包基线。
- `runs/2026-05-01_deep-real-sortie-prepared/` 与 `runs/2026-05-01_deep-comparison-prepared/`：deep baseline 真实架次和公开序列准备根。
- `runs/2026-05-01_full-loso-deep-comparison/`：公开 deep baseline 全 LOSO 对比。
- `runs/2026-04-29_case-study/`：case-study summary 与 ablation table。
- `runs/2026-05-02_feature-export-e-allwindow-clean/` 与 `runs/2026-05-02_feature-export-f-allwindow-clean/`：特征导出 clean roots。
- `runs/2026-04-27_feature-export-closure/`：带 causal fusion summary 的特征导出 closure，用于真实架次序列准备和 case-study 默认入口。
- `runs/2026-04-22_alignment-e-baseline/`、`runs/2026-04-22_alignment-f-full/`、`runs/2026-04-22_alignment-g-baseline/` 与 `runs/2026-04-22_alignment-g-min/`：alignment/fusion 早期诊断输入，供 support summary 追溯。

## 引用规则

- 当前状态先看 `docs/STATE.md`。
- 当前执行队列先看 `docs/implementation/TASKS.md`。
- 论文需求先看 `docs/requirements/SPEC.md`。
- 论文图表和实验表优先从 `runs/2026-07-03_thesis-protocol-snapshot/experiment_registry.csv` 与 `result_matrix_long.csv` 反查原始路径和边界。
- 公开 UAB/NASA 结果必须写成公开数据适配、校准或上下文构造第二输入流证据。
- 鼎新结果必须写成鼎新真实数据弱监督任务证据，不写成人工专家真值。
- LLM preprocessing 只写成字段语义归一、规则复核、semantic hints、runtime explanation 和 pending human review packet。
- 融合表示流结构评价历史结果单独成表，`evidence_quadrant = fusion_stream_structure`，作为无监督结构诊断，不与已确认的分类、回归和检索指标混算，也不替代主分类与回归任务；历史检索产物保留，但不再承担主叙事。

## 清理记录

- `cleanup/20260705-name-migration-cleanup.md`：本轮职责命名迁移和产物整理记录。
- `../maintenance/2026-07-05_name-migration-map.md`：old path 到 new path 的迁移表。
- 更早 cleanup 记录保留在 `cleanup/`，用于追溯历史删除、外置备份和 LFS 决策。
