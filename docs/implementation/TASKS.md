# Chronaris 当前任务

更新时间：2026-09-04

## 当前任务：论文主线连续对齐与可学习语义融合

唯一活动开发分支为 `research/thesis-continuous-semantic-fusion-202609`。当前执行合同为[论文级冻结实验执行清单 v3.2.3](../requirements/thesis-frozen-paper-evaluation-v3.2.3.md)，此前版本保留追溯；现场保护与基线预检见[论文主线现场保护与基线预检](../artifacts/runs/2026-09-01_thesis-mainline-preflight/report.md)。

执行顺序：

1. **已完成：**保护当前工作树，创建历史证据标签和唯一论文开发分支；完整测试 `429 passed, 8 skipped`，CUDA、数据与磁盘预检通过。
2. **已完成：**确定性训练、真实 validation checkpoint、单次归一化、训练折缩放、正确上下文时长、原生时间公开数据输入、lazy batch provider 和受控缓存已完成。证据见 [可信训练修复](../artifacts/runs/2026-09-01_training-trust-repair/report.md)与[原生时间输入修复](../artifacts/runs/2026-09-01_native-time-euler-repair/report.md)。
3. **已完成：**批量 mask Euler 子步、可学习事件语义查询、五分类显式时移和跨 group 事件—响应配对目标均已实现；仿真训练内验证冻结选择单步 Euler，0.5 秒子步对齐损失略高，四阶 Runge-Kutta 超过 2 倍运行时间门。证据见[可学习事件语义与训练目标实现](../artifacts/runs/2026-09-01_learnable-semantic-objectives/report.md)与[连续演化数值方案选择](../artifacts/runs/2026-09-01_ode-solver-validation/report.md)。
4. **已完成：**人工确认去重为四个唯一候选；seed 17 干净基线矩阵完成 29 个有效单元，鼎新、仿真波次 A、CogPilot 和 CLARE 五折均保持外层结果关闭。CogPilot safe-lag 训练内宏平均 F1 为 `0.3708`，高于两条单流；CLARE safe-lag 五折均值 `0.4680`，高于两条单流但低于旧融合 `0.5251`，且最差折 `0.0769`，只说明链路可运行和方差较高。证据见[干净基线矩阵](../artifacts/runs/2026-09-01_seed17-clean-baseline/report.md)。
5. **已完成：**v3.2 在提交 `41683e98` 上完成 144/144 个训练内单元，外层结果始终关闭；双目标候选的均衡五分类时移准确率为 `0.2000/0.2909/0.3091`，后两个随机种子高于 20% 和无显式时移候选，正确—错误事件配对相似度差为 `0.2016/0.1447/0.1950`。运动学目标在仿真、鼎新和 CogPilot 有有限有效计数，CLARE 明确不可用。双目标候选以两个新机制门独占第一并冻结，证据见[训练内候选冻结报告](../artifacts/runs/2026-09-02_candidate-screen-v3p2/report.md)。
6. **当前：**已按用户确认完成 v3.2.3 无观测表示与严格安全门修复，工程验收 6/6，CPU 全量测试 481 项通过、15 项跳过，CUDA 实际窗口重放通过；严格安全门 9/12 达标、3 个失败。已有 15 个主实验检查点、12 个消融检查点和完整干净/消融消费者不变，150 份成功压力表示已逐文件核对后复用。冻结修订后只恢复剩余压力、时间机制及严格总审计，所有不利结果保留，公开和鼎新外层继续关闭。详见[修复报告](../artifacts/runs/2026-09-04_no-observation-safety-repair/report.md)；不根据不利结果调参。

2026 年 7 月公开与研究分支结果保留用于追溯，但受协议缺陷影响的结果均标记 `superseded_due_to_protocol_defects`，不进入毕业论文主表。下方安全滞后感知融合任务作为历史研究记录保留。

## 当前任务：安全滞后感知融合新主线

两条鼎新真实数据研究线（简化下游评价、任务感知安全残差/残差激活）均已收口退出。当前唯一活动分支为 `research/safe-lag-aware-fusion-20260718`（安全滞后感知融合）。阶段 1 审计与阶段 2 计划/协议已完成：

- 审计：[当前 Chronaris + CogPilot/CLARE 审计](../artifacts/runs/2026-07-22_safe-lag-aware-fusion-audit/report.md)（逐项代码证据确认九项缺陷，物理约束健康；CogPilot 为公开主双流、CLARE 为辅助跨受试者）。
- 计划：[安全滞后感知融合研究计划](notes/safe-lag-aware-fusion-research-plan-2026-07-18.md)。
- 协议：[安全滞后感知融合评价协议 v2](../requirements/safe-lag-aware-fusion-evaluation-v2.md)。

### 下一步顺序（阶段 3–5）

1. **已完成：**新主干 `SafeLagAwareFusion`（生理单流旁路、航电单流旁路和安全门控跨模态残差）、`lag_aware_alignment_loss`（因果滞后容限对齐，替代同刻余弦）、`representation_diagnostics`（有效秩、协方差谱和维度利用率）；通过 `fusion_kind` 接入两条训练路径，协议哈希与 checkpoint 兼容校验区分融合类型；12+3 项聚焦测试通过，完整测试 `426 passed, 8 skipped`。
2. **部分完成（波次 A）：**G1 仿真 15 epoch 工程冒烟已验证机制——safe_lag 航电恢复 R² `0.944` > 旧 multiscale `0.914`、有效秩 `5.13` > `2.73`、安全门控 `0.022` 近回退。尚需：完整预算（50 epoch、多种子）鼎新 inner-validation 机动负迁移验证、波次 B（跨流增量）、波次 C（CogPilot/CLARE 预训练迁移）、波次 D（任务感知头）。
3. **待运行：**晋级门禁全满足后运行一次鼎新锁定分组确认（评价协议 v2 §5–6）。门禁未满足前不合入 main。

不根据任一已收口线已打开的外层结果继续调参；门控残差与安全回退作为新主线设计基础保留复用。

## 历史已收口线 1：简化鼎新下游评价（最新真实数据协议冻结确认）

在不依赖新增鼎新数据或人工标签的前提下，复用成熟代码基础，把默认研究入口缩减为导师要求的融合表示下游比较：六种方法输出任务无关统一表示，再由相同 Ridge/Logistic 消费者完成未来机动与未来生理状态预测。原分支 `codex/simple-downstream-rebuild-20260715`，evidence tag `evidence/simple-downstream-confirmation-20260716`。

### 收口顺序

1. **已完成：**固化协议、分支和文档入口，停止旧门禁自动续跑。
2. **已完成：**审计现有 checkpoint、270 份表示与新字段合同的兼容性；结论为必须按新字段合同重训。
3. **已完成：**实现并实跑 90 个完整未来目标、60 个唯一机动上下文、逐字段生理目标与持久性基线。
4. **已完成：**实现统一 Ridge/Logistic、机动去重聚合和字段级指标。
5. **已完成：**一个留一架次折、seed 17、六方法工程冒烟；预训练、完整表示和下游算法三段均通过，不以模型分数作为门禁。
6. **已完成：**使用冻结的 50 epoch、batch size 32、patience 8 配置完成两个留一架次折、三随机种子的一次正式确认；30/30 预训练单元、72/72 表示导出和 36/36 下游评价单元均完成。
7. **已完成：**复用既有仿真、时间机制和消融证据，保留有利与不利结果，形成论文报告、四张图、结果表和独立复核记录。
8. **已收口：**正式结果已经打开，不再根据两个架次调参，也不新增不能改变跨架次主结论的留一视图训练线。

### 正式结果判断（不利结果保留，不删除、不覆盖）

- 未来机动由航电单流领先：宏平均 F1 `0.8084`、平衡准确率 `0.8111`、Spearman `0.6165`、相对当前状态技能 `0.6685`。
- Chronaris 的未来机动宏平均 F1 为 `0.1948`，相对当前状态技能为 `-123.5859`，没有形成真实任务优势。
- 未来生理字段中六种方法的正技能字段比例均为 `0`；最佳标准化 RMSE 宏平均为航电单流 `7.0838`，Chronaris 为 `8.8901`，所有方法均未超过持久性基线。
- 既有仿真中 Chronaris 的时钟偏移和生理响应时延恢复误差最低，但随机缺失、连续缺失和部分组件消融结果不利。论文统一采用“时间机制局部有效，真实下游整体优势尚未成立”。

## 历史已收口线 2：任务感知安全残差与残差激活（晋级门禁未通过）

在不依赖新增鼎新数据或人工标签、不读取外层测试的前提下，完成教师辅助残差激活与任务解耦，验证 Chronaris 连续因果支路能否在安全基座之上形成跨验证支持稳定增量。原分支 `codex/dingxin-residual-activation-20260715`（含祖先 `codex/dingxin-task-aware-safe-residual-20260715`），evidence tag `evidence/dingxin-safe-residual-gap-20260715`。

### 残差激活结果与退出决定

- 8 个候选 × 6 个唯一训练内验证支持共 48 个运行全部完成；教师预测来自交叉拟合模型，正式推理不需要教师集成。
- 排名最高的样本条件门控教师辅助候选取得机动平均宏平均 F1 `0.9460`（相对安全基座 `0.9037`），连续响应中位 RMSE 比率降至 `0.8626`，高响应平均归一化平均精确率 `0.4042`。
- 但连续响应平均技能仍为 `-0.0171`，最佳候选跨支持中位残差贡献比仅 `0.0005`，全部 8 候选低于 `0.02` 激活下限，不能证明连续因果支路形成稳定增量。
- `activation_gate_passed=false`、`safety_gate_passed=true`、`research_gate_passed=false`、`allow_stage_3b=false`、`configuration_locked=false`、`outer_test_opened=false`。
- 证据入口：[教师辅助残差激活与任务解耦](../artifacts/runs/2026-07-15_dingxin-residual-activation-task-decoupled/gap_report.md)。

### 任务感知安全残差阶段 2 结果（冻结阶段无损）

- 六方法同预算任务基座：机动分类由航电单流领先，平均宏平均 F1 `0.9037`；连续响应由直接观测基座取得最低中位 RMSE 比率 `1.0071`；高响应平均归一化平均精确率 `0.3652`。
- 冻结安全残差（航电机动基座 + 直接观测响应基座）：机动平均/中位/最差宏平均 F1 `0.9164/0.9630/0.7500`，高响应平均/中位归一化平均精确率 `0.3773/0.4202`；三项无损支持数均为 `6/6`。
- 部分解冻比例 `0.02` 只改善高响应，`0.05` 只轻微改善连续响应，均只改善 `1/3` 任务；研究门禁失败，训练外教师蒸馏与配置锁定未启动。
- `frozen_safety_passed=true`、`partial_safety_passed=true`、`research_gate_passed=false`、`allow_teacher_distillation=false`。
- 证据入口：[鼎新任务感知安全残差阶段 2](../artifacts/runs/2026-07-15_dingxin-task-aware-safe-residual/gap_report.md)。

### 两条已收口线的统一禁止事项

- 不把代码仓回退到 `0a38171`；该提交只作历史复现参考。
- 不默认复用旧 checkpoint 或表示；产物必须先通过新协议兼容性审计。
- 不重跑端到端联合训练、教师蒸馏、仿真预训练适配或旧目标门禁。
- 不使用留一视图结果声称新架次泛化。
- 不根据正式留一架次或残差激活结果修改目标、字段、模型、消费者或划分。
- 不覆盖历史确认指标和论文协议快照。

## 历史里程碑：目标重构与统一条件比较已收口

本轮复用 6 个唯一主选模验证单元和第三训练池压力单元，六种方法统一删除 30 个时间类航电通道，并使用最多 50 epoch、patience 8 的相同无标签预算、64 维表示合同与固定容量任务头。42 个方法—验证单元训练和 84 份训练内表示导出全部完成；短预算筛选与完整预算确认独立留档，正式退出结论只读取完整预算确认。时间捷径抑制和四项新任务门禁均失败，按预注册规则停止，不启动安全融合、任务感知主干训练或教师蒸馏。

### 本轮结果与退出决定

- 未来机动连续分数由 ContiFormer 领先，6 个主选模单元的斯皮尔曼秩相关系数中位数为 `0.5979`，低于 `0.65` 门槛；Chronaris 为 `-0.6993`。
- 未来机动增强、稳定、减弱趋势由航电单流领先，训练池平衡平均/最差宏平均 F1 为 `0.6598/0.3535`，低于 `0.75/0.60` 门槛；Chronaris 平均值为 `0.2818`。
- 生理剩余响应由航电单流领先，中位 RMSE 比率为 `1.1614`、平均技能为 `-0.2529`，`0/6` 个单元技能为正；这表明增加训练预算后，重构幅值目标仍未形成稳定跨支持预测技能。
- 高剩余响应由航电单流领先，平均/中位归一化平均精确率为 `0.0833/0.0000`，只有 `1/6` 个单元为正。训练内四分位阈值在 `5/6` 个主验证单元产生零正样本，进一步确认风险标尺没有稳定迁移。
- 时间捷径抑制门禁未通过，较弱机动任务相对飞行进程诊断的标准化增益为 `-0.0874`；第三训练池完成两项连续目标压力评价，但不能替代其余门禁。
- 运行只以真实外层样本标识绑定隔离守卫，外层观测、标签、预测和指标均未打开；Chronaris 主干未修改，也未使用任务标签训练编码器。
- `allow_safe_fusion=false`、`allow_task_aware_research=false`。当前没有自动后继编码任务；若继续该研究线，必须建立新的预注册目标或新增可信监督信息，不能在本轮结果上原地调参。
- 紧凑证据入口：[鼎新目标重构与统一条件确认](../artifacts/runs/2026-07-15_dingxin-target-reconstruction-confirmation/gap_report.md)。

### 前一轮固定数据长程记录

- 选定配置 seed 17 开发确认完成前两个留一视图折；第一折 Chronaris 在 epoch 28 早停、最佳 epoch 20。该冗余开发队列已停止，后续由正式三 seed 五折协议承接。
- 第 1 折六方法 train/validation/outer-test 表示完成 18/18 导出；这只冻结输入，不提前运行 outer-test consumer。
- Chronaris 锁定训练中的连续对齐、物理一致性和因果方向损失已真实参与反向传播，公共自监督损失仍是唯一早停依据。
- 100 ms 公共因果时间箱五方法单 epoch 实跑为 8/8；四个基线 GPU 耗时 8.59–16.46 秒，Chronaris CPU 为 91.86 秒、GPU 为 40.07 秒。鼎新正式三随机种子 run 使用新 ID、全方法单 GPU 串行和逐 epoch 恢复。
- 正式 consumer 使用 validation 固定网格选择 Logistic/Ridge 与 MiniRocket 参数；机动分段使用两层残差因果 TCN、kernel 5、dilation 1/2、patience 6，并可在 GPU 上训练。
- G2 压力扩展包含 7 个单因素的 34 个等级版本和 1 个 mixed-severe 版本；48 条轨迹共 1,680 个观测场景，同轨迹复用相同 observation seed，7/7 验收通过。
- G1→G2 clean 表示与正式 consumer 均完成：18/18 单元、1152 条指标、768 条双流增益、90 条 48 轨迹配对统计和 7/7 门禁。压力表示完成 630/630 与 5/5 门禁，冻结 consumer 复用和退化斜率正在运行。
- 鼎新正式表示与 consumer 编排已实现：75 个 checkpoint 完整后才导出 270 份六方法三角色表示，validation 只负责选参，outer-test 只负责一次锁定评价。
- Chronaris 四项机制消融已接入锁定训练、checkpoint 回载、表示导出和相同下游 consumer；完整模型与消融按 48 条 G2 潜在轨迹配对。
- synthetic-to-real 轨道已完成 75/75 训练、270/270 表示和 90/90 consumer。Chronaris 相对 real-only 的三项主指标方向归一变化为 -0.0048、-0.0336、+0.0062；只有生理响应 RMSE 小幅改善，高生理响应识别的原有分项优势反而下降，因此这条轨道作为跨域适配诊断保留，不作为主结果替代方案。
- 鼎新统一表示和 consumer 已显式继承并校验上游表示族，real-only 与 synthetic-to-real 使用不同 family 字段和独立 run，不会因复用脚本而混入同一结果表。
- 时间偏移/响应时延恢复已实现四方法专用表示与下游探针：G1 train 拟合、G1 validation 选择 Ridge 强度，G2 35 场景只评价；主统计单位固定为 48 条潜在轨迹。
- 时间机制正式链路已完成：144/144 表示、24 个 G1 Ridge 探针、840 个 G2 场景评价、3360 条指标、630 条配对统计及 5/5、6/6 两级门禁。Chronaris 在时钟偏移/响应时延 MAE 与容差命中率上领先，MulT 在响应时延 Spearman 上领先。
- 仿真端到端微调辅助链路已实现并通过定向测试：五个可训练方法更新完整编码器，朴素同步只更新相同容量任务头；三任务联合损失只由 train 拟合、validation 早停，G2 held-out 只评价，输出独立 `end_to_end_finetuned_v1` 表。
- 论文证据包正式 run 已完成 8 层证据、7 图、54 项迁移配对和 9/9 门禁；七图全部抽查，读者可见文本不再暴露内部折 ID、轨迹 ID 或未解释英文。
- 公共自监督增强已移到 CPU 确定性执行，模型前向和 target tensor 才进入 GPU；该路径完成单 epoch CUDA 实跑，可避开此前 `_feature_age` 的 WSL 小算子 launch failure，同时保持训练配置和增强 realization 不变。
- clean、压力和机制恢复表示导出已支持 baseline CUDA / Chronaris CPU 混合设备；seed 17 完整六方法三角色冒烟导出 18/18、7/7 通过，正式三 seed clean 表示队列正在复用该配置。

### G1–G4.1 已完成

- 六方法生产表示、Chronaris 连续主干、五方法公共预训练和朴素时间同步训练折无监督变换均已可恢复。
- 16 条仿真 train 轨迹的 8/4/4 smoke 完成 5 个训练 checkpoint、18 个折外表示和 72 条线性指标。
- workload 真值只在五个 checkpoint 完成后打开；预训练路径不读取 oracle，仿真 validation/locked_test 路径零命中。
- 删除一个 Chronaris 留出折表示后仅重建该项，重建 SHA-256 与原输出一致；闭环 20/20 验收通过。
- G3b.4 完成时完整测试为 `301 passed, 8 skipped`；G4.1 完成后为 `312 passed, 8 skipped`；G4.2 目标归档为 `314 passed, 8 skipped`；原始上下文绑定后为 `316 passed, 8 skipped`；训练内划分后刷新为 `318 passed, 8 skipped`。
- 16 条仿真训练轨迹各取四个跨状态上下文，形成 64 个样本和 32/16/16 profile 隔离；六方法应用上下文表示共 18 份，恢复 18/18 复用。
- 线性、MiniROCKET 10,000 kernels、两层因果 TCN 与训练折持续时间解码共产生 384 条可计算指标、256 条融合增益和 30 条轨迹级配对统计。
- MiniROCKET 训练折方差过滤、TCN 随机流隔离和组件级恢复均已固化；删除 Chronaris 表示、MiniROCKET、TCN 后只重建目标组件，12/12 验收通过。
- G4.1 紧凑证据位于 `2026-07-11_application-consumer-smoke`，约 15 MB 模型、表示和预测留在被忽略目录；所有指标均为 smoke only。
- G4.2 已生成五折两个任务共 10 个独立目标 archive：分类保留 96 个上下文，原始点中位数生理响应保留 90/93 个完整未来区间，3 个末端候选结构化不可用。
- 10/10 archive 恢复复用，原 snapshot 哈希不变，目标归档 `12/12` 验收通过；本阶段没有模型训练或任务指标。
- 96 个目标上下文中 93 个具有完整 30 秒输入，三个 25.991 秒部分末窗不可用；机动分类/生理响应最终绑定 93/90 个唯一上下文。
- 12 生理 + 955 航电字段采用 78.6 MB 允许字段 CSR 缓存，20 个机动标签源在映射层删除；完整绑定审计 12.34 秒、峰值 767 MB，12/12 通过。
- 五个外层折已形成互斥 inner-train/validation/overlap embargo/outer-test；inner-train/validation 规模为 31/31、31/31、38/14、19/7、38/14，共享航电原始时间区间零重叠，11/11 通过。
- 当前 outer-train 阈值只用于固定配置 smoke；正式候选筛选必须以 inner-train 重新拟合阈值和目标，不能据此提前比较方法。
- 主协议首折已完成五个可训练方法各 31 step、六个 checkpoint 和 18 份三角色表示；18/18 恢复、同角色六方法对齐和 2.5 GB 内存门均通过。
- 完整首折累计训练 213.63 秒、峰值 1967.4 MB，Chronaris 训练 148.73 秒；约 79 MB 重型产物仅位于被忽略目录。本 run 未打开任务目标或计算 outer-test 指标。
- 流式训练与首折表示接入后完整测试为 `323 passed, 8 skipped`。
- 五折总计 30 个 checkpoint、90 份表示，五个子 run 60/60、聚合审计 13/13；所有 archive、checkpoint hash、fit hash、角色样本和公共 schema 已重验。
- 五折五方法累计训练 1066.45 秒，最高峰值 2047.1 MB；约 393 MB 重型产物仅位于被忽略目录，任务目标与 outer-test 指标保持关闭。
- 五折聚合审计接入后完整测试为 `325 passed, 8 skipped`。
- 五折冻结表示 consumer 形成 30 个方法—折组合、60 个组件、1680 条全可计算 smoke 指标和 1120 条双流增益；首次拟合 144.34 秒，恢复 60/60，15/15 通过。
- 五折实际使用 440 个机动分类角色上下文和 425 个生理响应角色上下文；36 MB 重型模型/预测被忽略，真实与仿真指标保持分层。
- consumer 工程冒烟接入后完整测试为 `327 passed, 8 skipped`。
- 五折 inner-train 嵌套目标形成 10 个确定性 archive，机动分类/生理响应可用角色数为 440/425；75 个机动类别与 51 个高响应标签相对工程冒烟口径改变，10/10 通过。
- 三个时间块 validation 不含低机动类；保留真实分布并固定三类 macro-F1 合同，禁止通过调阈值补类。
- 嵌套目标与固定类别指标合同接入后完整测试为 `328 passed, 8 skipped`。
- 嵌套 validation-only consumer 形成 30 个方法—折 bundle、840 条全可计算指标和 560 条增益；恢复 60/60，outer-test 零指标，12/12 通过。

### 当前输入与不可变边界

- 表示合同固定为 `[sample,96,64]`，G4 consumer 不得回读原始双流为某个方法追加特征。
- 鼎新 target 来源固定为 G1 审计中的训练折阈值、字段角色和 30 秒上下文；标签源航电字段仍禁止进入机动分类输入。
- 仿真 target 只允许读取 `ground_truth.npz` 中预先列入任务合同的 `true_time_s`、`workload`、`maneuver_state`、`maneuver_type` 和事件边界；物理残差、生成参数和真实时延不进入应用任务 consumer。
- consumer 只拟合 representation train role；validation 用于固定网格选择，held-out 每个锁定配置只评价一次。
- G4.1 复用 `2026-07-11_common-pretraining-loop-smoke` 的六个 checkpoint，但为 64 个跨状态上下文重新导出 18 份表示；没有读取仿真 validation 或 locked_test，也不产生模型排名。
- G4.2 必须使用鼎新 snapshot 原始点重新构造外层折输入，不能把仿真输入维数的 checkpoint 直接套到鼎新，也不能回用可能含标签源信息的历史对齐投影。

### 子任务 G4.2-a：鼎新独立 target archive（已完成）

#### 鼎新机动强度弱监督分类

1. 对每个 leave-one-view-out 外层折，读取 G1 `fold_label_thresholds.csv` 的训练折阈值和 `fold_task_labels.csv` 的上下文标签；不在 G4 重新计算全局分位数。
2. 目标 archive 记录 context、view、sortie、fold、label、阈值 hash、标签源字段 hash 和 `weak_supervision=true`。
3. 类别不足的训练折写 unavailable；禁止移动阈值、合并测试类或用 held-out 分布补齐。
4. 表示样本必须与 target context 一一匹配；漏样本、重复样本和跨 view checkpoint 直接失败。

#### 鼎新机动诱发生理响应回归

1. 复用训练折拟合的生理基线和字段有效性，目标为未来 5 秒相对当前上下文末端的多指标变化幅度。
2. 同时生成连续响应值和训练折高响应阈值，但 G4 主回归 consumer 只使用连续值；高响应识别留作分类辅助表。
3. 输入窗口截至 context end，目标区间为其后 0–5 秒；未来生理点绝不进入表示或归一化器。
4. 目标无有效生理字段时按样本 unavailable，不以零变化替代。

执行结果补充：正式目标使用冻结原始点的当前/未来 5 秒窗口中位数，不再使用 G1 的窗口均值兼容统计；原始点只到 181 秒，三个 `context_end_0035` 不足完整未来区间，因此正式可用上下文为 90 个。

#### 仿真负荷与机动状态（G4.1 已完成）

1. 负荷提前评估：使用输入结束后 0–5 秒 workload 均值，训练折三分位形成三类，并保留连续回归值。
2. 机动状态分段：把 oracle `maneuver_state` 重采样到 96 点查询轴，固定状态为稳态、进入、持续、退出/恢复；同一重采样规则供六方法共享。
3. 边界真值由状态变化点独立生成，记录原始秒数和查询点索引；consumer 不读取 event duration 或 test transition 统计。
4. target archive 与表示目录分离，写入允许字段列表；任一未声明 oracle 字段访问由测试拦截。

### 子任务 G4.1-b：统一线性与 MiniRocket consumer（已完成）

1. 保留已完成的 Logistic/Ridge 线性探针，提取为正式 `FrozenRepresentationConsumer` 接口。
2. MiniRocket 输入统一转为 `[N,64,96]`，固定 `aeon==1.5.0`、10,000 kernels、seed 17；变换器只在 representation train role 拟合。
3. 分类变换后使用 Logistic/RidgeClassifier 固定网格，回归使用 Ridge；网格、标准化和最大迭代数按任务固定，不按方法改变。
4. 内层 validation 选择超参数后冻结；同一任务六方法共享被选中的 consumer 配置，不能让每种表示选择不同容量。
5. 记录 transform hash、fit sample hash、kernel 数、训练秒数、峰值内存和输出维数；恢复时逐项校验。

### 子任务 G4.1-c：TCN emission 与持续时间约束 Viterbi（已完成）

1. TCN 固定两层、64 channels、kernel size 3、dilation 1/2、dropout 0.1，使用严格因果左 padding；输出每个查询点的状态 emission logits。
2. 训练 loss 使用有效查询 mask 的 class-balanced cross entropy；类别权重只由训练 role 统计。
3. 转移概率、初始概率和每类持续时间上下限只在训练 role 估计；加 Laplace 1 平滑并记录计数。
4. 自研 duration-constrained Viterbi 不依赖 `hmmlearn`；validation/held-out 标签不能参与 transition、duration 或后处理阈值拟合。
5. 同时保留 raw TCN 与 TCN+Viterbi 两组结果，以区分表示/发射模型能力和时序先验增益。

### 子任务 G4.1-d：任务指标与方向合同（已完成）

| 任务 | 主指标 | 辅助指标 | 指标方向 |
| --- | --- | --- | --- |
| 鼎新机动分类 | macro-F1 | balanced accuracy、macro-AUPRC、最差折 | 越高越好 |
| 鼎新生理响应 | RMSE | MAE、Spearman、高响应 AUPRC | 误差越低；相关越高 |
| 仿真负荷分类/回归 | macro-AUPRC、RMSE | macro-F1、MAE、Spearman | 分项方向 |
| 仿真机动分段 | segmental F1 | frame macro-F1、boundary F1、edit、检测延迟 | F1/edit 越高；延迟越低 |

具体要求：

1. `metric_long.csv` 每行包含 dataset、task、method、consumer、seed、fold、role、metric、value、direction 和 status。
2. boundary F1 固定 ±1 秒、±2 秒两个容差；segmental F1 固定 IoU 0.10/0.25/0.50；不根据结果选择容差。
3. 校准输出 Brier score 与 ECE 10 bins；小样本空 bin 跳过但记录有效 bin 数。
4. regression/分类/分段 unavailable 分开记录，不能用另一个任务指标补位。

### 子任务 G4.1-e：双流增益、压力斜率和配对统计（接口已完成）

1. 对每个 fusion 方法计算 `Score_fusion - max(Score_physiology, Score_vehicle)`；误差型指标先转换方向后再计算。
2. 六方法必须使用相同 held-out trajectory/context 配对；缺任一样本时整组配对统计 unavailable。
3. 仿真以 trajectory 为独立单位，鼎新以 view/sortie 外层折为独立单位；禁止把 96 个查询点当作独立样本做显著性检验。
4. 实现 trajectory-level paired bootstrap 和 exact/permutation test；smoke 只验证接口，正式 locked confirmation 才报告区间和 p 值。
5. stress slope 只接收锁定 checkpoint 的成对场景结果；G4 consumer 开发期不运行 G2 stress。

### 子任务 G4.1-f：仿真应用 consumer smoke（已完成）

已生成 `docs/artifacts/runs/2026-07-11_application-consumer-smoke/`，沿用 16 条仿真 train 轨迹的六个 checkpoint，并为四个时间位置重新导出 18 份冻结表示：

1. 从 ground truth 生成 workload 分类/回归和 96 点机动状态 target archive，验证允许字段审计。
2. 六方法运行线性、MiniRocket 分类/回归、TCN 和 TCN+Viterbi；小样本指标全部标记 `smoke_only=true`。
3. 所有 consumer 使用同一 train/validation/held-out role，完成 fit hash、恢复和方法不变超参数审计。
4. 删除 Chronaris 留出表示、MiniROCKET transform 和 TCN checkpoint 后各只重建对应项；未删除组件哈希保持不变，预测哈希一致。
5. 生成完整 `metric_long`、fusion gain、边界/分段指标和配对接口输出；不进入候选选择。

### 本里程碑预期产物

- `task_target_manifest.json`、`target_availability.csv`
- `consumer_protocol.json`、`consumer_fit_manifest.json`
- `minirocket_status.csv`、`tcn_training_status.csv`、`viterbi_parameter_manifest.json`
- `metric_long.csv`、`fold_metrics.csv`、`segment_metrics.csv`
- `fusion_gain.csv`、`paired_statistics.csv`
- `acceptance_checks.csv`
- `report.md`、`claim_boundary.md`、`progress.json`、`run.log`、`resume_command.txt` 和 `evidence_manifest.json`

transform、consumer checkpoint、emission、逐样本预测与逐点状态序列已写入被忽略的 `artifacts/application_evaluation/2026-07-11_application-consumer-smoke/`；紧凑 run 为 12/12 验收通过。

### G4.1 验收（已通过）

- 三类 target archive 的来源、时间边界、训练折阈值和允许 oracle 字段全部通过 fail-closed 测试。
- MiniRocket 10,000 kernels、TCN 架构和 Viterbi 参数在六方法间完全一致，fit lineage 只含 train role。
- raw TCN 与 Viterbi 后处理都产生 frame、segment、boundary 和 edit 指标；test 标签不进入 transition/duration 拟合。
- classification、regression、segmentation、校准、fusion gain 和配对统计接口均输出方向明确的结构化结果。
- consumer checkpoint/transform 的完整恢复和单项删除重建通过，表示 checkpoint hash 保持不变。
- 完整测试、`compileall`、`git diff --check`、术语、密钥、LFS 和重型产物忽略检查通过。

### G4.2 当前实现顺序

1. 已完成：生成鼎新机动分类与生理响应两个独立目标 archive，逐样本记录 G1 阈值、原始点中位数、snapshot 与外层折 hash。
2. 已完成：从固定 snapshot 构造与目标 context 一一对应的 30 秒原始异步双流，审计标签源字段排除、部分末窗和未来区间隔离。
3. 已完成：在 outer-train group 内建立防重叠 inner-train/validation，固化 leave-one-view-out 主协议和 leave-one-sortie-out 辅助协议。
4. 已完成：五折五方法公共预训练 checkpoint、朴素时间同步无监督变换和六方法 train/validation/test 表示全部生成。
5. 已完成：固定线性/MiniROCKET consumer 的真实数据 smoke，真实与仿真 metric root 分离。
6. 已完成：以 inner-train 重拟合机动阈值、生理字段尺度与高响应阈值，生成嵌套目标。
7. 已完成：只在 validation 复跑固定 consumer，并修正固定类别指标合同；outer-test 继续关闭。
8. 当前：冻结 G4 协议，运行 seed 17 四候选 screen；只使用仿真开发 validation 和鼎新 validation，不读取任何 outer-test/locked-test。

### G5 已完成的筛选基础设施

- A–D 候选已成为代码中的冻结配置：`64/1e-3/0.1`、`64/3e-4/0.1`、`32/1e-3/0.1`、`64/1e-3/0.2`；共同使用 2 层、4 heads 和 64 维输出合同。
- 候选 C 只缩小内部隐层，单流、MulT、ContiFormer 和 Chronaris 的对外表示仍为 `[N,96,64]`；旧 64 维 checkpoint 可严格加载。
- G1 只读取 96 个 train 与 24 个 validation profile 的 `raw_dual_stream.npz`；其中 23 个 validation 用于排序、1 个保留作开发确认，封存测试不进入批次。
- 验证增强固定为 seed 17、epoch 0；早停只使用遮挡重构、短期预测和时延判别三项公共损失，Chronaris 专属诊断不进入分数。
- 候选内逐 epoch 保存 `last.pt`，恢复时加载编码器、公共 head、优化器、最佳分数和 patience，从下一 epoch 继续；候选完成后直接复用 `best.pt`。
- 四候选逐损失做方法内 min-max，常量损失项归一化为 0；总分并列时按更小参数量、候选字母序确定唯一配置。
- 20 候选单 epoch smoke 已完成，耗时 5 分 03 秒、峰值内存 4.34 GB、重型 checkpoint 约 114 MB，6/6 验收通过。该结果只验证长跑链路，不作为正式候选结论。
- 本轮正式训练、鼎新下游、机制恢复与消融编排收口后完整测试为 `360 passed, 8 skipped, 317 warnings`，`compileall`、Ruff 与 `git diff --check` 通过。
- 正式 50 epoch/patience 8 筛选完成；Chronaris、MulT、ContiFormer、生理单流选择 A，航电单流选择 C。预留的第 24 个 G1 validation profile 只用于五个选定 checkpoint 的一次开发确认，不参与重新排序。
- Chronaris 参考轴采样已由逐样本重放改为数学等价的向量化查询，输出、审计与梯度对照通过；CPU 单 epoch 从约 60 秒降至 17.07 秒。RTX 4090 同批次为 49.87 秒，因此该主干正式 screen 使用 CPU，CUDA 支持保留给更适合并行的后续训练。

## 已锁定规范

- [固定数据证据策略](../requirements/foundation/fixed-data-evidence-strategy.md)
- [下游应用评估任务协议](../requirements/downstream-evaluation-spec.md)
- [仿真基准规格](../requirements/synthetic-benchmark-spec.md)
- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)
- [详细实施计划](notes/fixed-data-downstream-evaluation-2026-07-10.md)
- [长程运行手册](notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)

## 后续验收门

### G5：screen

- 已完成：seed 17、每个可训练方法四候选。
- 已完成：只用 G1 公共自监督验证损失排序，并在预留 profile 上确认选定配置。
- 已保持：未读取 G2 locked test，未使用结构诊断指标选择候选。
- 下一步：只重训每方法唯一选定配置并进入鼎新五折 validation；禁止让 20 个候选反复读取鼎新 validation。

### G6：locked confirmation

- 历史执行记录：seeds 17、29、43 的五个唯一配置重训曾在独占 GPU 重复失败后迁移到 CPU，Chronaris 使用同批基准更快的 CPU；该队列后续已完成并由下方各项收口记录覆盖。
- 已完成：仿真 seeds 17/29/43 五方法均完成 50 epoch，15 个训练单元、54 份六方法三角色表示、9/9 与 7/7 门禁全部通过。首次 consumer 运行确认默认 `lbfgs` 高维网格约需 29 分钟/方法；同一冻结变换上显式 OvR `liblinear` 三值网格合计 6.97 秒，已在不读取 G2 比较结果的前提下统一锁定并准备从空根重跑。
- 已完成：G2 压力 35 场景的三随机种子六方法表示共 630 份、5/5 门禁；任务真值与指标在表示阶段保持关闭。
- 已完成：G2 clean 正式指标。Chronaris 在线性低容量负荷探针与机动状态/片段质量上领先；MulT 在 MiniRocket 负荷任务和边界定位上领先。Chronaris 对最佳单流的分段 frame/segmental 平均增益为 0.1194/0.1109，但边界 F1@1s 平均增益为 -0.0053，不把边界任务写成优势。
- 已完成：端到端辅助表 18/18、864 条指标、7/7 门禁。MulT 在微调分类/分段领先，Chronaris 仅回归 RMSE 0.2666 略领先；Chronaris 三项对应指标均弱于冻结主表，因此把该表写成小样本适配与过拟合诊断，不替代冻结表示结论。
- 已完成：压力冻结 consumer 的持续时间约束批量解码。真实 `[192,96,5]` logits 与逐样本结果逐位一致，运行时约提升 43.95 倍；旧标量运行未完成场景单元并已隔离，正式压力 run 从空根恢复。
- 已完成：压力冻结 consumer 105/105 种子—场景、630 方法评估、20,160 条指标、4,032 条退化斜率和 630 条 48 轨迹配对统计，6/6 门禁通过。Chronaris 在最高单因素压力下 15/21 个主指标组合位于前二，但随机/连续缺失平均退化斜率均为第六，不形成全因素鲁棒性优势。
- 验证刷新：完整测试 `370 passed, 8 skipped, 319 warnings`，覆盖新加入的因果时间合并、显式 OvR 高维求解器和批量持续时间约束逐位等价路径。
- 已完成：鼎新 75 个选定配置锁定重训 `2026-07-12_dingxin-locked-pretraining-coalesced`；三个留一视图主折和两个留一架次辅助折共 75/75 单元、8/8 门禁通过。CPU 构造增强、五个可训练方法全部使用唯一 GPU；任务目标与 outer-test 访问数均为零，任何缺少 `model_input_contract.json` 的旧 checkpoint 均拒绝恢复。
- 已完成：从 75 个已验证 checkpoint 导出 3 seeds × 5 folds × 6 methods × train/validation/outer-test 共 270/270 份 `frozen_task_agnostic_v1` 表示，6/6 门禁通过；表示阶段不计算 outer-test 任务指标。
- 已完成：鼎新冻结 consumer 90/90 单元、5,040 条正式指标、3,360 条双流增益和 504 条主折汇总，8/8 门禁通过。Chronaris 在高生理响应识别上以 AUPRC 0.8839 居首，但机动强度分类和连续响应回归不领先。
- 待上游完成后自动执行：G1→G2 clean 三随机种子六方法表示、validation 选参、锁定 held-out 指标和 48 轨迹配对统计。
- 已实现待队列门禁打开：鼎新 270 份统一表示、主/辅助 split 正式 consumer，以及 Chronaris 四项固定消融的训练—表示—consumer 链路。
- 已完成：Chronaris 四项固定消融 × seeds 17/29/43 共 12/12 个 checkpoint、6/6 门禁；训练和保留确认全程不打开 G2 或任务真值。
- 已完成：12 个消融 checkpoint 导出 G1 train、G1 validation 和 G2 held-out 共 36/36 份统一表示，5/5 门禁通过；任务真值在表示阶段保持关闭。
- 已验证：无物理约束变体完成 G1 train/validation 与 G2 held-out 三角色表示导出，3/3 输出、5/5 验收通过；正式消融表示等待 12 个变体 checkpoint。
- 已完成：synthetic-to-real 三 seed 五折无标签适配 75/75、统一表示 270/270、consumer 90/90；54 项 real-only 主视图折配对键完整，迁移收益按指标方向统一解释。
- 已实现待上游门禁：端到端微调辅助表；三 seed 六方法、独立表示族和 48 轨迹统计均已编排，不得替代冻结表示主结果。

### G7：stress 与论文证据包

- 已完成：48 条 G2 轨迹 × 35 个严格成对观测版本，生成器不接收方法名，7/7 审计通过。
- 待锁定 checkpoint：跑全部单因素表示、冻结 consumer、退化斜率和 mixed-severe。
- 待锁定 checkpoint：导出 G1 六观测场景四方法表示，并在 G2 压力表示完成后运行绝对时钟偏移和主生理响应时延恢复。
- 结构诊断只放附录。
- 已实现待上游门禁：输出中文论文图表、证据矩阵、claim boundary 和新协议快照候选；正式运行后还需逐图抽查中文字体、长标签和数值可读性。

## 方法和任务固定项

### 方法

- 生理单流。
- 航电单流。
- 朴素时间同步。
- MulT。
- ContiFormer。
- Chronaris。

### 任务

- 机动强度弱监督分类。
- 机动诱发生理响应预测。
- 仿真负荷提前评估。
- 仿真机动状态分段。
- 时间偏移与响应时延恢复。

### 主下游算法

- Logistic/Ridge 线性探针。
- MiniRocket 10,000 kernels。
- 两层 64-channel TCN + 自研持续时间约束 Viterbi。

### 正式随机种子

- 开发：17。
- 锁定确认：17、29、43。

## 运行和产物规则

- Python：`/home/wangminan/env/anaconda3/envs/chronaris/bin/python`。
- RTX 4090 正式训练最多单 CUDA 进程；独占训练重复失败的长队列必须记录设备历史并迁移 CPU，不再无限同配置重试。短 TCN/微调可在张量自检通过后单独使用 GPU，重复失败则同样迁移。
- 紧凑可引用产物：`docs/artifacts/runs/YYYY-MM-DD_intent/`。
- raw snapshot、dense bundle、checkpoint 和逐样本预测：`artifacts/application_evaluation/`，禁止入仓。
- 每个正式 run 必须有 progress、resume、protocol、fold metrics、claim boundary 和 evidence manifest。
- 每通过一个验收门，同步 `STATE.md`、本文件和 `ARTIFACTS.md`。

## 当前禁止事项

- 不整体合并 `implement/fusion-stream-structure-20260707`。
- 不把旧 E3 run 结果写入论文主表。
- 不继续围绕 `chr_v2_residual_delta_h64` 做无边界调参。
- 不让仿真器读取方法名称或评价结果。
- 不把仿真、公开数据和鼎新真实指标混成单一平均分。
- 不在元数据不足时用所有航电字段构造机动标签。
- 不修改既有 confirmed metrics，直到新的 locked confirmation 和 review 完成。
- 不在本次上限门禁失败后自动启动任务感知主干开发、教师蒸馏或外层确认。

## 当前验证门

当前长程 goal 已完成训练内审计，但研究放行门禁未通过；工程验证结果如下：

- 上限门禁：三项均失败；安全融合门禁：未运行并结构化阻断；`allow_next_goal=false`。
- 本轮新增聚焦测试 `9 passed`；按仓库真实运行拓扑加载原始工作区的被忽略历史工件、同时使用本分支源码与测试执行完整 pytest，结果为 `381 passed, 8 skipped, 319 warnings`。
- 本轮改动文件 Ruff、`compileall src scripts tests` 和 `git diff --check` 已通过；重型因果查询缓存与候选状态保持在 Git 忽略目录。
- 三幅中文审计图已经逐图检查字体、指标方向、长标签和数值标注。

下列早期协议门禁继续保留作为追溯记录：

1. 已通过：两项鼎新 target archive 的时间边界、训练折阈值、样本覆盖、字段 lineage 和 unavailable 测试。
2. 已通过：机动标签源字段在原始映射中零命中，未来生理点在输入中零命中，三个部分末窗结构化不可用。
3. 已通过：leave-one-view-out 与 leave-one-sortie-out 的 inner-train/validation/outer-test group 或时间块无重叠；后续所有变换必须只用 inner-train。
4. 已通过：五折 target、原始点 context 与表示 sample ID 一一对应；漏样本、重复样本、跨折 checkpoint 直接失败。
5. 至少一个真实外层折完成六方法表示和固定 consumer smoke，所有鼎新结果继续标记弱监督并与仿真指标分层。
6. G1–G4.1 聚焦测试保持通过，并运行完整 `pytest`、`compileall`、`git diff --check`、术语、密钥、LFS 和重型产物忽略检查。
