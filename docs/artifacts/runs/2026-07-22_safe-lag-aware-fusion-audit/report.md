# 安全滞后感知融合：当前 Chronaris 与新数据审计

状态：completed（只读审计）。审计日期：2026-07-22。分支：`research/safe-lag-aware-fusion-20260718`。
本审计不训练模型、不修改既有确认指标，只为新主线提供诊断依据与数据清单。

## 一、审计目标与背景

阶段 0 已完成分支清理与远端同步（见 STATE.md）。两条鼎新真实数据研究线（简化下游评价、任务感知安全残差）均已收口退出。本审计按 `goal.md` 阶段 1 要求，逐项验证当前 Chronaris 的 10 个疑似缺陷，并对 PhysioNet、CLARE 两套新公开数据做只读盘点，为新主线 `research/safe-lag-aware-fusion-20260718` 的研究计划与评价协议提供事实依据。

## 二、鼎新真实数据失败证据（独立复核结论）

简化下游评价（evidence tag `evidence/simple-downstream-confirmation-20260716`）与论文证据复核的正式结果：

| 任务 | 航电单流 | ContiFormer（最强融合） | Chronaris | 含义 |
| --- | --- | --- | --- | --- |
| 未来机动三分类 macro-F1 | **0.808** | 0.384 | **0.195** | 航电单流领先；Chronaris 灾难性负迁移 |
| 未来机动连续分数 Spearman | **0.617** | — | 0.069 | 同上 |
| 相对当前机动状态技能 | **0.668** | — | **−123.586** | Chronaris 远不如“保持当前状态” |
| 未来生理字段标准化 RMSE 宏平均 | **7.084** | 7.227 | 8.890 | 航电单流仍领先 |
| 未来生理字段正技能比例 | 0/12 | 0/12 | 0/12 | 六方法均未超过持久性基线 |

机制层面（仿真）：Chronaris 时钟偏移恢复 MAE 0.888s、生理响应时延恢复 MAE 7.486s 均为四方法最低；但随机缺失/连续缺失方向归一化退化斜率为 −0.073/−0.126，排第六。

结论：当前 Chronaris 在真实下游任务上同时表现为（a）对航电强信号的灾难性破坏（机动任务），（b）对生理流的零增量（生理任务），（c）时间机制表达局部有效但缺失鲁棒性差。这正是新主线必须解决的三个核心问题。

## 三、当前 Chronaris 架构审计（10 项，逐项代码证据）

数据流（`file:line`）：原始异步双流 → `alignment_bridge.py:17` 合并同时刻观测 → 双流 ODE-RNN 主干（`prototype.py:320`，每流 `SingleStreamODERNNPrototype`）在 96 点查询轴上输出 `[B,96,64]` 潜态 → 多尺度因果滞后融合（`multiscale_causal.py:83`）→ `output_projection`（`Linear(256→64)`）→ `sequence.mean(dim=1)` 形成 `[B,64]` 窗口表示。主干约 123.8k 参数（两路 ODE-RNN 约 105k，融合仅约 18.5k，其中 17k 是单一 `output_projection`）。

1. **缺少绕过融合投影的单流残差出口 — CONFIRMED。** 唯一输出来自 `chronaris_continuous.py:261` 的 `fusion.sequence_embedding`；融合内 `merged = cat(phys, veh, attended, phys−attended)`（`multiscale_causal.py:139-147`）再经单一 `Linear(256→64)`。没有任何形如 `output = projection(merged) + phys_private` 或 `+ vehicle_private` 的旁路；残差项 `phys−attended` 是生理中心的，不是对称单流旁路。
2. **航电强信号在融合与 64 维压缩后丢失风险 — PARTIALLY/确认有风险。** 航电进入 `merged` 第二块与多尺度航电上下文，但被强制经 256→64 单一瓶颈，无受保护子空间；注意力几何不对称：`query=physiology, key/value=vehicle`（`multiscale_causal.py:88-90`），无航电反向查询生理的路径。
3. **对 96 时间点直接均值池化 — CONFIRMED。** 所有适配器 `pooled = sequence.mean(dim=1)`（`chronaris_continuous.py:303` 等），且被表示合同强制：`contracts.py:209-221` 断言 `pooled_embedding` 等于有效掩码均值，否则抛 `RepresentationContractError`。末端状态、趋势、局部事件与滞后结构全部被均值抹平。
4. **同刻潜态相似损失与生理滞后响应假设冲突 — CONFIRMED。** `continuous_alignment = 1 − cos(phys_ref_t, veh_ref_t)`（`chronaris_auxiliary.py:37-55, 131-143`），两流在同一 96 点参考网格上零滞后配对，主动推动两流同刻共线，与“生理在 t 对应航电在 t−τ”的滞后假设直接矛盾。
5. **辅助损失前 10 epoch 为零 — CONFIRMED。** `chronaris_auxiliary_weight_schedule`（`pretext.py:141-158`）：epoch≤10 时 fraction=0；11–19 线性升；≥20 满权重（continuous_alignment 0.2、physical 0.1、causal 0.1）。机制项前 10 epoch 完全不参与梯度。
6. **早停只依据公共重构损失 — CONFIRMED。** 早停分数只用 `PUBLIC_SELECTION_WEIGHTS`（masked_reconstruction 0.5 / short_horizon 0.25 / lag_discrimination 0.25，`candidate_screen.py:41-45`；`chronaris_locked_training.py:264-267`），辅助损失计入训练梯度（`:232`）但**不计入**选择分数；patience 8、max 50。checkpoint 元数据明确写 `selection_uses_public_pretext_only=True`。典型最优 checkpoint 落在辅助损失尚未升满处。
7. **尺度门控可塌缩到单一滞后窗口 — CONFIRMED。** `scale_gate = LayerNorm(320)+Linear(320,3)` 后对三尺度做无约束 softmax（`multiscale_causal.py:74-77, 133-135, 216-229`），仅按可用性掩码，无熵/温度/平衡正则；`scale_gate_weights` 返回但从不进入任何损失，可退化为 one-hot。
8. **表示有效秩/维度塌缩/种子不稳定/跨折漂移无监测 — CONFIRMED。** 全仓 `modeling/`、`representation/` 无 effective_rank/svd/eigvalsh/nuclear 诊断（`representation_diagnostics/` 目录为空，仅 `__pycache__`）；训练循环只记标量损失，不记表示几何。均值池化、同刻共线、无约束 softmax、无方差/正交惩罚共同构成塌缩压力。
9. **物理约束作用在正确的航电变量与潜态 — CONFIRMED（健康）。** `chronaris_physics.py:81-88` 以 `mode="feature_only"` 在 `alignment_output.vehicle.reconstructions` 与生理解码重构上计算刚体残差（`physics.py:371-417`、`physics_residuals.py:10-60`：d(speed)/dt−acc 等），**不**让共享 64 维潜态承担物理残差；缺语义字段时结构化 `unavailable`，不静默回退。此项无需修改，保留即可。
10. **公共预训练目标过于偏向原值重构 — CONFIRMED。** `CommonPretextWeights`（`pretext.py:14-18`）：masked_reconstruction 1.0 + short_horizon_prediction 0.5 = 1.5/1.7 为原始拼接值重构；两 head 均为 `Linear(64,F)` 预测 `cat(phys,veh).values`（`pretext_targets.py:108-131`）。唯一涉及流间关系的是 lag_discrimination（0.2，二值时移检测），无任何“跨流增量预测”目标。

审计结论：除物理约束（Q9）外，其余九项均确认存在缺陷，且与鼎新失败证据（航电被破坏、生理零增量、机制局部有效）一一对应。

## 四、新数据审计

### 4.1 PhysioNet 虚拟飞行任务（主公开真实双流外部验证）

来源：PhysioNet `virtual-reality-piloting` 1.0.0（`/home/wangminan/dataset/chronaris/physio_net`）。许可：`LICENSE.txt`（PhysioNet 受限数据，按受限数据处理，不公开可识别信息）。官方 SHA 清单 `SHA256SUMS.txt`（9021 行）随数据发布，作为权威完整性校验。

- 规模：约 40.5 GiB，8899 文件，8246 个 CSV；35 名参与者；419 次 ILS 进近任务运行 + 安静基线运行；4 个难度等级（PerfMetrics 各 104/106/104/105 次）。
- 生理流（人体）：`lslshimmerecg`（ECG）、`lslshimmereda`（EDA，约 10 GiB）、`lslshimmeremg`（EMG）、`lslrespitrace`（呼吸，约 1024 Hz）、`lslshimmerresp`、`lslshimmertorsoacc`（躯干加速度，约 128 Hz）、`lslhtcviveeye`（眼动/瞳孔，约 16.9 GiB）。
- 机器/任务流（飞机）：`lslxp11xpcac`（X-Plane 飞机加速度等）、`lslxp11xpcplt`（X-Plane 飞机位姿/轨迹）、`lslxp11`（X-Plane 状态）。这正是所需双流：人体生理 + 模拟飞机状态/操纵/性能。
- 时间戳：`time_dn` 为 MATLAB datenum（天，自公元 0 年）；各流独立采样、不同速率，通过 LSL 时间戳对齐到公共时钟。
- 标签：`PerfMetrics.csv`（419 行）含难度等级与累计总误差（`cumulative_total_error`，按难度上升：1≈3155、2≈2931、3≈3651、4≈4685），按参与者—试次分组。
- 可构造任务（语义依据）：(a) 飞行任务难度/负荷四分类（标签：PerfMetrics.difficulty，分组：参与者，预存）；(b) 着陆/飞行误差回归（标签：cumulative_total_error，预存）；(c) 飞机/操纵事件条件下生理响应残差（输入侧事件定义，衍生）；(d) event-to-response 滞后/匹配（机制任务）。
- 划分：以参与者或参与者—试次为分组，主评价 LOSO 或 group-held-out，同一参与者窗口不跨训练/测试。
- 不可发布：原始波形与可识别信息；只发布清单、匿名统计、哈希、指标、图。

### 4.2 CLARE（跨受试者多模态生理与认知负荷辅助验证）

来源：CLARE（Cognitive Load Assessment in REaltime，MATB-II 认知负荷电池；Bhatti et al., arXiv:2404.17098；Borealis DOI 10.5683/SP3/H0AELT）。路径 `/home/wangminan/dataset/chronaris/clare`，约 6.3 GiB，511 文件。本地无 LICENSE 文件，按最严格默认处理，发布前须核对 Borealis 许可。

- 参与者：本地 20 名（论文称 24，缺 4；EEG 仅 19 名，缺 1936）。4 次实验会话（exp_0..3，各约 9 分钟）+ 各会话前基线。
- 模态与速率：EEG（TP9/AF7/AF8/TP10，256 Hz，独立时钟）；ECG（3 导联 RAW+CAL，约 512 Hz）；EDA（GSR RAW/Resistance/Conductance，128 Hz）；Gaze（瞳孔/注视/扫视 32 列，约 50 Hz）。ECG/EDA/Gaze 共享同一时钟，EEG 独立——存在跨时钟对齐缺口（无硬件同步标记）。
- 双流语义划分：**中枢（EEG）vs 外周（ECG+EDA+Gaze）**，语义清晰，正好验证异步多模态融合与模态缺失鲁棒性。
- 标签：`Labels/<sub>.csv`，每会话 54 行 × 4 列，10 秒自评认知负荷 1–9（预存，非衍生）。subject 1674 仅 3 会话。
- 可构造任务：跨受试者认知负荷回归/分类（LOSO，分组=参与者）、双流增量（中枢 vs 外周）、模态缺失鲁棒性、负迁移分析、静息 vs 任务对比。
- 数据质量告警：Gaze 含 INT32 极值哨兵（Gaze X/Y ±2.147e9、Saccade Peak Velocity 溢出）、subject 1026 瞳孔恒为 5.0；ECG/EDA 在共享网格上稀疏；需强制滤波。
- 划分：subject-wise / LOSO；同一受试者窗口不跨角色；标签为 10 秒，窗口不得跨越会话边界或标签边界。

## 五、诊断与设计驱动（驱动新主线）

依据第二、三节失败与缺陷映射，新主线 `safe-lag-aware-fusion` 的设计驱动：

1. **安全融合与单流保真**：输出改为 `z_out = [z_phys_private, z_vehicle_private, gate·z_cross]`，门控初始化接近安全回退，复用任务感知安全残差线的门控残差代码（`evidence/dingxin-safe-residual-gap-20260715`）。直接修复 Q1/Q2 与鼎新机动负迁移。
2. **滞后感知条件对齐**：以“历史航电预测随后生理变化 / 正确滞后优于错误滞后 / event-conditioned 对比”替换同刻余弦相似，修复 Q4 与生理零增量。
3. **更合理时序汇聚**：冻结轨道采用多统计量汇聚（mean/std/slope/last 或注意力池化），任务感知轨道消费完整序列；修复 Q3。
4. **分阶段与多准则 checkpoint**：辅助损失提前启用（非前 10 epoch 全零），checkpoint 选择结合公共损失 + 时间机制验证 + 单流保真 + 训练内下游 + 负迁移门禁；修复 Q5/Q6。
5. **尺度门控抗塌缩**：加入熵/温度/平衡正则；修复 Q7。
6. **跨流增量预训练目标**：在原值重构之外加入“跨流预测/残差”目标；修复 Q10。
7. **表示几何诊断**：填充空的 `representation_diagnostics/`，监控有效秩、协方差谱、维度利用率，并作为晋级门禁；修复 Q8。
8. **物理约束保留**（Q9 健康）：继续只在航电特征空间作用，不强制共享潜态承担物理残差。

## 六、审计产物与边界

- 本审计为只读；未训练模型、未修改既有确认指标、未打开鼎新外层结果。
- 数据哈希：PhysioNet 采用官方 `SHA256SUMS.txt`；CLARE 顶层 `README.txt`/`MANIFEST.TXT` 哈希已记录（10571a73…、70c17697…），完整逐文件哈希按需补算。
- 原始数据路径不进入需公开传播的报告；只发布清单、匿名统计、哈希、指标与图。
- 下一阶段：基于本审计提交研究计划与评价协议（不含 `final` 命名），再开发新主干。
