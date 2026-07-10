# Chronaris 当前任务

更新时间：2026-07-11

## 当前长程 goal

在不依赖新增鼎新数据或人工标签的前提下，完成真实弱监督应用任务、模型无关半物理仿真、六方法统一融合表示、Chronaris 连续融合主干、下游评估、锁定实验、图表证据和状态文档闭环。

当前分支：`codex/fixed-data-downstream-evaluation-20260710`。

## 当前里程碑：G3b.3 Chronaris 连续融合生产主干

本里程碑把论文方法中的双流连续潜态、物理一致性和秒级因果滞后融合连接成一个任务无关编码器，接入既有训练折归一化、检查点注册与留出折导出。此处先证明机制路径和四项固定消融真实可执行；五个可训练编码器的公共自监督训练在本门通过后统一启动。

### G1–G3b.2 已完成

- 固定鼎新数据、原始异步点、仿真真值、统一表示与留出折来源合同均已固化。
- 生理单流、航电单流、朴素时间同步、MulT 与 ContiFormer 已具备任务头前生产适配器。
- 两轮适配器冒烟共完成 10 个仿真/鼎新留出折输出；浅层基线 14/14、深度基线 15/15 验收通过。
- 所有生产适配器当前与历史查询均不受未来观测扰动；两个双流深度基线对两种历史输入均有非零响应。
- 完整测试为 `272 passed, 8 skipped`；检查点和稠密表示均位于被忽略目录。

### 当前可复用实现与禁止替代路径

- 连续潜态原型：`src/chronaris/models/alignment/prototype.py` 中的 `DualStreamODERNNPrototype`、`SingleStreamODERNNPrototype`。
- 观测编码与连续演化：`src/chronaris/models/alignment/encoders.py`、`ode_cells.py` 和 `torch_batch.py`。
- 已有物理损失：`src/chronaris/models/alignment/physics.py`、`physics_features.py`、`physics_state_mapping.py` 与 `losses.py`。
- 历史融合：`src/chronaris/models/fusion/causal.py` 目前支持因果与固定点数窗口，可复用投影/注意力思路，不能继续用点数窗口作为新主协议。
- 公共表示输入、训练折变换与折外导出：`src/chronaris/representation/`。
- 历史 `ChronarisPrivateTaskAwareWrapper` 和任务训练后的 pooled embedding 不得作为本里程碑的 Chronaris 主路径。
- G3b.2 证据：`docs/artifacts/runs/2026-07-11_deep-baseline-adapter-smoke/`。

### 子任务 G3b.3-a：原始异步双流到连续潜态

1. 新建任务无关 `ChronarisContinuousFusionEncoder`；输入直接使用每个模态的原始时间、数值、字段有效掩码和长度，不能先用公共前向填充把不规则采样抹平。
2. 生理和航电各自使用独立 ObservationEncoder、ODE 演化与 GRU 观测更新；共享隐藏维和求解配置，但不共享输入投影参数。
3. 将 `DualStreamObservationBatch` 明确转换为现有 `TorchAlignmentBatch` 或等价的严格张量合同；padding 时间不得触发 ODE 或 GRU 更新。
4. 在统一 96 点查询轴上读取两条连续潜态；查询轴之前没有观测时输出无效掩码，不能使用未来第一个观测回填。
5. 增加运行计数器或 trace，逐样本记录 ODE 演化步数、GRU 更新步数、查询次数和最大时间间隔，供路径审计使用。

### 子任务 G3b.3-b：秒级多尺度因果融合

1. 新建生产级多尺度因果融合模块，窗口固定为近时延 0–5 秒、中时延 5–15 秒、长时延 15–30 秒。
2. 每个生理查询只能读取对应窗口内的当前或历史航电潜态；时间差由真实秒数计算，不由查询点序号近似。
3. 三个尺度分别计算注意力上下文；空窗口输出结构化不可用并从尺度门控 softmax 中排除，不能用全零上下文参与归一化。
4. 门控输入只允许当前/历史生理潜态、三个尺度上下文和模态可用性；门控权重按查询点归一化并记录尺度利用率。
5. 将生理连续潜态、航电当前潜态和门控跨流上下文投影为 64 维表示；最终有效掩码继续使用六方法公共查询合同。
6. 历史 `lag_window_points` 仅保留 artifact replay 兼容入口，新 checkpoint 清单必须记录 `lag_ranges_s=[[0,5],[5,15],[15,30]]`。

### 子任务 G3b.3-c：物理一致性可用性与损失接口

1. 为仿真 12 状态航电字段和鼎新元数据分别构造物理语义映射；映射只根据字段名/元数据和训练折统计，不读取仿真 oracle。
2. 复用 rigid-body/full family 的物理残差实现，但把训练返回值扩展为每项 `status`、`active`、`count`、`raw_value`、`weighted_value` 和缺失原因。
3. 仿真至少审计速度/姿态/角速度/加速度可用项；鼎新按实际字段逐项启用，无法满足输入语义的项标记 `unavailable`。
4. 物理损失只在有效查询与有效字段上聚合；没有样本时不返回数值零，避免“未计算等于完全一致”的误读。
5. 物理损失是训练辅助量，不写入 `FusionStreamBatch`；表示 manifest 只记录组件配置和可用性摘要。
6. 本工程冒烟不声称模型已被物理目标训练；正式权重仍按第 1–10 epoch 为 0、第 11–20 epoch 线性升至 0.1 的训练协议执行。

### 子任务 G3b.3-d：四项固定消融

建立单一枚举配置，禁止为消融另开超参数搜索：

| 变体 | 唯一变化 | 保持不变 |
| --- | --- | --- |
| 完整 Chronaris | ODE-RNN、物理、多尺度因果融合全部启用 | 统一基准 |
| 无连续演化 | 相邻观测间不执行 ODE 演化，只保留观测更新和查询保持 | 编码器、隐藏维、融合、损失预算 |
| 无物理一致性 | 物理权重为 0，仍计算可用性审计 | ODE-RNN、因果融合、公共目标 |
| 无因果掩码 | 多尺度融合允许对称时间可见域，仅用于消融 | ODE-RNN、物理、尺度与参数预算 |
| 单尺度时延 | 使用 0–30 秒单一历史窗口，不使用三尺度门控 | ODE-RNN、物理、输出投影 |

每个变体 manifest 必须列出与完整配置的字段级 diff；自动测试断言 diff 仅命中表中目标字段。

### 子任务 G3b.3-e：检查点与任务无关导出

1. 检查点保存主干配置、模型权重、训练折归一化器、随机种子、代码路径版本和物理语义映射摘要。
2. `label_used_for_encoder_training=false`；本 smoke 的 `training_invoked=false`，不得加载旧分类/回归 checkpoint。
3. 仿真使用训练、验证、锁定测试三个不同 profile；鼎新使用三个不同 view，沿用一训练、一验证、一留出折的工程冒烟划分。
4. 完整 Chronaris 导出仿真和鼎新各一个留出折表示；四项消融先完成前向、路径与配置审计，不产生论文任务指标。
5. 二次运行逐项校验 checkpoint、normalizer、输入和表示哈希，完整项必须恢复复用；任一来源哈希变化时拒绝复用。

### 子任务 G3b.3-f：针对性测试矩阵

单元与集成测试至少覆盖：

1. 不规则时间间隔会改变 ODE 演化结果；禁用连续演化后该差异消失。
2. padding 时间、无效字段和整段缺失模态不会触发伪观测更新。
3. 未来原始点扰动不改变当前及历史完整模型输出；无因果掩码消融应被反例测试检出未来敏感性。
4. 精确位于 5、15、30 秒边界的键只进入预先规定的尺度；31 秒历史不进入任何主尺度。
5. 三个尺度均可用时门控和为 1；部分尺度空缺时只在可用尺度归一化。
6. 仿真物理项出现 active 且 count 大于 0；语义字段不足的 fixture 返回 unavailable 而非零损失。
7. 四项消融每项只改变一个目标机制，参数和 checkpoint 元数据可复核。
8. 检查点保存/加载前后输出、trace 和配置一致；折外导出与恢复合同保持通过。

### 本里程碑预期产物

紧凑 run `docs/artifacts/runs/2026-07-11_chronaris-continuous-adapter-smoke/`：

- `adapter_protocol.json`
- `continuous_path_audit.csv`
- `lag_scale_boundary_audit.csv`
- `physics_availability.csv`
- `ablation_config_diff.csv`
- `attention_causality_audit.csv`
- `dual_stream_sensitivity.csv`
- `parameter_budget.csv`
- `fold_transform_manifest.json`
- `checkpoint_registry.json`
- `representation_export_manifest.json`
- `acceptance_checks.csv`
- `report.md`、`claim_boundary.md`、`progress.json`、`resume_command.txt` 和 `evidence_manifest.json`

检查点和稠密表示继续写入 `artifacts/application_evaluation/2026-07-11_chronaris-continuous-adapter-smoke/`，禁止入仓。

### G3b.3 验收

- 完整主干输出 `[B,96,64]`，样本、查询时间、有效掩码、训练折来源与五个对照方法合同一致。
- 路径 trace 证明两条 ODE-RNN、统一查询、物理可用性计算和秒级多尺度因果融合真实执行。
- 未来扰动最大变化在数值容差内为 0；三条秒级边界、空窗口门控和整段模态缺失测试通过。
- 物理清单至少在仿真出现 active 项；所有未计算项都有明确 unavailable 原因。
- 四项消融配置 diff 均只命中目标机制，完整模型与消融都能完成有限值前向。
- 任务无关 checkpoint round-trip、折外导出和完整恢复复用通过；文件中不存在标签、logits 或下游预测。
- 完整测试、`compileall`、`git diff --check`、读者术语、密钥、LFS 与被忽略重型产物检查通过。

## 已锁定规范

- [固定数据证据策略](../requirements/foundation/fixed-data-evidence-strategy.md)
- [下游应用评估任务协议](../requirements/downstream-evaluation-spec.md)
- [仿真基准规格](../requirements/synthetic-benchmark-spec.md)
- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)
- [详细实施计划](notes/fixed-data-downstream-evaluation-2026-07-10.md)
- [长程运行手册](notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)

## 后续验收门

### G3b.4：五个可训练编码器公共自监督训练

- 训练对象为生理单流、航电单流、MulT、ContiFormer 和 Chronaris；朴素时间同步只拟合训练折归一化与无监督投影。
- 公共目标固定为 masked reconstruction 1.0、短期预测 0.5、时延判别 0.2；目标构造、遮挡位置和错误时移由共享 augmentation ID 派生。
- Chronaris 的连续对齐、物理一致性与因果方向正则在第 1–10 epoch 为 0，第 11–20 epoch 线性升至 0.2/0.1/0.1，之后保持。
- 首先运行六方法、单一仿真 fold、候选 A、1 epoch 的训练—导出—线性探针闭环烟雾测试；通过后再启动 seed 17 开发筛选。
- 训练器记录每个方法/epoch 的公共损失、方法特有损失、有效样本数、参数量、吞吐、峰值显存和 checkpoint 哈希；任务标签不得进入表示预训练。

### G4：下游 consumer

- 线性分类/回归。
- `aeon==1.5.0` MiniRocket。
- TCN + duration-constrained Viterbi。
- fusion gain、stress slope、trajectory-level paired statistics。

### G5：screen

- seed 17、每个深度方法四候选。
- 只使用 G1 validation 和鼎新外层 train groups。
- 不读取 G2 locked test，不使用结构诊断指标选择候选。

### G6：locked confirmation

- seeds 17、29、43。
- 鼎新主/辅助 split、G1 -> G2、synthetic-to-real。
- frozen consumer 主表和端到端微调辅助表。
- fixed Chronaris 四项消融。

### G7：stress 与论文证据包

- locked checkpoint 跑全部单因素 stress 和 mixed-severe。
- 结构诊断只放附录。
- 输出中文论文图表、证据矩阵、claim boundary 和新协议快照候选。

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

## 当前验证门

当前 G3b.3 实现提交前必须通过：

1. ODE-RNN 真实时间演化、padding 隔离、查询前无观测和 checkpoint round-trip 测试。
2. 0–5、5–15、15–30 秒可见域边界、空尺度门控和未来扰动不变性测试。
3. 仿真/鼎新物理项 active/unavailable 审计与缺字段不冒充零损失测试。
4. 完整 Chronaris 与四项固定消融的单一机制 diff 和有限值前向测试。
5. 仿真与鼎新各完成完整 Chronaris 留出折导出、恢复复用和路径审计。
6. G1–G3b.2 全部聚焦测试保持通过，并运行完整 `pytest`、`compileall` 和 `git diff --check`。
7. 读者可见术语、未来信息、仿真真值隔离、密钥、LFS 和 Git ignore 审计通过。
8. `git status` 中不存在原始点、完整仿真 bundle、稠密表示或检查点。
