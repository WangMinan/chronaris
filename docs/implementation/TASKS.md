# Chronaris 当前任务

更新时间：2026-07-11

## 当前长程 goal

在不依赖新增鼎新数据或人工标签的前提下，完成真实弱监督应用任务、模型无关半物理仿真、六方法统一融合表示、Chronaris 连续融合主干、下游评估、锁定实验、图表证据和状态文档闭环。

当前分支：`codex/fixed-data-downstream-evaluation-20260710`。

## 当前里程碑：G3b.2 MulT 与 ContiFormer 生产适配器

本里程碑从历史任务 wrapper 中分离两个深度基线的任务头前时序状态，接入 G3a 统一表示与留出折导出器。只做生产适配器、检查点可恢复性和因果边界验证；公共自监督训练在五个可训练编码器都具备后统一启动。

### G1–G3b.1 已完成

- 固定数据、原始点、仿真真值、统一表示与留出折来源合同均已固化。
- 生理单流、航电单流和朴素时间同步已成为生产适配器，不再由合同探针承担主表示。
- 仿真三划分与鼎新三个不同视图共完成 6 个生产导出，恢复 6/6 复用，14/14 验收通过。
- 未来观测与非激活模态扰动的历史输出最大变化均为 0。
- 完整测试为 `265 passed, 8 skipped`；本阶段开始前工作树不包含入仓检查点或稠密表示。

### 当前输入

- 公共因果查询层：`src/chronaris/modeling/fusion_encoders/causal_query.py`。
- 现有历史 wrapper：`ChronarisMulTWrapper` 与 `ChronarisContiFormerWrapper`，仅作为结构与权重命名参考，不直接导出任务训练后的 pooled embedding。
- vendored MulT：`third_party/mult/`；需先确认自注意力和跨模态注意力的掩码语义。
- vendored ContiFormer：已支持 `causal=True`，历史 wrapper 默认仍为 false。
- 统一检查点、表示序列化和恢复：`src/chronaris/representation/`。
- G3b.1 证据：`docs/artifacts/runs/2026-07-11_shallow-baseline-adapter-smoke/`。

### 子任务 G3b.2-a：双流查询输入

1. 两流分别经过公共因果查询层，形成 96 点值、字段有效性和观测年龄；适配器不直接读取原始未来点。
2. 生理与航电各自使用独立输入投影，输入维数允许不同，隐藏维固定为 64。
3. 整段模态缺失时保留内部模态可用性掩码，并保证统一查询有效掩码跨方法一致。
4. 双流归一化只在训练折拟合；MulT 与 ContiFormer 共享同一 transform hash。

### 子任务 G3b.2-b：MulT 生产适配器

1. 复用 vendored Transformer block，但新增严格因果的自注意力与跨模态注意力掩码。
2. 生理查询航电、航电查询生理两个方向都只能关注当前及历史查询点。
3. 两个方向的时序状态在任务头前合并并投影到 64 维，不导出 logits 或旧任务 pooled embedding。
4. 增加未来点扰动、单向模态扰动、整段模态缺失和注意力掩码测试。

### 子任务 G3b.2-c：ContiFormer 生产适配器

1. 拼接两流的因果查询值、字段掩码和观测年龄，使用 `ContiFormerEncoder(causal=True)`。
2. 时序状态直接作为任务头前 sequence embedding；模型隐藏维即 64，不再追加任务特征。
3. 兼容历史 wrapper 的非因果默认行为，但生产适配器必须在清单中记录 `causal_attention=true`。
4. 增加未来点扰动、时间轴变化、模态缺失和检查点 round-trip 测试。

### 子任务 G3b.2-d：公平预算与导出

1. 两个适配器使用相同隐藏维 64、两层、四头、dropout 0.1 和随机种子 17。
2. 两者共享训练折归一化与同一 augmentation realization 合同；本 smoke 不执行训练增强。
3. 仿真训练/验证/G2 锁定轨迹和鼎新三个不同视图分别完成两个方法的留出折导出。
4. 记录参数量、检查点哈希、输入字段、导出耗时、未来扰动、双流敏感性和恢复状态。

### 预期产物

紧凑 run `docs/artifacts/runs/2026-07-11_deep-baseline-adapter-smoke/`：

- `adapter_protocol.json`
- `attention_causality_audit.csv`
- `dual_stream_sensitivity.csv`
- `parameter_budget.csv`
- `fold_transform_manifest.json`
- `checkpoint_registry.json`
- `representation_export_manifest.json`
- `acceptance_checks.csv`
- `report.md`、`claim_boundary.md`、`progress.json`、`resume_command.txt` 和 `evidence_manifest.json`

### G3b.2 验收

- 两个方法均输出 `[B,96,64]`，样本标识、查询时间、有效掩码和留出折来源一致。
- 修改查询时刻之后的任一模态观测，不能改变该时刻及之前的输出。
- 分别扰动生理或航电历史时，两个双流适配器都必须产生非零但有限的表示变化。
- 生产表示来自任务头之前，文件中不存在标签、logits、预测值或方法专属诊断量。
- 两个方法的归一化拟合样本哈希一致且不含验证/留出样本。
- 检查点保存/加载后输出一致；恢复运行只重建缺失项，完整项校验后复用。

## 已锁定规范

- [固定数据证据策略](../requirements/foundation/fixed-data-evidence-strategy.md)
- [下游应用评估任务协议](../requirements/downstream-evaluation-spec.md)
- [仿真基准规格](../requirements/synthetic-benchmark-spec.md)
- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)
- [详细实施计划](notes/fixed-data-downstream-evaluation-2026-07-10.md)
- [长程运行手册](notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)

## 后续验收门

### G3b.3：Chronaris 连续融合生产主干

- 打通双流 ODE-RNN、统一查询、物理一致性和秒级三尺度因果融合。
- 路径测试证明连续演化、物理项与因果掩码真实执行，不由任务 wrapper 替代。
- 物理项逐项记录 active/unavailable/count/value；缺字段时不得写零值冒充启用。
- 固定无连续演化、无物理、无因果掩码和单尺度时延四项消融。

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

当前 G3b.2 实现提交前必须通过：

1. MulT/ContiFormer 因果注意力、未来扰动不变性、双流敏感性和检查点 round-trip 测试。
2. G1–G3b.1 全部聚焦测试保持通过。
3. 仿真与鼎新各完成两个深度基线的留出折导出。
4. `git diff --check`、完整 `pytest` 和 `compileall`。
5. 读者可见术语、未来信息、真值隔离和密钥审计。
6. `git status` 中不存在原始点、完整仿真 bundle、稠密表示或检查点。
