# Chronaris 当前任务

更新时间：2026-07-11

## 当前长程 goal

在不依赖新增鼎新数据或人工标签的前提下，完成真实弱监督应用任务、模型无关半物理仿真、六方法统一融合表示、Chronaris 连续融合主干、下游评估、锁定实验、图表证据和状态文档闭环。

当前分支：`codex/fixed-data-downstream-evaluation-20260710`。

## 当前里程碑：G3b.1 两个单流与朴素时间同步生产适配器

本里程碑把 G3a 的三个合同探针替换为可训练、可导出的生产基线。只实现生理单流、航电单流和朴素时间同步；MulT、ContiFormer 与 Chronaris 分别进入 G3b.2 和 G3b.3，避免一次改动同时跨越所有模型主干。

### G1–G3a 已完成

- 固定鼎新数据审计：111 个窗口、96/93 个应用上下文、5 个外层折全部完成。
- 原始点冻结：57,648 个共享航电点、2,715 个生理点，6/6 点数和 20/20 标签源排除检查通过。
- 仿真正式基准：训练/验证/锁定测试 96/24/48 条潜在架次、1,008 个场景、19/19 验收通过。
- 统一输入与表示：鼎新 12/955 字段、仿真 7/12 字段、96 点查询轴、64 维表示和训练折隔离均已固化。
- 合同冒烟验证：六个方法接口 6/6 输出、6/6 恢复复用、14/14 验收通过；相关聚焦测试 20 个通过。
- 合同探针只验证接口，不作为任何模型效果或候选选择依据。

### 当前输入

- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)。
- G3a 证据：`docs/artifacts/runs/2026-07-11_representation-contract-smoke/`。
- 生产接口：`src/chronaris/representation/` 中的批次、训练折变换、检查点注册表和留出折导出器。
- 现有连续时间基础块：`src/chronaris/modeling/common/deep_models.py`；复用前先确认其时间、掩码与任务头边界。
- 仿真训练/验证只使用 G1 生成族；G2 事件样条锁定测试在适配器开发期间不可读取任务真值。

### 子任务 G3b.1-a：公共因果查询层

1. 实现只使用 `timestamp <= query_time` 的批量查询重采样器，支持 forward-fill、观测年龄和整段模态缺失。
2. 对相同时间的多 measurement 观测采用稳定顺序聚合，不引入未来点。
3. 输出每个查询点的值、字段有效掩码、模态有效掩码和观测年龄，不改变 96 点查询轴。
4. 增加未来值扰动、未来时间戳插入、首点晚于查询时刻和重复时间戳测试。

### 子任务 G3b.1-b：两个单流编码器

1. 新增一个共享 `ContinuousTimeSingleStreamEncoder`，生理单流和航电单流只通过 `active_stream` 配置切换，不复制模型实现。
2. 输入投影同时消费数值、字段掩码、观测年龄和相对时间；主干隐藏维固定 64。
3. 单流缺失时输出无效查询掩码，不用全零向量冒充有效状态。
4. 表示来自任务头之前；适配器不得接收标签、任务名或预测目标。
5. 检查两个单流的参数量差异只能来自输入投影维数，主干层数、隐藏维和训练预算相同。

### 子任务 G3b.1-c：朴素时间同步

1. 两流分别通过公共因果查询层投影到 96 点，不允许默认线性插值跨越未来观测。
2. 拼接同步值、字段有效掩码和截断观测年龄；中位数/四分位距归一化只在训练折拟合。
3. 无监督主成分分析最多保留 64 个分量，不足 64 维时右侧补零；拟合样本哈希进入表示清单。
4. 朴素同步不训练任务头，不读取仿真真值或鼎新弱监督标签。

### 子任务 G3b.1-d：统一导出与冒烟验证

1. 三个生产适配器接入现有检查点注册表和留出折导出器，替换对应合同探针。
2. 仿真使用一条训练、一条验证和一条 G2 锁定轨迹验证来源隔离；鼎新使用一个 30 秒上下文验证 955 维稀疏航电输入。
3. 记录输入字段数、参数量、查询有效率、拟合样本哈希、检查点哈希、运行耗时和峰值内存。
4. 保留同一批次的合同探针作为接口对照，但报告中明确区分“生产适配器”和“合同探针”。

### 预期产物

紧凑 run `docs/artifacts/runs/2026-07-11_shallow-baseline-adapter-smoke/`：

- `adapter_protocol.json`
- `causal_query_audit.csv`
- `parameter_budget.csv`
- `fold_transform_manifest.json`
- `checkpoint_registry.json`
- `representation_export_manifest.json`
- `acceptance_checks.csv`
- `report.md`、`claim_boundary.md`、`progress.json`、`resume_command.txt` 和 `evidence_manifest.json`

### G3b.1 验收

- 修改任一查询时刻之后的观测，不能改变该时刻及之前的输出。
- 生理单流与航电单流复用同一主干类，且非激活模态的数值变化不影响输出。
- 朴素同步的归一化和主成分分析拟合样本不含验证或留出测试样本。
- 三个生产适配器均输出 `[B,96,64]`，池化严格等于有效查询点均值。
- 删除一个已完成导出后，恢复运行只重建缺失项；完整项校验后复用。
- 仿真和鼎新冒烟验证通过，且没有标签、输出分数、预测值或诊断量进入表示文件。

## 已锁定规范

- [固定数据证据策略](../requirements/foundation/fixed-data-evidence-strategy.md)
- [下游应用评估任务协议](../requirements/downstream-evaluation-spec.md)
- [仿真基准规格](../requirements/synthetic-benchmark-spec.md)
- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)
- [详细实施计划](notes/fixed-data-downstream-evaluation-2026-07-10.md)
- [长程运行手册](notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)

## 后续验收门

### G3b.2：MulT 与 ContiFormer 生产适配器

- 从任务头之前导出时序状态，增加统一时间/掩码适配层和 64 维投影。
- 不复用旧回归任务 checkpoint 作为论文主表示初始化。
- 两者共享公共自监督目标、增强 realization、候选数和训练预算。
- 仿真与鼎新各完成一折留出导出后才能进入 Chronaris 主干改造。

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

当前 G3b.1 实现提交前必须通过：

1. 因果查询、未来扰动不变性、单流模态隔离和朴素同步训练折隔离测试。
2. G1–G3a 全部聚焦测试保持通过。
3. 仿真与鼎新各至少一个 30 秒上下文通过三个生产适配器。
4. `git diff --check`、完整 `pytest` 和 `compileall`。
5. 读者可见术语、未来信息、真值隔离和密钥审计。
6. `git status` 中不存在原始点、完整仿真 bundle、稠密表示或检查点。
