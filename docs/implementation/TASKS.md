# Chronaris 当前任务

更新时间：2026-07-10

## 当前长程 goal

在不依赖新增鼎新数据或人工标签的前提下，完成真实弱监督应用任务、模型无关半物理仿真、六方法统一融合表示、Chronaris 连续融合主干、下游评估、锁定实验、图表证据和状态文档闭环。

当前分支：`codex/fixed-data-downstream-evaluation-20260710`。

## 当前里程碑：G2a 鼎新原始输入冻结

本里程碑只执行现有两个 sortie 的只读原始点冻结和输入合同审计，不启动完整模型训练。

### G1 已完成

- 正式 run：`docs/artifacts/runs/2026-07-10_fixed-data-audit/`。
- 规模：2 个 sortie、3 个 view、111 个窗口、96 个分类上下文、93 个响应上下文。
- 划分：3 个 leave-one-view-out fold、2 个 leave-one-sortie-out fold，全部完成。
- 字段：每个 sortie 10 个载机机动标签源；12 个唯一 EEG/SpO₂ 响应字段。
- 动态标签语义：3 轴加速度、俯仰、滚转；训练折双 IQR 为 0 的速度、航向和过载已排除。
- 边界：历史对齐后投影可能编码机动标签源，不进入新的防泄漏分类主结果。
- 测试：7 个 focused tests 通过；没有启动训练或修改既有确认指标。

### 当前输入

- G1 `label_field_manifest.json` 和 `label_feature_overlap_audit.csv`。
- 既有 E/F clean manifest 的白名单 sortie、时间范围和视图合同。
- 本机 MySQL/InfluxDB 当前副本，只允许只读查询。
- 现有 `access` 读取器、sortie profile、`RawPoint` 和 `SortieBundle` 合同。

### 需要编码

1. `dataset/application_evaluation`：snapshot contract、字段排除和一致性校验。
2. `evaluation/application_tasks`：冻结 orchestrator、compact manifest 和 unavailable writer。
3. `scripts/evaluation/application_tasks/freeze_fixed_data.py`：只解析 CLI 和调用核心模块。
4. 测试：sortie 白名单、查询只读、secret 不落盘、点数/时间范围一致、标签字段排除、hash 可复现。

### 预期产物

本机重型目录 `artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot/`：

- 两流规范化原始点分片。
- 本机 snapshot index 和校验 hash。

G1 紧凑 run 追加或新建的可引用产物：

- `raw_snapshot_manifest.json`
- `raw_snapshot_consistency.csv`
- `raw_input_field_exclusion.csv`
- `raw_snapshot_unavailable.json`（仅失败时）
- 更新后的 `report.md`、`progress.json` 和 `evidence_manifest.json`

### G2a 验收

- 仅出现两个白名单 sortie，没有扩大数据范围。
- 原始点时间范围覆盖现有 37 个窗口，点数差异有结构化解释。
- 六方法原始输入字段合同明确排除 G1 标签源及其确定性派生项。
- snapshot 路径被 Git 忽略，紧凑 manifest 不含原始高频值或密钥。
- 数据不可读时产生可复现 unavailable 状态，不伪造原始流。
- 本里程碑不改既有确认指标、不运行结构诊断、不启动大规模训练。

## 已锁定规范

- [固定数据证据策略](../requirements/foundation/fixed-data-evidence-strategy.md)
- [下游应用评估任务协议](../requirements/downstream-evaluation-spec.md)
- [仿真基准规格](../requirements/synthetic-benchmark-spec.md)
- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)
- [详细实施计划](notes/fixed-data-downstream-evaluation-2026-07-10.md)
- [长程运行手册](notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)

## 后续验收门

### G2b：G1/G2 仿真器

- 方法无关、seed 可复现、生成族隔离。
- 96/24/48 潜在架次和成对 observation 场景。
- 状态、负荷、物理 residual、时钟与响应 lag oracle 完整。
- 数据质量和中文图件审计通过。

### G3a：统一表示基础设施

- `DualStreamObservationBatch` / `FusionStreamBatch` 合同。
- train-only normalizer、checkpoint registry、OOF exporter、resume。
- 六方法同 sample/query 顺序和 64 维输出。

### G3b：六方法编码器

- 两个单流、朴素同步、MulT、ContiFormer、Chronaris。
- Chronaris 确实调用连续 ODE-RNN、物理约束和秒级因果融合。
- 公共 pretext、增强和候选预算一致。

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

当前 G2a 实现提交前必须通过：

1. snapshot 与 compact manifest schema 测试。
2. G1 全部 focused tests 保持通过。
3. `git diff --check` 和完整 `pytest`。
4. `compileall src scripts tests`。
5. 读者可见术语与密钥审计。
6. `git status` 中不存在 raw point、dense bundle 或 checkpoint。
