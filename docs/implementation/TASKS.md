# Chronaris 当前任务

更新时间：2026-07-10

## 当前长程 goal

在不依赖新增鼎新数据或人工标签的前提下，完成真实弱监督应用任务、模型无关半物理仿真、六方法统一融合表示、Chronaris 连续融合主干、下游评估、锁定实验、图表证据和状态文档闭环。

当前分支：`codex/fixed-data-downstream-evaluation-20260710`。

## 当前里程碑：G1 固定数据审计

本里程碑只实现数据/任务合同和审计，不启动完整模型训练。

### 输入

- `docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json`
- `docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json`
- 当前 MySQL 航电字段元数据。
- 当前三个 view 的 window/raw summary 和 feature bundle。

### 需要编码

1. `dataset/application_evaluation` 包：
   - 数据来源、字段角色和上下文 dataclass。
   - E/F clean manifest loader。
   - 30 秒上下文与连续性 builder。
   - fold-fitted 机动标签和生理响应目标。
   - 标签源字段及确定性派生字段排除器。
   - leave-one-view-out / leave-one-sortie-out split builder。
2. 固定数据审计 CLI：
   - 读取现有数据和 MySQL 元数据。
   - 写数据、字段、缺失、标签覆盖、split 和泄漏审计。
   - 支持结构化 blocked/unavailable。
3. 测试：
   - test group 不参与任何 fit。
   - 未知字段不回退全字段。
   - 标签字段与输入字段交集为零。
   - 30 秒连续上下文数量和分组正确。
   - 阈值、IQR 和样本 hash 可追溯。

### 预期产物

`docs/artifacts/runs/2026-07-10_fixed-data-audit/`：

- `data_manifest.json`
- `field_role_manifest.csv`
- `sampling_interval_summary.csv`
- `missingness_summary.csv`
- `context_sample_manifest.jsonl`
- `label_field_manifest.json`
- `fold_label_thresholds.csv`
- `label_feature_overlap_audit.csv`
- `split_manifest.json`
- `report.md`
- `evidence_manifest.json`

### G1 验收

- 无排除时分类/响应上下文数量为 96/93；实际排除均有原因。
- 每个 fold 的标签阈值只由 train sample hash 计算。
- 标签源字段和可确定性重建标签的字段不进入模型输入。
- 每个 fold 的类别/目标覆盖足够；不足时结构化标记，不伪造标签。
- 本里程碑不改 confirmed metrics，不运行 E3，不启动大规模训练。

## 已锁定规范

- [固定数据证据策略](../requirements/foundation/fixed-data-evidence-strategy.md)
- [下游应用评估任务协议](../requirements/downstream-evaluation-spec.md)
- [仿真基准规格](../requirements/synthetic-benchmark-spec.md)
- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)
- [详细实施计划](notes/fixed-data-downstream-evaluation-2026-07-10.md)
- [长程运行手册](notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)

## 后续验收门

### G2a：鼎新原始输入冻结

- 只读导出现有两个 sortie 的原始异步点。
- 本机重型 snapshot 写入被忽略的 `artifacts/application_evaluation/`。
- 与现有 feature-export 时间范围和抽样点数一致。
- 读取失败时进入对齐后真实证据 fallback，不伪造原始流。

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

本轮详细规格完成后必须通过：

1. 文档路径和相互链接存在。
2. `git diff --check`。
3. `compileall src scripts tests`，确认文档变更未破坏当前代码。
4. 读者可见术语审计。
5. `git status` 中不存在重型产物。
