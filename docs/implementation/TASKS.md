# Chronaris 当前任务

更新时间：2026-07-11

## 当前长程 goal

在不依赖新增鼎新数据或人工标签的前提下，完成真实弱监督应用任务、模型无关半物理仿真、六方法统一融合表示、Chronaris 连续融合主干、下游评估、锁定实验、图表证据和状态文档闭环。

当前分支：`codex/fixed-data-downstream-evaluation-20260710`。

## 当前里程碑：G3a 统一双流输入与融合表示基础设施

本里程碑只建立六方法共用的输入、训练折变换、checkpoint、OOF 导出和 resume 合同；暂不宣称任何模型指标。

### G1–G2b 已完成

- 固定鼎新数据审计：111 个窗口、96/93 个应用上下文、5 个外层折全部完成。
- 原始点冻结：57,648 个共享航电点、2,715 个生理点，6/6 点数和 20/20 标签源排除检查通过。
- 仿真 smoke：4 条潜在轨迹、8 个场景、13/13 检查和 4 张中文图通过。
- 仿真正式集：训练/验证/锁定测试 96/24/48 条潜在架次、1,008 个场景、19/19 验收通过。
- 仿真低/中/高负荷占比 23.3%/43.0%/33.6%，干净场景响应时延 ±1 秒命中率 100%。
- 约 1,018 MB 正式仿真 bundle 与 5.3 MB 鼎新 snapshot 均只在被忽略目录；compact audit 可入仓。
- 以上阶段均未训练六种待比较方法或修改既有确认指标。

### 当前输入

- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)。
- 鼎新 snapshot manifest：`docs/artifacts/runs/2026-07-10_dingxin-input-snapshot/raw_snapshot_manifest.json`。
- 仿真 audit：`docs/artifacts/runs/2026-07-10_aviation-simulation-audit/`；模型 loader 只允许读取 heavy run 中的 `raw_dual_stream.npz`。
- G1 split/profile/seed manifest 和 G2 locked test 隔离边界。

### 需要编码

1. `DualStreamObservationBatch`：两流 observed time、value、valid mask、query grid、sample ID 和特征名校验。
2. `FusionStreamBatch`：统一 `[B,T,64]` sequence、pooled embedding、valid mask、fold/checkpoint lineage；禁止 logits、标签、预测和 diagnostics 字段。
3. 仿真/鼎新输入 loader 与 30 秒 context collator；oracle 文件只能由任务 builder 和机制审计读取。
4. train-only robust normalizer、可选 PCA/projector registry；fit sample hash 必须可追溯。
5. checkpoint registry、fold status、OOF exporter、sample/query order hash 和中断恢复。
6. 公共 augmentation realization：由 sample ID、epoch 和 seed 派生，后续六方法共享。
7. `pyproject.toml` 增加 `application-eval = ["aeon==1.5.0"]`；结构诊断依赖保持独立可选。
8. smoke CLI 与测试：oracle 注入拒绝、test group 不参与 fit、OOF checkpoint 隔离、resume 和 64 维合同。

### 预期产物

紧凑 run `docs/artifacts/runs/2026-07-11_representation-contract-smoke/`：

- `input_schema.json`
- `representation_schema.json`
- `sample_order_manifest.json`
- `fold_transform_manifest.json`
- `checkpoint_registry.json`
- `oof_export_manifest.json`
- `acceptance_checks.csv`
- `report.md`、`claim_boundary.md`、`progress.json`、`resume_command.txt` 和 `evidence_manifest.json`

### G3a 验收

- observed loader 无法读取或返回 oracle/label 字段。
- train-only normalizer/PCA 的 fit hash 不含 test sample。
- OOF test sample 只来自 held-out fold checkpoint。
- 所有方法适配器必须保持完全相同的 sample ID、query timestamp、mask 和 64 维输出顺序。
- 删除一个已完成 fold 的下一个输出后，`--resume` 只重建缺失 fold。
- 仿真 smoke 和鼎新 snapshot 都能通过同一输入合同校验。

## 已锁定规范

- [固定数据证据策略](../requirements/foundation/fixed-data-evidence-strategy.md)
- [下游应用评估任务协议](../requirements/downstream-evaluation-spec.md)
- [仿真基准规格](../requirements/synthetic-benchmark-spec.md)
- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)
- [详细实施计划](notes/fixed-data-downstream-evaluation-2026-07-10.md)
- [长程运行手册](notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)

## 后续验收门

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

当前 G3a 实现提交前必须通过：

1. 输入/表示 schema、forbidden field、fit isolation、OOF lineage 和 resume 测试。
2. G1–G2b 全部 focused tests 保持通过。
3. 仿真与鼎新各至少一个 smoke context 通过相同 collator。
4. `git diff --check`、完整 `pytest` 和 `compileall`。
5. 读者可见术语、oracle 隔离和密钥审计。
6. `git status` 中不存在 raw point、完整仿真 bundle、dense representation 或 checkpoint。
