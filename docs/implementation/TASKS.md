# Chronaris 当前任务

更新时间：2026-07-10

## 当前长程 goal

在不依赖新增鼎新数据或人工标签的前提下，完成真实弱监督应用任务、模型无关半物理仿真、六方法统一融合表示、Chronaris 连续融合主干、下游评估、锁定实验、图表证据和状态文档闭环。

当前分支：`codex/fixed-data-downstream-evaluation-20260710`。

## 当前里程碑：G2b 方法无关半物理仿真器

本里程碑实现两个与待比较方法解耦的航空人机异步双流生成族、成对观测压力场景和 oracle 审计，不启动六方法完整训练。

### G1/G2a 已完成

- G1 run：`docs/artifacts/runs/2026-07-10_fixed-data-audit/`；111 个窗口形成 96/93 个应用上下文，5 个外层折全部完成。
- G1 字段：每个 sortie 10 个载机标签源，12 个唯一 EEG/SpO₂ 响应字段；动态标签语义为 3 轴加速度、俯仰和滚转。
- G2a run：`docs/artifacts/runs/2026-07-10_dingxin-input-snapshot/`。
- G2a 原始点：2 份共享航电共 57,648 点，3 份生理共 2,715 点；6/6 历史点数一致。
- G2a 存储：5.3 MB 原始值只在被忽略目录，仓库只保留约 60 KB 紧凑 manifest、对账和排除表。
- G2a 排除：20/20 机动标签源在 snapshot 中可见且全部禁止进入机动分类输入。
- G2a resume：哈希复核后不重复查询数据库；G1/G2a focused tests 共 10 个通过。
- 两个里程碑均未启动训练或修改既有确认指标。

### 当前输入

- [仿真基准规格](../requirements/synthetic-benchmark-spec.md)。
- G2a 真实点的采样率、字段规模、缺失和时间间隔摘要，只用于仿真参数范围校准，不复制真实值。
- G1 任务上下文：过去 30 秒、基础窗口 5 秒、负荷提前 10 秒、五类机动状态。
- 正式开发 seed 17，训练/验证/锁定测试潜在架次分别为 96/24/48。

### 需要编码

1. `simulation/aviation_dual_stream`：配置、半马尔可夫机动、车辆动力学、潜在负荷、生理响应、观测时钟和 generator。
2. G1：切换线性状态空间、受控输入、一阶滞后响应。
3. G2：事件驱动样条、非线性饱和和非一阶生理滤波，不复用 Chronaris ODE 方程。
4. paired observation generator：同一 latent trajectory 生成 clean 和 stress，不重新抽样状态或个体参数。
5. oracle 与 validator：状态边界、负荷、时钟偏移/漂移、响应 lag、物理 residual、seed 和 family。
6. CLI：`generate_aviation_dual_stream.py`、`audit_aviation_dual_stream.py`。
7. 测试：方法无关 API、seed 复现、family split、状态覆盖、物理范围、响应 lag、同轨迹配对。

### 预期产物

本机重型目录 `artifacts/application_evaluation/2026-07-10_aviation-simulation/`：

- 生成后的潜在轨迹、原始异步双流、oracle 和训练/验证/测试 bundle。

紧凑 run `docs/artifacts/runs/2026-07-10_aviation-simulation-audit/`：

- `simulation_manifest.json`
- `scenario_coverage.csv`
- `oracle_validation.csv`
- `paired_observation_audit.csv`
- `generator_family_split.json`
- 中文数据质量图与 `figure_manifest.json`
- `report.md`、`claim_boundary.md`、`progress.json`、`resume_command.txt` 和 `evidence_manifest.json`

### G2b 验收

- generator API 和配置中不存在方法名、候选名或 checkpoint。
- 相同 latent ID 的 clean/stress oracle 完全一致，仅观测过程改变。
- G1 train/validation 与 G2 locked test 的 family、profile、seed 无交集。
- 状态转移、负荷范围、响应 lag 和时钟真值可由 oracle 重建。
- 96/24/48 潜在架次全部生成或明确记录失败，不静默补样本。
- 图表中文可读且仿真结论不与鼎新真实指标混算。

## 已锁定规范

- [固定数据证据策略](../requirements/foundation/fixed-data-evidence-strategy.md)
- [下游应用评估任务协议](../requirements/downstream-evaluation-spec.md)
- [仿真基准规格](../requirements/synthetic-benchmark-spec.md)
- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)
- [详细实施计划](notes/fixed-data-downstream-evaluation-2026-07-10.md)
- [长程运行手册](notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)

## 后续验收门

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

当前 G2b 实现提交前必须通过：

1. 仿真方法无关、seed 复现、family split 和 paired latent 测试。
2. G1/G2a 全部 focused tests 保持通过。
3. smoke 生成、oracle audit 和关键 PNG 抽查。
4. `git diff --check`、完整 `pytest` 和 `compileall`。
5. 读者可见术语与真实/仿真证据隔离审计。
6. `git status` 中不存在完整仿真 bundle、raw point、dense bundle 或 checkpoint。
