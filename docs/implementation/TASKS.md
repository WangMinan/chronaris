# Chronaris 当前任务

更新时间：2026-07-11

## 当前长程 goal

在不依赖新增鼎新数据或人工标签的前提下，完成真实弱监督应用任务、模型无关半物理仿真、六方法统一融合表示、Chronaris 连续融合主干、下游评估、锁定实验、图表证据和状态文档闭环。

当前分支：`codex/fixed-data-downstream-evaluation-20260710`。

## 当前里程碑：G4.2 鼎新弱监督目标与真实外层折接入

G4.1 已把仿真负荷与机动状态真值接入六方法冻结表示，完成统一下游消费者与指标。G4.2 已完成鼎新两项独立目标 archive、防泄漏原始点上下文和五折任务绑定，当前进入训练内 validation 与真实外层折公共预训练；未完成前不进入候选 screen。

### G1–G4.1 已完成

- 六方法生产表示、Chronaris 连续主干、五方法公共预训练和朴素时间同步训练折无监督变换均已可恢复。
- 16 条仿真 train 轨迹的 8/4/4 smoke 完成 5 个训练 checkpoint、18 个折外表示和 72 条线性指标。
- workload 真值只在五个 checkpoint 完成后打开；预训练路径不读取 oracle，仿真 validation/locked_test 路径零命中。
- 删除一个 Chronaris 留出折表示后仅重建该项，重建 SHA-256 与原输出一致；闭环 20/20 验收通过。
- G3b.4 完成时完整测试为 `301 passed, 8 skipped`；G4.1 完成后为 `312 passed, 8 skipped`；G4.2 目标归档为 `314 passed, 8 skipped`；原始上下文绑定后刷新为 `316 passed, 8 skipped`。
- 16 条仿真训练轨迹各取四个跨状态上下文，形成 64 个样本和 32/16/16 profile 隔离；六方法应用上下文表示共 18 份，恢复 18/18 复用。
- 线性、MiniROCKET 10,000 kernels、两层因果 TCN 与训练折持续时间解码共产生 384 条可计算指标、256 条融合增益和 30 条轨迹级配对统计。
- MiniROCKET 训练折方差过滤、TCN 随机流隔离和组件级恢复均已固化；删除 Chronaris 表示、MiniROCKET、TCN 后只重建目标组件，12/12 验收通过。
- G4.1 紧凑证据位于 `2026-07-11_application-consumer-smoke`，约 15 MB 模型、表示和预测留在被忽略目录；所有指标均为 smoke only。
- G4.2 已生成五折两个任务共 10 个独立目标 archive：分类保留 96 个上下文，原始点中位数生理响应保留 90/93 个完整未来区间，3 个末端候选结构化不可用。
- 10/10 archive 恢复复用，原 snapshot 哈希不变，目标归档 `12/12` 验收通过；本阶段没有模型训练或任务指标。
- 96 个目标上下文中 93 个具有完整 30 秒输入，三个 25.991 秒部分末窗不可用；机动分类/生理响应最终绑定 93/90 个唯一上下文。
- 12 生理 + 955 航电字段采用 78.6 MB 允许字段 CSR 缓存，20 个机动标签源在映射层删除；完整绑定审计 12.34 秒、峰值 767 MB，12/12 通过。

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
3. 当前：在 outer-train group 内建立防重叠 inner-train/validation，固化 leave-one-view-out 主协议和 leave-one-sortie-out 辅助协议。
4. 训练五方法公共预训练 checkpoint、朴素时间同步无监督变换并导出六方法 train/validation/test 表示。
5. 接入固定线性/MiniROCKET consumer 的真实数据 smoke；真实与仿真 metric root 分离。G4.2 通过后再进入 seed 17 四候选 screen。

## 已锁定规范

- [固定数据证据策略](../requirements/foundation/fixed-data-evidence-strategy.md)
- [下游应用评估任务协议](../requirements/downstream-evaluation-spec.md)
- [仿真基准规格](../requirements/synthetic-benchmark-spec.md)
- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)
- [详细实施计划](notes/fixed-data-downstream-evaluation-2026-07-10.md)
- [长程运行手册](notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)

## 后续验收门

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

当前 G4.2 提交前必须通过：

1. 已通过：两项鼎新 target archive 的时间边界、训练折阈值、样本覆盖、字段 lineage 和 unavailable 测试。
2. 已通过：机动标签源字段在原始映射中零命中，未来生理点在输入中零命中，三个部分末窗结构化不可用。
3. leave-one-view-out 与 leave-one-sortie-out 的 inner-train/validation/outer-test group 或时间块无重叠，所有变换只用 inner-train。
4. target、原始点 context 与表示 sample ID 一一对应；漏样本、重复样本、跨折 checkpoint 直接失败。
5. 至少一个真实外层折完成六方法表示和固定 consumer smoke，所有鼎新结果继续标记弱监督并与仿真指标分层。
6. G1–G4.1 聚焦测试保持通过，并运行完整 `pytest`、`compileall`、`git diff --check`、术语、密钥、LFS 和重型产物忽略检查。
