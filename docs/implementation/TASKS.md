# Chronaris 当前任务

更新时间：2026-07-11

## 当前长程 goal

在不依赖新增鼎新数据或人工标签的前提下，完成真实弱监督应用任务、模型无关半物理仿真、六方法统一融合表示、Chronaris 连续融合主干、下游评估、锁定实验、图表证据和状态文档闭环。

当前分支：`codex/fixed-data-downstream-evaluation-20260710`。

## 当前里程碑：G3b.4 公共自监督训练与六方法单折闭环

本里程碑先建立一条可恢复、可审计的最小完整实验链：五个可训练编码器在同一仿真训练折使用相同增强与公共目标训练 1 epoch，朴素时间同步拟合相同训练折的无监督变换；六种表示随后由同一线性下游算法消费。这个 smoke 只验证训练、导出、任务消费和报告闭环，不进入论文主指标或候选选择。

### G1–G3b.3 已完成

- 固定鼎新数据、方法无关仿真、统一表示、五个对照适配器和 Chronaris 连续融合主干均已固化。
- Chronaris 已连接双流 ODE-RNN、96 点连续查询、结构化物理可用性和秒级三尺度因果融合；四项固定消融可执行。
- G3b.3 仿真/鼎新 2 个留出折输出恢复 2/2 复用，21/21 验收通过；完整模型未来扰动最大变化为 0。
- 完整测试为 `281 passed, 8 skipped`；原始点、完整仿真、checkpoint 和稠密表示均不进入 Git。

### 当前输入与严格边界

- 五个可训练主干：`ContinuousTimeSingleStreamEncoder` 两个实例、`CausalMulTFusionEncoder`、`CausalContiFormerFusionEncoder` 和 `ChronarisContinuousFusionEncoder`。
- 不训练方法：`NaiveTimeSyncEncoder`，只允许训练折中位数/四分位距归一化与无监督主成分投影。
- 共享增强计划已存在于 `src/chronaris/representation/augmentation.py`，目前只有 realization 生成器，尚未实现对原始异步双流的实际变换。
- 表示导出、checkpoint 来源、恢复与禁止字段合同位于 `src/chronaris/representation/`。
- smoke 只从仿真 `train` split 的 G1 生成族选样并在内部划分 train/validation/held-out；不得读取 `locked_test`，不得使用鼎新留出视图调参。
- 训练编码器不得读取 workload、maneuver、alignment、lag 等 oracle；oracle 只允许在表示冻结后由下游任务构造器按明确清单读取。

### 子任务 G3b.4-a：增强执行器与查询来源追踪

1. 实现 `apply_augmentation_realizations`，逐样本消费既有 realization；API 只接收 batch、realization 和固定 policy，不接收方法名。
2. 先应用整段模态 dropout，再应用连续缺失段和单点随机缺失；两个模态不得同时被整段删除。
3. 时间抖动与时钟偏移只改变训练输入时间；变换后按 `(timestamp, original_index)` 稳定排序，相同时间的观测顺序可复现。
4. 时间变换越过 0/30 秒边界的点从训练输入移除，不裁剪回边界制造重复点；padding、feature mask 和 observation age 重新计算。
5. 扩展因果查询返回 source observation index；masked reconstruction mask 由“原查询来源存在、增强后该来源被删除或替换”确定，不能用数值恰好相等推测。
6. 每个样本/epoch 写出 augmentation ID、两流保留点数、遮挡字段数、时钟偏移和被删模态；五个方法对应记录必须逐项一致。

### 子任务 G3b.4-b：公共 pretext 目标

所有目标先在训练折归一化后的原始 batch 上构造 target，再把增强 batch 送入编码器：

| 目标 | 权重 | 有效位置 | 头部输出 |
| --- | ---: | --- | --- |
| 遮挡重构 | 1.0 | 查询来源被增强移除且原字段有效 | 64 维状态到生理+航电字段 |
| 短期预测 | 0.5 | 当前和下一查询均有效，最后一点排除 | 预测下一查询的生理+航电字段 |
| 时延判别 | 0.2 | 正配对与固定错误时移各半 | 当前/池化状态到二分类 logit |

具体约束：

1. 重构与预测使用按字段有效数归一的 Huber loss；某批次无有效位置时返回 unavailable，不以零值计入目标分母。
2. 错误时移从冻结集合 `{-10,-5,5,10}` 秒按 augmentation ID 选择，移动一个模态时间戳后重新排序；不得读取真值时延。
3. 五个可训练方法使用同一个 target archive、遮挡 mask 和正负配对顺序；单流方法仍接受同一任务，但 inactive stream 的输入不会被偷偷补回。
4. 三个公共头的结构、初始化 seed 与参数量规则相同；头部只服务预训练，正式 `FusionStreamBatch` 仍来自任务头之前。

### 子任务 G3b.4-c：统一训练适配器

1. 为五个主干实现可微分 `encode_for_pretraining`，返回 `[B,96,64]`、内部可用性和方法特有辅助量，不经过适配器的 `inference_mode`。
2. 单流、MulT、ContiFormer 与 Chronaris 共享 `PretextHeadBundle` 类和 `CommonPretextLoss` 聚合器；不得复制五套损失实现。
3. Chronaris 额外暴露 continuous alignment、物理一致性和 causal direction 三项；其他方法明确为 not_applicable，不写零值混入公共平均。
4. Chronaris 升权函数固定：epoch 1–10 为 0，epoch 11–20 线性升至 0.2/0.1/0.1，之后保持；1 epoch smoke 中三项权重必须为 0，但可用性仍审计。
5. 每个 step 验证输出有限、有效目标数量非负、公共 augmentation ID 一致；梯度裁剪前后范数和跳过原因写入 batch log。

### 子任务 G3b.4-d：优化器、预算与 checkpoint

smoke 候选 A 固定为：

- seed 17、1 epoch、batch size 4；显存不足时只允许降至 2/1，隐藏维不变。
- AdamW，学习率 `3e-4`，weight decay `1e-4`，梯度裁剪 `1.0`。
- 每个方法使用相同样本顺序和 batch 数；参数量差异单独报告，不用提前停止制造训练量差异。
- `last.pt` 与 `best.pt` 分开；1 epoch smoke 两者可指向相同权重，但 manifest 必须分别登记。
- checkpoint 包含主干、公共头、normalizer、optimizer、epoch、augmentation policy、输入/split hash、损失权重和 `label_used_for_encoder_training=false`。
- `--resume` 只复用配置、输入和代码路径 hash 全部一致的 completed method/fold；中断 batch 从最近原子 checkpoint 恢复。

### 子任务 G3b.4-e：仿真训练折 smoke 数据

1. 仅选择 `train/g1_state_space` 的 clean-asynchronous 观测，按 profile/trajectory 分成 8 个训练、4 个验证、4 个留出样本；三组 profile 与 latent seed 不重叠。
2. 编码器预训练加载器只能打开 `raw_dual_stream.npz`；路径审计对 `oracle.npz`、任务标签和生成器内部状态为零读取。
3. 表示冻结后，下游 smoke 才加载同一 16 条轨迹的 oracle，构造一个三类仿真负荷窗口任务和一个连续负荷预测任务。
4. 下游标签 archive 与预训练输入分目录、分 manifest；测试断言在 checkpoint 完成前实例化标签读取器会失败。
5. smoke 样本量只用于贯通协议，不写入论文表；正式 screen 使用锁定的完整 G1 train/validation 配置。

### 子任务 G3b.4-f：六方法表示与线性 consumer

1. 五个训练模型和朴素时间同步均导出 train/validation/held-out 三种 role 的 `[B,96,64]` 表示；所有方法样本、查询轴和有效掩码一致。
2. 线性分类使用 Logistic Regression，线性回归使用 Ridge；smoke 固定 `C=1.0`、`alpha=1.0`，不做方法特异调参。
3. scaler、标签阈值和线性模型只在表示 train role 拟合；validation 只检查运行，held-out 只计算一次。
4. 输出 macro-F1、balanced accuracy、AUPRC、MAE、RMSE 和 Spearman；若单折类别不足，写结构化 unavailable，不改标签阈值补齐类别。
5. 指标仅标记 `smoke_only=true`，不得进入 confirmed metrics、候选 Pareto 或论文主图。

### 本里程碑预期产物

紧凑 run `docs/artifacts/runs/2026-07-11_common-pretraining-loop-smoke/`：

- `data_manifest.json`、`split_manifest.json`
- `augmentation_protocol.json`、`augmentation_alignment.csv`
- `pretext_target_manifest.json`、`pretext_loss_audit.csv`
- `training_protocol.json`、`training_status.csv`、`resource_budget.csv`
- `checkpoint_registry.json`、`representation_export_manifest.json`
- `downstream_protocol.json`、`fold_metrics.csv`、`metric_long.csv`
- `acceptance_checks.csv`
- `report.md`、`claim_boundary.md`、`progress.json`、`run.log`、`resume_command.txt` 和 `evidence_manifest.json`

逐 batch log、checkpoint、稠密表示、target archive 和逐样本预测写入 `artifacts/application_evaluation/2026-07-11_common-pretraining-loop-smoke/`，禁止入仓。

### G3b.4 验收

- 五个方法在每个样本/epoch 上的 augmentation ID、遮挡来源和错误时移完全一致；增强执行器源码/API 不含方法参数。
- 三个公共目标在至少一个训练 batch 中均为 active、count 大于 0、loss 有限；无有效位置时以 unavailable 处理。
- 五个 checkpoint 均未使用下游标签，六方法三种 role 共 18 个表示输出来源完整且恢复可复用。
- Chronaris smoke 的三个特有权重为 0，与冻结升权计划一致；物理可用性仍被记录。
- 线性分类与回归使用完全相同的训练折、超参数和单次 held-out 评价；指标标记只用于 smoke。
- 任一预训练代码在 checkpoint 完成前尝试读取 oracle 时测试失败；`locked_test` 路径审计为零命中。
- 重跑 `--resume` 时已完成方法/role 全部复用，删除一个输出后只重建该项。
- 完整测试、`compileall`、`git diff --check`、术语、密钥、LFS 与重型产物忽略检查通过。

## 已锁定规范

- [固定数据证据策略](../requirements/foundation/fixed-data-evidence-strategy.md)
- [下游应用评估任务协议](../requirements/downstream-evaluation-spec.md)
- [仿真基准规格](../requirements/synthetic-benchmark-spec.md)
- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)
- [详细实施计划](notes/fixed-data-downstream-evaluation-2026-07-10.md)
- [长程运行手册](notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)

## 后续验收门

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

当前 G3b.4 实现提交前必须通过：

1. 增强执行器的 method-free API、稳定排序、双模态不同时删除和查询来源追踪测试。
2. 遮挡重构、短期预测、时延判别三项目标 active/unavailable、有效计数、权重与有限梯度测试。
3. 五个方法共享 augmentation ID、target hash、样本顺序、step 数和优化器协议审计。
4. 预训练阶段 oracle/下游标签 fail-closed，仿真 `locked_test` 路径零读取测试。
5. 五个训练 checkpoint、朴素同步变换和六方法 train/validation/held-out 共 18 个表示输出来源完整。
6. 固定 Logistic/Ridge 在同一折完成分类/回归 smoke，所有指标明确标记不进入论文结果。
7. 完整恢复复用和单项删除重建测试；输入、配置或代码路径 hash 改变时拒绝复用。
8. G1–G3b.3 聚焦测试保持通过，并运行完整 `pytest`、`compileall`、`git diff --check`、术语、密钥、LFS 和重型产物忽略检查。
