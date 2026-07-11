# 固定数据下游评估与完整论文实验实施计划

日期：2026-07-10  
分支：`codex/fixed-data-downstream-evaluation-20260710`  
状态：长程 goal 执行中；当前进入鼎新真实外层折公共预训练与统一表示导出

## 1. 最终交付

本长程任务结束时，仓库应具备：

1. 现有鼎新两架次的可复现输入快照或明确的对齐后 fallback。
2. 防泄漏的机动强度弱监督分类和机动诱发生理响应预测。
3. 方法无关的 G1/G2 航空人机异步双流仿真器。
4. 六方法统一 `[B,T,64]` 表示接口。
5. 真正连接连续 ODE-RNN、物理约束和因果融合的 Chronaris 编码器。
6. 线性、MiniRocket、TCN/Viterbi 下游 consumer。
7. real-only、synthetic、synthetic-to-real 三条实验轨道。
8. 三 seed 锁定确认、消融、鲁棒性曲线和论文证据包。
9. 与实际代码和 artifact 一致的 `STATE/TASKS/ARTIFACTS`。

## 2. 开始状态

- 新分支从 `origin/main` 建立；该基线只包含 2026-07-06 E3 规划，不包含后续 E3 实现分支。
- `implement/fusion-stream-structure-20260707` 保留历史，不整体合并。
- 当前已确认 111 个真实窗口、2 个 sortie、3 个 view；旧机动标签在全量数据上计算阈值，旧生理目标为下一窗口绝对聚合值。
- 当前任务评价中的 Chronaris wrapper 与论文 ODE-RNN 主干不是同一条完整路径。
- 当前环境有 `aeon 1.5.0`，没有 `hmmlearn`；状态平滑使用自研动态规划。

## 3. 包结构原则

核心逻辑不放脚本。新增代码按职责拆分，单文件目标小于 500 行：

```text
src/chronaris/
├── dataset/application_evaluation/
├── simulation/aviation_dual_stream/
├── modeling/fusion_streams/
└── evaluation/application_tasks/

scripts/
├── simulation/
└── evaluation/application_tasks/
```

- `dataset`：快照、样本、字段角色、fold-fitted 标签和 split。
- `simulation`：潜在过程、观测过程、oracle 与审计。
- `modeling/fusion_streams`：统一编码器、trainer、OOF 导出和 manifest。
- `evaluation/application_tasks`：consumer、指标、统计和报告。
- `scripts`：只解析 CLI 和调用上述模块。

## 4. 工作包 A：固定数据审计与协议基线

### 4.1 目标

在任何新训练前确认当前真实数据到底能支持哪些任务，并产生可机器检查的泄漏报告。

### 4.2 实现

新增：

- 数据来源与字段角色 dataclass。
- 从 E/F clean feature-export manifest 读取三个 view 的稳定 loader。
- 字段元数据解析器：把 `BUS...code...` 映射到中文名称、单位和物理类别。
- 30 秒上下文 builder 和连续性检查。
- fold-fitted 机动标签、生理响应目标与字段排除器。
- leave-one-view-out / leave-one-sortie-out split manifest builder。

CLI：

```text
scripts/evaluation/application_tasks/audit_fixed_data.py
```

关键参数：

```text
--e-run-manifest
--f-run-manifest
--run-id
--output-root docs/artifacts/runs
--strict-label-field-metadata
```

### 4.3 产物

首个紧凑 run：`docs/artifacts/runs/2026-07-10_fixed-data-audit/`。

至少包含：

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

### 4.4 测试

- 全量分位数不得被新 builder 调用。
- 测试 fold 不参与 median/IQR/threshold。
- label-source 与 model-input 交集为零。
- 连续窗口数量在无排除时为 96/93。
- 元数据不足时 fail closed，不回退全字段。
- split 中 train/test group 不相交。

### 4.5 验收门 G1

- 审计报告列出真实可用字段、任务样本数和每折类别分布。
- 所有阈值可追溯到训练样本 hash。
- 不训练模型，不改 confirmed metrics。
- 若 R1 字段元数据不足，明确 blocked 原因后仍继续 R2 和仿真工作。

### 4.6 2026-07-10 执行结果

- G1 正式 run 为 `docs/artifacts/runs/2026-07-10_fixed-data-audit/`，状态 `completed`。
- 111 个窗口形成 96 个分类上下文和 93 个未来响应上下文；3 个 leave-one-view-out fold 与 2 个 leave-one-sortie-out fold 全部完成。
- MySQL 元数据解析错误为 0；每个 sortie 选择 10 个载机标签源，目标机与质量字段不入选。
- 训练折动态语义实际为 3 轴加速度、俯仰和滚转；速度、航向和过载因双 IQR 为 0 按 fold 排除。
- 生理响应有 12 个唯一 EEG/SpO₂ 字段。G1 使用窗口均值验证合同，G2a 后按任务规格切换到原始点窗口中位数。
- 既有对齐后投影可能已经编码机动标签源，因此只保留为历史诊断输入，不进入新的防泄漏分类主结果。

## 5. 工作包 B：现有原始数据冻结

### 5.1 目标

从当前 MySQL/InfluxDB 只读导出同两个 sortie 的原始异步点，减少后续对数据库在线状态的依赖。

### 5.2 实现

复用现有：

- `access` 读取器。
- sortie profile 和日期补全规则。
- `SortieBundle`、`RawPoint` 和 `AlignmentBatch` 语义。

新增本机 snapshot writer，保存：

- 原始时间戳、measurement、values、tags 的规范化数组。
- feature name/unit/source 映射。
- 每 view 的 point/field/time-range summary。
- snapshot SHA-256 和数据库查询范围。

CLI：

```text
scripts/evaluation/application_tasks/freeze_fixed_data.py
```

重型输出：

```text
artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot/
```

紧凑 manifest 追加到固定数据审计 run。

### 5.3 安全要求

- 只允许白名单中的两个 sortie ID。
- 连接密钥只从环境或 `docs/SECRETS.md` 读取，不进入日志。
- snapshot 不入 Git/LFS。
- 导出前后数据库只读，不执行 DDL/DML。

### 5.4 fallback

如果读取失败：

- 写 `raw_snapshot_unavailable.json`。
- real-only 六方法原始输入实验不伪造。
- 基于现有 reference projection 的实验标记为 `post_alignment_real_evidence`。
- 工作包 C–H 的仿真轨道继续。

### 5.5 验收门 G2a

- snapshot hash、样本范围和现有 feature-export 时间范围一致。
- 随机抽查每个 view 至少两个窗口的原始点数与现有 window manifest 相符。
- `git status` 不出现重型 bundle。

### 5.6 2026-07-10 执行结果

- G2a 正式 run 为 `docs/artifacts/runs/2026-07-10_dingxin-input-snapshot/`，状态 `completed`。
- 锁定时间范围内每个 sortie 有 28,824 个航电点；每个 view 有 905 个生理点，6/6 与既有 37 窗口 manifest 对账一致。
- snapshot 采用每 sortie 一份共享航电、每 view 一份生理的 5 文件布局，使用确定性 gzip JSONL 和 SHA-256。
- 5.3 MB 原始高频值只保存在被忽略目录；仓库只保留约 60 KB 紧凑证据。
- 20 个机动标签源字段在 snapshot 中全部可见，并全部进入原始/统计/差分/变化率/标准化副本排除合同。
- `--resume` 校验 5 个文件 hash 后在 2.47 秒内完成，没有重复数据库查询。

## 6. 工作包 C：模型无关仿真器

### 6.1 实现顺序

1. config/profile/contracts。
2. SemiMarkov/event plan。
3. G1 vehicle/workload/physiology。
4. G2 spline/event/kernel。
5. observation process。
6. oracle/task builder。
7. storage/resume。
8. validation/report。

CLI：

```text
scripts/simulation/generate_aviation_dual_stream.py
scripts/simulation/audit_aviation_dual_stream.py
```

### 6.2 smoke

先生成：

- G1/G2 各 2 条、60 秒。
- clean 与 mixed-severe 各一个观测版本。
- 不进入正式指标。

smoke 通过后生成 96/24/48 潜在架次和规范场景；正式压力 sweep 最后生成，避免前期浪费存储。

### 6.3 产物

- 本机重型：`artifacts/application_evaluation/<simulation_run_id>/`。
- 紧凑验证：`docs/artifacts/runs/2026-07-10_aviation-simulation-validation/`。

### 6.4 验收门 G2b

- synthetic spec 第 14–15 节全部测试通过。
- G1/G2 split、profile、seed 无交集。
- latent hash 在成对 observation 场景中一致。
- 生成器源码方法名称审计为零命中。
- 关键图件中文标签可读。

### 6.5 2026-07-10 执行结果

- smoke 生成 G1/G2 各 2 条、clean/mixed-severe 共 8 个场景，13/13 检查通过；4 张图经过两轮渲染和人工可读性抽查。
- 正式生成训练/验证/锁定测试 96/24/48 条潜在架次，每条 6 个规范观测版本，共 1,008 个场景，19/19 验收通过。
- 三个 split 的飞行员参数档案、latent seed 和生成族无交集；每个 split 均覆盖五种机动类型。
- 全局低/中/高仿真负荷占比 23.3%/43.0%/33.6%；每个锁定测试 profile 均有高负荷区间。
- G1 物理残差中位数最坏 0.0114，G2 残差 95% 分位最坏 0.0421；干净场景时延 ±1 秒命中率 100%。
- 正式重型 bundle 约 1,018 MB，只在被忽略目录；紧凑审计约 1.2 MB。
- `audit_aviation_dual_stream.py` 可在 3.13 秒内重建紧凑审计，不重新生成 bundle。

## 7. 工作包 D：统一表示基础设施

### 7.1 通用模块

实现：

- `DualStreamObservationBatch` 校验和 collator。
- `FusionStreamBatch` 及禁止字段检查。
- train-only normalizer/PCA registry。
- fold trainer、checkpoint registry、OOF exporter。
- representation manifest、fold status、progress/resume。
- sample-order/hash 校验器。

### 7.2 选择性复用 E3 分支

手工移植而非 cherry-pick：

- checkpoint/fold manifest 的 schema 思路。
- OOF 表示校验和 resume/skip-completed 机制。
- run observer heartbeat。

不移植：

- 候选优化器。
- E3 专用 pooled representation family。
- 任何旧 run artifact。

### 7.3 依赖

在 `pyproject.toml` 增加：

```toml
[project.optional-dependencies]
application-eval = ["aeon==1.5.0"]
structure-diagnostics = ["claspy>=0.2,<0.3", "stumpy>=1.14,<2"]
```

MiniRocket 是主实验依赖；ClaSP/STUMPY 保持可选 gated import。无需 `hmmlearn`。

### 7.4 验收门 G3a

- synthetic smoke batch 上六方法能够导出相同 sample/query 顺序和 64 维表示。
- OOF test sample 只能来自 held-out checkpoint。
- 禁止字段注入测试能拒绝 logits/labels/diagnostics。
- 中断一个 fold 后 `--resume` 能从下一个未完成 fold 继续。

### 7.5 2026-07-11 执行结果

- 新增 `chronaris.representation` 基础层，冻结 30 秒上下文、96 点查询轴和 64 维融合表示的张量、掩码与来源合同。
- 仿真加载器只接受独立观测归档；真值字段或额外字段注入直接失败。鼎新加载器从固定 snapshot 形成 12 个生理字段和 955 个跨架次同序航电字段，20 个机动标签源全部排除。
- 实现训练折中位数/四分位距归一化、无监督主成分投影、检查点注册表、严格表示序列化、留出折导出、覆盖检查和恢复复用。
- 公共训练增强 realization 只由样本标识、epoch 和随机种子派生，API 不接收方法名。
- 六个方法接口使用合同探针完成 6/6 输出与 6/6 恢复复用，样本标识、查询时间、有效掩码和留出折检查点来源完全一致；探针输出不作为模型结果。
- `aeon==1.5.0` 已进入 `application-eval` 可选依赖；统一合同正式冒烟验证 14/14 通过，相关聚焦测试 20 个通过。
- 紧凑 run 为 `docs/artifacts/runs/2026-07-11_representation-contract-smoke/`；探针检查点和稠密表示只保存在被忽略的 `artifacts/application_evaluation/`。

## 8. 工作包 E：六方法编码器

### 8.1 单流与朴素同步

- 生理/航电单流使用同一个连续时间 encoder 类，不复制两套实现。
- 朴素同步只使用过去/当前观测，train-only scaler/PCA。
- 输出统一投影到 64 维。

### 8.2 MulT/ContiFormer

- 从现有 deep wrapper 读取 pre-head sequence embedding。
- 将不同原生维度投影到 64。
- 增加独立时间/mask 适配器，保留原 wrapper logits 兼容性。
- 不使用旧回归任务 checkpoint 作为主表示初始化。

### 8.3 Chronaris

- 把 `DualStreamODERNNPrototype` 提升为任务无关连续融合编码器。
- 保留已有 `from_torch_alignment_batch` 兼容入口。
- 物理 loss 返回逐项 active/unavailable/count/value。
- `CausalFusionConfig` 新增 seconds-based lag ranges；旧 `lag_window_points` 保留 deprecated 读取，仅用于历史 artifact replay。
- 三尺度输出经过门控和投影成为 `[B,T,64]`。

### 8.4 训练器

- pretext 目标和升权计划按融合表示合同固定。
- augmentation realization 由 sample ID + epoch + seed 派生，六方法共享。
- candidate grid 只在配置层声明一次。
- 每个方法生成参数量、训练时间、峰值显存和每 epoch 样本吞吐。

### 8.5 验收门 G3b

- 路径测试证明 Chronaris 调用了 ODE-RNN、physics、seconds-based causal mask。
- 四个 Chronaris 消融各只关闭目标组件。
- 六方法相同数据/增强/预算审计通过。
- 不允许文件超过 800 行；接近 500 行时拆分。

### 8.6 2026-07-11 G3b.1 执行结果

- 实现公共因果查询层，按字段选择查询时刻之前最近一次真实观测，并保留字段有效性、模态可用性和观测年龄。
- 扩展 vendored ContiFormer 连续时间块的可选因果注意力；默认保持关闭以兼容历史 wrapper，两个新单流生产适配器显式启用。
- 生理单流和航电单流复用同一主干类；朴素时间同步只做当前/历史 forward-fill、训练折归一化和无监督主成分投影。
- 三个生产适配器都支持检查点保存/加载、留出折来源校验、严格 64 维表示导出和恢复复用。
- 仿真训练/验证/锁定轨迹与鼎新三个不同视图共完成 6 个导出，14/14 验收通过；恢复复核 6/6 复用。
- 未来观测扰动对当前及历史查询的最大变化为 0；非激活模态扰动对两个单流输出的最大变化为 0。
- 本里程碑没有公共自监督训练或下游任务指标；紧凑证据为 `docs/artifacts/runs/2026-07-11_shallow-baseline-adapter-smoke/`，检查点和稠密表示只保存在被忽略目录。

### 8.7 2026-07-11 G3b.2 执行结果

- MulT 与 ContiFormer 已从历史任务 wrapper 中分离为任务头前生产适配器，两者统一输出 96 点、64 维因果时序表示。
- MulT 双向跨模态注意力与后续自注意力均加入严格上三角屏蔽和 key padding mask；ContiFormer 生产路径显式使用 `causal=True`。
- 仿真与鼎新各完成两个方法的留出折导出，共 4 个输出；独立恢复复核 4/4 复用，15/15 验收通过。
- 未来观测扰动对当前及历史输出的最大变化为 0；生理和航电历史扰动均产生非零且有限的表示变化。
- 两个方法在同一数据集内共享训练折归一化拟合样本哈希，检查点清单明确记录未使用任务标签。
- 本里程碑没有公共自监督训练或下游任务指标；紧凑证据为 `docs/artifacts/runs/2026-07-11_deep-baseline-adapter-smoke/`，约 11 MB 检查点和稠密表示只保存在被忽略目录。

### 8.8 2026-07-11 G3b.3 执行结果

- 新增 `ChronarisContinuousFusionEncoder`，直接消费原始异步点，经两条 ObservationEncoder—ODE 演化—GRU 更新路径在 96 点公共查询轴读取连续潜态。
- ODE-RNN 输出新增路径 trace 和查询有效性；`enable_continuous_evolution=false` 会关闭真实 ODE 步骤，构成可审计消融而非报告字段变化。
- 新增按真实秒数定义的 0–5、5–15、15–30 秒互斥可见域、空尺度屏蔽和三尺度门控；无因果掩码消融使用对称时间可见域，单尺度消融固定为 0–30 秒。
- 物理项逐项记录 active/disabled/unavailable、有效残差对数量、原始值、加权值与原因；仿真/鼎新分别有 5/4 项可计算，不可用项不以零值冒充启用。
- 完整主干与四项消融均在仿真和鼎新完成前向；字段级配置 diff 只命中目标机制，无因果消融在两套数据上均出现非零未来反事实变化。
- 仿真与鼎新各完成一个 Chronaris 留出折导出，恢复复核 2/2 复用，21/21 验收通过；完整模型未来扰动最大历史变化为 0。
- 本里程碑没有公共自监督训练或下游任务指标；紧凑证据为 `docs/artifacts/runs/2026-07-11_chronaris-continuous-adapter-smoke/`，约 2.0 MB 检查点和稠密表示只保存在被忽略目录。

### 8.9 2026-07-11 G3b.4 执行结果

- 增强 realization 已从计划对象升级为真实执行器，支持模态删除、连续/随机缺失、时间抖动和时钟偏移；逐点 provenance 保证遮挡目标来自真实被移除查询来源。
- 五个可训练编码器统一使用 `CommonPretextHeadBundle`，公共目标权重固定为 1.0/0.5/0.2；无有效位置以 unavailable 处理，Chronaris 第 1 epoch 三个特有权重按协议保持 0。
- 通用训练 checkpoint 保存主干、公共头、optimizer、normalizer、fold、augmentation、输入与代码协议 hash；协议变化会拒绝错误 resume。
- 仿真 train split 选择 16 个不同 G1 profile 的 clean-asynchronous 轨迹，按 8/4/4 profile 隔离；五个方法各完成 1 epoch、2 step，30 条公共 loss 记录全部 active。
- 六个方法 train/validation/held-out 共导出 18 个表示并完成恢复复用；删除 Chronaris held-out 表示后仅重建该项且 hash 一致。
- 五个训练 checkpoint 完成后才读取 30–35 秒 workload 真值，固定 Logistic/Ridge 输出 72 条 smoke-only 指标；不读取 locked_test，不更新 confirmed metrics。
- 正式紧凑证据为 `docs/artifacts/runs/2026-07-11_common-pretraining-loop-smoke/`，20/20 验收通过；约 37 MB 重型产物只在被忽略目录。

## 9. 工作包 F：下游 consumer 与指标

### 9.1 实现

- Linear classification/regression probe。
- aeon MiniRocket transformer + 同一线性 estimator 网格。
- TCN emission model。
- duration-constrained Viterbi 与训练折 transition/duration estimator。
- real/synthetic 指标、fusion gain、stress slope。
- trajectory-level bootstrap/permutation。

CLI：

```text
scripts/evaluation/application_tasks/run_application_benchmark.py
```

主参数：

```text
--dataset-manifest
--representation-root
--tasks
--methods
--consumers
--split-protocol
--seeds
--run-id
--resume
```

### 9.2 验收门 G4

- 同一 representation 输入在重复 seed 下确定性一致。
- consumer 超参数网格按方法完全相同。
- TCN/Viterbi 不读取 oracle transition 以外的 test 标签。
- fusion gain 的高/低方向转换测试通过。
- metrics long 表能完整反查 fold、seed、method、consumer 和 checkpoint。

### 9.3 2026-07-11 G4.1 执行结果

- G4.1 没有直接复用只有单一 0–30 秒上下文的旧表示，而是从 16 条 G1 仿真训练轨迹分别提取 30/60/90/120 秒四个上下文，形成覆盖五类机动状态的 64 个样本和 32/16/16 profile 隔离划分。
- 六方法复用 G3b.4 的冻结 checkpoint，在新上下文上导出 train/validation/held-out 共 18 份 `[N,96,64]` 表示；真值只在五个可训练 checkpoint 与全部表示完成后打开。
- 固定线性探针、MiniROCKET 10,000 kernels、两层因果 TCN 和训练折持续时间解码已形成统一可恢复接口。MiniROCKET 对单窗口内恒定潜在维执行同一训练折方差规则；TCN 初始化、dropout 和优化步骤共用隔离随机流。
- 分类输出 macro-F1、balanced accuracy、macro-AUPRC、Brier 和 ECE；回归输出 MAE、RMSE 和 Spearman；分段输出 frame macro-F1、三档 segmental F1、两档 boundary F1、edit 和检测延迟。
- 六方法共产生 384 条可计算的 smoke-only 指标、256 条方向归一融合增益和 30 条以 4 条留出轨迹为独立单位的配对统计接口；这些结果不参与模型选择。
- 删除 Chronaris 留出表示、MiniROCKET 和 TCN 后分别只重建目标组件，未删除模型 SHA-256 保持不变，重建预测哈希一致；公共预训练 checkpoint 前后哈希不变。
- 紧凑证据位于 `docs/artifacts/runs/2026-07-11_application-consumer-smoke/`，12/12 验收通过；约 15 MB 表示、消费者模型和逐样本预测只在被忽略目录。
- G4 尚未整体完成。下一步先完成鼎新两项弱监督 target archive、固定 snapshot 上下文和真实外层折接入，再进入 G5 screen。

### 9.4 2026-07-11 G4.2 目标归档执行结果

- 机动强度弱监督分类直接继承 G1 五个外层折的训练折拟合阈值、fit sample hash 和 96 个上下文标签；20 个标签源航电字段继续全部标记为禁止输入。
- 生理响应没有沿用 G1 的窗口均值兼容值，而是从冻结生理原始点重新计算当前与未来 5 秒窗口中位数的绝对变化；每折只用可用训练上下文拟合字段覆盖、IQR 缩放和高响应四分位阈值。
- 原始 snapshot 在每条流约 181 秒结束，三个 view 的 `context_end_0035` 只能获得约 1 秒未来点。正式协议将它们标记为 `future_interval_not_fully_observed`，从 93 个审计候选中保留 90 个完整生理响应目标。
- 五折原始点中位数目标与 G1 窗口均值兼容目标的 Spearman 为 0.9349–0.9432；两者保持对照 lineage，但只使用原始点中位数目标进入后续正式实验。
- 五折两个任务共写出 10 个确定性 archive 和独立阈值文件；真实 resume 复用 10/10，重写 hash 稳定，五个 snapshot 文件前后 SHA-256 不变。
- 紧凑证据为 `docs/artifacts/runs/2026-07-11_dingxin-application-targets/`，12/12 验收通过；约 248 KB 目标 archive 位于被忽略目录。本 run 未训练模型或生成指标。
- 下一步为按 archive context 时间范围构造 30 秒防泄漏原始异步双流，并接入真实外层折表示训练。

### 9.5 2026-07-11 G4.2 原始上下文绑定执行结果

- 目标 catalog 的 96 个上下文中有 93 个严格满足 30 秒输入合同；三个 `context_end_0036` 只覆盖 155–180.991 秒，作为部分末窗不可用，不把名义 181–185 秒补造出来。
- 五折任务绑定中，机动分类有 93 个唯一可用输入，生理响应再叠加完整未来 5 秒要求后为 90 个；每个不可用原因都按 fold/task 保留。
- 原始输入 schema 为 12 个生理字段和 955 个跨架次同序航电字段。20 个机动标签源字段在 raw-to-index 映射阶段删除；93 个可用上下文的最大相对时间为 29.999 秒。
- 不生成全量稠密上下文 bundle。首次 Python `RawPoint` 缓存方案虽把运行缩短到约 70 秒，但峰值达到约 3.9 GB，已被替换为允许字段 CSR 缓存。
- 最终 CSR 缓存包含 10,255,756 个 float32 值、数组净大小 78.6 MB；完整 96 上下文审计耗时 12.34 秒、峰值 767 MB。缓存与直接 gzip 切片逐值一致。
- 10 个目标 archive、阈值文件和 5 个 snapshot 文件重新校验哈希；外层 train/test group 无交集，12/12 验收通过。
- 紧凑证据为 `docs/artifacts/runs/2026-07-11_dingxin-context-bindings/`；本 run 未训练模型或生成指标。下一步进入 outer-train 内部 validation 和真实五折公共预训练。

### 9.6 2026-07-11 G4.2 训练内验证划分执行结果

- 五个固定外层折都被展开为 inner-train、validation、overlap embargo 和 outer-test 四种互斥角色；93 个完整输入在每折恰好出现一次。
- 外层训练组含两个不同架次时，完整留出后一个训练架次作为 validation；只含同一架次时，以最后七个唯一时间块作为 validation，并把与其 30 秒原始窗口重叠的中间上下文放入 embargo。
- 五折 inner-train/validation 数量依次为 31/31、31/31、38/14、19/7、38/14；共享航电流上的训练/验证原始时间区间重叠数为 0。
- 每折分类的 inner-train、validation、outer-test 都覆盖低、中、高三类；生理响应三个角色都有有限连续目标和高/非高两类。
- 当前目标阈值仍由 outer-train 拟合，只允许后续固定配置 smoke 使用；正式 screen 必须先以 inner-train 生成嵌套目标 archive，不能复用这些阈值做候选选择。
- 紧凑证据为 `docs/artifacts/runs/2026-07-11_dingxin-inner-splits/`，11/11 验收通过；本 run 未训练模型、未读取 outer-test 指标，也未形成方法排名。

### 9.7 2026-07-11 G4.2 主协议首折公共预训练执行结果

- 归一化、公共预训练和 OOF 导出已支持按样本小批量读取固定 snapshot；精确中位数/四分位距只聚合 inner-train 的观察值，不生成整折稠密原始 bundle。
- 朴素同步在因果 forward-fill 后使用固定 seed 的随机化主成分分析（PCA），solver 与随机种子进入 checkpoint；缺失新元数据的旧 checkpoint 会显式重建。
- 留一视图第一折的 inner-train、validation、outer-test 各 31 个上下文。生理单流、航电单流、MulT、ContiFormer 和 Chronaris 各训练 1 epoch、31 step；朴素同步只拟合无监督变换。
- 五方法累计训练 213.63 秒，Chronaris 为 148.73 秒；完整成功链路峰值内存为 1967.4 MB，低于 2.5 GB 门限。
- 六方法共注册 6 个 checkpoint，并导出 train/validation/outer-test 共 18 份 `[N,96,64]` 表示；恢复复核 18/18 复用，同角色六方法样本、查询轴和 source hash 对齐。
- checkpoint solver 元数据缺失、重训后 registry hash 更新和非恢复模式下恢复校验三个问题均由安全门实际暴露并修复；最终 run 12/12 通过。
- 紧凑证据为 `docs/artifacts/runs/2026-07-11_dingxin-fold-pretraining-smoke/`；约 79 MB checkpoint 与表示位于被忽略目录。预训练未打开两项任务目标，outer-test 不生成指标或排名。

### 9.8 2026-07-11 G4.2 五折公共预训练与表示闭环

- 同一流式训练入口已扩展到三个留一视图主协议折和两个留一架次辅助折；按固定 split manifest 分别使用 31/31/38/19/38 个 inner-train 上下文，不改动统一 1 epoch 预算。
- 每折注册生理单流、航电单流、朴素时间同步、MulT、ContiFormer 和 Chronaris 共 6 个 checkpoint，并导出 train/validation/outer-test 各 6 份表示；五折合计 30 个 checkpoint 和 90 份 `[N,96,64]` 表示。
- 聚合审计逐项读取 90 份 archive 与 manifest，重验文件 SHA-256、sample/source hash、角色数量、checkpoint lineage 和 inner-train fit hash；每折第二遍恢复均复用 18/18。
- 五个子 run 合计 60/60 验收，聚合层 13/13；五折五方法累计训练 1066.45 秒，所有运行最高峰值内存 2047.1 MB，低于 2.5 GB 门限。
- 约 393 MB checkpoint 与稠密表示只位于被忽略目录；紧凑子 run 与聚合清单进入 `docs/artifacts/runs/`。全部预训练保持任务目标关闭、outer-test 指标关闭，未形成方法排名。
- 聚合证据为 `docs/artifacts/runs/2026-07-11_dingxin-five-fold-pretraining/`。下一步使用这 90 份冻结表示接入统一线性与 MiniROCKET 工程冒烟，并在正式 screen 前重建 inner-train 嵌套目标。

### 9.9 2026-07-11 G4.2 五折固定 consumer 工程冒烟

- 五个固定外层折的 90 份冻结表示已接入方法不变的线性模型与 MiniROCKET 10,000 kernels；分类 C、回归 alpha、随机 seed 和训练折方差过滤合同对六方法完全一致。
- 每个方法—折组合同时拟合机动强度三分类、生理响应连续回归和高生理响应二分类，形成 30 个 bundle、60 个消费者组件；组件第二遍恢复 60/60，预测文件哈希 30/30 一致。
- 时间 embargo 后，五折角色清单累计使用 440 个机动分类上下文和 425 个生理响应上下文；未来区间不足的三个末端样本继续不可用，没有补零或复制目标。
- validation 与 outer-test 共生成 1680 条 smoke-only 指标，全部可计算；方向归一双流增益接口生成 1120 条。首次 consumer 拟合累计 144.34 秒。
- 约 36 MB 模型与逐样本预测位于被忽略目录；紧凑指标、协议、资源、增益和验收位于 `docs/artifacts/runs/2026-07-11_dingxin-consumer-smoke/`，15/15 通过。
- 当前指标仍使用 outer-train 目标阈值，只验证真实链路、指标方向和恢复，不进入候选选择或论文确认表。下一步必须按 inner-train 重拟合所有目标参数。

### 9.10 2026-07-11 G4.2 inner-train 嵌套目标执行结果

- 机动分类不复用 outer-train score，而是从既有对齐窗口的原始航电统计重新拟合 inner-train 语义中位数/IQR、分位边界，再应用到 validation/outer-test。
- 生理响应从冻结 snapshot 重新计算当前/未来 5 秒原始点中位数差，并只用 inner-train 选择字段、拟合字段 IQR 和高响应四分位阈值。
- 五折共生成 10 个确定性 archive；机动分类 440 个角色上下文，生理响应 425 个可用角色上下文。两轮完整重建哈希一致，snapshot 文件哈希不变。
- 相对 outer-train 工程冒烟目标，机动类别改变 75/440，高响应标签改变 51/425；连续响应分数 Spearman 为 0.9787–0.9971，说明嵌套尺度重拟合产生了实质影响。
- 三个时间块 validation 只覆盖中/高机动类。该分布漂移按真实结果保留，不通过移动阈值或合并类别补齐；后续分类指标固定三类标签集合。
- 紧凑证据为 `docs/artifacts/runs/2026-07-11_dingxin-nested-targets/`，重型 archive 约 108 KB，10/10 通过；本 run 不训练模型、生成指标或形成排名。

## 10. 运行与收口入口

工作包 G（开发筛选、锁定训练、压力测试和消融）、工作包 H（附录诊断、论文证据包）、恢复策略、失败处理和最终验证拆分到：

- [fixed-data-downstream-evaluation-runbook-2026-07-10.md](fixed-data-downstream-evaluation-runbook-2026-07-10.md)

拆分后本文件保持“实现哪些能力”的主计划，runbook 负责“如何长程运行并收口”。
