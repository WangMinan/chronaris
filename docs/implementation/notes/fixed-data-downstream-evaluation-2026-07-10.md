# 固定数据下游评估与完整论文实验实施计划

日期：2026-07-10  
分支：`codex/fixed-data-downstream-evaluation-20260710`  
状态：长程 goal 已启动；当前处于详细规格与实施入口冻结阶段

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

## 10. 运行与收口入口

工作包 G（开发筛选、锁定训练、压力测试和消融）、工作包 H（附录诊断、论文证据包）、恢复策略、失败处理和最终验证拆分到：

- [fixed-data-downstream-evaluation-runbook-2026-07-10.md](fixed-data-downstream-evaluation-runbook-2026-07-10.md)

拆分后本文件保持“实现哪些能力”的主计划，runbook 负责“如何长程运行并收口”。
