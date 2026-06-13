# Chronaris 当前任务

更新时间：2026-06-07

## 文档定位

本文件是当前唯一主动执行入口。总路线、阶段结构、当前任务队列和默认工作方式统一维护在这里；历史计划与阶段笔记保留在 `notes/`。

## 总路线

1. 读取指定架次的人机多源数据及元信息。
2. 建立统一 schema、统一时间参考和统一样本组织。
3. 实现双流连续潜态建模。
4. 实现物理一致性约束时间对齐。
5. 实现因果掩码跨模态融合。
6. 输出标准化融合特征与中间态接口。
7. 面向典型任务开展对比、消融和案例验证。

## 阶段结构

- `Stage A/B/C`：已完成。
- `Stage D`：后置，保留为后续数据集工程化工作。
- `Stage E0/E/F/G(min)/H`：已完成并收口，作为历史基线与后续依赖。
- `Stage I`：
  - `Phase A/B/C`：已接上统一骨干、真实 weak-label 联合训练、private/thesis 分层资产和中期证据包。
  - `Phase D`：刚体运动物理约束补强已完成二轮真实 smoke / ablation，`translation + vertical` 已真实启用。
  - `Phase E`：语义事件融合补强已完成多 view support 产物，覆盖当前 Stage H `validation` profile 的 3 个双流 view。
  - `Phase F`：runtime inference 已完成服务化 replay 补强，支持 `batch / incremental / both` 和 schema diagnostics。

## 默认工作方式

每轮实现默认按下面顺序收敛：

1. 目标锁定。
2. 代码实现。
3. 测试闭环。
4. 文档回写。
5. 冗余清理。

运行 Python 脚本、测试、基准或阶段命令前，默认使用：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python
```

## 当前工作区与推送状态

当前事实：

- `main` 当前基线 `HEAD=57ca739 feat: expand stage i rigid-body support and runtime service`。
- `origin/main=57ca739`，本地 Phase D/E/F 代码、文档与资产已经推送到远端。
- 本轮文档重组前工作区干净；本轮会产生 `docs/STATE.md`、`docs/implementation/TASKS.md` 以及必要索引文档的未提交修改。
- 已进入 git 历史的最新 Stage I 真实 replay/support/ablation 资产包括：
  - `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/`
  - `docs/artifacts/assets/stage_i_semantic_event_support/20260607T-stage-i-semantic-support-r2/`
  - `docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/`
  - `docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/`
  - `docs/artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md`
  - `docs/artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md`
  - `docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md`

收敛顺序：

1. 先完成本轮 `STATE.md`、`TASKS.md` 和必要产物索引的事实同步。
2. 再进入中期前主动任务队列：先统一 runner，再做论文本体深挖、private proxy 机制诊断、public adapter 有界校准、公开迁移边界表和 `rotation` 字段核验。
3. 每个新增实验都必须明确 `evidence_layer`，不能把 public adapter、private proxy、thesis weak-label 混写。
4. 推送前复查 `git status --short --untracked-files=all`，确保没有临时 stdout、半成品资产或大文件误入。

验收：

- `git diff --check` 通过。
- 状态文档中的 `HEAD`、远端同步状态、当前任务队列和编码缺口一致。
- 新增报告路径和 asset 路径必须能从 `ARTIFACTS.md` 或本文件追溯。

## 已完成 P0：冻结 Phase D/E/F 工作区并提交

结果：`Phase D/E/F` 主代码、测试与 runtime sample exporter 已经进入本地 git 历史。

本轮复查范围：

- 复查 `git status --short --untracked-files=all` 中全部未提交项，区分三类改动：
  - `Phase D/E/F` 代码、测试和文档同步。
  - 为 runtime checkpoint feature schema 做的必要兼容改动。
  - 与本阶段无关的临时改动或新产物。
- 保留并复查 `Phase D rigid_body` 关键文件：
  - `src/chronaris/models/alignment/physics_state_mapping.py`
  - `src/chronaris/models/alignment/physics_residuals.py`
  - `src/chronaris/models/alignment/physics.py`
  - `src/chronaris/models/alignment/physics_features.py`
  - `src/chronaris/models/alignment/__init__.py`
  - `src/chronaris/pipelines/alignment_preview.py`
  - `scripts/run_stage_e_relative_preview.py`
  - `tests/test_alignment_model_losses.py`
- 保留并复查 `Phase E semantic event fusion` 关键文件：
  - `src/chronaris/models/fusion/semantic_event.py`
  - `src/chronaris/models/fusion/__init__.py`
  - `src/chronaris/pipelines/causal_fusion.py`
  - `src/chronaris/pipelines/stage_i/stage_i_support_builders.py`
  - `src/chronaris/pipelines/stage_i/stage_i_support_reporting.py`
  - `tests/test_stage_i_support.py`
- 保留并复查 `Phase F runtime inference` 关键文件：
  - `src/chronaris/dataset/streaming_windows.py`
  - `src/chronaris/dataset/__init__.py`
  - `src/chronaris/serving/runtime_inference.py`
  - `src/chronaris/serving/__init__.py`
  - `scripts/run_stage_i_runtime_inference.py`
  - `src/chronaris/pipelines/stage_i/stage_i_multitask_train.py`
  - `tests/test_runtime_inference.py`
- 确认新增代码默认输出路径统一落到 `docs/artifacts/assets/...`，Markdown 报告统一落到 `docs/artifacts/stage_i/...` 或既有阶段报告目录。
- 本轮补跑并保留结果，完整覆盖 torch runtime 用例时需要显式开启测试开关：

```bash
CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 \
  /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest \
  tests.test_alignment_model_losses \
  tests.test_alignment_pipeline \
  tests.test_stage_i_support \
  tests.test_stage_i_multitask_train \
  tests.test_runtime_inference
```

当前最近一次结果：`Ran 33 tests in 7.962s`，`OK`。

本轮提交：

- `0f4db72 feat: add stage i runtime sample exporter`
- `9ef4f64 feat: add rigid-body physics semantic event runtime inference`
- `890a315 docs: record stage i runtime semantic rigid-body artifacts`
- `57ca739 feat: expand stage i rigid-body support and runtime service`

## 已完成 P1：补真实资产上的 Stage I multitask 联合训练证据

结果：Phase C 已经不再停留在合成烟测，而是在真实 Stage H all-window clean 资产上形成了可引用证据。

前置编码任务：

- 已新增 `scripts/run_stage_i_multitask_train.py`。
- 已复用 `collect_stage_i_backbone_samples()` 路线并补 `view_id::raw_window_sample_id` sample-id contract，生成真实 `E0ExperimentSample`。
- 已复用 `load_aligned_private_records()` 与 `build_stage_i_real_task_payload()` 构造 weak-label task entries。
- 已修正 workload proxy 的归一化尺度，并把 event replay pair 调整为优先近邻配对，避免验证/测试分区丢失 retrieval 监督。
- 默认输出已切到 `docs/artifacts/assets/stage_i_multitask/<run_id>/`。

建议真实资产入口：

- `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json`
- `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json`

本轮输出：

- `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_checkpoint.pt`
- `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_summary.json`
- `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`
- `docs/artifacts/stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md`

报告边界：

- 已明确写成 `thesis weak-label evidence`。
- 已保持“不写成人工真值任务”的边界。
- 当前产物按 `mainline closure evidence` 使用，不宣称人工真值最优。

## 已完成 P2：刷新 private benchmark 分层资产

结果：已用当前 Phase C 代码重跑 private benchmark，使历史 `T1/T2/T3` proxy evidence 与 thesis weak-label evidence 在资产层明确拆开。

建议命令形态：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_private_benchmark.py \
  --e-run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json \
  --f-run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json \
  --output-root docs/artifacts/assets/stage_i_private \
  --report-root docs/artifacts \
  --enable-optimized-chronaris \
  --export-optimized-package
```

本轮验收：

- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_proxy_task_manifest.jsonl`。
- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_proxy_task_summary.json`。
- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/thesis_task_manifest.jsonl`。
- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/thesis_task_summary.json`。
- `private_benchmark_summary.json` 已包含 `evidence_layers.proxy_evidence` 和 `evidence_layers.thesis_task_evidence`。
- CLI 已打印 thesis task manifest / summary 路径。

## 已完成编码 P3：补 Stage F 刚体运动物理约束

结果：`rigid_body` physics family 已进入工作区，并保持旧 `minimal / full` 路线不回归。

实际落点：

- `src/chronaris/models/alignment/physics_state_mapping.py`
- `src/chronaris/models/alignment/physics_residuals.py`
- `src/chronaris/models/alignment/physics.py`
- `src/chronaris/models/alignment/physics_features.py`
- `src/chronaris/pipelines/alignment_preview.py`
- `scripts/run_stage_e_relative_preview.py`
- `tests/test_alignment_model_losses.py`

本轮验收：

- `rigid_body` family 可被显式选择。
- 已新增刚体状态映射与残差模块，能诊断启用/缺失项。
- `tests.test_alignment_model_losses` 已覆盖 family 选择、缺失诊断和垂直/姿态残差。

## 已完成编码 P4：补 Stage G 语义事件融合

结果：`G(min)` 之上已新增 `SemanticQueryBank / EventTokenExtractor / CausalEventFusion`，并把事件级归因摘要接到 support 报告链路。

实际落点：

- `src/chronaris/models/fusion/semantic_event.py`
- `src/chronaris/models/fusion/__init__.py`
- `src/chronaris/pipelines/causal_fusion.py`
- `src/chronaris/pipelines/stage_i/stage_i_support_builders.py`
- `src/chronaris/pipelines/stage_i/stage_i_support_reporting.py`
- `tests/test_stage_i_support.py`

本轮验收：

- 已输出 event token。
- 已输出 query-to-event attention。
- 已输出事件级归因摘要。
- `tests.test_stage_i_support` 已验证 support 报告能区分“时间步注意力”和“事件级归因”。

## 已完成编码 P5：补 runtime inference

结果：已新增真正的 checkpoint-backed runtime inference 入口，支持样本序列化、本地 replay 输入和流式窗口缓存。

实际落点：

- `src/chronaris/dataset/streaming_windows.py`
- `src/chronaris/dataset/__init__.py`
- `src/chronaris/serving/runtime_inference.py`
- `src/chronaris/serving/__init__.py`
- `scripts/run_stage_i_runtime_inference.py`
- `src/chronaris/pipelines/stage_i/stage_i_multitask_train.py`
- `tests/test_runtime_inference.py`

本轮验收：

- mock stream 或本地回放流可以增量产出窗口。
- 可以加载 checkpoint 做风险/负荷/事件预测。
- 可以输出 attention / semantic event attribution 解释。
- `tests.test_runtime_inference` 已覆盖窗口缓存、样本序列化和端到端推理闭环。

## 已完成主体 P6：把 Phase D/E/F 从代码完成推进到产物闭环

结果：runtime replay、语义事件融合 support、刚体约束 smoke / ablation 都已经落盘；并且在切到本地 `127.0.0.1:3306` MySQL 后，`rigid_body` 已经不再只是 fallback evidence，`vehicle_rigid_body_translation` 已经在真实链路上启用。

本轮完成：

- 已补 runtime sample exporter：
  - `scripts/export_stage_i_runtime_samples.py`
- 已完成 runtime replay：
  - `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-replay-r1/runtime_samples.jsonl`
  - `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-replay-r1/runtime_inference_summary.json`
  - `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-replay-r1/runtime_inference_predictions.csv`
  - `docs/artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-replay-r1.md`
  - 当前 replay 规模：`111` 个样本、`3` 个 view、`2` 个 sortie，`sample_id_mode=view_prefixed`。
  - 当前 runtime task heads：`risk_proxy` 分类、`workload_proxy` 回归、`event_replay_tag` 检索。
- 已完成语义事件融合 preview + support：
  - `docs/artifacts/stage_i/stage-i-semantic-event-20260607T-stage-i-semantic-event-r1.md`
  - `docs/artifacts/stage_i/assets/stage-i-semantic-event-20260607T-stage-i-semantic-event-r1/causal_fusion_summary.json`
  - `docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r1/support_summary.json`
  - `docs/artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r1.md`
  - `support_summary.json` 已包含 `alignment_support`、`causal_support.semantic_event`、`main_ablation_rows` 和 overview plot。
- 已完成 `minimal / full / rigid_body` 真实 smoke / ablation：
  - `docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r1/rigid_body_ablation_summary.json`
  - `docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r1.md`
  - 本次更新后 `vehicle_field_metadata.status=loaded`、`field_count=96`。
  - `rigid_body` 的 `vehicle_rigid_body_translation=1.413177490234375`，`vehicle_rigid_body_vertical=0`，`vehicle_rigid_body_rotation=0`。

## 中期前主动推进顺序

1. 先建立统一 evidence runner，避免后续五条线靠手工命令散跑。
2. 深挖现有私有 Stage H 数据与论文本体模型：围绕 `risk_proxy / workload_proxy / event_replay_tag` 做小网格和消融表。
3. 拆解 `chronaris_opt` 机制贡献：把它从“private proxy 最优结果”推进成“表示学习、对齐、因果掩码贡献可诊断”。
4. 有界补做 CPU-heavy `sklearn` / UAB torch 候选：只作为 public adapter baseline / calibration baseline，不扩写成本体闭环。
5. 整理 NASA/UAB 公开适配器迁移边界：用表格说明公开代理数据与私有双流数据的模态、标签、任务粒度和时间基准差异。
6. 核验 `rigid_body rotation` 字段：能找到成对角速度字段就补 rotation ablation；找不到则固化为真实字段缺口诊断。

## 已完成 P7：扩充 rigid_body 的字段语义覆盖

结果：`rigid_body` 已经不再只启用 `translation`，本轮通过真实 MySQL label + token 扩充，`vertical` 也已经在真实链路上启用。

本轮完成：

- `vehicle_field_metadata` 已确认可通过本地 `127.0.0.1:3306` 正常加载。
- `BUS6000019110020` 当前加载字段数为 `96`。
- 已扩充中文 token，并新增刚体字段映射诊断导出。
- 新的 `rigid_body` 真实 run：

```bash
CHRONARIS_MYSQL_HOST=127.0.0.1 CHRONARIS_MYSQL_PORT=3306 \
CHRONARIS_MYSQL_USER=wangminan CHRONARIS_MYSQL_PASSWORD=... \
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_e_relative_preview.py \
  --enable-physics-constraints \
  --physics-constraint-family rigid_body \
  --input-normalization-mode zscore_train \
  --epoch-count 1 \
  --batch-size 8 \
  --strict-mysql-field-labels \
  --device cpu \
  --report-path docs/artifacts/stage_i/stage-i-rigid-body-rigid-body-20260607T-stage-i-rigid-body-r2.md
```

本轮验收：

- `vehicle_field_metadata.status=loaded`
- `enabled_residuals=['translation','vertical']`
- `vehicle_rigid_body_translation=1.133332371711731`
- `vehicle_rigid_body_vertical=3.9466116428375244`
- 汇总：`docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/rigid_body_ablation_summary.json`
- 报告：`docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md`

## 已完成 P8：放大 semantic event support 证据

结果：已经从单 preview summary 扩到当前 Stage H `validation` profile 的 3 个双流 view，并生成了 view-level semantic ranking。

本轮完成：

- 新增 runner：`scripts/run_stage_i_semantic_event_support.py`
- 新增多 view summary：
  - `docs/artifacts/assets/stage_i_semantic_event_support/20260607T-stage-i-semantic-support-r2/semantic_event_support_summary.json`
- 新增 support 聚合：
  - `docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`
- 新增报告：
  - `docs/artifacts/stage_i/stage-i-semantic-event-support-20260607T-stage-i-semantic-support-r2.md`
  - `docs/artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md`

本轮验收：

- 覆盖 `3` 个 view、`111` 个样本。
- `support_summary.json` 已包含 `causal_support.semantic_event.view_rows`。
- support 报告已经能回答：
  - top view：`20251005_四01_ACT-4_云_J20_22#01__pilot_10033`
  - dominant query：`risk_proxy`
  - top offset：`0.626197s`

## 已完成 P9：runtime inference 服务化补强

结果：runtime inference 现在支持 batch / incremental / both 三种 replay 模式、schema 对齐诊断和 JSONL 导出。

本轮完成：

- `StreamingWindowBuffer` 已支持：
  - `max_cached_points_per_stream`
  - `allow_out_of_order`
  - `diagnostics`
- `runtime_inference.py` 已支持：
  - `replay_mode=batch|incremental|both`
  - `batch_size`
  - `max_windows`
  - `strict_feature_schema`
  - `input_normalization_stats` 作为旧 checkpoint 的 schema fallback
  - latency / throughput / chunk_count / feature_schema_status diagnostics
- CLI 已支持：
  - `--max-windows`
  - `--batch-size`
  - `--emit-jsonl`
  - `--strict-feature-schema`
  - `--replay-mode`
- 新产物：
  - `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/runtime_inference_summary.json`
  - `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/runtime_inference_predictions.jsonl`
  - `docs/artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md`

本轮验收：

- `replay_mode=both`
- `batch_sample_count=40`
- `incremental_sample_count=40`
- `sample_count_match=True`
- `feature_schema_status=aligned`
- `feature_schema_source=input_normalization_stats`

## 已完成 P10：统一论文闭环评测 harness

目标：把 Phase C/D/E/F 以及中期前新增的五条证据线整合成一个可重复跑的 evidence runner，减少后续论文补图、补表时的手工步骤。

建议新增：

- `src/chronaris/pipelines/stage_i/stage_i_evidence_runner.py`
- `scripts/run_stage_i_evidence_closure.py`
- `tests/test_stage_i_evidence_runner.py`

职责：

- 串联 Stage I multitask checkpoint、rigid_body ablation、semantic support、runtime replay、private proxy diagnostics、public adapter calibration summary。
- 统一 run id 规则、artifact root、report root 和 manifest 输出。
- 生成 `evidence_manifest.json`，记录输入、输出、命令参数、git commit、测试结果摘要和 `evidence_layer`。
- 支持 `--skip-heavy`、`--reuse-existing`、`--only multitask|rigid_body|semantic|runtime|private_proxy|public_adapter|rotation|all`。
- 对 heavy public adapter 分支默认只登记已有产物；只有显式打开 `--run-heavy-public-adapter` 才真实执行。

本轮结果：

- 统一入口：
  - `src/chronaris/pipelines/stage_i/stage_i_evidence_runner.py`
  - `scripts/run_stage_i_evidence_closure.py`
  - `tests/test_stage_i_evidence_runner.py`
- 稳定 manifest：
  - `docs/artifacts/assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/evidence_manifest.json`
- 稳定报告：
  - `docs/artifacts/stage_i/stage-i-evidence-closure-20260607T-stage-i-evidence-closure-r2.md`
- 当前 `r2` 已纳入：
  - `multitask / rigid_body / semantic / runtime / private_proxy / public_adapter / rotation`
- `skip-heavy` 已切到 bounded 路线：
  - `multitask` 使用 `stage_h_window_stats_proxy`
  - `rigid_body / semantic / runtime` 复用稳定资产

## 已完成 P11：私有 Stage H 论文本体模型深挖

目标：围绕现有私有 Stage H 双流数据和 Phase C multitask checkpoint，把 `risk_proxy / workload_proxy / event_replay_tag` 做成更充分的 thesis weak-label evidence，而不是只保留一轮训练结果。

建议新增或扩展：

- `src/chronaris/pipelines/stage_i/stage_i_multitask_sweep.py`
- `scripts/run_stage_i_multitask_sweep.py`
- `tests/test_stage_i_multitask_sweep.py`

实验设计：

- 小网格，不做无边界扩搜：
  - `physics_constraint_family=minimal|full|rigid_body`
  - `causal_weight=0|0.05|0.1`
  - `task_loss_weight=0.5|1.0`
  - `causal_lag_window_points=None|3`
- 固定输入：
  - `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json`
  - `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json`
- 每个 run 输出 `multitask_summary.json`、`thesis_task_manifest.jsonl`、`checkpoint_metadata`、训练/验证/测试指标。

本轮结果：

- 新增：
  - `src/chronaris/pipelines/stage_i/stage_i_multitask_sweep.py`
  - `scripts/run_stage_i_multitask_sweep.py`
  - `tests/test_stage_i_multitask_sweep.py`
- 稳定产物：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/multitask_sweep_summary.json`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/thesis_weak_label_multitask_ablation.csv`
  - `docs/artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260607T-stage-i-evidence-closure-r2-multitask.md`
- 当前 `skip-heavy` 路线说明：
  - `sample_source=stage_h_window_stats_proxy`
  - 仍明确写成 `thesis weak-label evidence`
  - 不包装成人工真值任务

## 已完成 P12：`chronaris_opt` 机制诊断

代码落点：

- `src/chronaris/pipelines/stage_i/stage_i_private_optimization.py`
- `src/chronaris/pipelines/stage_i/stage_i_private_benchmark.py`
- `scripts/run_stage_i_private_benchmark.py`
- `tests/test_stage_i_private_optimization.py`

目标：

- 把 `chronaris_opt` 从“private proxy benchmark 最优候选”推进成“机制贡献可诊断”的证据。
- 在 `T1/T2/T3` 上拆解：
  - 去掉因果掩码。
  - 去掉时间残差或 lag-aware residual。
  - 去掉 task-aware head。
  - 仅保留 E/F/G/H 既有 baseline。
- 输出 `chronaris_opt_component_ablation.csv/json` 和中文机制诊断报告。

本轮结果：

- 新增：
  - `src/chronaris/pipelines/stage_i/stage_i_private_component_ablation.py`
  - `scripts/run_stage_i_private_component_ablation.py`
  - `tests/test_stage_i_private_component_ablation.py`
- 稳定产物：
  - `docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/chronaris_opt_component_ablation.json`
  - `docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/chronaris_opt_component_ablation.csv`
  - `docs/artifacts/stage_i/stage-i-private-component-ablation-20260607T-stage-i-evidence-closure-r2-private-proxy.md`
- 当前报告已拆出：
  - `remove_causal_mask`
  - `remove_time_residual`
  - `remove_task_aware_head`

## 已完成 P13：public adapter 有界校准 baseline

目标：中期前补做有限预算的 CPU-heavy `sklearn` / UAB torch 候选，用于公开代理数据的 adapter baseline / calibration baseline，而不是改写成双流本体闭环。

建议范围：

- UAB：
  - 默认仍以 torch heat-specialist / robust-prior 公开线为主。
  - 只允许 3 到 5 个候选组合，固定 seed、固定 LOSO、固定 `selected_subset`。
  - CPU-heavy `sklearn uab_hybrid` 必须显式 `--allow-cpu-heavy-sklearn`，并在 manifest 写入 `heavy_reason` 与 runtime。
- NASA：
  - 继续以 `NASA enhanced round 1` 为主。
  - 只补必要的 calibration 对照，不扩成新的深度模型竞赛。

建议落点：

- 扩展 `scripts/run_stage_i_public_opt.py` 的 run manifest metadata。
- 新增 `src/chronaris/pipelines/stage_i/stage_i_public_adapter_calibration.py` 或复用 public mainline report builder。
- 新增 `tests/test_stage_i_public_opt.py` 中的 evidence layer / heavy guard 回归。

本轮结果：

- 新增：
  - `src/chronaris/pipelines/stage_i/stage_i_public_adapter_calibration.py`
  - `scripts/run_stage_i_public_adapter_calibration.py`
  - `tests/test_stage_i_public_transfer_boundary.py`
- 稳定产物：
  - `docs/artifacts/assets/stage_i_public_adapter_calibration/20260607T-stage-i-evidence-closure-r2-public-adapter/public_adapter_calibration_summary.json`
  - `docs/artifacts/stage_i/stage-i-public-adapter-calibration-20260607T-stage-i-evidence-closure-r2-public-adapter.md`
- 当前 summary 已区分：
  - `public_adapter_baseline`
  - `calibration_baseline`
  - `legacy_public_opt`
  - `torch_uab`

## 已完成 P14：NASA/UAB 迁移与校准边界报告

目标：把 NASA/UAB 公开适配器结果整理成论文可引用的“迁移边界”证据，说明公开代理数据和私有真实双流数据之间的差异。

建议新增：

- `src/chronaris/pipelines/stage_i/stage_i_public_transfer_boundary.py`
- `scripts/build_stage_i_public_transfer_boundary.py`
- `tests/test_stage_i_public_transfer_boundary.py`

报告内容：

- 数据边界表：
  - 私有 Stage H：真实生理流 + 真实飞机时序流 + sortie/pilot/view。
  - UAB：公开 physiology + task/context proxy + subjective workload。
  - NASA：公开 physiology + scenario/context proxy + attention state。
- 任务边界表：
  - `risk_proxy / workload_proxy / event_replay_tag` 属于 thesis weak-label。
  - `T1/T2/T3` 属于 private proxy。
  - UAB/NASA 属于 public adapter / calibration evidence。
- 性能表：
  - 引用 `stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md`。
  - 引用新的 P13 calibration summary。

本轮结果：

- 新增：
  - `src/chronaris/pipelines/stage_i/stage_i_public_transfer_boundary.py`
  - `scripts/build_stage_i_public_transfer_boundary.py`
- 稳定产物：
  - `docs/artifacts/assets/stage_i_public_transfer_boundary/20260607T-stage-i-evidence-closure-r2-transfer-boundary/public_transfer_boundary_summary.json`
  - `docs/artifacts/stage_i/stage-i-public-transfer-boundary-20260607T-stage-i-evidence-closure-r2-transfer-boundary.md`
- 当前报告已覆盖：
  - 数据边界表
  - 任务边界表
  - public adapter/calibration 性能引用表

## 已完成 P15：`rigid_body rotation` 字段核验与消融

目标：中期前补完 `rigid_body` 的最后一个关键缺口：`rotation`。能找到真实成对角速度字段就补实验；找不到就把缺口固化成可引用诊断。

建议落点：

- `src/chronaris/models/alignment/physics_state_mapping.py`
- `scripts/run_stage_e_relative_preview.py`
- `tests/test_alignment_model_losses.py`

核验顺序：

1. 从 MySQL label 和 Stage H feature schema 中复查 `真航向`、俯仰、横滚、角速度、航向角速度等字段。
2. 若存在可用 rate field，扩充 token 并重跑 `minimal / full / rigid_body`。
3. 若只存在角度、没有 rate field，保留 `rotation` disabled，并输出 `missing_requirements.rotation` 诊断。

本轮结果：

- 代码与测试：
  - `src/chronaris/models/alignment/physics_state_mapping.py`
  - `src/chronaris/pipelines/stage_i/stage_i_rigid_body_rotation_audit.py`
  - `scripts/run_stage_i_rigid_body_rotation_audit.py`
  - `tests/test_alignment_model_losses.py`
  - `tests/test_stage_i_rotation_audit.py`
- 稳定产物：
  - `docs/artifacts/assets/stage_i_rotation_audit/20260607T-stage-i-rotation-audit-r2/rigid_body_rotation_audit_summary.json`
  - `docs/artifacts/stage_i/stage-i-rigid-body-rotation-audit-20260607T-stage-i-rotation-audit-r2.md`
- 当前结论：
  - `BUS6000019110020.code1031 = 真航向` 已归入 `yaw`
  - `yaw_rate` 仍缺失
  - `rotation_status=disabled`

## 下一步 P16：论文案例与消融表稳定化

目标：把现有和 P11-P15 新增 evidence 转成论文可直接引用的表格与案例材料，减少后期靠手工复制指标。

代码落点：

- `src/chronaris/evaluation/` 或 `src/chronaris/pipelines/stage_i/`
  - 新增 report table builder：统一导出 thesis weak-label ablation、private proxy component ablation、public adapter calibration、transfer boundary、rigid_body rotation、runtime prediction examples、semantic event attribution cases。
  - 对每张表附带 `evidence_layer`、`source_path`、`metric_definition`。
- `docs/artifacts/stage_i/`
  - 输出论文案例报告，保留中文解释、边界说明和引用路径。

验收：

- 至少生成六张稳定表：论文本体 weak-label 小网格、`chronaris_opt` 组件诊断、public adapter calibration、公开迁移边界、物理约束消融、runtime/semantic case。
- 每张表都能追溯到 JSON/CSV 原始产物。
- 表述不越界：public adapter、private proxy、thesis weak-label、case support 分层清楚。

## 下一步 P17：系统封装与部署边界

目标：为毕业设计系统实现章节补齐“离线/准实时推理服务”的工程闭环，而不是只停留在训练脚本和报告。

建议落点：

- `src/chronaris/serving/`
  - 增加纯 Python service facade：加载 checkpoint、接收 window payload、返回 response schema。
  - 明确配置对象：checkpoint path、feature schema、window policy、device、diagnostics mode。
- `scripts/serve_stage_i_runtime.py` 或 `scripts/run_stage_i_runtime_smoke.py`
  - 如果暂不做常驻服务，先做本地 smoke CLI，输入 JSONL，输出 predictions JSONL。
- `docs/implementation/`
  - 补系统封装说明，明确不是重建上游接收器，也不把原始大文件入仓。

验收：

- 冷启动加载 checkpoint 成功。
- 输入 1 个 view 的 replay JSONL，输出 predictions JSONL 与 summary JSON。
- 有错误样例：缺 checkpoint、缺字段、空窗口、schema mismatch。
- 与 `tests.test_runtime_inference` 形成自动化覆盖。

## 中期前边界管理

下面几类工作现在纳入中期前主动任务，但必须按证据分层写清楚，不能因为加做实验就改变论文边界。

- CPU-heavy `sklearn` 或 UAB torch 候选搜索：中期前允许有限预算复现/补跑；只能作为公开 adapter baseline / calibration baseline。
- NASA/UAB 公开数据适配器结果：中期前要整理成 public adapter evidence 和 transfer boundary；不能改写成论文双流本体闭环。
- `chronaris_opt` 与 `T1/T2/T3`：中期前要补机制诊断；仍只能写成 private proxy benchmark evidence，不能写成人工真值 thesis task fully closed。
- `risk_proxy / workload_proxy / event_replay_tag`：中期前要补小网格与消融；仍只能写成 thesis weak-label evidence。
- `rigid_body rotation`：中期前必须核验字段；启用或缺失都要以 diagnostics 形式固化。
- 上游接收器、入库链路和原始大文件入仓：中期前不重建；论文系统封装时可说明现有 MySQL / InfluxDB 接入边界，必要时补轻量接口说明或部署文档。

## 历史计划入口

- [notes/coding-roadmap.md](notes/coding-roadmap.md)
- [notes/stage-i-thesis-mainline-roadmap-2026-05-15.md](notes/stage-i-thesis-mainline-roadmap-2026-05-15.md)
- [notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md](notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md)
- [notes/thesis-coding-gap.md](notes/thesis-coding-gap.md)
- [notes/iteration-playbook.md](notes/iteration-playbook.md)
