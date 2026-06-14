# Chronaris 当前任务

更新时间：2026-06-14

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

- 本地 `main` 已同步推送到 `origin/main`。
- `P10-P15` 主动证据工具、测试、报告、索引和可引用汇总资产已经进入远端历史，主体功能与资产提交为 `70b651a feat: add stage i evidence closure tools`。
- `Phase D/E/F` 代码、文档与资产已经进入历史基线；当前最新主动证据入口为 `docs/artifacts/assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/evidence_manifest.json`。
- 已进入 git 历史的最新 Stage I 真实 replay/support/ablation 资产包括：
  - `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/`
  - `docs/artifacts/assets/stage_i_semantic_event_support/20260607T-stage-i-semantic-support-r2/`
  - `docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/`
  - `docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/`
  - `docs/artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md`
  - `docs/artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md`
  - `docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md`

后续收敛顺序：

1. 当前 P20 已完成 DeepSeek 在线时序数据预处理首轮实现、小样本真实 run 与 schema harness；后续若继续编码，优先扩展更多 cards 和人工复核封装。
2. 中期报告写作优先从 `docs/midterm/` 的事实清单、边界风险说明和 claims matrix 进入；P20 当前可写成已实现在线 LLM preprocessing context，但不能写成真值标注、OpenAI 默认接入或核心因果证据。
3. 保留并维护 `P18` 的 `P11 stable/partial` 与 `P17 schema-contract` 当前入口，避免后续继续回退到纯手工解释或旧 r1 demo。
4. 持续维护 evidence runner 的 `skip-heavy / reuse-existing` 策略；如需更大 `live_influx` 网格，先明确预算，再从当前 `2` 组合稳定版扩展。
5. 若后续发现可用角速度字段，在 `rotation audit` 基础上复跑 `minimal / full / rigid_body`；若没有，继续保持 `rotation disabled` diagnostics 口径。
6. 若后续要把 runtime/service 继续收紧到“exact schema only”，优先围绕当前 `native_feature_schema_status=aligned` 的 missing vehicle groups 做采样契约补齐，而不是重建上游接收器。
7. 展开文献检索前，先用 `docs/midterm/claims-matrix-2026-06-13.md` 约束论文 claim 强度，再按异构时序对齐、连续潜态、物理约束、因果融合、航空人因 weak-label、LLM 辅助时序预处理六组关键词搜索。

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

## 已完成 P11+：补一轮 `live_influx` thesis weak-label sweep

目标：在当前 bounded `stage_h_window_stats_proxy` sweep 之外，使用本地 `127.0.0.1` 的 MySQL / InfluxDB CLI 形成 `live_influx` sample collection 证据，并与 proxy 路线并排展示。

本轮结果：

- 真实 `live_influx` child run 已完成并用于稳定汇总：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/runs/20260613T-stage-i-p11-live-influx-r1-01-minimal-cw0p00-tlw0p50-lagnone/multitask_summary.json`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/runs/20260613T-stage-i-p11-live-influx-r1-02-minimal-cw0p00-tlw0p50-lag3/multitask_summary.json`
- 稳定汇总产物：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r2/multitask_sweep_summary.json`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r2/thesis_weak_label_multitask_ablation.csv`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r2/proxy_vs_live_influx_comparison.csv`
  - `docs/artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r2.md`
- 当前对比口径：
  - `sample_source=live_influx`
  - `sample_count=111`
  - `task_entry_count=333`
  - `combination_count=2`
  - `best_test_total=1153.8985701851223`
- 当前 blocker 已保留但未伪造成稳定证据：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/progress.json`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/run.log`
  - `20260613T-stage-i-p11-live-influx-r1` 的 `4` 组合尝试在 `run_index=3/4` 处因 runtime cost 手动中断；稳定 `r2` 只复用其中已完成的两个 live child run，与 proxy 的 `2` 组合口径对齐。

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

## 已完成 P16：论文案例、消融表与说明图稳定化

目标：把现有和 P11-P15 新增 evidence 转成论文可直接引用的表格、案例材料和说明图件，减少后期靠手工复制指标或临时画图。

代码落点：

- `src/chronaris/pipelines/stage_i/stage_i_thesis_materials.py`
- `scripts/run_stage_i_thesis_materials.py`
- `docs/artifacts/stage_i/`
  - 输出论文案例报告，保留中文解释、边界说明、图表引用路径。
- `docs/artifacts/assets/stage_i_thesis_figures/<run_id>/`
  - 输出 `.png` 图件和对应 `figure_manifest.json`，记录每张图的源数据、用途、证据层级和可复现命令。

本轮结果：

- 稳定报告：
  - `docs/artifacts/stage_i/stage-i-thesis-materials-20260613T-stage-i-thesis-materials-r1.md`
- 稳定 manifest：
  - `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r1/table_manifest.json`
  - `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r1/figure_manifest.json`
- 六张稳定表：
  - `evidence_layer_overview.csv`
  - `weak_label_sweep_ablation.csv`
  - `chronaris_opt_component_ablation.csv`
  - `public_transfer_boundary.csv`
  - `runtime_semantic_case.csv`
  - `rigid_body_rotation_audit.csv`
- 六张稳定说明图：
  - `evidence_layer_overview.png`
  - `weak_label_sweep_ablation.png`
  - `chronaris_opt_component_ablation.png`
  - `public_transfer_boundary.png`
  - `runtime_semantic_case.png`
  - `rigid_body_rotation_audit.png`
- 当前 stable root：
  - `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r1/`
- 当前说明图和表共用同一批 source path，`figure_manifest.json` 已分别记录 `metric_definition` 或 `case_definition`，并保持 `public adapter / private proxy / thesis weak-label / runtime support` 分层清楚。

## 已完成 P17：系统封装与部署边界

目标：为毕业设计系统实现章节补齐“离线/准实时推理服务”的工程闭环和说明图，而不是只停留在训练脚本和报告。

代码落点：

- `src/chronaris/serving/`
  - `src/chronaris/serving/runtime_service_smoke.py`
  - `src/chronaris/serving/__init__.py`
- `scripts/run_stage_i_runtime_smoke.py`
- `tests/test_runtime_service_smoke.py`

本轮结果：

- 真实单 view 输入：
  - `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/input_view_runtime_samples.jsonl`
  - 当前 `view_id=20251005_四01_ACT-4_云_J20_22#01__pilot_10033`
  - 当前 `sample_count=37`
- 稳定服务 smoke root：
  - `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/`
- 关键输出：
  - `runtime_service_smoke_summary.json`
  - `runtime_inference/20260613T-stage-i-runtime-service-smoke-r1-runtime/runtime_inference_predictions.jsonl`
  - `runtime_inference/20260613T-stage-i-runtime-service-smoke-r1-runtime/runtime_inference_summary.json`
  - `runtime_error_cases.json`
  - `figure_manifest.json`
  - `docs/artifacts/stage_i/stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r1.md`
- 三张系统说明图：
  - `runtime_service_flow.png`
  - `runtime_payload_schema.png`
  - `runtime_error_cases.png`
- 当前错误样例均已固化为 `expected_failure`：
  - `missing_checkpoint`
  - `missing_fields`
  - `empty_window`
  - `schema_mismatch`
- 当前真实 smoke 结论：
  - checkpoint 冷启动成功
  - 单 view replay JSONL 成功输出 predictions JSONL 与 summary JSON
  - `feature_schema_status=aligned`
  - 说明当前 runtime facade 仍依赖 `input_normalization_stats` 做 schema 对齐，若后续要收紧到 exact schema，需要继续补输入契约而不是重建上游接收器

## 已完成 P18：P11/P17 风险收口优化

目标：把 P11 的“部分完成但不中断证据链”和 P17 的“aligned 但还不是 exact schema”变成可复现、可解释、可验收的工程能力，而不是靠人工口头说明。

### 已完成 P18-A：P11 live_influx sweep partial/resume

当前事实：

- `r1` 的更大 `4` 组合尝试没有伪造成成功结果，blocker 已保留在 `progress.json / run.log`。
- 当前已新增 stable resume 版与 partial blocked 版：
  - stable resume：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/`
  - partial blocked：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/`

代码落点：

- `src/chronaris/pipelines/stage_i/stage_i_multitask_sweep.py`
  - 每个 child run 完成后即时写入 `partial_summary.json` 和临时 CSV。
  - 中断或失败时输出 `status=partial_blocked`、`completed_child_runs`、`blocked_at_run_index`、`blocker_log_path`。
  - 支持从已有 child run 恢复汇总，避免重复跑已完成组合。
- `scripts/run_stage_i_multitask_sweep.py`
  - 增加 `--resume-existing` 与 `--resume-run-root`，允许从指定历史 run root 复用已完成 child run。
  - 增加 `--max-runtime-seconds` 或明确的预算 guard，避免 live_influx 大网格无限拖住。
- `tests/test_stage_i_multitask_sweep.py`
  - 覆盖 partial summary、resume、blocker 不伪造成 completed。

本轮结果：

- stable resume summary：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json`
  - `docs/artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r3-resume.md`
  - 当前已包含：
    - `derived_from_run_id=20260613T-stage-i-p11-live-influx-r1`
    - `completed_child_run_paths`
    - `blocked_attempt_log_paths`
    - `blocked_at_run_index=3`
    - `evidence_layer=thesis_weak_label`
- partial blocked summary：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/partial_summary.json`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/thesis_weak_label_multitask_ablation.partial.csv`
  - 当前 `status=partial_blocked`
  - 当前 `completed_child_runs=2`
  - 当前 `blocked_at_run_index=3`
  - 当前 `blocker_log_path=docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/run.log`
- blocker 继续保留：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/progress.json`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/run.log`
- 当前没有伪造更大 live_influx 网格成功；`r4-partial` 只是一个 bounded resume/blocker smoke，用来固化证据链，而不是把未完成组合包装成 completed。

### 已完成 P18-B：P17 runtime schema contract 与 exact 边界

当前事实：

- P17 真实 smoke 成功，但 `feature_schema_status=aligned`，`feature_schema_source=input_normalization_stats`。
- 当前 checkpoint 没有显式 `feature_schema`，runtime fallback 到 `input_normalization_stats`。
- 当前单 view 输入为 `965` 个 vehicle features；checkpoint 期望 `1930` 个 vehicle features。缺口集中在另一半 BUS measurement：`BUS6000019110021` 到 `BUS6000019110026`。

代码落点：

- 新增 `src/chronaris/serving/runtime_schema_contract.py`
  - 从 checkpoint 导出 expected schema、schema hash、stream feature counts、measurement group counts。
  - 对比 runtime JSONL 输入 schema，输出 missing/extra feature 分组和 `exact_possible`。
  - 输出 `runtime_schema_contract.json`，作为 P17 后续部署契约事实源。
- 扩展 `src/chronaris/serving/runtime_service_smoke.py`
  - `StageIRuntimeSmokeConfig` 增加 `strict_feature_schema: bool`。
  - summary 增加 `schema_contract_path`、`native_feature_schema_status`、`canonical_feature_schema_status`。
  - error cases 继续保留 `schema_mismatch`，但错误摘要优先输出分组统计，避免几百个字段刷屏。
- 扩展 `scripts/run_stage_i_runtime_smoke.py`
  - 增加 `--strict-feature-schema`。
  - 增加 `--export-canonical-payload` 或等价参数，输出 `canonical_runtime_samples.jsonl`。
- 测试：
  - `tests/test_runtime_service_smoke.py`
  - 新增 `tests/test_runtime_schema_contract.py`

本轮结果：

- 新增 schema contract root：
  - `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/`
- 关键产物：
  - `runtime_service_smoke_summary.json`
  - `runtime_schema_contract.json`
  - `canonical_runtime_samples.jsonl`
  - `runtime_inference/20260613T-stage-i-runtime-service-smoke-r2-contract-canonical/runtime_inference_summary.json`
  - `docs/artifacts/stage_i/stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r2-contract.md`
- 当前 native 单 view 输入已清楚记录：
  - `native_feature_schema_status=aligned`
  - `expected_vehicle_feature_count=1930`
  - `input_vehicle_feature_count=965`
  - `missing_vehicle_feature_count=965`
  - `missing_vehicle_measurement_group_counts` 覆盖：
    - `BUS6000019110021`
    - `BUS6000019110022`
    - `BUS6000019110023`
    - `BUS6000019110024`
    - `BUS6000019110025`
    - `BUS6000019110026`
- strict native smoke 已作为 expected failure 写入：
  - `runtime_error_cases.json`
  - `strict_native_feature_schema_probe`
- canonical payload 路线已生成：
  - `canonical_runtime_samples.jsonl`
  - `runtime_schema_contract.json`
  - `canonical_feature_schema_status=exact`
- 当前报告口径已固定：
  - `native aligned` 是当前真实部署边界
  - `canonical exact` 是服务层契约化 payload 能力
  - 不把 canonical exact 写成“原始上游输入 exact”
  - 不重建上游接收器，不做原始大文件入仓

### 已完成 P18-C：P16 图表同步刷新

目标：P18 完成后，刷新 P16 thesis materials，让论文图表同步反映 P11/P17 的真实边界。

本轮结果：

- 刷新后的 thesis materials：
  - `docs/artifacts/stage_i/stage-i-thesis-materials-20260613T-stage-i-thesis-materials-r2-p18.md`
  - `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/`
- 当前 `weak_label_sweep_ablation.png/csv` 已增加：
  - `summary_status`
  - `derived_from_run_id`
  - `blocked_at_run_index`
  - `blocked_attempt_log_path_count`
  - partial resume/blocker 注记
- 当前 `runtime_semantic_case.png/csv` 已体现：
  - `native_feature_schema_status=aligned`
  - `canonical_feature_schema_status=exact`
  - `expected_vehicle_feature_count=1930`
  - `input_vehicle_feature_count=965`
  - `missing_vehicle_feature_count=965`
  - `native_missing_measurement_group_count=6`
- 当前 `figure_manifest.json` 已记录：
  - `runtime_schema_contract.json`
  - `stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json`
  - `stage-i-p11-live-influx-r4-partial/partial_summary.json`

## 已完成 P19：中期报告材料冻结与 docs 入口清理

目标：在搜索论文和正式展开中期报告前，把当前可写事实、证据边界、风险说明和报告 claim 强度冻结成文档，供后续本地 clone 后直接作为中期报告写作基础。

本轮结果：

- 新增中期写作入口：
  - `docs/midterm/README.md`
  - `docs/midterm/midterm-fact-sheet-2026-06-13.md`
  - `docs/midterm/boundaries-and-risks-2026-06-13.md`
  - `docs/midterm/claims-matrix-2026-06-13.md`
- 清理过时入口：
  - `docs/artifacts/mid-term/README.md` 已从旧 `20260509` r2 中期包改指向当前 `20260607` r3 和 `docs/midterm/`。
  - `docs/artifacts/mid-term/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md` 兼容链接已删除。
  - `docs/artifacts/mid-term/stage-i-midterm-20260607T-stage-i-midterm-r3.md` 兼容链接已新增。
- 更新导航与状态：
  - `docs/README.md` 已新增 `midterm` 目录说明。
  - `docs/artifacts/ARTIFACTS.md` 已新增中期事实清单、边界说明和 claims matrix。
  - `docs/artifacts/stage_i/README.md` 已把 P16/P17 当前入口改为 `r2-p18 / r2-contract`，首轮 r1 降为历史入口。
  - `docs/STATE.md` 已把 P16/P17 r1 标为首轮历史，并补入 P18/p18 和中期写作材料入口。

验收口径：

- 中期报告写作先读 `docs/midterm/README.md`。
- 当前事实引用先读 `docs/midterm/midterm-fact-sheet-2026-06-13.md`。
- 边界和答辩风险先读 `docs/midterm/boundaries-and-risks-2026-06-13.md`。
- 文献检索和正文 claim 先用 `docs/midterm/claims-matrix-2026-06-13.md` 约束证据强度。

## 已完成 P20：DeepSeek 在线时序数据预处理

目标：在现有 MySQL / InfluxDB 私有数据链路和 Stage H / Stage I 资产之上，接入 DeepSeek 在线大模型，形成中期可写的 LLM 辅助时序数据预处理能力。

当前状态：

- 已完成文档计划：`docs/implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md`。
- 已新增 DeepSeek/OpenAI-compatible provider contract、strict response contract、schema-repair harness 和下游消费 helper。
- 已完成 mock provider 测试，并覆盖 schema repair retry。
- 已完成 DeepSeek v4-pro 小样本真实 run：`docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r1/llm_preprocessing_summary.json`。
- 当前真实 run `request_count=5`、`error_count=0`、`field_semantic_count=24`、`weak_label_review_count=3`、`semantic_query_hint_count=4`、`runtime_explanation_count=4`。

默认 provider：

- `CHRONARIS_LLM_PROVIDER=deepseek`
- `CHRONARIS_LLM_MODEL=deepseek-v4-pro`
- 不默认使用 OpenAI，除非用户后续明确解除信息安全顾虑。

实际代码落点：

- `src/chronaris/llm/provider.py`
- `src/chronaris/llm/schemas.py`
- `src/chronaris/llm/prompts.py`
- `src/chronaris/pipelines/stage_i/stage_i_llm_preprocessing.py`
- `scripts/run_stage_i_llm_preprocessing.py`
- `tests/test_stage_i_llm_preprocessing.py`

建议输入：

- MySQL 字段 label、measurement id、code id、sortie / view 元信息。
- InfluxDB 派生的 Stage H 窗口统计摘要。
- 当前 `2` 个 sortie、`3` 个双流 view、`111` 个窗口样本的 weak-label task summary。
- runtime schema contract、runtime semantic case 和 schema gap summary。

建议输出：

- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r1/llm_preprocessing_context.json`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r1/llm_field_semantics.jsonl`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r1/field_semantic_dictionary.csv`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r1/llm_weak_label_review.jsonl`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r1/weak_label_llm_comparison.csv`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r1/llm_schema_gap_policy.json`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r1/runtime_llm_explanations.jsonl`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r1/llm_request_response_audit.jsonl`
- `docs/artifacts/stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r1.md`

验收：

- mock provider 测试通过，离线环境不依赖真实 API。
- DeepSeek 小样本真实 run 完成并落盘。
- 请求、响应、prompt version、input hash、错误样例、成本/延迟摘要可追溯。
- 输出只作为字段语义、预处理建议、weak-label 复核和 runtime 解释证据，不替代人工真值或核心因果证据。
- 文档回写 `docs/STATE.md`、本文件、`docs/artifacts/ARTIFACTS.md` 和 `docs/midterm/claims-matrix-*.md`。

## 中期前边界管理

下面几类工作现在纳入中期前主动任务，但必须按证据分层写清楚，不能因为加做实验就改变论文边界。

- CPU-heavy `sklearn` 或 UAB torch 候选搜索：中期前允许有限预算复现/补跑；只能作为公开 adapter baseline / calibration baseline。
- NASA/UAB 公开数据适配器结果：中期前要整理成 public adapter evidence 和 transfer boundary；不能改写成论文双流本体闭环。
- `chronaris_opt` 与 `T1/T2/T3`：中期前要补机制诊断；仍只能写成 private proxy benchmark evidence，不能写成人工真值 thesis task fully closed。
- `risk_proxy / workload_proxy / event_replay_tag`：中期前要补小网格与消融；仍只能写成 thesis weak-label evidence。
- DeepSeek 在线 LLM 预处理：P20 已接入字段语义归一、weak-label 复核、schema gap policy 和 runtime 解释；仍不能写成 OpenAI 默认接入、人工真值替代、原始全量数据外发或核心因果证据。
- `rigid_body rotation`：中期前必须核验字段；启用或缺失都要以 diagnostics 形式固化。
- 上游接收器、入库链路和原始大文件入仓：中期前不重建；论文系统封装时可说明现有 MySQL / InfluxDB 接入边界，必要时补轻量接口说明或部署文档。

## 历史计划入口

- [notes/coding-roadmap.md](notes/coding-roadmap.md)
- [notes/stage-i-thesis-mainline-roadmap-2026-05-15.md](notes/stage-i-thesis-mainline-roadmap-2026-05-15.md)
- [notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md](notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md)
- [notes/thesis-coding-gap.md](notes/thesis-coding-gap.md)
- [notes/iteration-playbook.md](notes/iteration-playbook.md)
