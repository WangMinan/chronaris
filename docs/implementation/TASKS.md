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
  - `Phase D`：刚体运动物理约束补强代码已完成，首轮真实 smoke / ablation 已落盘。
  - `Phase E`：语义事件融合补强代码已完成，真实 summary / support 产物已落盘。
  - `Phase F`：runtime inference 代码已完成，真实 replay / runtime 报告已落盘。

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

## 当前工作区收敛：文档与产物归档

当前事实：

- `main` 当前 `HEAD=0f4db72 feat: add stage i runtime sample exporter`。
- `main...origin/main [ahead 2]`，领先提交为 `0f4db72` 和 `9ef4f64`；本地 Phase D/E/F 代码已经提交，但尚未推到 `origin/main`。
- 当前没有未提交的 `src/` 或 `scripts/` 代码文件。
- 当前未提交修改为 `docs/STATE.md`、`docs/implementation/TASKS.md`。
- 当前未跟踪产物为本轮 Stage I 真实 replay/support/ablation 资产，主要位于：
  - `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-replay-r1/`
  - `docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r1/`
  - `docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r1/`
  - `docs/artifacts/stage_i/assets/stage-i-semantic-event-20260607T-stage-i-semantic-event-r1/`
  - `docs/artifacts/stage_i/stage-i-*-20260607T-*.md`

收敛顺序：

1. 先提交 `STATE.md` 和 `TASKS.md` 的事实同步。
2. 再决定新增产物是否整体进入 git 历史；建议至少保留 JSON/CSV/Markdown 报告，checkpoint 与 PNG 可按仓库容量再判断。
3. 若本轮产物入库，同步更新 `docs/artifacts/ARTIFACTS.md` 的“当前最常引用产物”。
4. 推送前复查 `git status --short --untracked-files=all`，确保没有临时 stdout、半成品资产或大文件误入。

验收：

- `git diff --check` 通过。
- 状态文档中的 `HEAD`、领先远端状态、未跟踪产物和当前编码缺口一致。
- 若提交产物，报告路径和 asset 路径必须能从 `ARTIFACTS.md` 或本文件追溯。

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

## 中期前最小收敛顺序

1. 直接引用最新 Phase C 真实联合训练证据与 private benchmark 分层资产。
2. 直接引用最新中期证据包：`docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md`。
3. 直接引用本轮 runtime replay：`docs/artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-replay-r1.md`。
4. 直接引用本轮 semantic support：`docs/artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r1.md`。
5. 直接引用本轮 rigid_body ablation：`docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r1.md`。

## 当前 P7：扩充 rigid_body 的字段语义覆盖

目标：在 `translation` 已经启用的基础上，继续让 `vertical / rotation` 残差在真实链路上稳定启用。

现状：

- `vehicle_field_metadata` 已确认可通过本地 `127.0.0.1:3306` 正常加载。
- `BUS6000019110020` 当前加载字段数为 `96`。
- 现有 token 映射已经触发 `speed + acceleration`，所以 `translation` 生效。
- `altitude + vertical_speed` 和姿态角/角速度轴对没有在当前真实链路稳定匹配，导致 `vertical / rotation` 分量仍为 0。

代码落点：

- `scripts/run_stage_e_relative_preview.py`
  - 复用 `_resolve_vehicle_field_labels()` 和 `MySQLRealBusContextReader`，不要绕过既有 MySQL 元信息访问。
  - 新增或扩展字段映射诊断导出，至少能列出每个 `rigid_body` group 的匹配字段、原始 label、未匹配候选。
- `src/chronaris/models/alignment/physics_state_mapping.py`
  - 扩充 `_TRANSLATION_GROUP_TOKENS` 与 `_ROTATION_AXIS_TOKENS`。
  - 优先加真实字段 label 的精确或半精确 token，不要用过宽的泛化词导致误配。
  - 保留 `enabled_residuals()` 与 `missing_requirements()` 的可解释输出。
- `src/chronaris/models/alignment/physics_features.py`
  - 保持 Stage F 旧 semantic grouping 与 `rigid_body` mapping 口径一致，避免 `full` 与 `rigid_body` 对同一 label 给出冲突解释。
- `src/chronaris/models/alignment/physics_residuals.py`
  - 若新增字段组合后 vertical/rotation 数值尺度异常，优先在残差内部做单位/差分尺度诊断，不直接调大 loss weight 掩盖问题。
- `tests/test_alignment_model_losses.py`
  - 补真实中文 label token 的单元测试。
  - 覆盖 `altitude + vertical_speed`、`pitch + pitch_rate`、`roll + roll_rate`、`yaw + yaw_rate` 至少一组 rotation 生效路径。

实跑命令形态：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_e_relative_preview.py \
  --report-path docs/artifacts/stage_i/stage-i-rigid-body-rigid-body-20260607T-stage-i-rigid-body-r2.md \
  --enable-physics-constraints \
  --physics-constraint-family rigid_body \
  --input-normalization-mode zscore_train \
  --epoch-count 1 \
  --batch-size 8 \
  --strict-mysql-field-labels \
  --device cpu
```

验收：

- MySQL metadata 仍为 `loaded`，且字段数没有意外降为 0。
- `rigid_body` diagnostics 中 `enabled_residuals` 至少包含 `translation` 和 `vertical`；若真实字段具备姿态角/角速度，再包含 `rotation`。
- `vehicle_rigid_body_vertical` 或 `vehicle_rigid_body_rotation` 至少一个从 0 变为非零。
- `full` / `rigid_body` 的 `test_total` 仍在同量级，不因字段误配出现数量级爆炸。
- 相关 unittest 通过，并重建 `docs/artifacts/assets/stage_i_rigid_body/<new_run_id>/rigid_body_ablation_summary.json`。

## 下一步 P8：放大 semantic event support 证据

目标：把当前一条 preview-scale 语义事件证据，扩展成多 view / 多 sortie 的 support matrix，使论文中“语义事件融合”不只依赖单点样例。

代码落点：

- `src/chronaris/pipelines/causal_fusion.py`
  - 支持从 Stage H manifest 批量读取多个 view 的 causal summary 输入。
  - 输出 view-level semantic event metrics：`event_token_count`、`query_entropy`、`top_query_name`、`top_event_attribution`、`top_query_event_offset_s`。
- `src/chronaris/pipelines/stage_i/stage_i_support_builders.py`
  - 将 `causal_support.semantic_event` 从单 summary 聚合改成可接收 summary list。
  - 给每个 view 保留 `sortie_id / pilot_id / view_id / source_summary_path`。
- `src/chronaris/pipelines/stage_i/stage_i_support_reporting.py`
  - 增加 semantic event support 表与 view-level 排名表。
  - 报告中继续区分 `private proxy evidence`、`thesis weak-label evidence`、`case support evidence`。
- `tests/test_stage_i_support.py`
  - 增加多 summary、多 view、缺失 semantic_event 字段的 fallback 测试。

验收：

- 至少覆盖当前 Stage H `validation` profile 的 3 个双流 view。
- `support_summary.json` 中有 view-level semantic rows，而不是只保存均值。
- 支持报告能回答“哪个 view / 哪个 query / 哪个 offset 贡献最大”。
- 不把 `T1/T2/T3` 或 weak label 写成人工真值。

## 下一步 P9：runtime inference 服务化补强

目标：把 checkpoint-backed replay 从“可跑脚本”推进到“可封装准实时推理接口”，为毕业论文系统实现章节准备稳定边界。

代码落点：

- `src/chronaris/dataset/streaming_windows.py`
  - 明确窗口缓存策略：按 `view_id` 分桶、可配置 stride、最大缓存长度、乱序点处理。
  - 增加 schema drift 诊断：缺列、额外列、时间戳不单调、采样间隔异常。
- `src/chronaris/serving/runtime_inference.py`
  - 输出标准化 response schema：prediction、confidence/score、attention summary、semantic event summary、diagnostics。
  - 增加 batch replay 与 incremental replay 两种入口，避免脚本层自己拼业务逻辑。
  - 对 checkpoint feature schema mismatch 给出可读错误和 fallback 策略。
- `scripts/run_stage_i_runtime_inference.py`
  - 增加 `--max-windows`、`--batch-size`、`--emit-jsonl`、`--strict-feature-schema`。
- `tests/test_runtime_inference.py`
  - 覆盖乱序输入、缺列、批次 replay、strict schema mismatch、解释输出裁剪。

验收：

- 同一 `runtime_samples.jsonl` 在 batch replay 与 incremental replay 下输出样本数一致。
- `runtime_inference_summary.json` 增加 latency / throughput / diagnostics 统计。
- 单条 prediction JSON 可直接作为系统接口样例纳入论文或附录。

## 下一步 P10：统一论文闭环评测 harness

目标：把 Phase C/D/E/F 的真实命令整合成一个可重复跑的 evidence runner，减少后续论文补图、补表时的手工步骤。

建议新增：

- `src/chronaris/pipelines/stage_i/stage_i_evidence_runner.py`
- `scripts/run_stage_i_evidence_closure.py`
- `tests/test_stage_i_evidence_runner.py`

职责：

- 串联 Stage I multitask checkpoint、rigid_body ablation、semantic support、runtime replay。
- 统一 run id 规则、artifact root、report root 和 manifest 输出。
- 生成 `evidence_manifest.json`，记录输入、输出、命令参数、git commit、测试结果摘要。
- 支持 `--skip-heavy`、`--reuse-existing`、`--only rigid_body|semantic|runtime|all`。

验收：

- 一条命令能重建核心 JSON/CSV/Markdown。
- 每个子任务失败时保留 partial manifest，且不会覆盖上一轮稳定产物。
- `docs/artifacts/ARTIFACTS.md` 可直接链接最新 evidence manifest。

## 下一步 P11：论文案例与消融表稳定化

目标：把现有 evidence 转成论文可直接引用的表格与案例材料，减少后期靠手工复制指标。

代码落点：

- `src/chronaris/evaluation/` 或 `src/chronaris/pipelines/stage_i/`
  - 新增 report table builder：统一导出 ablation table、runtime prediction examples、semantic event attribution cases。
  - 对每张表附带 `evidence_layer`、`source_path`、`metric_definition`。
- `docs/artifacts/stage_i/`
  - 输出论文案例报告，保留中文解释、边界说明和引用路径。

验收：

- 至少生成三张稳定表：物理约束消融、语义事件 support、runtime replay 样例。
- 每张表都能追溯到 JSON/CSV 原始产物。
- 表述不越界：public adapter、private proxy、thesis weak-label、case support 分层清楚。

## 下一步 P12：系统封装与部署边界

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

## 中期前降级处理

下面几类工作不是永久排除，而是中期答辩前不抢占主线投入；后续毕业论文整理和系统封装阶段，可以按边界清楚、证据分层、可复现的方式适当纳入。

- CPU-heavy `sklearn` 或 UAB torch 候选搜索：中期前不再扩搜；论文封装阶段可作为公开 adapter baseline / calibration baseline 的补充材料。
- NASA/UAB 公开数据适配器结果：中期前不改写成论文双流本体闭环；论文中可作为 public adapter evidence，用于说明方法在公开代理数据上的迁移与校准边界。
- `chronaris_opt` 与 `T1/T2/T3`：中期前不写成人工真值 thesis task fully closed；论文中可作为 private proxy benchmark evidence，用于支撑表示学习、对齐和因果融合增益。
- 上游接收器、入库链路和原始大文件入仓：中期前不重建；论文系统封装时可说明现有 MySQL / InfluxDB 接入边界，必要时补轻量接口说明或部署文档。

## 历史计划入口

- [notes/coding-roadmap.md](notes/coding-roadmap.md)
- [notes/stage-i-thesis-mainline-roadmap-2026-05-15.md](notes/stage-i-thesis-mainline-roadmap-2026-05-15.md)
- [notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md](notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md)
- [notes/thesis-coding-gap.md](notes/thesis-coding-gap.md)
- [notes/iteration-playbook.md](notes/iteration-playbook.md)
