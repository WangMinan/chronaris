# Chronaris 当前状态

更新时间：2026-06-07

## 一句话状态

项目已经具备中期答辩可用的历史实验资产与最新主线补充证据：真实链路 `Stage E/F/G(min)/H`、Stage I 历史公开 benchmark、`chronaris_opt` 私有代理证据、public adapter 支撑线、Phase C 真实 Stage H multitask 联合训练证据和重建后的中期证据包都已形成并进入 git 历史；同时，`Phase D/E/F` 已经从“代码闭环”推进到“二轮真实资产闭环”，包括 `rigid_body translation+vertical` 真实启用、多 view 语义事件 support、以及具备 `batch/incremental/both` 模式和 schema diagnostics 的 runtime replay。当前下一步主线已经切到 `P10`：把这些分散命令收敛成统一 evidence runner。

## 当前阶段

- 阶段 A/B/C：已完成。
- 阶段 E0：已完成 preview 路径。
- 阶段 E/F/G(min)：已完成真实链路收口，作为历史基线保留。
- 阶段 H：已完成标准化特征导出收口，`validation` profile 可稳定导出 3 个双流 view。
- 阶段 I 历史公开 benchmark：`Phase 0/1/2/3` 已完成并收口。
- 阶段 I thesis mainline：
  - `Phase A/B` 已经进入 git 历史，主线边界校准、统一骨干训练入口、checkpoint inference export contract 已具备。
  - `Phase C` 已进入 git 历史，内容包括统一任务头、任务监督损失、因果正则接入、`risk_proxy / workload_proxy / event_replay_tag` weak-label thesis task builder、`stage_i_multitask_train` 联合训练入口，以及 private benchmark 中 `proxy_evidence / thesis_task_evidence` 分层。
  - `Phase D` 已完成首轮真实 smoke / ablation：
    - 首轮：`docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r1/`
    - 二轮：`docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/`
    - 最新汇总报告：`docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md`
    - 当前 `rigid_body` 已经在真实链路上启用 `translation + vertical`：
      - `vehicle_rigid_body_translation=1.133332371711731`
      - `vehicle_rigid_body_vertical=3.9466116428375244`
    - `rotation` 仍为 `0`，当前主要原因是缺少成对角速度字段，而不是 MySQL 元数据问题。
  - `Phase E` 已完成真实语义事件融合 support：
    - 单 view preview：`docs/artifacts/stage_i/stage-i-semantic-event-20260607T-stage-i-semantic-event-r1.md`
    - 多 view support summary：`docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`
    - 多 view support 报告：`docs/artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md`
    - 当前 semantic support 已覆盖 `3` 个双流 view、`111` 个样本，并给出 view-level ranking。
  - `Phase F` 已完成真实 runtime replay：
    - 首轮 replay：`docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-replay-r1/`
    - 服务化补强 replay：`docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/`
    - 最新报告：`docs/artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md`
    - 当前 runtime replay 已支持 `batch / incremental / both`，并输出 `latency / throughput / feature_schema_status / feature_schema_source`。

## Git 与工作区核对

- 当前分支为 `main`，当前 `HEAD=0f4db72`，提交信息为 `feat: add stage i runtime sample exporter`。
- `main...origin/main [ahead 2]`：本地领先远端 `origin/main=14dd0f0` 两个提交，尚未推送。
  - `0f4db72 feat: add stage i runtime sample exporter`
  - `9ef4f64 feat: add rigid-body physics semantic event runtime inference`
- 关键实现提交 `a5fda40` 覆盖 Stage I thesis mainline `Phase C`：真实 Stage H multitask 联合训练、private/thesis 分层资产和中期证据包。
- 更早的关键实现提交 `2055dec` 覆盖 Stage I thesis mainline `Phase A/B`：public adapter/proxy 边界、backbone train、Stage H checkpoint inference contract 和相关测试。
- 当前 `Phase D/E/F` 主代码和 runtime sample exporter 已经进入本地 git 历史；相对 `origin/main` 的代码差异为 `22 files changed, 2386 insertions(+), 20 deletions(-)`。
  - 刚体物理：`physics_state_mapping.py`、`physics_residuals.py`、`physics.py`、`physics_features.py`、`run_stage_e_relative_preview.py`。
  - 语义事件融合：`semantic_event.py`、`causal_fusion.py`、`stage_i_support_builders.py`、`stage_i_support_reporting.py`。
  - runtime inference：`streaming_windows.py`、`runtime_inference.py`、`run_stage_i_runtime_inference.py`、`export_stage_i_runtime_samples.py`。
  - 测试覆盖：`tests/test_alignment_model_losses.py`、`tests/test_stage_i_support.py`、`tests/test_runtime_inference.py`。
- 当前工作区剩余未提交项主要是：
  - 本轮状态文档刷新：`docs/STATE.md`、`docs/implementation/TASKS.md`
  - 本轮新增产物：`docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-replay-r1/`、`docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r1/`、`docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r1/`、`docs/artifacts/stage_i/assets/stage-i-semantic-event-20260607T-stage-i-semantic-event-r1/` 以及对应 Markdown 报告。
  - 当前没有未提交的 `src/` 或 `scripts/` 代码文件。
- 已进入历史的关键前置产物：
  - Phase C 真实联合训练：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_summary.json`
  - thesis weak-label 报告：`docs/artifacts/stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md`
  - private benchmark 分层资产：`docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json`
  - 最新中期证据包：`docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md`

## 当前主线事实

- 当前鼎新私有任务验证主线仍是 `chronaris_opt`，但它属于 `private proxy benchmark / proxy evidence`。
- 当前公开支撑线为 `public opt closed`，但 `UAB robust-prior adapter / target_prior_median` 只能写成 `public adapter / calibration evidence`，不能写成双流连续对齐或因果融合模块本体的直接胜利。
- 当前公开第二模态应写成 `context proxy / public adapter evidence`，不是论文严格意义上的真实航电流。
- `T1/T2/T3` 是私有代理任务；`risk_proxy / workload_proxy / event_replay_tag` 是 thesis weak-label task builder，不等价于人工真值任务。
- `20251110_单01_ACT-2_涛_J20_26#01` 仍是 vehicle-only partial-data，不是双流 Stage H view。

## 编码层面还需要做什么

1. 选择性整理本轮新增代码与产物：决定 `r2` 版 rigid_body、semantic support、runtime service 结果哪些进入 git 历史，并同步 `docs/artifacts/ARTIFACTS.md`。
2. 补 `rotation` 方向的真实字段覆盖：重点确认 `真航向` 与潜在角速度字段是否存在，决定是扩 token 还是明确写成“当前 sortie 无可用 rate field”。
3. 建立统一 evidence runner：把 rigid_body、semantic support、runtime replay 收进同一套 manifest 与 CLI。
4. 在 runtime inference 上补更细的窗口状态缓存与错误样例。
5. 系统封装阶段补离线服务边界：明确输入 manifest、checkpoint、样本流、输出 schema、错误处理和准实时延迟指标。

## 实验层面还需要做什么

1. 已完成 Phase C 真实联合训练证据：
   - 输入：`docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json`
   - 输入：`docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json`
   - 输出：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/`
2. 已完成 private benchmark 分层资产刷新：
   - 输出：`docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/`
   - `private_benchmark_summary.json` 已包含 `evidence_layers.proxy_evidence / thesis_task_evidence`
3. 已完成中期证据包重建：
   - 输出：`docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/`
   - 报告：`docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md`
4. 本轮新增的 `Phase D/E/F` 产物已经落盘并已更新到 `r2`：
   - runtime service-style replay：`docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/`
     - `sample_count=40`
     - `replay_mode=both`
     - `sample_count_match=True`
     - `feature_schema_status=aligned`
     - `feature_schema_source=input_normalization_stats`
   - semantic event support：`docs/artifacts/assets/stage_i_semantic_event_support/20260607T-stage-i-semantic-support-r2/`
     - `view_count=3`
     - `top_view_id=20251005_四01_ACT-4_云_J20_22#01__pilot_10033`
   - support 聚合：`docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/`
     - `support_summary.json` 已包含 `causal_support.semantic_event.view_rows`
   - rigid_body ablation：`docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/`
     - `vehicle_field_metadata.status=loaded`
     - `enabled_residuals=['translation','vertical']`
     - `vehicle_rigid_body_translation=1.133332371711731`
     - `vehicle_rigid_body_vertical=3.9466116428375244`
     - `vehicle_rigid_body_rotation=0`

## 当前关键入口

- 当前执行入口与任务队列：[implementation/TASKS.md](implementation/TASKS.md)
- 论文需求入口：[requirements/SPEC.md](requirements/SPEC.md)
- 产物索引：[artifacts/ARTIFACTS.md](artifacts/ARTIFACTS.md)
- 中期前目标笔记：[implementation/notes/midterm-goal-2026-06-07.md](implementation/notes/midterm-goal-2026-06-07.md)

## 本轮验证

使用指定 conda 解释器并显式启用 torch runtime 测试执行：

```bash
CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 \
  /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest \
  tests.test_alignment_model_losses \
  tests.test_alignment_pipeline \
  tests.test_stage_i_support \
  tests.test_stage_i_multitask_train \
  tests.test_runtime_inference
```

结果：`Ran 33 tests in 7.962s`，`OK`。此外，本轮真实命令已完成：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/export_stage_i_runtime_samples.py \
  --run-id 20260607T-stage-i-runtime-replay-r1 \
  --run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json \
  --output-root docs/artifacts/assets/stage_i_runtime_inference

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_runtime_inference.py \
  --run-id 20260607T-stage-i-runtime-replay-r1 \
  --checkpoint-path docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_checkpoint.pt \
  --sample-jsonl docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-replay-r1/runtime_samples.jsonl \
  --artifact-root docs/artifacts/assets/stage_i_runtime_inference \
  --report-root docs/artifacts/stage_i \
  --device cpu

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_support.py \
  --run-id 20260607T-stage-i-support-semantic-r1 \
  --artifact-root docs/artifacts/assets/stage_i_support \
  --report-root docs/artifacts/stage_i \
  --causal-g-summary-path docs/artifacts/stage_i/assets/stage-i-semantic-event-20260607T-stage-i-semantic-event-r1/causal_fusion_summary.json
```

刚体对比另行落到了 `docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r1/` 和 `docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r1.md`；与上一版不同的是，本次已确认 `vehicle_field_metadata.status=loaded`，且 `vehicle_rigid_body_translation` 已经非零。

## 中期前降级处理

下面几类工作不是永久排除，而是中期答辩前不抢占主线投入；后续毕业论文整理和系统封装阶段，可以按边界清楚、证据分层、可复现的方式适当纳入。

- CPU-heavy `sklearn` 或 UAB torch 候选搜索：中期前不再扩搜；论文封装阶段可作为公开 adapter baseline / calibration baseline 的补充材料。
- NASA/UAB 公开数据适配器结果：中期前不改写成论文双流本体闭环；论文中可作为 public adapter evidence，用于说明方法在公开代理数据上的迁移与校准边界。
- `chronaris_opt` 与 `T1/T2/T3`：中期前不写成人工真值 thesis task fully closed；论文中可作为 private proxy benchmark evidence，用于支撑表示学习、对齐和因果融合增益。
- 上游接收器、入库链路和原始大文件入仓：中期前不重建；论文系统封装时可说明现有 MySQL / InfluxDB 接入边界，必要时补轻量接口说明或部署文档。
