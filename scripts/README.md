# Scripts

运行约定：

- 本文档涉及的所有 Python 脚本默认显式使用 `chronaris` 解释器：`/home/wangminan/env/anaconda3/envs/chronaris/bin/python`
- 若命令前需要环境变量，例如 `CHRONARIS_MYSQL_HOST=127.0.0.1`，应写成 `CHRONARIS_MYSQL_HOST=127.0.0.1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python <script>`
- Stage I 长任务默认同时看 CLI `INFO` 输出、artifact 目录下的 `run.log` 与 `progress.json`；断连后优先从这两个文件判断 dataset/candidate/fold/输出路径进度。
- 脚本只承载 CLI 编排；可复用逻辑必须回收到 `src/chronaris`。

## 顶层脚本

- `run_stage_e_relative_preview.py`：Stage E/F/G(min) overlap-focused preview 与物理/因果融合对照。
- `run_stage_h_export.py`：Stage H 标准化融合特征导出，支持 `preview / validation / full_clip` profile。

## Stage I 脚本目录

Stage I 入口已按职责拆到 `scripts/stage_i/<category>/`。根目录不再保留 `run_stage_i_*.py`、`build_stage_i_*.py`、`prepare_stage_i_*.py`、`export_stage_i_*.py` 旧脚本文件。

### data

- `scripts/stage_i/data/prepare_dataset.py`：构建公开 UAB/NASA `task_manifest.jsonl` 与 `feature_table.parquet`。
- `scripts/stage_i/data/prepare_sequences.py`：构建 deep/public opt 所需 sequence contract。

### training

- `scripts/stage_i/training/train_backbone.py`：从 Stage H view 样本训练可复用 alignment backbone。
- `scripts/stage_i/training/train_multitask.py`：训练 thesis weak-label multitask checkpoint。

### public

- `scripts/stage_i/public/run_opt.py`：公开 UAB/NASA public opt 统一入口。
- `scripts/stage_i/public/run_opt_torch_uab.py`：UAB torch-native heat-specialist 分支。
- `scripts/stage_i/public/run_deep_baseline.py`：单个 deep baseline。
- `scripts/stage_i/public/run_deep_comparison.py`：固定顺序 deep comparison。
- `scripts/stage_i/public/run_fusion_screen.py`：public fusion screening。
- `scripts/stage_i/public/run_public_fusion_refresh.py`：P28 `chronaris_public_fusion` screen + confirm refresh。
- `scripts/stage_i/public/run_public_fusion_gpuopt.py`：P28 GPU optimization profiling，不替代 confirmed metrics。
- `scripts/stage_i/public/run_public_fusion_ablation.py`：P31 public adapter/context-proxy component ablation。
- `scripts/stage_i/public/build_public_model_comparison.py`：P27 public model comparison 图表/CSV/报告整编。
- `scripts/stage_i/public/build_mainline_report.py`：public mainline report builder。

### private

- `scripts/stage_i/private/run_benchmark.py`：私有 Stage H `T1/T2/T3` proxy benchmark 与 `chronaris_opt` 证据。
- `scripts/stage_i/private/run_leakage_safe_ablation.py`：P24 `protocol=leakage_safe_v1` 标签-特征审计与防泄漏组件消融。
- `scripts/stage_i/private/run_private_thirdparty_comparison.py`：P30 private Stage H third-party comparison canonical 入口。
- `scripts/stage_i/private/run_task_head_optimization.py`：P34 task-aware heads optimization / confirm 入口。

### evidence

- `scripts/stage_i/evidence/run_closure.py`：P10-P15/P18 evidence runner。
- `scripts/stage_i/evidence/run_weak_label_sweep.py`：P11 thesis weak-label multitask sweep。
- `scripts/stage_i/evidence/run_private_component_ablation.py`：P12 `chronaris_opt` 组件诊断。
- `scripts/stage_i/evidence/run_public_adapter_calibration.py`：P13 public adapter calibration。
- `scripts/stage_i/evidence/build_public_transfer_boundary.py`：P14 public transfer boundary 报告。
- `scripts/stage_i/evidence/run_rigid_body_rotation_audit.py`：P15 rigid-body rotation audit。
- `scripts/stage_i/evidence/build_thesis_materials.py`：P16 thesis tables/figures。
- `scripts/stage_i/evidence/build_support.py`：Stage I support 聚合报告。
- `scripts/stage_i/evidence/build_semantic_event_support.py`：多 view semantic event support。
- `scripts/stage_i/evidence/build_midterm_evidence.py`：中期证据包整编。
- `scripts/stage_i/evidence/export_anchors.py`：关键工况 anchor 导出。
- `scripts/stage_i/evidence/run_case_study.py`：Stage I case study。
- `scripts/stage_i/evidence/build_cross_evidence_matrix.py`：P32 cross-evidence matrix 构建。
- `scripts/stage_i/evidence/build_optimized_chronaris_reevaluation.py`：P36 optimized Chronaris re-evaluation 聚合。
- `scripts/stage_i/evidence/build_optimized_model_summary.py`：optimized model summary 聚合。
- `scripts/stage_i/evidence/run_stream_role_fusion_eval.py`：P35 stream-role-aware fusion routing evaluation canonical 入口。
- `scripts/stage_i/evidence/build_thesis_protocol.py`：P38 论文协议冻结，统一 P30/P31/P32/P34/P35/P36/P37 registry、result matrix 和 claim boundary。

### runtime

- `scripts/stage_i/runtime/export_runtime_samples.py`：从 Stage H manifest 导出 runtime replay JSONL。
- `scripts/stage_i/runtime/run_inference.py`：checkpoint-backed runtime inference。
- `scripts/stage_i/runtime/run_smoke.py`：runtime/service smoke 与 schema contract 边界。
- `scripts/stage_i/runtime/run_demo.py`：历史 thesis-facing runtime demo。

### llm

- `scripts/stage_i/llm/run_preprocessing.py`：P20 DeepSeek/OpenAI-compatible LLM preprocessing context、规则复核、semantic hints 与 runtime explanation。
- `scripts/stage_i/llm/run_preprocessing_comparison.py`：P21 LLM preprocessing A0-A4 对比实验。

### legacy

- `scripts/stage_i/legacy/run_baseline.py`：历史 Stage I baseline suite。
- `scripts/stage_i/legacy/run_phase3.py`：历史 public Phase 3 closure。

## 维护规则

- 新增 Stage I 脚本必须进入上面的分类目录，不要恢复根目录 `stage_i` 前缀脚本。
- 长任务脚本必须保留 `run.log / progress.json` 或等价进度记录。
- 新增输出路径应能从 `docs/artifacts/ARTIFACTS.md`、`docs/implementation/TASKS.md` 或对应报告 manifest 追溯。
