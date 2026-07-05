# 2026-07-05 Name Migration Map

本表记录本轮本地命名迁移的 top-level old path 到 new path。子文件随所属目录一起移动；核心 CSV、JSON、PNG、MD 均保留。

## Code

| Old path | New path |
|---|---|
| `src/chronaris/pipelines/stage_h/` | `src/chronaris/feature_export/` |
| `src/chronaris/pipelines/stage_i/common/` | `src/chronaris/modeling/common/` |
| `src/chronaris/pipelines/stage_i/training/` | `src/chronaris/modeling/training/` |
| `src/chronaris/pipelines/stage_i/private/` | `src/chronaris/evaluation/dingxin/pipelines/` |
| `src/chronaris/pipelines/stage_i/public/` | `src/chronaris/evaluation/public_datasets/pipelines/` |
| `src/chronaris/pipelines/stage_i/evidence/` | `src/chronaris/evidence/` |
| `src/chronaris/pipelines/stage_i/llm/` | `src/chronaris/llm_preprocessing/` |
| `src/chronaris/pipelines/stage_i/legacy/` | `src/chronaris/archive/legacy_public_benchmark/` |
| `scripts/stage_i/` | `scripts/<responsibility>/` |
| `scripts/run_stage_h_export.py` | `scripts/feature_export/run_export.py` |
| `tests/test_stage_i_*.py` | `tests/<responsibility>/test_*.py` |
| `tests/test_stage_h_export.py` | `tests/feature_export/test_export.py` |

## Current Artifact Roots

| Old path | New path |
|---|---|
| `docs/artifacts/assets/stage_i_thesis_protocol/20260703T-stage-i-thesis-protocol-r1/` | `docs/artifacts/runs/2026-07-03_thesis-protocol-snapshot/` |
| `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/` | `docs/artifacts/runs/2026-07-02_metric-calibration/` |
| `docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/` | `docs/artifacts/runs/2026-07-02_selected-model-summary/` |
| `docs/artifacts/assets/stage_i_optimized_reevaluation/20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20/` | `docs/artifacts/runs/2026-07-02_selected-model-reevaluation/` |
| `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/` | `docs/artifacts/runs/2026-07-02_stream-role-fusion/` |
| `docs/artifacts/assets/stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20/` | `docs/artifacts/runs/2026-07-02_task-head-calibration/` |
| `docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/` | `docs/artifacts/runs/2026-07-02_cross-evidence-matrix/` |
| `docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/` | `docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/` |
| `docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/` | `docs/artifacts/runs/2026-07-02_public-fusion-ablation/` |
| `docs/artifacts/assets/stage_i_public_model_comparison/20260701T-stage-i-public-model-comparison-r1/` | `docs/artifacts/runs/2026-07-01_public-model-comparison/` |
| `docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/` | `docs/artifacts/runs/2026-07-01_public-fusion-calibration/` |
| `docs/artifacts/assets/stage_i_midterm_runtime/20260509T065500Z-stage-i-public-fusion-nasa-full-confirm/` | `docs/artifacts/runs/2026-05-09_public-fusion-nasa-full-confirm/` |
| `docs/artifacts/assets/stage_i_public_mainline/20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1/` | `docs/artifacts/runs/2026-05-08_public-mainline-uab-robust-prior-r1/` |
| `docs/artifacts/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/` | `docs/artifacts/runs/2026-05-08_public-opt-uab-robust-prior-r1/` |
| `docs/artifacts/assets/stage_i_public_opt_torch/20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1/` | `docs/artifacts/runs/2026-05-08_public-opt-uab-heat-specialist-r1/` |
| `docs/artifacts/assets/stage_i_public_opt/20260508T080318Z-stage-i-public-opt-nasa-prepared-v2/` | `docs/artifacts/runs/2026-05-08_public-opt-nasa-prepared-v2/` |
| `docs/artifacts/assets/stage_i_public_opt_torch/20260506T165558Z-stage-i-public-opt-uab-torch/` | `docs/artifacts/runs/2026-05-06_public-opt-uab-torch/` |
| `docs/artifacts/assets/stage_i_public_opt/20260506T161500Z-stage-i-public-opt-nasa-round1/` | `docs/artifacts/runs/2026-05-06_public-opt-nasa-round1/` |
| `docs/artifacts/assets/stage_i_public_opt/20260506T124000Z-stage-i-public-opt-nasa-prepared/` | `docs/artifacts/runs/2026-05-06_public-opt-nasa-prepared/` |
| `docs/artifacts/assets/stage_i_public_opt/20260506T121000Z-stage-i-public-opt-uab/` | `docs/artifacts/runs/2026-05-06_public-opt-uab/` |
| `docs/artifacts/assets/stage_i_public_fusion_screen/20260506T-stage-i-public-fusion-screen-round2/` | `docs/artifacts/runs/2026-05-06_public-fusion-screen-round2/` |
| `docs/artifacts/assets/stage_i_public_opt/20260504T161500Z-stage-i-public-opt-uab-prepared/` | `docs/artifacts/runs/2026-05-04_public-opt-uab-prepared/` |
| `docs/artifacts/assets/stage_i/20260506T-nasa-public-fusion-confirm/` | `docs/artifacts/runs/2026-05-06_public-fusion-nasa-confirm/` |
| `docs/artifacts/assets/stage_i/20260506T-uab-public-fusion-confirm/` | `docs/artifacts/runs/2026-05-06_public-fusion-uab-confirm/` |
| `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/` | `docs/artifacts/runs/2026-06-21_thesis-materials-report-figures/` |
| `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/` | `docs/artifacts/runs/2026-06-19_dingxin-leakage-safe-ablation/` |
| `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/` | `docs/artifacts/runs/2026-06-14_llm-preprocessing-context/` |
| `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/` | `docs/artifacts/runs/2026-06-14_llm-preprocessing-comparison/` |
| `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/` | `docs/artifacts/runs/2026-06-13_runtime-schema-contract/` |
| `docs/artifacts/assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/` | `docs/artifacts/runs/2026-06-07_evidence-closure/` |
| `docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/` | `docs/artifacts/runs/2026-06-07_dingxin-weak-label-multitask-sweep/` |
| `docs/artifacts/assets/stage_i_rotation_audit/20260607T-stage-i-evidence-closure-r2-rotation/` | `docs/artifacts/runs/2026-06-07_rotation-audit-closure/` |
| `docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/` | `docs/artifacts/runs/2026-06-07_rigid-body-diagnostics/` |
| `docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support/` | `docs/artifacts/runs/2026-05-06_semantic-support-baseline/` |
| `docs/artifacts/assets/stage_i_anchor/20260506T165435Z-stage-i-anchor/` | `docs/artifacts/runs/2026-05-06_anchor-windows/` |
| `docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/` | `docs/artifacts/runs/2026-05-04_dingxin-opt-package/` |
| `docs/artifacts/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/` | `docs/artifacts/runs/2026-05-01_deep-comparison-prepared/` |
| `docs/artifacts/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/` | `docs/artifacts/runs/2026-05-01_deep-real-sortie-prepared/` |
| `docs/artifacts/assets/stage_i/20260501T-full-loso-deep-comparison/` | `docs/artifacts/runs/2026-05-01_full-loso-deep-comparison/` |
| `docs/artifacts/assets/stage_i/20260429T000000Z-stage-i-phase2-case-study/` | `docs/artifacts/runs/2026-04-29_case-study/` |
| `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/` | `docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/` |
| `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/` | `docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/` |
| `docs/artifacts/assets/stage_h/20260427T000000Z-stage-h-closure/` | `docs/artifacts/runs/2026-04-27_feature-export-closure/` |
| `docs/artifacts/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-g-min/` | `docs/artifacts/runs/2026-04-22_alignment-g-min/` |
| `docs/artifacts/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-f-baseline/` | `docs/artifacts/runs/2026-04-22_alignment-g-baseline/` |
| `docs/artifacts/assets/alignment-preview-stage-f-closure-2026-04-22-stage-f-full/` | `docs/artifacts/runs/2026-04-22_alignment-f-full/` |
| `docs/artifacts/assets/alignment-preview-stage-f-closure-2026-04-22-e-baseline/` | `docs/artifacts/runs/2026-04-22_alignment-e-baseline/` |

## Archive

All remaining stage-numbered report directories and legacy assets under `docs/artifacts/` were moved under `docs/artifacts/archive/`.
