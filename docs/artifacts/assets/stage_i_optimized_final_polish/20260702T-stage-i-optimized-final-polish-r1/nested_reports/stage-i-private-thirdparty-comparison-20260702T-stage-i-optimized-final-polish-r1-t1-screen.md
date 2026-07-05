# Stage I Dingxin Real-Data Third-party Comparison - 20260702T-stage-i-optimized-final-polish-r1-t1-screen

## Executive Summary

On the Dingxin / Stage H real dual-stream dataset, Chronaris is compared with MulT and ContiFormer under the same leakage-safe split manifest. The comparison uses real physiology and real vehicle time-series streams, with label-source fields and identity/time-position features excluded from model inputs. Across 分类任务、回归任务和检索任务 weak-label constructed tasks, the report provides model-level leaderboard, fold-level stability and Chronaris-vs-third-party deltas.

## Dataset and protocol

- evidence_role: dingxin_real_dual_stream
- sample_facts: `{'sample_count': 111, 'view_count': 3, 'sortie_count': 2}`
- split_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/split_manifest.json`

## Leakage-safe audit

- audit_json: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/label_feature_overlap_audit.json`
- audit_csv: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/label_feature_overlap_audit.csv`

## Main leaderboard

| task_name | split_strategy | model_name | metric | value_mean | value_std | seed_count |
| --- | --- | --- | --- | --- | --- | --- |
| 分类任务：机动强度分类 | leave_one_view_out | p37_t1_focal_gamma1_gate0p65_ls0p05 | macro_f1 | 0.1986 | 0.0000 | 1 |
| 分类任务：机动强度分类 | leave_one_view_out | p37_t1_focal_gamma1_gate0p65_ls0p05 | balanced_accuracy | 0.3504 | 0.0000 | 1 |
| 分类任务：机动强度分类 | leave_one_sortie_out | p37_t1_focal_gamma1_gate0p65_ls0p05 | macro_f1 | 0.2107 | 0.0000 | 1 |
| 分类任务：机动强度分类 | leave_one_sortie_out | p37_t1_focal_gamma1_gate0p65_ls0p05 | balanced_accuracy | 0.3269 | 0.0000 | 1 |
| 分类任务：机动强度分类 | leave_one_view_out | p37_t1_focal_gamma2_gate0p75_vehicle_skip_collapse0p05 | macro_f1 | 0.1700 | 0.0000 | 1 |
| 分类任务：机动强度分类 | leave_one_view_out | p37_t1_focal_gamma2_gate0p75_vehicle_skip_collapse0p05 | balanced_accuracy | 0.3248 | 0.0000 | 1 |
| 分类任务：机动强度分类 | leave_one_sortie_out | p37_t1_focal_gamma2_gate0p75_vehicle_skip_collapse0p05 | macro_f1 | 0.2362 | 0.0000 | 1 |
| 分类任务：机动强度分类 | leave_one_sortie_out | p37_t1_focal_gamma2_gate0p75_vehicle_skip_collapse0p05 | balanced_accuracy | 0.3397 | 0.0000 | 1 |
| 分类任务：机动强度分类 | leave_one_view_out | p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10 | macro_f1 | 0.1733 | 0.0000 | 1 |
| 分类任务：机动强度分类 | leave_one_view_out | p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10 | balanced_accuracy | 0.3333 | 0.0000 | 1 |
| 分类任务：机动强度分类 | leave_one_sortie_out | p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10 | macro_f1 | 0.2362 | 0.0000 | 1 |
| 分类任务：机动强度分类 | leave_one_sortie_out | p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10 | balanced_accuracy | 0.3397 | 0.0000 | 1 |

## Improvement over third-party baselines

_No rows._

## Figure index

- `fig_private_third_party_task_leaderboard`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_third_party_task_leaderboard.png`
- `fig_private_third_party_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_third_party_delta_heatmap.png`
- `fig_private_third_party_fold_variance`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_third_party_fold_variance.png`
- `fig_private_third_party_training_curves`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_third_party_training_curves.png`
- `fig_private_third_party_retrieval_topk`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_third_party_retrieval_topk.png`
- `fig_private_third_party_gpu_throughput`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_third_party_gpu_throughput.png`
- `fig_private_thirdparty_t1_macro_f1`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_thirdparty_t1_macro_f1.png`
- `fig_private_thirdparty_t2_rmse`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_thirdparty_t2_rmse.png`
- `fig_private_thirdparty_t3_retrieval`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_thirdparty_t3_retrieval.png`
- `fig_private_thirdparty_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_thirdparty_delta_heatmap.png`
- `fig_private_thirdparty_fold_stability`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_thirdparty_fold_stability.png`
- `fig_private_thirdparty_confusion_t1`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_thirdparty_confusion_t1.png`
- `fig_private_thirdparty_t2_error_distribution`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_thirdparty_t2_error_distribution.png`
- `fig_private_thirdparty_t3_retrieval_curve`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/fig_private_thirdparty_t3_retrieval_curve.png`

## Reproducibility

- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen`
- config: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/private_thirdparty_config.json`
- evidence_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/evidence_manifest.json`
- run_log: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/run.log`
- progress: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/20260702T-stage-i-optimized-final-polish-r1-t1-screen/progress.json`

## Midterm-ready wording

自有鼎新 / Stage H 分支在同一 leakage-safe 任务协议下比较 Chronaris、MulT 与 ContiFormer，量化真实生理流和真实航电流连续对齐场景中的模型适配性。分类任务、回归任务和检索任务 均保持 weak-label constructed task 标注边界，结果以 fold-level stability、mean/std 和 Chronaris-vs-baseline delta 展示。
