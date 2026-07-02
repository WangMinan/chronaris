# Stage I Optimized Final Polish - 20260702T-stage-i-optimized-final-polish-r1

## Executive Summary
- status: `completed`
- runtime_device: `cuda`
- boundary: P30/P31/P34/P35/P36 are fixed references; public rows are context-proxy evidence.
- decision: accept P37 T1 calibration and public route calibration where they improve confirmed references; keep P34 T3 retrieval as the confirmed result.

## Fixed references and protocol boundary
- P34 reference: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20`
- P35 reference: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20`

## T3 final polish
- metrics: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/t3_final_polish_metrics.csv`
- delta vs P34: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/p37_delta_vs_p34.csv`
| metric | p34_value | p37_model | p37_value | delta_positive_is_better |
| --- | --- | --- | --- | --- |
| top1 | 0.0315315 | p37_t3_info_nce_temp0p05_hardw2 | 0.0315315 | 0 |
| top3 | 0.0855856 | p37_t3_info_nce_temp0p05_hardw2 | 0.0855856 | 0 |
| top5 | 0.13964 | p37_t3_info_nce_temp0p05_hardw2 | 0.13964 | 0 |
| mrr | 0.117939 | p37_t3_info_nce_temp0p05_hardw2 | 0.117939 | 0 |

## T1 calibration
- metrics: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/t1_calibration_metrics.csv`
| split_strategy | metric | p34_value | p37_model | p37_value | delta_positive_is_better |
| --- | --- | --- | --- | --- | --- |
| leave_one_view_out | macro_f1 | 0.187489 | p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10 | 0.204614 | 0.0171246 |
| leave_one_view_out | balanced_accuracy | 0.344729 | p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10 | 0.353276 | 0.00854701 |
| leave_one_sortie_out | macro_f1 | 0.216065 | p37_t1_focal_gamma1_gate0p65_ls0p05 | 0.220099 | 0.00403356 |
| leave_one_sortie_out | balanced_accuracy | 0.34188 | p37_t1_focal_gamma1_gate0p65_ls0p05 | 0.337607 | -0.0042735 |
| task_name | split_strategy | model_name | metric | value_mean |
| --- | --- | --- | --- | --- |
| T1_maneuver_intensity_class | leave_one_view_out | p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10 | macro_f1 | 0.204614 |
| T1_maneuver_intensity_class | leave_one_view_out | p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10 | balanced_accuracy | 0.353276 |
| T1_maneuver_intensity_class | leave_one_sortie_out | p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10 | macro_f1 | 0.202212 |
| T1_maneuver_intensity_class | leave_one_sortie_out | p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10 | balanced_accuracy | 0.33547 |
| T1_maneuver_intensity_class | leave_one_view_out | p37_t1_focal_gamma1_gate0p65_ls0p05 | macro_f1 | 0.197315 |
| T1_maneuver_intensity_class | leave_one_view_out | p37_t1_focal_gamma1_gate0p65_ls0p05 | balanced_accuracy | 0.344729 |
| T1_maneuver_intensity_class | leave_one_sortie_out | p37_t1_focal_gamma1_gate0p65_ls0p05 | macro_f1 | 0.220099 |
| T1_maneuver_intensity_class | leave_one_sortie_out | p37_t1_focal_gamma1_gate0p65_ls0p05 | balanced_accuracy | 0.337607 |

## Public route calibration
- metrics: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/public_route_calibration_metrics.csv`
- gate calibration: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/route_gate_calibration.csv`
| dataset_id | variant_id | primary_metric | selection_score | combined_macro_f1 | mean_rmse |
| --- | --- | --- | --- | --- | --- |
| nasa_csm | p37_public_force_adaptive_context_gate | combined_macro_f1 | 0.443919 | 0.443919 |  |
| nasa_csm | p37_public_context_adapter_only_cap2x_do0p2 | combined_macro_f1 | 0.404797 | 0.404797 |  |
| uab_workload_dataset | p37_public_force_adaptive_context_gate | mean_rmse | 3.19181 |  | 3.19181 |
| uab_workload_dataset | p37_public_context_adapter_only_cap2x_do0p2 | mean_rmse | 3.32712 |  | 3.32712 |
| dataset_id | metric | p37_variant | p37_value | reference_value | delta_positive_is_better |
| --- | --- | --- | --- | --- | --- |
| nasa_csm | combined_macro_f1 | p37_public_force_adaptive_context_gate | 0.443919 | 0.432193 | 0.0117263 |
| nasa_csm | combined_macro_f1 | p37_public_context_adapter_only_cap2x_do0p2 | 0.404797 | 0.432193 | -0.027396 |
| uab_workload_dataset | mean_rmse | p37_public_force_adaptive_context_gate | 3.19181 | 3.37478 | 0.182968 |
| uab_workload_dataset | mean_rmse | p37_public_context_adapter_only_cap2x_do0p2 | 3.32712 | 3.37478 | 0.0476603 |
| scope | dataset_id | metric | reference | p37_value | reference_value | delta_positive_is_better |
| --- | --- | --- | --- | --- | --- | --- |
| private |  | macro_f1 | P30 chronaris_full | 0.204614 | 0.173333 | 0.0312803 |
| private |  | balanced_accuracy | P30 chronaris_full | 0.353276 | 0.333333 | 0.019943 |
| private |  | macro_f1 | P30 chronaris_full | 0.220099 | 0.173333 | 0.0467657 |
| private |  | balanced_accuracy | P30 chronaris_full | 0.337607 | 0.333333 | 0.0042735 |
| private |  | top1 | P30 chronaris_full | 0.0315315 | 0.027027 | 0.0045045 |
| private |  | top3 | P30 chronaris_full | 0.0855856 | 0.0945946 | -0.00900901 |
| private |  | top5 | P30 chronaris_full | 0.13964 | 0.148649 | -0.00900901 |
| private |  | mrr | P30 chronaris_full | 0.117939 | 0.119948 | -0.00200877 |
|  | nasa_csm | combined_macro_f1 | P31 no_lag_window | 0.443919 | 0.394103 | 0.0498162 |
|  | nasa_csm | combined_macro_f1 | P31 no_lag_window | 0.404797 | 0.394103 | 0.0106939 |
|  | uab_workload_dataset | mean_rmse | P31 context_only | 3.19181 | 4.6763 | 1.48448 |
|  | uab_workload_dataset | mean_rmse | P31 context_only | 3.32712 | 4.6763 | 1.34918 |

## Accepted / rejected candidates
- accepted: `{'accepted': [{'scope': 'T1', 'reason': 'P37 improves macro-F1 on at least one split.'}, {'scope': 'public_route', 'reason': 'P37 public route improves at least one P35 public metric.'}]}`
- rejected: `{'rejected': [{'scope': 'T3', 'reason': 'No P37 retrieval metric exceeded P34 in available rows.'}]}`

## GPU runtime summary
- GPU summary: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/gpu_perf_summary.json`

## Figure index
- fig_p37_t3_retrieval_leaderboard: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_t3_retrieval_leaderboard.png`
- fig_p37_t3_delta_vs_p34: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_t3_delta_vs_p34.png`
- fig_p37_t1_macro_f1_leaderboard: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_t1_macro_f1_leaderboard.png`
- fig_p37_t1_gate_sweep: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_t1_gate_sweep.png`
- fig_p37_t1_confusion_matrix_best: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_t1_confusion_matrix_best.png`
- fig_p37_public_route_nasa_macro_f1: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_public_route_nasa_macro_f1.png`
- fig_p37_public_route_uab_rmse: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_public_route_uab_rmse.png`
- fig_p37_route_gate_profile: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_route_gate_profile.png`
- fig_p37_public_route_delta_heatmap: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_public_route_delta_heatmap.png`
- fig_p37_overall_acceptance_summary: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_overall_acceptance_summary.png`
- fig_p37_gpu_runtime: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_gpu_runtime.png`
- fig_p37_t3_hard_negative_margin: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_t3_hard_negative_margin.png`
- fig_p37_t3_similarity_distribution: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_t3_similarity_distribution.png`

## Reproducibility manifest
- manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/evidence_manifest.json`
- resume: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/resume_command.txt`

## Midterm / thesis-ready wording
P37 final polish focuses on the two remaining diagnostic gaps after P34/P35: retrieval ranking and public context-proxy routing. The accepted P37 candidate is used only where it improves the P34/P35 confirmed metric under the same split and leakage boundary; otherwise P34/P35 remain the confirmed optimized result.
