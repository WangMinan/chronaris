# Stage I Private Third-party Comparison - 20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1

## Executive Summary

On the private Dingxin / Stage H real dual-stream dataset, Chronaris is compared with MulT and ContiFormer under the same leakage-safe split manifest. The comparison uses real physiology and real vehicle time-series streams, with label-source fields and identity/time-position features excluded from model inputs. Across T1/T2/T3 proxy tasks, the report provides model-level leaderboard, fold-level stability and Chronaris-vs-third-party deltas.

## Dataset and protocol

- evidence_role: `private_real_dual_stream`
- sample_facts: `{'sample_count': 111, 'view_count': 3, 'sortie_count': 2}`
- split_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/split_manifest.json`

## Leakage-safe audit

- audit_json: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/label_feature_overlap_audit.json`
- audit_csv: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/label_feature_overlap_audit.csv`

## Main leaderboard

| task_name | split_strategy | model_name | metric | value_mean | value_std | seed_count |
| --- | --- | --- | --- | --- | --- | --- |
| T1_maneuver_intensity_class | leave_one_view_out | chronaris_full | macro_f1 | 0.1733 | 0.0000 | 3 |
| T1_maneuver_intensity_class | leave_one_view_out | chronaris_full | balanced_accuracy | 0.3333 | 0.0000 | 3 |
| T1_maneuver_intensity_class | leave_one_sortie_out | chronaris_full | macro_f1 | 0.1733 | 0.0000 | 3 |
| T1_maneuver_intensity_class | leave_one_sortie_out | chronaris_full | balanced_accuracy | 0.3333 | 0.0000 | 3 |
| T1_maneuver_intensity_class | leave_one_view_out | mult | macro_f1 | 0.2035 | 0.0156 | 3 |
| T1_maneuver_intensity_class | leave_one_view_out | mult | balanced_accuracy | 0.3533 | 0.0081 | 3 |
| T1_maneuver_intensity_class | leave_one_sortie_out | mult | macro_f1 | 0.2132 | 0.0148 | 3 |
| T1_maneuver_intensity_class | leave_one_sortie_out | mult | balanced_accuracy | 0.3397 | 0.0052 | 3 |
| T1_maneuver_intensity_class | leave_one_view_out | contiformer | macro_f1 | 0.2096 | 0.0255 | 3 |
| T1_maneuver_intensity_class | leave_one_view_out | contiformer | balanced_accuracy | 0.3561 | 0.0145 | 3 |
| T1_maneuver_intensity_class | leave_one_sortie_out | contiformer | macro_f1 | 0.2230 | 0.0122 | 3 |
| T1_maneuver_intensity_class | leave_one_sortie_out | contiformer | balanced_accuracy | 0.3376 | 0.0060 | 3 |
| T1_maneuver_intensity_class | leave_one_view_out | classical_baseline | macro_f1 | 0.2118 | 0.0000 | 3 |
| T1_maneuver_intensity_class | leave_one_view_out | classical_baseline | balanced_accuracy | 0.3504 | 0.0000 | 3 |
| T1_maneuver_intensity_class | leave_one_sortie_out | classical_baseline | macro_f1 | 0.1888 | 0.0000 | 3 |
| T1_maneuver_intensity_class | leave_one_sortie_out | classical_baseline | balanced_accuracy | 0.3333 | 0.0000 | 3 |
| T2_next_window_physiology_response | leave_one_view_out | chronaris_full | rmse | 838.1210 | 0.0000 | 3 |
| T2_next_window_physiology_response | leave_one_view_out | chronaris_full | mae | 762.5579 | 0.0000 | 3 |
| T2_next_window_physiology_response | leave_one_view_out | chronaris_full | nrmse | 8.7810 | 0.0000 | 3 |
| T2_next_window_physiology_response | leave_one_sortie_out | chronaris_full | rmse | 1111.9789 | 0.0000 | 3 |
| T2_next_window_physiology_response | leave_one_sortie_out | chronaris_full | mae | 992.5239 | 0.0000 | 3 |
| T2_next_window_physiology_response | leave_one_sortie_out | chronaris_full | nrmse | 11.3163 | 0.0000 | 3 |
| T2_next_window_physiology_response | leave_one_view_out | mult | rmse | 344.2890 | 0.3901 | 3 |
| T2_next_window_physiology_response | leave_one_view_out | mult | mae | 273.3819 | 0.4420 | 3 |
| T2_next_window_physiology_response | leave_one_view_out | mult | nrmse | 2.2379 | 0.0045 | 3 |
| T2_next_window_physiology_response | leave_one_sortie_out | mult | rmse | 413.4298 | 0.4768 | 3 |
| T2_next_window_physiology_response | leave_one_sortie_out | mult | mae | 314.7452 | 0.6979 | 3 |
| T2_next_window_physiology_response | leave_one_sortie_out | mult | nrmse | 2.0426 | 0.0011 | 3 |
| T2_next_window_physiology_response | leave_one_view_out | contiformer | rmse | 344.3351 | 0.0458 | 3 |
| T2_next_window_physiology_response | leave_one_view_out | contiformer | mae | 273.5299 | 0.0409 | 3 |
| T2_next_window_physiology_response | leave_one_view_out | contiformer | nrmse | 2.2361 | 0.0014 | 3 |
| T2_next_window_physiology_response | leave_one_sortie_out | contiformer | rmse | 414.2889 | 0.2751 | 3 |
| T2_next_window_physiology_response | leave_one_sortie_out | contiformer | mae | 315.9286 | 0.3334 | 3 |
| T2_next_window_physiology_response | leave_one_sortie_out | contiformer | nrmse | 2.0463 | 0.0027 | 3 |
| T2_next_window_physiology_response | leave_one_view_out | classical_baseline | rmse | 463.1366 | 0.0000 | 3 |
| T2_next_window_physiology_response | leave_one_view_out | classical_baseline | mae | 360.9690 | 0.0000 | 3 |
| T2_next_window_physiology_response | leave_one_view_out | classical_baseline | nrmse | 4.1115 | 0.0000 | 3 |
| T2_next_window_physiology_response | leave_one_sortie_out | classical_baseline | rmse | 11284.2992 | 0.0000 | 3 |
| T2_next_window_physiology_response | leave_one_sortie_out | classical_baseline | mae | 8571.0432 | 0.0000 | 3 |
| T2_next_window_physiology_response | leave_one_sortie_out | classical_baseline | nrmse | 141.4624 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | chronaris_full | top1 | 0.0270 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | chronaris_full | top3 | 0.0946 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | chronaris_full | top5 | 0.1486 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | chronaris_full | mrr | 0.1199 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_sortie_out | chronaris_full | top1 | 0.0270 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_sortie_out | chronaris_full | top3 | 0.0946 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_sortie_out | chronaris_full | top5 | 0.1486 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_sortie_out | chronaris_full | mrr | 0.1199 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | mult | top1 | 0.0270 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | mult | top3 | 0.0901 | 0.0064 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | mult | top5 | 0.1441 | 0.0064 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | mult | mrr | 0.1178 | 0.0030 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | contiformer | top1 | 0.0270 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | contiformer | top3 | 0.0856 | 0.0064 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | contiformer | top5 | 0.1396 | 0.0064 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | contiformer | mrr | 0.1157 | 0.0030 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | naive_time_sync | top1 | 0.0270 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | naive_time_sync | top3 | 0.0811 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | naive_time_sync | top5 | 0.1351 | 0.0000 | 3 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | naive_time_sync | mrr | 0.1136 | 0.0000 | 3 |

## Improvement over third-party baselines

| task_name | split_strategy | metric | baseline_model | chronaris_value | baseline_value | delta_abs | delta_rel_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1_maneuver_intensity_class | leave_one_view_out | macro_f1 | mult | 0.1733 | 0.2035 | -0.0301 | -14.8070 |
| T1_maneuver_intensity_class | leave_one_view_out | macro_f1 | contiformer | 0.1733 | 0.2096 | -0.0362 | -17.2862 |
| T1_maneuver_intensity_class | leave_one_view_out | macro_f1 | classical_baseline | 0.1733 | 0.2118 | -0.0385 | -18.1762 |
| T1_maneuver_intensity_class | leave_one_view_out | balanced_accuracy | mult | 0.3333 | 0.3533 | -0.0199 | -5.6452 |
| T1_maneuver_intensity_class | leave_one_view_out | balanced_accuracy | contiformer | 0.3333 | 0.3561 | -0.0228 | -6.4000 |
| T1_maneuver_intensity_class | leave_one_view_out | balanced_accuracy | classical_baseline | 0.3333 | 0.3504 | -0.0171 | -4.8780 |
| T1_maneuver_intensity_class | leave_one_sortie_out | macro_f1 | mult | 0.1733 | 0.2132 | -0.0399 | -18.6946 |
| T1_maneuver_intensity_class | leave_one_sortie_out | macro_f1 | contiformer | 0.1733 | 0.2230 | -0.0497 | -22.2691 |
| T1_maneuver_intensity_class | leave_one_sortie_out | macro_f1 | classical_baseline | 0.1733 | 0.1888 | -0.0154 | -8.1802 |
| T1_maneuver_intensity_class | leave_one_sortie_out | balanced_accuracy | mult | 0.3333 | 0.3397 | -0.0064 | -1.8868 |
| T1_maneuver_intensity_class | leave_one_sortie_out | balanced_accuracy | contiformer | 0.3333 | 0.3376 | -0.0043 | -1.2658 |
| T1_maneuver_intensity_class | leave_one_sortie_out | balanced_accuracy | classical_baseline | 0.3333 | 0.3333 | 0.0000 | 0.0000 |
| T2_next_window_physiology_response | leave_one_view_out | rmse | mult | 838.1210 | 344.2890 | -493.8321 | -143.4353 |
| T2_next_window_physiology_response | leave_one_view_out | rmse | contiformer | 838.1210 | 344.3351 | -493.7859 | -143.4027 |
| T2_next_window_physiology_response | leave_one_view_out | rmse | classical_baseline | 838.1210 | 463.1366 | -374.9844 | -80.9663 |
| T2_next_window_physiology_response | leave_one_view_out | mae | mult | 762.5579 | 273.3819 | -489.1760 | -178.9351 |
| T2_next_window_physiology_response | leave_one_view_out | mae | contiformer | 762.5579 | 273.5299 | -489.0280 | -178.7842 |
| T2_next_window_physiology_response | leave_one_view_out | mae | classical_baseline | 762.5579 | 360.9690 | -401.5889 | -111.2530 |
| T2_next_window_physiology_response | leave_one_view_out | nrmse | mult | 8.7810 | 2.2379 | -6.5431 | -292.3790 |
| T2_next_window_physiology_response | leave_one_view_out | nrmse | contiformer | 8.7810 | 2.2361 | -6.5449 | -292.6953 |
| T2_next_window_physiology_response | leave_one_view_out | nrmse | classical_baseline | 8.7810 | 4.1115 | -4.6695 | -113.5723 |
| T2_next_window_physiology_response | leave_one_sortie_out | rmse | mult | 1111.9789 | 413.4298 | -698.5492 | -168.9644 |
| T2_next_window_physiology_response | leave_one_sortie_out | rmse | contiformer | 1111.9789 | 414.2889 | -697.6900 | -168.4066 |
| T2_next_window_physiology_response | leave_one_sortie_out | rmse | classical_baseline | 1111.9789 | 11284.2992 | 10172.3202 | 90.1458 |
| T2_next_window_physiology_response | leave_one_sortie_out | mae | mult | 992.5239 | 314.7452 | -677.7786 | -215.3420 |
| T2_next_window_physiology_response | leave_one_sortie_out | mae | contiformer | 992.5239 | 315.9286 | -676.5953 | -214.1608 |
| T2_next_window_physiology_response | leave_one_sortie_out | mae | classical_baseline | 992.5239 | 8571.0432 | 7578.5194 | 88.4200 |
| T2_next_window_physiology_response | leave_one_sortie_out | nrmse | mult | 11.3163 | 2.0426 | -9.2738 | -454.0296 |
| T2_next_window_physiology_response | leave_one_sortie_out | nrmse | contiformer | 11.3163 | 2.0463 | -9.2701 | -453.0185 |
| T2_next_window_physiology_response | leave_one_sortie_out | nrmse | classical_baseline | 11.3163 | 141.4624 | 130.1461 | 92.0005 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | top1 | mult | 0.0270 | 0.0270 | 0.0000 | 0.0000 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | top1 | contiformer | 0.0270 | 0.0270 | 0.0000 | 0.0000 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | top1 | naive_time_sync | 0.0270 | 0.0270 | 0.0000 | 0.0000 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | top1 | classical_baseline | 0.0270 | 0.0676 | -0.0405 | -60.0000 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | top3 | mult | 0.0946 | 0.0901 | 0.0045 | 5.0000 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | top3 | contiformer | 0.0946 | 0.0856 | 0.0090 | 10.5263 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | top3 | naive_time_sync | 0.0946 | 0.0811 | 0.0135 | 16.6667 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | top3 | classical_baseline | 0.0946 | 0.1216 | -0.0270 | -22.2222 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | top5 | mult | 0.1486 | 0.1441 | 0.0045 | 3.1250 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | top5 | contiformer | 0.1486 | 0.1396 | 0.0090 | 6.4516 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | top5 | naive_time_sync | 0.1486 | 0.1351 | 0.0135 | 10.0000 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | top5 | classical_baseline | 0.1486 | 0.1757 | -0.0270 | -15.3846 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | mrr | mult | 0.1199 | 0.1178 | 0.0021 | 1.8083 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | mrr | contiformer | 0.1199 | 0.1157 | 0.0043 | 3.6832 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | mrr | naive_time_sync | 0.1199 | 0.1136 | 0.0064 | 5.6285 |
| T3_paired_pilot_window_retrieval | leave_one_view_out | mrr | classical_baseline | 0.1199 | 0.1530 | -0.0330 | -21.5819 |
| T3_paired_pilot_window_retrieval | leave_one_sortie_out | top1 | naive_time_sync | 0.0270 | 0.0270 | 0.0000 | 0.0000 |
| T3_paired_pilot_window_retrieval | leave_one_sortie_out | top1 | classical_baseline | 0.0270 | 0.0676 | -0.0405 | -60.0000 |
| T3_paired_pilot_window_retrieval | leave_one_sortie_out | top3 | naive_time_sync | 0.0946 | 0.0811 | 0.0135 | 16.6667 |
| T3_paired_pilot_window_retrieval | leave_one_sortie_out | top3 | classical_baseline | 0.0946 | 0.1216 | -0.0270 | -22.2222 |
| T3_paired_pilot_window_retrieval | leave_one_sortie_out | top5 | naive_time_sync | 0.1486 | 0.1351 | 0.0135 | 10.0000 |
| T3_paired_pilot_window_retrieval | leave_one_sortie_out | top5 | classical_baseline | 0.1486 | 0.1757 | -0.0270 | -15.3846 |
| T3_paired_pilot_window_retrieval | leave_one_sortie_out | mrr | naive_time_sync | 0.1199 | 0.1136 | 0.0064 | 5.6285 |
| T3_paired_pilot_window_retrieval | leave_one_sortie_out | mrr | classical_baseline | 0.1199 | 0.1530 | -0.0330 | -21.5819 |

## Figure index

- `fig_private_third_party_task_leaderboard`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_third_party_task_leaderboard.png`
- `fig_private_third_party_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_third_party_delta_heatmap.png`
- `fig_private_third_party_fold_variance`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_third_party_fold_variance.png`
- `fig_private_third_party_training_curves`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_third_party_training_curves.png`
- `fig_private_third_party_retrieval_topk`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_third_party_retrieval_topk.png`
- `fig_private_third_party_gpu_throughput`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_third_party_gpu_throughput.png`
- `fig_private_thirdparty_t1_macro_f1`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t1_macro_f1.png`
- `fig_private_thirdparty_t2_rmse`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t2_rmse.png`
- `fig_private_thirdparty_t3_retrieval`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t3_retrieval.png`
- `fig_private_thirdparty_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png`
- `fig_private_thirdparty_fold_stability`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_fold_stability.png`
- `fig_private_thirdparty_confusion_t1`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_confusion_t1.png`
- `fig_private_thirdparty_t2_error_distribution`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t2_error_distribution.png`
- `fig_private_thirdparty_t3_retrieval_curve`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t3_retrieval_curve.png`

## Reproducibility

- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1`
- config: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/private_thirdparty_config.json`
- evidence_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/evidence_manifest.json`
- run_log: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/run.log`
- progress: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/progress.json`

## Midterm-ready wording

自有鼎新 / Stage H 分支在同一 leakage-safe 任务协议下比较 Chronaris、MulT 与 ContiFormer，量化真实生理流和真实航电流连续对齐场景中的模型适配性。T1/T2/T3 均保持 proxy task 标注边界，结果以 fold-level stability、mean/std 和 Chronaris-vs-baseline delta 展示。
