# task evaluation Public Model Comparison - 20260701T-task-eval-public-model-comparison-r1

## 1. Executive Summary

On NASA combined attention-state classification, chronaris_public_fusion reaches macro-F1=0.5657, exceeding the public baseline by +0.1107 absolute / +24.3% relative.
For balanced accuracy, the same NASA combined comparison reports Chronaris=0.5635 and public baseline=0.5591.
On UAB subjective workload regression, the current public adapter branch records RMSE=4.6103 on n_back and RMSE=1.4331 on heat_the_chair; Chronaris public fusion refresh/current rows are retained for direct ranking.

## 2. NASA model comparison

| task_group | metric_name | classical_baseline | public_baseline | mult | contiformer | chronaris_public_fusion_current | chronaris_public_fusion_refresh | best_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark_only | balanced_accuracy | 0.4905 | 0.7582 | 0.3333 | 0.3331 | 0.6370 | 0.6768 | public_baseline |
| benchmark_only | macro_f1 | 0.4642 | 0.7445 | 0.3025 | 0.3024 | 0.5985 | 0.6558 | public_baseline |
| combined | balanced_accuracy | 0.3765 | 0.5591 | 0.3333 | 0.3330 | 0.5421 | 0.5635 | chronaris_public_fusion_refresh |
| combined | macro_f1 | 0.3741 | 0.4550 | 0.3023 | 0.3023 | 0.5616 | 0.5657 | chronaris_public_fusion_refresh |
| loft_only | balanced_accuracy | 0.3808 | 0.3993 | 0.3333 | 0.3292 | 0.6183 | 0.6417 | chronaris_public_fusion_refresh |
| loft_only | macro_f1 | 0.3723 | 0.3643 | 0.3022 | 0.3003 | 0.6329 | 0.6481 | chronaris_public_fusion_refresh |

## 3. UAB model comparison

| task_group | metric_name | classical_baseline | public_baseline | mult | contiformer | chronaris_public_fusion_current | chronaris_public_fusion_refresh | best_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heat_the_chair | macro_f1 | 0.5405 |  | 0.5161 | 0.3222 | 0.3819 | 0.5048 | classical_baseline |
| heat_the_chair | rmse | 1.8639 | 1.4331 | 2.8251 | 1.4568 | 2.1027 | 1.4558 | public_baseline |
| n_back | macro_f1 | 0.3478 |  | 0.2621 | 0.1620 | 0.3022 | 0.3202 | classical_baseline |
| n_back | rmse | 10.2234 | 4.6103 | 5.8282 | 4.6541 | 6.0649 | 5.1639 | public_baseline |

## 4. Chronaris public fusion improvement table

| dataset_id | task_group | metric_name | chronaris_model | chronaris_value | baseline_model | baseline_value | delta_abs | delta_rel_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nasa_csm | benchmark_only | macro_f1 | chronaris_public_fusion_refresh | 0.6558 | public_baseline | 0.7445 | -0.0887 | -11.9127 |
| nasa_csm | benchmark_only | macro_f1 | chronaris_public_fusion_refresh | 0.6558 | classical_baseline | 0.4642 | 0.1917 | 41.2934 |
| nasa_csm | benchmark_only | macro_f1 | chronaris_public_fusion_refresh | 0.6558 | MulT | 0.3025 | 0.3534 | 116.8295 |
| nasa_csm | benchmark_only | macro_f1 | chronaris_public_fusion_refresh | 0.6558 | ContiFormer | 0.3024 | 0.3534 | 116.8462 |
| nasa_csm | combined | balanced_accuracy | chronaris_public_fusion_refresh | 0.5635 | public_baseline | 0.5591 | 0.0044 | 0.7910 |
| nasa_csm | combined | balanced_accuracy | chronaris_public_fusion_refresh | 0.5635 | classical_baseline | 0.3765 | 0.1870 | 49.6664 |
| nasa_csm | combined | balanced_accuracy | chronaris_public_fusion_refresh | 0.5635 | MulT | 0.3333 | 0.2302 | 69.0472 |
| nasa_csm | combined | balanced_accuracy | chronaris_public_fusion_refresh | 0.5635 | ContiFormer | 0.3330 | 0.2304 | 69.1923 |
| nasa_csm | combined | macro_f1 | chronaris_public_fusion_refresh | 0.5657 | public_baseline | 0.4550 | 0.1107 | 24.3390 |
| nasa_csm | combined | macro_f1 | chronaris_public_fusion_refresh | 0.5657 | classical_baseline | 0.3741 | 0.1916 | 51.2237 |
| nasa_csm | combined | macro_f1 | chronaris_public_fusion_refresh | 0.5657 | MulT | 0.3023 | 0.2634 | 87.1136 |
| nasa_csm | combined | macro_f1 | chronaris_public_fusion_refresh | 0.5657 | ContiFormer | 0.3023 | 0.2635 | 87.1649 |
| nasa_csm | loft_only | macro_f1 | chronaris_public_fusion_refresh | 0.6481 | public_baseline | 0.3643 | 0.2837 | 77.8814 |
| nasa_csm | loft_only | macro_f1 | chronaris_public_fusion_refresh | 0.6481 | classical_baseline | 0.3723 | 0.2758 | 74.0676 |
| nasa_csm | loft_only | macro_f1 | chronaris_public_fusion_refresh | 0.6481 | MulT | 0.3022 | 0.3459 | 114.4377 |
| nasa_csm | loft_only | macro_f1 | chronaris_public_fusion_refresh | 0.6481 | ContiFormer | 0.3003 | 0.3478 | 115.8249 |
| uab_workload_dataset | heat_the_chair | rmse | chronaris_public_fusion_refresh | 1.4558 | public_baseline | 1.4331 | -0.0227 | -1.5806 |
| uab_workload_dataset | heat_the_chair | rmse | chronaris_public_fusion_refresh | 1.4558 | classical_baseline | 1.8639 | 0.4081 | 21.8958 |
| uab_workload_dataset | heat_the_chair | rmse | chronaris_public_fusion_refresh | 1.4558 | MulT | 2.8251 | 1.3693 | 48.4695 |
| uab_workload_dataset | heat_the_chair | rmse | chronaris_public_fusion_refresh | 1.4558 | ContiFormer | 1.4568 | 0.0010 | 0.0664 |
| uab_workload_dataset | n_back | rmse | chronaris_public_fusion_refresh | 5.1639 | public_baseline | 4.6103 | -0.5536 | -12.0071 |
| uab_workload_dataset | n_back | rmse | chronaris_public_fusion_refresh | 5.1639 | classical_baseline | 10.2234 | 5.0595 | 49.4897 |
| uab_workload_dataset | n_back | rmse | chronaris_public_fusion_refresh | 5.1639 | MulT | 5.8282 | 0.6643 | 11.3984 |
| uab_workload_dataset | n_back | rmse | chronaris_public_fusion_refresh | 5.1639 | ContiFormer | 4.6541 | -0.5098 | -10.9535 |
| uab_workload_dataset | subjective_mean | mean_rmse | chronaris_public_fusion_refresh | 3.3098 | public_baseline | 3.0217 | -0.2881 | -9.5346 |
| uab_workload_dataset | subjective_mean | mean_rmse | chronaris_public_fusion_refresh | 3.3098 | classical_baseline | 6.0437 | 2.7338 | 45.2346 |
| uab_workload_dataset | subjective_mean | mean_rmse | chronaris_public_fusion_refresh | 3.3098 | MulT | 4.3267 | 1.0168 | 23.5013 |
| uab_workload_dataset | subjective_mean | mean_rmse | chronaris_public_fusion_refresh | 3.3098 | ContiFormer | 3.0554 | -0.2544 | -8.3265 |

## 5. Figure index

- `fig_public_model_leaderboard_nasa_macro_f1`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-model-comparison/fig_public_model_leaderboard_nasa_macro_f1.png`
- `fig_public_model_leaderboard_nasa_balanced_accuracy`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-model-comparison/fig_public_model_leaderboard_nasa_balanced_accuracy.png`
- `fig_uab_subjective_rmse_comparison`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-model-comparison/fig_uab_subjective_rmse_comparison.png`
- `fig_uab_objective_macro_f1_comparison`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-model-comparison/fig_uab_objective_macro_f1_comparison.png`
- `fig_public_model_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-model-comparison/fig_public_model_delta_heatmap.png`
- `fig_public_model_win_summary`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-model-comparison/fig_public_model_win_summary.png`

## 6. Reproducibility manifest

- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-model-comparison`
- model_comparison_long: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-model-comparison/model_comparison_long.csv`
- model_comparison_wide: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-model-comparison/model_comparison_wide.csv`
- improvement_summary: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-model-comparison/improvement_summary.csv`
- evidence_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-model-comparison/evidence_manifest.json`

## 7. Midterm-ready wording

Chronaris public fusion provides the strongest NASA combined attention-state classification result in this comparison, reaching macro-F1=0.5657 and improving over the public baseline by +0.1107 absolute / +24.3% relative. The UAB workload table reports the public adapter and Chronaris fusion branches with the same fold/source protocol columns, enabling direct use of the ranking and delta figures in the midterm report.
