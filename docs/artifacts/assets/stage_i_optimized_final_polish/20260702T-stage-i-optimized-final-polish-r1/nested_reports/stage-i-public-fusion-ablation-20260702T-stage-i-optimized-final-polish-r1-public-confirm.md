# Stage I Public Fusion Ablation - 20260702T-stage-i-optimized-final-polish-r1-public-confirm

## Executive Summary

公开融合消融 decomposes the 公开融合刷新 chronaris_public_fusion refresh result on NASA/UAB. The public branch is evaluated as public_adapter_context_proxy_evidence, with the second stream recorded as context_derived_second_stream rather than Dingxin vehicle telemetry. The ablation table reports absolute and relative deltas for lag window, event bias, fusion normalization, stream contribution, fusion head, target transform and regression-loss settings.

## Dataset and public evidence role

- evidence_role: `public_adapter_context_proxy_evidence`
- source_prepared_roots: `{'nasa_csm': '/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/nasa_csm', 'uab_workload_dataset': '/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/uab_workload_dataset'}`
- 公开融合刷新 source run: `20260701T-stage-i-public-fusion-refresh-r1`

## Ablation design

- variant_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/ablation_variant_manifest.json`
- config: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/public_ablation_config.json`

## Main leaderboard

| dataset_id | variant_id | seed | primary_metric | selection_score | secondary_score | fold_count |
| --- | --- | --- | --- | --- | --- | --- |
| nasa_csm | p37_public_force_adaptive_context_gate | 42 | combined_macro_f1 | 0.4439 | 0.4505 | 17 |
| nasa_csm | p37_public_context_adapter_only_cap2x_do0p2 | 42 | combined_macro_f1 | 0.4048 | 0.4103 | 17 |
| uab_workload_dataset | p37_public_force_adaptive_context_gate | 42 | mean_rmse | 3.1918 | 2.5464 | 17 |
| uab_workload_dataset | p37_public_context_adapter_only_cap2x_do0p2 | 42 | mean_rmse | 3.3271 | 2.6217 | 17 |

## Ablation deltas

| dataset_id | task_group | metric | variant_id | value_mean | delta_abs_mean | delta_rel_pct_mean |
| --- | --- | --- | --- | --- | --- | --- |
| nasa_csm | combined | combined_macro_f1 | p37_public_force_adaptive_context_gate | 0.4439 |  |  |
| nasa_csm | combined | combined_balanced_accuracy | p37_public_force_adaptive_context_gate | 0.4505 |  |  |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | p37_public_force_adaptive_context_gate | 0.3698 |  |  |
| nasa_csm | loft_only | loft_only_macro_f1 | p37_public_force_adaptive_context_gate | 0.5077 |  |  |
| nasa_csm | combined | combined_macro_f1 | p37_public_context_adapter_only_cap2x_do0p2 | 0.4048 |  |  |
| nasa_csm | combined | combined_balanced_accuracy | p37_public_context_adapter_only_cap2x_do0p2 | 0.4103 |  |  |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | p37_public_context_adapter_only_cap2x_do0p2 | 0.3722 |  |  |
| nasa_csm | loft_only | loft_only_macro_f1 | p37_public_context_adapter_only_cap2x_do0p2 | 0.5549 |  |  |
| uab_workload_dataset | n_back | n_back_rmse | p37_public_force_adaptive_context_gate | 4.9175 |  |  |
| uab_workload_dataset | n_back | n_back_mae | p37_public_force_adaptive_context_gate | 3.9265 |  |  |
| uab_workload_dataset | heat_the_chair | heat_the_chair_rmse | p37_public_force_adaptive_context_gate | 1.4661 |  |  |
| uab_workload_dataset | heat_the_chair | heat_the_chair_mae | p37_public_force_adaptive_context_gate | 1.1663 |  |  |
| uab_workload_dataset | n_back_macro | n_back_macro_f1 | p37_public_force_adaptive_context_gate | 0.3468 |  |  |
| uab_workload_dataset | n_back_balanced | n_back_balanced_accuracy | p37_public_force_adaptive_context_gate | 0.3661 |  |  |
| uab_workload_dataset | heat_the_chair_macro | heat_the_chair_macro_f1 | p37_public_force_adaptive_context_gate | 0.4941 |  |  |
| uab_workload_dataset | heat_the_chair_balanced | heat_the_chair_balanced_accuracy | p37_public_force_adaptive_context_gate | 0.5365 |  |  |
| uab_workload_dataset | uab_mean | mean_rmse | p37_public_force_adaptive_context_gate | 3.1918 |  |  |
| uab_workload_dataset | n_back | n_back_rmse | p37_public_context_adapter_only_cap2x_do0p2 | 5.1919 |  |  |
| uab_workload_dataset | n_back | n_back_mae | p37_public_context_adapter_only_cap2x_do0p2 | 4.0838 |  |  |
| uab_workload_dataset | heat_the_chair | heat_the_chair_rmse | p37_public_context_adapter_only_cap2x_do0p2 | 1.4623 |  |  |
| uab_workload_dataset | heat_the_chair | heat_the_chair_mae | p37_public_context_adapter_only_cap2x_do0p2 | 1.1595 |  |  |
| uab_workload_dataset | n_back_macro | n_back_macro_f1 | p37_public_context_adapter_only_cap2x_do0p2 | 0.3610 |  |  |
| uab_workload_dataset | n_back_balanced | n_back_balanced_accuracy | p37_public_context_adapter_only_cap2x_do0p2 | 0.3613 |  |  |
| uab_workload_dataset | heat_the_chair_macro | heat_the_chair_macro_f1 | p37_public_context_adapter_only_cap2x_do0p2 | 0.5101 |  |  |
| uab_workload_dataset | heat_the_chair_balanced | heat_the_chair_balanced_accuracy | p37_public_context_adapter_only_cap2x_do0p2 | 0.5361 |  |  |
| uab_workload_dataset | uab_mean | mean_rmse | p37_public_context_adapter_only_cap2x_do0p2 | 3.3271 |  |  |

## Figure index

- `fig_public_ablation_nasa_macro_f1`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/fig_public_ablation_nasa_macro_f1.png`
- `fig_public_ablation_nasa_balanced_accuracy`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/fig_public_ablation_nasa_balanced_accuracy.png`
- `fig_public_ablation_uab_rmse`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/fig_public_ablation_uab_rmse.png`
- `fig_public_ablation_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/fig_public_ablation_delta_heatmap.png`
- `fig_public_ablation_win_summary`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/fig_public_ablation_win_summary.png`
- `fig_public_ablation_config_sensitivity`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/fig_public_ablation_config_sensitivity.png`
- `fig_public_ablation_context_contribution`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/fig_public_ablation_context_contribution.png`
- `fig_public_ablation_component_contribution`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/fig_public_ablation_component_contribution.png`
- `fig_public_ablation_fold_stability`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/fig_public_ablation_fold_stability.png`
- `fig_public_ablation_gpu_throughput`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/fig_public_ablation_gpu_throughput.png`

## Reproducibility

- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm`
- screen_leaderboard: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/screen_leaderboard.csv`
- confirm_leaderboard: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/confirm_leaderboard.csv`
- fold_metrics: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/fold_metrics.csv`
- training_curves: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/training_curves.csv`
- predictions: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/fold_predictions.csv`
- evidence_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/evidence_manifest.json`
- run_log: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/run.log`
- progress: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/progress.json`

## Midterm-ready wording

公开融合消融 public ablation decomposes the 公开融合刷新 chronaris_public_fusion refresh result on NASA/UAB. The table reports how lag-aware fusion, event bias, public context stream use, causal fusion head, target transform and regression loss contribute to NASA attention-state classification and UAB workload regression under the same prepared public split protocol.
