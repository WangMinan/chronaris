# Stage I Public Fusion Ablation - 20260702T-stage-i-public-fusion-ablation-gpuopt-r1

## Executive Summary

公开融合消融 decomposes the 公开融合刷新 chronaris_public_fusion refresh result on NASA/UAB. The public branch is evaluated as public_adapter_context_proxy_evidence, with the second stream recorded as context_derived_second_stream rather than Dingxin vehicle telemetry. The ablation table reports absolute and relative deltas for lag window, event bias, fusion normalization, stream contribution, fusion head, target transform and regression-loss settings.

## Dataset and public evidence role

- evidence_role: `public_adapter_context_proxy_evidence`
- source_prepared_roots: `{'nasa_csm': '/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/nasa_csm', 'uab_workload_dataset': '/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/uab_workload_dataset'}`
- 公开融合刷新 source run: `20260701T-stage-i-public-fusion-refresh-r1`

## Ablation design

- variant_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/ablation_variant_manifest.json`
- config: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/public_ablation_config.json`

## Main leaderboard

| dataset_id | variant_id | seed | primary_metric | selection_score | secondary_score | fold_count |
| --- | --- | --- | --- | --- | --- | --- |
| nasa_csm | no_lag_window | 42 | combined_macro_f1 | 0.3941 | 0.3859 | 17 |
| nasa_csm | full | 42 | combined_macro_f1 | 0.3272 | 0.3460 | 17 |
| nasa_csm | mse_loss | 42 | combined_macro_f1 | 0.3272 | 0.3460 | 17 |
| nasa_csm | no_event_bias | 42 | combined_macro_f1 | 0.3272 | 0.3460 | 17 |
| nasa_csm | no_target_transform | 42 | combined_macro_f1 | 0.3272 | 0.3460 | 17 |
| nasa_csm | context_only | 42 | combined_macro_f1 | 0.3023 | 0.3333 | 17 |
| nasa_csm | no_causal_fusion | 42 | combined_macro_f1 | 0.3023 | 0.3333 | 17 |
| nasa_csm | physiology_only | 42 | combined_macro_f1 | 0.3023 | 0.3333 | 17 |
| uab_workload_dataset | context_only | 42 | mean_rmse | 3.0666 | 2.4531 | 17 |
| uab_workload_dataset | physiology_only | 42 | mean_rmse | 3.1867 | 2.5596 | 17 |
| uab_workload_dataset | no_causal_fusion | 42 | mean_rmse | 3.2604 | 2.5806 | 17 |
| uab_workload_dataset | full | 42 | mean_rmse | 3.2791 | 2.5928 | 17 |
| uab_workload_dataset | mse_loss | 42 | mean_rmse | 3.3082 | 2.6328 | 17 |
| uab_workload_dataset | no_lag_window | 42 | mean_rmse | 3.3561 | 2.6575 | 17 |
| uab_workload_dataset | no_event_bias | 42 | mean_rmse | 3.3912 | 2.6838 | 17 |
| uab_workload_dataset | no_target_transform | 42 | mean_rmse | 3.4331 | 2.6908 | 17 |

## Ablation deltas

| dataset_id | task_group | metric | variant_id | value_mean | delta_abs_mean | delta_rel_pct_mean |
| --- | --- | --- | --- | --- | --- | --- |
| nasa_csm | combined | combined_macro_f1 | no_lag_window | 0.3941 | -0.0669 | -16.9781 |
| nasa_csm | combined | combined_balanced_accuracy | no_lag_window | 0.3859 | -0.0399 | -10.3286 |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | no_lag_window | 0.3026 | 0.0000 | 0.0000 |
| nasa_csm | loft_only | loft_only_macro_f1 | no_lag_window | 0.5267 | -0.0298 | -5.6561 |
| nasa_csm | combined | combined_macro_f1 | full | 0.3272 | 0.0000 | 0.0000 |
| nasa_csm | combined | combined_balanced_accuracy | full | 0.3460 | 0.0000 | 0.0000 |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | full | 0.3026 | 0.0000 | 0.0000 |
| nasa_csm | loft_only | loft_only_macro_f1 | full | 0.4969 | 0.0000 | 0.0000 |
| nasa_csm | combined | combined_macro_f1 | mse_loss | 0.3272 | 0.0000 | 0.0000 |
| nasa_csm | combined | combined_balanced_accuracy | mse_loss | 0.3460 | 0.0000 | 0.0000 |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | mse_loss | 0.3026 | 0.0000 | 0.0000 |
| nasa_csm | loft_only | loft_only_macro_f1 | mse_loss | 0.4969 | 0.0000 | 0.0000 |
| nasa_csm | combined | combined_macro_f1 | no_event_bias | 0.3272 | 0.0000 | 0.0000 |
| nasa_csm | combined | combined_balanced_accuracy | no_event_bias | 0.3460 | 0.0000 | 0.0000 |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | no_event_bias | 0.3026 | 0.0000 | 0.0000 |
| nasa_csm | loft_only | loft_only_macro_f1 | no_event_bias | 0.4969 | 0.0000 | 0.0000 |
| nasa_csm | combined | combined_macro_f1 | no_target_transform | 0.3272 | 0.0000 | 0.0000 |
| nasa_csm | combined | combined_balanced_accuracy | no_target_transform | 0.3460 | 0.0000 | 0.0000 |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | no_target_transform | 0.3026 | 0.0000 | 0.0000 |
| nasa_csm | loft_only | loft_only_macro_f1 | no_target_transform | 0.4969 | 0.0000 | 0.0000 |
| nasa_csm | combined | combined_macro_f1 | context_only | 0.3023 | 0.0248 | 8.2174 |
| nasa_csm | combined | combined_balanced_accuracy | context_only | 0.3333 | 0.0127 | 3.8043 |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | context_only | 0.3025 | 0.0001 | 0.0300 |
| nasa_csm | loft_only | loft_only_macro_f1 | context_only | 0.3022 | 0.1947 | 64.4210 |
| nasa_csm | combined | combined_macro_f1 | no_causal_fusion | 0.3023 | 0.0248 | 8.2174 |
| nasa_csm | combined | combined_balanced_accuracy | no_causal_fusion | 0.3333 | 0.0127 | 3.8043 |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | no_causal_fusion | 0.3025 | 0.0001 | 0.0300 |
| nasa_csm | loft_only | loft_only_macro_f1 | no_causal_fusion | 0.4519 | 0.0450 | 9.9664 |
| nasa_csm | combined | combined_macro_f1 | physiology_only | 0.3023 | 0.0248 | 8.2174 |
| nasa_csm | combined | combined_balanced_accuracy | physiology_only | 0.3333 | 0.0127 | 3.8043 |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | physiology_only | 0.3025 | 0.0001 | 0.0300 |
| nasa_csm | loft_only | loft_only_macro_f1 | physiology_only | 0.3022 | 0.1947 | 64.4210 |
| uab_workload_dataset | n_back | n_back_rmse | context_only | 4.6763 | -0.4195 | -8.2314 |
| uab_workload_dataset | n_back | n_back_mae | context_only | 3.7461 | -0.2739 | -6.8142 |
| uab_workload_dataset | heat_the_chair | heat_the_chair_rmse | context_only | 1.4570 | -0.0055 | -0.3735 |
| uab_workload_dataset | heat_the_chair | heat_the_chair_mae | context_only | 1.1600 | -0.0056 | -0.4823 |
| uab_workload_dataset | n_back_macro | n_back_macro_f1 | context_only | 0.2652 | 0.0925 | 34.8615 |
| uab_workload_dataset | n_back_balanced | n_back_balanced_accuracy | context_only | 0.3305 | 0.0507 | 15.3445 |
| uab_workload_dataset | heat_the_chair_macro | heat_the_chair_macro_f1 | context_only | 0.5205 | -0.0132 | -2.5438 |
| uab_workload_dataset | heat_the_chair_balanced | heat_the_chair_balanced_accuracy | context_only | 0.5225 | 0.0120 | 2.2911 |

## Figure index

- `fig_public_ablation_nasa_macro_f1`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/fig_public_ablation_nasa_macro_f1.png`
- `fig_public_ablation_nasa_balanced_accuracy`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/fig_public_ablation_nasa_balanced_accuracy.png`
- `fig_public_ablation_uab_rmse`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/fig_public_ablation_uab_rmse.png`
- `fig_public_ablation_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/fig_public_ablation_delta_heatmap.png`
- `fig_public_ablation_win_summary`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/fig_public_ablation_win_summary.png`
- `fig_public_ablation_config_sensitivity`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/fig_public_ablation_config_sensitivity.png`
- `fig_public_ablation_context_contribution`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/fig_public_ablation_context_contribution.png`
- `fig_public_ablation_component_contribution`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/fig_public_ablation_component_contribution.png`
- `fig_public_ablation_fold_stability`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/fig_public_ablation_fold_stability.png`
- `fig_public_ablation_gpu_throughput`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/fig_public_ablation_gpu_throughput.png`

## Reproducibility

- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1`
- screen_leaderboard: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/screen_leaderboard.csv`
- confirm_leaderboard: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/confirm_leaderboard.csv`
- fold_metrics: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/fold_metrics.csv`
- training_curves: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/training_curves.csv`
- raw_predictions: pruned from docs artifacts after `fold_metrics.csv`, summary tables and figures were materialized; rerun from `resume_command.txt` if row-level predictions are needed.
- evidence_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/evidence_manifest.json`
- run_log: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/run.log`
- progress: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/progress.json`

## Midterm-ready wording

公开融合消融 public ablation decomposes the 公开融合刷新 chronaris_public_fusion refresh result on NASA/UAB. The table reports how lag-aware fusion, event bias, public context stream use, causal fusion head, target transform and regression loss contribute to NASA attention-state classification and UAB workload regression under the same prepared public split protocol.
