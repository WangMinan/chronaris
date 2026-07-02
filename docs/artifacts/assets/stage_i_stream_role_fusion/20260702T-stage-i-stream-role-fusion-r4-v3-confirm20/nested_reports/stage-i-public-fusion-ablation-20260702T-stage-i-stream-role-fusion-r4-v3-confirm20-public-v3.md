# Stage I Public Fusion Ablation - 20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3

## Executive Summary

P31 decomposes the P28 chronaris_public_fusion refresh result on NASA/UAB. The public branch is evaluated as public_adapter_context_proxy_evidence, with the second stream recorded as context_proxy rather than private vehicle telemetry. The ablation table reports absolute and relative deltas for lag window, event bias, fusion normalization, stream contribution, fusion head, target transform and regression-loss settings.

## Dataset and public evidence role

- evidence_role: `public_adapter_context_proxy_evidence`
- source_prepared_roots: `{'nasa_csm': '/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/nasa_csm', 'uab_workload_dataset': '/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/uab_workload_dataset'}`
- P28 source run: `20260701T-stage-i-public-fusion-refresh-r1`

## Ablation design

- variant_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/ablation_variant_manifest.json`
- config: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/public_ablation_config.json`

## Main leaderboard

| dataset_id | variant_id | seed | primary_metric | selection_score | secondary_score | fold_count |
| --- | --- | --- | --- | --- | --- | --- |
| nasa_csm | v3_force_private_causal | 17 | combined_macro_f1 | 0.4822 | 0.4725 | 17 |
| nasa_csm | v3_force_private_causal | 42 | combined_macro_f1 | 0.4674 | 0.4465 | 17 |
| nasa_csm | v3_no_role_gate | 17 | combined_macro_f1 | 0.4573 | 0.4588 | 17 |
| nasa_csm | v3_stream_role | 17 | combined_macro_f1 | 0.4573 | 0.4588 | 17 |
| nasa_csm | v3_context_adapter_only | 17 | combined_macro_f1 | 0.4560 | 0.4513 | 17 |
| nasa_csm | v3_context_adapter_only | 42 | combined_macro_f1 | 0.4517 | 0.4429 | 17 |
| nasa_csm | v3_no_role_gate | 42 | combined_macro_f1 | 0.4434 | 0.4394 | 17 |
| nasa_csm | v3_stream_role | 42 | combined_macro_f1 | 0.4434 | 0.4394 | 17 |
| nasa_csm | v3_force_private_causal | 29 | combined_macro_f1 | 0.4370 | 0.4228 | 17 |
| nasa_csm | v3_context_adapter_only | 29 | combined_macro_f1 | 0.3963 | 0.4137 | 17 |
| nasa_csm | v3_no_role_gate | 29 | combined_macro_f1 | 0.3959 | 0.4117 | 17 |
| nasa_csm | v3_stream_role | 29 | combined_macro_f1 | 0.3959 | 0.4117 | 17 |
| uab_workload_dataset | v3_force_private_causal | 29 | mean_rmse | 3.2252 | 2.5837 | 17 |
| uab_workload_dataset | v3_force_private_causal | 42 | mean_rmse | 3.2351 | 2.5494 | 17 |
| uab_workload_dataset | v3_force_private_causal | 17 | mean_rmse | 3.2503 | 2.5984 | 17 |
| uab_workload_dataset | v3_context_adapter_only | 42 | mean_rmse | 3.2521 | 2.5668 | 17 |
| uab_workload_dataset | v3_context_adapter_only | 17 | mean_rmse | 3.2632 | 2.5969 | 17 |
| uab_workload_dataset | v3_stream_role | 42 | mean_rmse | 3.3428 | 2.6545 | 17 |
| uab_workload_dataset | v3_no_role_gate | 42 | mean_rmse | 3.3436 | 2.6409 | 17 |
| uab_workload_dataset | v3_no_role_gate | 17 | mean_rmse | 3.3456 | 2.6730 | 17 |
| uab_workload_dataset | v3_context_adapter_only | 29 | mean_rmse | 3.3502 | 2.6666 | 17 |
| uab_workload_dataset | v3_stream_role | 17 | mean_rmse | 3.3899 | 2.7070 | 17 |
| uab_workload_dataset | v3_no_role_gate | 29 | mean_rmse | 3.3908 | 2.6907 | 17 |
| uab_workload_dataset | v3_stream_role | 29 | mean_rmse | 3.3916 | 2.6913 | 17 |

## Ablation deltas

| dataset_id | task_group | metric | variant_id | value_mean | delta_abs_mean | delta_rel_pct_mean |
| --- | --- | --- | --- | --- | --- | --- |
| nasa_csm | combined | combined_macro_f1 | v3_force_private_causal | 0.4622 |  |  |
| nasa_csm | combined | combined_balanced_accuracy | v3_force_private_causal | 0.4473 |  |  |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | v3_force_private_causal | 0.3098 |  |  |
| nasa_csm | loft_only | loft_only_macro_f1 | v3_force_private_causal | 0.5035 |  |  |
| nasa_csm | combined | combined_macro_f1 | v3_no_role_gate | 0.4322 |  |  |
| nasa_csm | combined | combined_balanced_accuracy | v3_no_role_gate | 0.4366 |  |  |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | v3_no_role_gate | 0.3595 |  |  |
| nasa_csm | loft_only | loft_only_macro_f1 | v3_no_role_gate | 0.4966 |  |  |
| nasa_csm | combined | combined_macro_f1 | v3_stream_role | 0.4322 |  |  |
| nasa_csm | combined | combined_balanced_accuracy | v3_stream_role | 0.4366 |  |  |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | v3_stream_role | 0.5550 |  |  |
| nasa_csm | loft_only | loft_only_macro_f1 | v3_stream_role | 0.4966 |  |  |
| nasa_csm | combined | combined_macro_f1 | v3_context_adapter_only | 0.4347 |  |  |
| nasa_csm | combined | combined_balanced_accuracy | v3_context_adapter_only | 0.4360 |  |  |
| nasa_csm | benchmark_only | benchmark_only_macro_f1 | v3_context_adapter_only | 0.3597 |  |  |
| nasa_csm | loft_only | loft_only_macro_f1 | v3_context_adapter_only | 0.5048 |  |  |
| uab_workload_dataset | n_back | n_back_rmse | v3_force_private_causal | 5.0185 |  |  |
| uab_workload_dataset | n_back | n_back_mae | v3_force_private_causal | 4.0008 |  |  |
| uab_workload_dataset | heat_the_chair | heat_the_chair_rmse | v3_force_private_causal | 1.4552 |  |  |
| uab_workload_dataset | heat_the_chair | heat_the_chair_mae | v3_force_private_causal | 1.1535 |  |  |
| uab_workload_dataset | n_back_macro | n_back_macro_f1 | v3_force_private_causal | 0.3187 |  |  |
| uab_workload_dataset | n_back_balanced | n_back_balanced_accuracy | v3_force_private_causal | 0.3251 |  |  |
| uab_workload_dataset | heat_the_chair_macro | heat_the_chair_macro_f1 | v3_force_private_causal | 0.4889 |  |  |
| uab_workload_dataset | heat_the_chair_balanced | heat_the_chair_balanced_accuracy | v3_force_private_causal | 0.5376 |  |  |
| uab_workload_dataset | uab_mean | mean_rmse | v3_force_private_causal | 3.2369 |  |  |
| uab_workload_dataset | n_back | n_back_rmse | v3_context_adapter_only | 5.1188 |  |  |
| uab_workload_dataset | n_back | n_back_mae | v3_context_adapter_only | 4.0632 |  |  |
| uab_workload_dataset | heat_the_chair | heat_the_chair_rmse | v3_context_adapter_only | 1.4582 |  |  |
| uab_workload_dataset | heat_the_chair | heat_the_chair_mae | v3_context_adapter_only | 1.1570 |  |  |
| uab_workload_dataset | n_back_macro | n_back_macro_f1 | v3_context_adapter_only | 0.3233 |  |  |
| uab_workload_dataset | n_back_balanced | n_back_balanced_accuracy | v3_context_adapter_only | 0.3314 |  |  |
| uab_workload_dataset | heat_the_chair_macro | heat_the_chair_macro_f1 | v3_context_adapter_only | 0.4590 |  |  |
| uab_workload_dataset | heat_the_chair_balanced | heat_the_chair_balanced_accuracy | v3_context_adapter_only | 0.5429 |  |  |
| uab_workload_dataset | uab_mean | mean_rmse | v3_context_adapter_only | 3.2885 |  |  |
| uab_workload_dataset | n_back | n_back_rmse | v3_stream_role | 5.2898 |  |  |
| uab_workload_dataset | n_back | n_back_mae | v3_stream_role | 4.2090 |  |  |
| uab_workload_dataset | heat_the_chair | heat_the_chair_rmse | v3_stream_role | 1.4598 |  |  |
| uab_workload_dataset | heat_the_chair | heat_the_chair_mae | v3_stream_role | 1.1596 |  |  |
| uab_workload_dataset | n_back_macro | n_back_macro_f1 | v3_stream_role | 0.3511 |  |  |
| uab_workload_dataset | n_back_balanced | n_back_balanced_accuracy | v3_stream_role | 0.3683 |  |  |

## Figure index

- `fig_public_ablation_nasa_macro_f1`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/fig_public_ablation_nasa_macro_f1.png`
- `fig_public_ablation_nasa_balanced_accuracy`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/fig_public_ablation_nasa_balanced_accuracy.png`
- `fig_public_ablation_uab_rmse`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/fig_public_ablation_uab_rmse.png`
- `fig_public_ablation_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/fig_public_ablation_delta_heatmap.png`
- `fig_public_ablation_win_summary`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/fig_public_ablation_win_summary.png`
- `fig_public_ablation_config_sensitivity`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/fig_public_ablation_config_sensitivity.png`
- `fig_public_ablation_context_contribution`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/fig_public_ablation_context_contribution.png`
- `fig_public_ablation_component_contribution`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/fig_public_ablation_component_contribution.png`
- `fig_public_ablation_fold_stability`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/fig_public_ablation_fold_stability.png`
- `fig_public_ablation_gpu_throughput`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/fig_public_ablation_gpu_throughput.png`

## Reproducibility

- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3`
- screen_leaderboard: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/screen_leaderboard.csv`
- confirm_leaderboard: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/confirm_leaderboard.csv`
- fold_metrics: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/fold_metrics.csv`
- training_curves: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/training_curves.csv`
- predictions: pruned from git during 2026-07-02 docs cleanup; aggregate metrics and figures remain.
- evidence_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/evidence_manifest.json`
- run_log: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/run.log`
- progress: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/progress.json`

## Midterm-ready wording

P31 public ablation decomposes the P28 chronaris_public_fusion refresh result on NASA/UAB. The table reports how lag-aware fusion, event bias, public context stream use, causal fusion head, target transform and regression loss contribute to NASA attention-state classification and UAB workload regression under the same prepared public split protocol.
