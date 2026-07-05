# Stage I Public Fusion Refresh - 20260701T-stage-i-public-fusion-refresh-r1

## Executive Summary

公开融合刷新 refresh enlarged chronaris_public_fusion from the previous light screen setting to confirm runs. On NASA combined attention-state classification, the best refreshed chronaris_public_fusion achieved macro-F1=0.5657, outperforming the public baseline by +0.1107 absolute / +24.3% relative.
On UAB subjective workload regression, the best refreshed candidate achieved mean RMSE=3.3098, with per-task RMSE 5.1639 / 1.4558.

## Leaderboards

- status: `completed`
- screen_leaderboard: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/screen_leaderboard.csv`
- confirm_leaderboard: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/confirm_leaderboard.csv`
- fold_metrics: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fold_metrics.csv`
- training_curves: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/training_curves.csv`
- run_log: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/run.log`
- progress: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/progress.json`

## Best confirm rows

| dataset_id | candidate_id | seed | primary_metric | selection_score | secondary_score | fold_count | summary_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nasa_csm | fusion_h64_l2_hd4_do0p1_bias0p25_lag16_lr0p001_bs128_smooth_l1_td1_zscore_train_wd1em05 | 42 | combined_macro_f1 | 0.5657 | 0.5635 | 17 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/confirm/nasa_csm/fusion_h64_l2_hd4_do0p1_bias0p25_lag16_lr0p001_bs128_smooth_l1_td1_zscore_train_wd1em05__seed42/deep_baseline_summary.json |
| uab_workload_dataset | fusion_h64_l2_hd4_do0p1_bias0p25_lag16_lr0p001_bs128_smooth_l1_td1_zscore_train_wd1em05 | 42 | mean_rmse | 3.3098 | 2.6122 | 17 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/confirm/uab_workload_dataset/fusion_h64_l2_hd4_do0p1_bias0p25_lag16_lr0p001_bs128_smooth_l1_td1_zscore_train_wd1em05__seed42/deep_baseline_summary.json |

## Figure index

- `fig_public_fusion_refresh_screen_leaderboard`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fig_public_fusion_refresh_screen_leaderboard.png`
- `fig_public_fusion_refresh_confirm_vs_baselines`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fig_public_fusion_refresh_confirm_vs_baselines.png`
- `fig_public_fusion_refresh_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fig_public_fusion_refresh_delta_heatmap.png`
- `fig_public_fusion_config_sensitivity`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fig_public_fusion_config_sensitivity.png`
- `fig_public_fusion_training_curves_best`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fig_public_fusion_training_curves_best.png`
- `fig_public_fusion_best_confusion_nasa_combined`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fig_public_fusion_best_confusion_nasa_combined.png`
- `fig_public_fusion_win_summary`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fig_public_fusion_win_summary.png`

## Reproducibility

- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1`
- config: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fusion_refresh_config.json`
- candidate_grid: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/candidate_grid.json`
- evidence_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/evidence_manifest.json`
