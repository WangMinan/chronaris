# task evaluation Deep Baseline - uab_workload_dataset - chronaris_public_fusion

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-fusion-calibration/confirm/uab_workload_dataset/fusion_h64_l2_hd4_do0p1_bias0p25_lag16_lr0p001_bs128_smooth_l1_td1_zscore_train_wd1em05__seed42`
- prepared root: `/tmp/chronaris_task_eval_public_fusion_refresh/20260701T-task-eval-public-fusion-refresh-r1/uab_workload_dataset`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.320230 | 0.342825 | 28052 | 16 |
| `heat_the_chair` | 0.504836 | 0.533471 | 5440 | 17 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 5.163879 | 4.073223 | -0.263240 | -0.102675 | 28052 | 16 |
| `heat_the_chair` | 1.455792 | 1.151207 | -0.083054 | -0.695269 | 5440 | 17 |
