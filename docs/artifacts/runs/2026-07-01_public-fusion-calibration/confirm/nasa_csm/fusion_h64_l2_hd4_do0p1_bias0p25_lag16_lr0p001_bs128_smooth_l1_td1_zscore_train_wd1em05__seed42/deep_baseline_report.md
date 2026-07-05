# task evaluation Deep Baseline - nasa_csm - chronaris_public_fusion

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-fusion-calibration/confirm/nasa_csm/fusion_h64_l2_hd4_do0p1_bias0p25_lag16_lr0p001_bs128_smooth_l1_td1_zscore_train_wd1em05__seed42`
- prepared root: `/tmp/chronaris_task_eval_public_fusion_refresh/20260701T-task-eval-public-fusion-refresh-r1/nasa_csm`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `benchmark_only` | 0.655822 | 0.676794 | 1451 | 17 |
| `loft_only` | 0.648086 | 0.641700 | 1359 | 17 |
| `combined` | 0.565732 | 0.563491 | 2810 | 17 |
