# task evaluation Deep Baseline - nasa_csm - chronaris_public_fusion

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-01_public-fusion-calibration/screen/nasa_csm/fusion_h96_l2_hd4_do0p2_bias0p5_lag8_lr0p001_bs128_mse_td1_none_wd0__seed42`
- prepared root: `/tmp/chronaris_task_eval_public_fusion_refresh/20260701T-task-eval-public-fusion-refresh-r1/nasa_csm`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `benchmark_only` | 0.524228 | 0.631925 | 172 | 2 |
| `loft_only` | 0.649089 | 0.597222 | 180 | 2 |
| `combined` | 0.454009 | 0.452773 | 352 | 2 |
