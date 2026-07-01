# Stage I Deep Baseline - nasa_csm - chronaris_public_fusion

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/screen/nasa_csm/fusion_h64_l2_hd4_do0p1_bias0p25_lag16_lr0p001_bs128_smooth_l1_td1_zscore_train_wd1em05__seed42`
- prepared root: `/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/nasa_csm`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `benchmark_only` | 0.629728 | 0.695775 | 172 | 2 |
| `loft_only` | 0.649089 | 0.597222 | 180 | 2 |
| `combined` | 0.460323 | 0.455018 | 352 | 2 |
