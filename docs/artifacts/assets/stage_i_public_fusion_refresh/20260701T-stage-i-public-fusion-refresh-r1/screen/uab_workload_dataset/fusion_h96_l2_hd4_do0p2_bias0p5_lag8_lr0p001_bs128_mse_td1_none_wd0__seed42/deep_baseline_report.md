# Stage I Deep Baseline - uab_workload_dataset - chronaris_public_fusion

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/screen/uab_workload_dataset/fusion_h96_l2_hd4_do0p2_bias0p5_lag8_lr0p001_bs128_mse_td1_none_wd0__seed42`
- prepared root: `/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/uab_workload_dataset`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.310726 | 0.337503 | 3764 | 2 |
| `heat_the_chair` | 0.463296 | 0.506693 | 638 | 2 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 6.036484 | 5.568420 | 0.016041 | 0.066199 | 3764 | 2 |
| `heat_the_chair` | 2.136205 | 1.840540 | -0.143788 | -0.780034 | 638 | 2 |
