# Stage I Deep Baseline - uab_workload_dataset - chronaris_public_fusion

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260701T-stage-i-public-fusion-ablation-r1/screen/uab_workload_dataset/lag_window_8__seed42`
- prepared root: `/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/uab_workload_dataset`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.314714 | 0.342708 | 3764 | 2 |
| `heat_the_chair` | 0.458852 | 0.506575 | 638 | 2 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 6.063463 | 5.581747 | 0.007227 | 0.098748 | 3764 | 2 |
| `heat_the_chair` | 2.072773 | 1.767705 | -0.076869 | -0.724506 | 638 | 2 |

