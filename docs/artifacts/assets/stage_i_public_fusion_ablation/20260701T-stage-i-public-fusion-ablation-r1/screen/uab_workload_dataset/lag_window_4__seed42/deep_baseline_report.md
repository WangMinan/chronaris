# Stage I Deep Baseline - uab_workload_dataset - chronaris_public_fusion

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260701T-stage-i-public-fusion-ablation-r1/screen/uab_workload_dataset/lag_window_4__seed42`
- prepared root: `/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/uab_workload_dataset`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.302966 | 0.327438 | 3764 | 2 |
| `heat_the_chair` | 0.429819 | 0.504403 | 638 | 2 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 5.966237 | 5.388284 | 0.038809 | 0.235639 | 3764 | 2 |
| `heat_the_chair` | 2.091306 | 1.789386 | -0.096212 | -0.747059 | 638 | 2 |

