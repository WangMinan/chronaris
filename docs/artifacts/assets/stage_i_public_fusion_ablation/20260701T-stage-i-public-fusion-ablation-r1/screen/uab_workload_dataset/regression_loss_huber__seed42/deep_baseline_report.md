# Stage I Deep Baseline - uab_workload_dataset - chronaris_public_fusion

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260701T-stage-i-public-fusion-ablation-r1/screen/uab_workload_dataset/regression_loss_huber__seed42`
- prepared root: `/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/uab_workload_dataset`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.308763 | 0.334184 | 3764 | 2 |
| `heat_the_chair` | 0.455383 | 0.504963 | 638 | 2 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 6.053349 | 5.474464 | 0.010536 | 0.061121 | 3764 | 2 |
| `heat_the_chair` | 2.095348 | 1.792898 | -0.100455 | -0.774424 | 638 | 2 |

