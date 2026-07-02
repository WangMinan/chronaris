# Stage I Deep Baseline - uab_workload_dataset - chronaris_public_fusion

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260701T-stage-i-public-fusion-ablation-r1/screen/uab_workload_dataset/event_bias_0p75__seed42`
- prepared root: `/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/uab_workload_dataset`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.304040 | 0.328668 | 3764 | 2 |
| `heat_the_chair` | 0.451684 | 0.506398 | 638 | 2 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 6.109038 | 5.590671 | -0.007754 | -0.030229 | 3764 | 2 |
| `heat_the_chair` | 2.095500 | 1.793186 | -0.100613 | -0.765100 | 638 | 2 |

