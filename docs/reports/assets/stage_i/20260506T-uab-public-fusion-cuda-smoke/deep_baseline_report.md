# Stage I Deep Baseline - uab_workload_dataset - chronaris_public_fusion

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260506T-uab-public-fusion-cuda-smoke`
- prepared root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt/20260504T161500Z-stage-i-public-opt-uab-prepared`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.219333 | 0.333665 | 1888 | 1 |
| `heat_the_chair` | 0.336066 | 0.500000 | 324 | 1 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 5.819862 | 5.277568 | -0.002017 | 0.189026 | 1888 | 1 |
| `heat_the_chair` | 1.895964 | 1.740301 | -5.391515 | -0.030362 | 324 | 1 |

