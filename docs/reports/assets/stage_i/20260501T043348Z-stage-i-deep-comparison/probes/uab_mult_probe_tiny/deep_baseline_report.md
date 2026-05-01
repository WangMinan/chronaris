# Stage I Deep Baseline - uab_workload_dataset - mult

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/probes/uab_mult_probe_tiny`
- prepared root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/uab_sequences`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.178240 | 0.332925 | 1888 | 1 |
| `heat_the_chair` | 0.336066 | 0.500000 | 324 | 1 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 7.834171 | 7.148945 | -0.815665 | -0.005675 | 1888 | 1 |
| `heat_the_chair` | 3.174344 | 3.074411 | -16.916436 | -0.032106 | 324 | 1 |

## Existing Classical Reference

| group | classical macro-F1 | classical balanced acc |
| --- | ---: | ---: |
| `n_back` | 0.347765 | 0.362466 |
| `heat_the_chair` | 0.540477 | 0.540470 |

| group | classical RMSE | classical MAE |
| --- | ---: | ---: |
| `n_back` | 10.223413 | 4.798697 |
| `heat_the_chair` | 1.863911 | 1.278513 |
