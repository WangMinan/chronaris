# Stage I Deep Baseline - uab_workload_dataset - mult

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/comparison_probe_all/uab_workload_dataset/mult`
- prepared root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/uab_sequences`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.160081 | 0.204060 | 1888 | 1 |
| `heat_the_chair` | 0.336066 | 0.500000 | 324 | 1 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 6.865815 | 5.779193 | -0.394548 | 0.016881 | 1888 | 1 |
| `heat_the_chair` | 1.245245 | 0.946868 | -1.757106 | -0.068950 | 324 | 1 |

## Existing Classical Reference

| group | classical macro-F1 | classical balanced acc |
| --- | ---: | ---: |
| `n_back` | 0.347765 | 0.362466 |
| `heat_the_chair` | 0.540477 | 0.540470 |

| group | classical RMSE | classical MAE |
| --- | ---: | ---: |
| `n_back` | 10.223413 | 4.798697 |
| `heat_the_chair` | 1.863911 | 1.278513 |
