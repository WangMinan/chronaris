# Stage I Deep Baseline - uab_workload_dataset - contiformer

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T-full-loso-deep-comparison/uab_workload_dataset/contiformer`
- prepared root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/uab_sequences`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.162030 | 0.333333 | 28052 | 16 |
| `heat_the_chair` | 0.322203 | 0.500000 | 5440 | 17 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 4.654091 | 3.812269 | -0.026132 | -0.405381 | 28052 | 16 |
| `heat_the_chair` | 1.456759 | 1.163693 | -0.084493 | -0.792231 | 5440 | 17 |

## Existing Classical Reference

| group | classical macro-F1 | classical balanced acc |
| --- | ---: | ---: |
| `n_back` | 0.347765 | 0.362466 |
| `heat_the_chair` | 0.540477 | 0.540470 |

| group | classical RMSE | classical MAE |
| --- | ---: | ---: |
| `n_back` | 10.223413 | 4.798697 |
| `heat_the_chair` | 1.863911 | 1.278513 |
