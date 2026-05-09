# Stage I Deep Baseline - uab_workload_dataset - contiformer

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_midterm_runtime/20260509T031000Z-stage-i-uab-fairness-contiformer`
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

