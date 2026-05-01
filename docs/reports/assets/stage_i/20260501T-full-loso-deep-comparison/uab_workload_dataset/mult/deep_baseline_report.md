# Stage I Deep Baseline - uab_workload_dataset - mult

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T-full-loso-deep-comparison/uab_workload_dataset/mult`
- prepared root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/uab_sequences`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.262130 | 0.304878 | 28052 | 16 |
| `heat_the_chair` | 0.516098 | 0.521129 | 5440 | 17 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 5.828201 | 4.670543 | -0.609173 | -0.232170 | 28052 | 16 |
| `heat_the_chair` | 2.825106 | 2.353416 | -3.078696 | -0.148280 | 5440 | 17 |

## Existing Classical Reference

| group | classical macro-F1 | classical balanced acc |
| --- | ---: | ---: |
| `n_back` | 0.347765 | 0.362466 |
| `heat_the_chair` | 0.540477 | 0.540470 |

| group | classical RMSE | classical MAE |
| --- | ---: | ---: |
| `n_back` | 10.223413 | 4.798697 |
| `heat_the_chair` | 1.863911 | 1.278513 |
