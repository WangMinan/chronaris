# task evaluation Deep Baseline - uab_workload_dataset - chronaris_public_fusion

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-05-06_public-fusion-uab-confirm`
- prepared root: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-05-04_public-opt-uab-prepared`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.302164 | 0.327583 | 3764 | 2 |
| `heat_the_chair` | 0.381882 | 0.503636 | 638 | 2 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 6.064891 | 5.408402 | 0.006759 | 0.045748 | 3764 | 2 |
| `heat_the_chair` | 2.102711 | 1.799039 | -0.108202 | -0.781983 | 638 | 2 |

