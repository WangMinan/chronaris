# Stage I Deep Baseline - nasa_csm - contiformer

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T-full-loso-deep-comparison/nasa_csm/contiformer`
- prepared root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/nasa_sequences`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `benchmark_only` | 0.302437 | 0.333057 | 1451 | 17 |
| `loft_only` | 0.300283 | 0.329193 | 1359 | 17 |
| `combined` | 0.302264 | 0.333047 | 2810 | 17 |

## Existing Classical Reference

| group | classical macro-F1 | classical balanced acc |
| --- | ---: | ---: |
| `benchmark_only` | 0.464156 | 0.490542 |
| `loft_only` | 0.372319 | 0.380775 |
| `combined` | 0.374102 | 0.376498 |

