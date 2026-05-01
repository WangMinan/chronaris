# Stage I Deep Baseline - nasa_csm - mult

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T-full-loso-deep-comparison/nasa_csm/mult`
- prepared root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/nasa_sequences`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `benchmark_only` | 0.302460 | 0.333333 | 1451 | 17 |
| `loft_only` | 0.302226 | 0.333333 | 1359 | 17 |
| `combined` | 0.302347 | 0.333333 | 2810 | 17 |

## Existing Classical Reference

| group | classical macro-F1 | classical balanced acc |
| --- | ---: | ---: |
| `benchmark_only` | 0.464156 | 0.490542 |
| `loft_only` | 0.372319 | 0.380775 |
| `combined` | 0.374102 | 0.376498 |

