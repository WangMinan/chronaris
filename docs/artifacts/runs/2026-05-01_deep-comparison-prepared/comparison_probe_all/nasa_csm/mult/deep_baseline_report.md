# Deep Baseline - nasa_csm - mult

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-05-01_deep-comparison-prepared/comparison_probe_all/nasa_csm/mult`
- prepared root: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-05-01_deep-comparison-prepared/nasa_sequences`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `benchmark_only` | 0.299578 | 0.333333 | 87 | 1 |
| `loft_only` | 0.308244 | 0.333333 | 100 | 1 |
| `combined` | 0.304264 | 0.333333 | 187 | 1 |

## Existing Classical Reference

| group | classical macro-F1 | classical balanced acc |
| --- | ---: | ---: |
| `benchmark_only` | 0.464156 | 0.490542 |
| `loft_only` | 0.372319 | 0.380775 |
| `combined` | 0.374102 | 0.376498 |

