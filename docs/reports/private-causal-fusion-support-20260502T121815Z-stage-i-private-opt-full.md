# Private Causal Fusion Support - 20260502T121815Z-stage-i-private-opt-full

- causal gain supported: `True`
- diagnostic supported: `True`
- target variant: `chronaris_opt`

## T1

| variant | macro-F1 | balanced accuracy |
| --- | ---: | ---: |
| `f_full` | 0.276605 | 0.350427 |
| `chronaris_opt` | 1.000000 | 1.000000 |
| `chronaris_opt_no_causal_mask` | 0.173333 | 0.333333 |

## T2

| variant | RMSE | MAE | Spearman |
| --- | ---: | ---: | ---: |
| `f_full` | 339.895398 | 193.241477 | 0.389261 |
| `chronaris_opt` | 201.489565 | 113.851926 | 0.349179 |
| `chronaris_opt_no_causal_mask` | 313.232477 | 173.648719 | -0.430984 |

## Diagnostics

| variant | attention entropy | top-event concentration | event-mask interference |
| --- | ---: | ---: | ---: |
| `f_full` | 0.000000 | 0.000000 | 0.000000 |
| `chronaris_opt` | 0.934591 | 0.397426 | 0.002123 |
| `chronaris_opt_no_causal_mask` | 0.998820 | 0.068461 | 0.001137 |
