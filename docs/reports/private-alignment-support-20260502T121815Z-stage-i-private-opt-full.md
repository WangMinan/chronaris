# Private Alignment Support - 20260502T121815Z-stage-i-private-opt-full

- alignment gain supported: `False`

## T1

| variant | macro-F1 | balanced accuracy |
| --- | ---: | ---: |
| `naive_sync` | 0.788235 | 0.777778 |
| `e_baseline` | 0.262981 | 0.333333 |
| `f_full` | 0.276605 | 0.350427 |

## T2

| variant | RMSE | MAE | Spearman |
| --- | ---: | ---: | ---: |
| `naive_sync` | 77849.499529 | 45043.259293 | 0.481281 |
| `e_baseline` | 333.744316 | 182.924813 | -0.018558 |
| `f_full` | 339.895398 | 193.241477 | 0.389261 |
