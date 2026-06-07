# Private Alignment Support - 20260607T-stage-i-private-opt-package-r2

- benchmark_role: `private_proxy_benchmark`
- task_role: `proxy_task`
- `T1/T2/T3` 仅作为 private proxy tasks，用于验证表示学习与融合增益，不等价于论文风险/负荷/复盘人工真值任务。
- thesis weak-label layer: `thesis_task_weak_label_benchmark`
- alignment gain supported: `False`

## T1 Proxy Task

| variant | macro-F1 | balanced accuracy |
| --- | ---: | ---: |
| `naive_sync` | 0.788235 | 0.777778 |
| `e_baseline` | 0.262981 | 0.333333 |
| `f_full` | 0.276605 | 0.350427 |

## T2 Proxy Task

| variant | RMSE | MAE | Spearman |
| --- | ---: | ---: | ---: |
| `naive_sync` | 77849.499529 | 45043.259293 | 0.481281 |
| `e_baseline` | 333.744316 | 182.924813 | -0.018558 |
| `f_full` | 339.895398 | 193.241477 | 0.389261 |
