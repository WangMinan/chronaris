# Dingxin Alignment Support - 20260607T-stage-i-private-opt-package-r2

- benchmark_role: dingxin_weak_label_benchmark
- task_role: weak_label_task
- 分类任务、回归任务和检索任务 仅作为 Dingxin weak-label tasks，用于验证表示学习与融合增益，不等价于论文风险/负荷/复盘人工真值任务。
- thesis weak-label layer: `thesis_task_weak_label_benchmark`
- alignment gain supported: `False`

## 分类任务 weak-label Task

| variant | macro-F1 | balanced accuracy |
| --- | ---: | ---: |
| `naive_sync` | 0.788235 | 0.777778 |
| `e_baseline` | 0.262981 | 0.333333 |
| `f_full` | 0.276605 | 0.350427 |

## 回归任务 weak-label Task

| variant | RMSE | MAE | Spearman |
| --- | ---: | ---: | ---: |
| `naive_sync` | 77849.499529 | 45043.259293 | 0.481281 |
| `e_baseline` | 333.744316 | 182.924813 | -0.018558 |
| `f_full` | 339.895398 | 193.241477 | 0.389261 |
