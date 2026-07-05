# Dingxin Real-Data Optimality Summary - 20260504T120000Z-stage-i-private-opt-package

- Dingxin optimality supported: `True`
- best 分类任务 variant: `chronaris_opt`
- best 回归任务 variant: `chronaris_opt`
- best 检索任务 variant: `chronaris_opt`
- best deep 分类任务 model: `mult`
- best deep 回归任务 model: `mult`

## Retrieval

| variant | top-1 accuracy | MRR |
| --- | ---: | ---: |
| `naive_sync` | 0.270270 | 0.496761 |
| `e_baseline` | 0.027027 | 0.113546 |
| `f_full` | 0.067568 | 0.152959 |
| `g_min` | 0.027027 | 0.113556 |
| `g_no_causal_mask` | 0.027027 | 0.119938 |
| `chronaris_opt` | 1.000000 | 1.000000 |
| `chronaris_opt_no_causal_mask` | 0.027027 | 0.113556 |
