# Dingxin Real-Data Optimality Summary - 20260607T-stage-i-private-opt-package-r2

- benchmark_role: dingxin_weak_label_benchmark
- 分类任务、回归任务和检索任务 只按 Dingxin weak-label tasks 解读，不把它们写成人工真值 thesis tasks。
- thesis weak-label tasks: `risk_proxy, workload_proxy, event_replay_tag`
- Dingxin optimality supported: `True`
- best 分类任务 variant: `chronaris_opt`
- best 回归任务 variant: `chronaris_opt`
- best 检索任务 variant: `chronaris_opt`
- best deep 分类任务 model: `contiformer`
- best deep 回归任务 model: `contiformer`

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
