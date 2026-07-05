# Dingxin Real-Data Optimization Summary - 20260607T-stage-i-private-opt-package-r2

- benchmark_role: dingxin_weak_label_benchmark
- 分类任务、回归任务和检索任务 的最优性只说明 Dingxin weak-label benchmark 收敛，不代表论文真值任务已经闭环。
- thesis weak-label manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`
- target variant: `chronaris_opt`
- no-mask variant: `chronaris_opt_no_causal_mask`
- Dingxin optimality supported: `True`

## Criteria

| check | pass |
| --- | ---: |
| `t1_chronaris_opt_beats_module_baselines` | `True` |
| `t2_chronaris_opt_beats_module_baselines` | `True` |
| `t3_chronaris_opt_beats_module_baselines` | `True` |
| `t1_chronaris_opt_beats_best_deep` | `True` |
| `t2_chronaris_opt_beats_best_deep` | `True` |
| `t1_chronaris_opt_beats_chronaris_opt_no_causal_mask` | `True` |
| `t2_chronaris_opt_beats_chronaris_opt_no_causal_mask` | `True` |
| `t3_chronaris_opt_beats_chronaris_opt_no_causal_mask` | `True` |

## Target Metrics

| task | variant | primary metrics |
| --- | --- | --- |
| 分类任务：机动强度分类 | `chronaris_opt` | macro_f1=1.000000, balanced_accuracy=1.000000 |
| 分类任务：机动强度分类 | `chronaris_opt_no_causal_mask` | macro_f1=0.173333, balanced_accuracy=0.333333 |
| 回归任务：下一窗口生理响应 | `chronaris_opt` | rmse=201.489565, mae=113.851926 |
| 回归任务：下一窗口生理响应 | `chronaris_opt_no_causal_mask` | rmse=313.232477, mae=173.648719 |
| 检索任务：配对飞行员窗口检索 | `chronaris_opt` | top1_accuracy=1.000000, mrr=1.000000 |
| 检索任务：配对飞行员窗口检索 | `chronaris_opt_no_causal_mask` | top1_accuracy=0.027027, mrr=0.113556 |

## Artifacts

- optimized candidate summary: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_summary.json`
- optimized candidate metrics: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_metrics.csv`
- optimized candidate package: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_package.json`
- optimized package report: `/home/wangminan/projects/chronaris/docs/artifacts/private-optimized-package-20260607T-stage-i-private-opt-package-r2.md`
