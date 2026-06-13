# Stage I Private Proxy Component Ablation - 20260607T-stage-i-evidence-closure-r2-private-proxy

- evidence_layer: `private_proxy`
- task_boundary: `t1_t2_t3_are_proxy_tasks_not_direct_thesis_tasks`
- source_manifests: `{'e_run_manifest_path': 'docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json', 'f_run_manifest_path': 'docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json'}`

## Reading

1. 当前全部结果都属于 `private proxy benchmark evidence`，`T1/T2/T3` 不是人工真值 thesis task fully closed。
2. `chronaris_opt` 作为 full candidate，分别对比移除因果掩码、移除时间残差、移除 task-aware head 后的退化情况。

## Component Table

| task | variant | component | primary_metric | value | delta_vs_full | note |
| --- | --- | --- | --- | ---: | ---: | --- |
| `T1_maneuver_intensity_class` | `naive_sync` | `module_baseline` | `macro_f1` | 0.788235 | 0.211765 | positive means score drop |
| `T1_maneuver_intensity_class` | `e_baseline` | `module_baseline` | `macro_f1` | 0.262981 | 0.737019 | positive means score drop |
| `T1_maneuver_intensity_class` | `f_full` | `module_baseline` | `macro_f1` | 0.276605 | 0.723395 | positive means score drop |
| `T1_maneuver_intensity_class` | `g_min` | `module_baseline` | `macro_f1` | 0.173333 | 0.826667 | positive means score drop |
| `T1_maneuver_intensity_class` | `g_no_causal_mask` | `module_baseline` | `macro_f1` | 0.173333 | 0.826667 | positive means score drop |
| `T1_maneuver_intensity_class` | `chronaris_opt` | `full_candidate` | `macro_f1` | 1.000000 | 0.000000 | positive means score drop |
| `T1_maneuver_intensity_class` | `chronaris_opt_no_causal_mask` | `remove_causal_mask` | `macro_f1` | 0.173333 | 0.826667 | positive means score drop |
| `T1_maneuver_intensity_class` | `chronaris_opt_no_time_residual` | `remove_time_residual` | `macro_f1` | 0.173333 | 0.826667 | positive means score drop |
| `T1_maneuver_intensity_class` | `chronaris_opt_no_task_head` | `remove_task_aware_head` | `macro_f1` | 0.527419 | 0.472581 | positive means score drop |
| `T2_next_window_physiology_response` | `naive_sync` | `module_baseline` | `rmse` | 77849.499529 | 77648.009964 | positive means worse rmse |
| `T2_next_window_physiology_response` | `e_baseline` | `module_baseline` | `rmse` | 333.744316 | 132.254750 | positive means worse rmse |
| `T2_next_window_physiology_response` | `f_full` | `module_baseline` | `rmse` | 339.895398 | 138.405833 | positive means worse rmse |
| `T2_next_window_physiology_response` | `g_min` | `module_baseline` | `rmse` | 278.042433 | 76.552868 | positive means worse rmse |
| `T2_next_window_physiology_response` | `g_no_causal_mask` | `module_baseline` | `rmse` | 295.802909 | 94.313343 | positive means worse rmse |
| `T2_next_window_physiology_response` | `chronaris_opt` | `full_candidate` | `rmse` | 201.489565 | 0.000000 | positive means worse rmse |
| `T2_next_window_physiology_response` | `chronaris_opt_no_causal_mask` | `remove_causal_mask` | `rmse` | 313.232477 | 111.742912 | positive means worse rmse |
| `T2_next_window_physiology_response` | `chronaris_opt_no_time_residual` | `remove_time_residual` | `rmse` | 313.232477 | 111.742912 | positive means worse rmse |
| `T2_next_window_physiology_response` | `chronaris_opt_no_task_head` | `remove_task_aware_head` | `rmse` | 148134.520363 | 147933.030798 | positive means worse rmse |
| `T3_paired_pilot_window_retrieval` | `naive_sync` | `module_baseline` | `top1_accuracy` | 0.270270 | 0.729730 | positive means score drop |
| `T3_paired_pilot_window_retrieval` | `e_baseline` | `module_baseline` | `top1_accuracy` | 0.027027 | 0.972973 | positive means score drop |
| `T3_paired_pilot_window_retrieval` | `f_full` | `module_baseline` | `top1_accuracy` | 0.067568 | 0.932432 | positive means score drop |
| `T3_paired_pilot_window_retrieval` | `g_min` | `module_baseline` | `top1_accuracy` | 0.027027 | 0.972973 | positive means score drop |
| `T3_paired_pilot_window_retrieval` | `g_no_causal_mask` | `module_baseline` | `top1_accuracy` | 0.027027 | 0.972973 | positive means score drop |
| `T3_paired_pilot_window_retrieval` | `chronaris_opt` | `full_candidate` | `top1_accuracy` | 1.000000 | 0.000000 | positive means score drop |
| `T3_paired_pilot_window_retrieval` | `chronaris_opt_no_causal_mask` | `remove_causal_mask` | `top1_accuracy` | 0.027027 | 0.972973 | positive means score drop |
| `T3_paired_pilot_window_retrieval` | `chronaris_opt_no_time_residual` | `remove_time_residual` | `top1_accuracy` | 0.040541 | 0.959459 | positive means score drop |
| `T3_paired_pilot_window_retrieval` | `chronaris_opt_no_task_head` | `remove_task_aware_head` | `top1_accuracy` | 0.216216 | 0.783784 | positive means score drop |
