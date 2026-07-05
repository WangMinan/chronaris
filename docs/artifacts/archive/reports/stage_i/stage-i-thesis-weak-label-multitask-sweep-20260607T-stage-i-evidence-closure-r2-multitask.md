# Stage I Thesis Weak-Label Multitask Sweep - 20260607T-stage-i-evidence-closure-r2-multitask

- evidence_layer: `thesis_weak_label`
- source_manifests: `{'e_run_manifest_path': 'docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json', 'f_run_manifest_path': 'docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json'}`
- combination_count: `2`

## Reading

1. 所有结果都属于 `thesis weak-label evidence`，任务仍是 `risk_proxy / workload_proxy / event_replay_tag`，不是人工真值闭环。
2. 当前表按 `test_total` 升序排序，便于快速定位在共享骨干 + 任务监督 + 因果正则组合下的相对稳定配置。

## Ablation Table

| run_id | physics_family | causal_weight | task_loss_weight | lag_window | test_total | test_task_total | test_causal_total | checkpoint |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `20260607T-stage-i-evidence-closure-r2-multitask-01-minimal-cw0p00-tlw0p50-lagnone` | `minimal` | 0.00 | 0.50 | `None` | 1024.809990 | 2.929292 | 0.934360 | `docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/runs/20260607T-stage-i-evidence-closure-r2-multitask-01-minimal-cw0p00-tlw0p50-lagnone/multitask_checkpoint.pt` |
| `20260607T-stage-i-evidence-closure-r2-multitask-02-minimal-cw0p00-tlw0p50-lag3` | `minimal` | 0.00 | 0.50 | `3` | 1025.084839 | 2.957319 | 0.935175 | `docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/runs/20260607T-stage-i-evidence-closure-r2-multitask-02-minimal-cw0p00-tlw0p50-lag3/multitask_checkpoint.pt` |

## Best Run

- child_run_id: `20260607T-stage-i-evidence-closure-r2-multitask-01-minimal-cw0p00-tlw0p50-lagnone`
- physics_constraint_family: `minimal`
- test_total: `1024.8099895974865`
- checkpoint_path: `docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/runs/20260607T-stage-i-evidence-closure-r2-multitask-01-minimal-cw0p00-tlw0p50-lagnone/multitask_checkpoint.pt`
