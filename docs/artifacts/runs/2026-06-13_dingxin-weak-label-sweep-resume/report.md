# task evaluation Thesis Weak-Label Multitask Sweep - 20260613T-task-eval-p11-live-influx-r3-resume

- status: `completed`
- evidence_layer: `thesis_weak_label`
- source_manifests: `{'e_run_manifest_path': 'docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json', 'f_run_manifest_path': 'docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json'}`
- combination_count: `2`
- derived_from_run_id: `20260613T-task-eval-p11-live-influx-r3-resume`
- blocked_at_run_index: `3`
- blocked_attempt_log_paths: `['docs/artifacts/runs/2026-06-13_dingxin-weak-label-sweep-child-r1/run.log']`

## Reading

1. 所有结果都属于 `thesis weak-label evidence`，任务仍是 `risk_proxy / workload_proxy / event_replay_tag`，不是人工真值闭环。
2. 当前表按 `test_total` 升序排序，便于快速定位在共享骨干 + 任务监督 + 因果正则组合下的相对稳定配置。

## Ablation Table

| run_id | physics_family | causal_weight | task_loss_weight | lag_window | test_total | test_task_total | test_causal_total | checkpoint |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `20260613T-task-eval-p11-live-influx-r1-01-minimal-cw0p00-tlw0p50-lagnone` | `minimal` | 0.00 | 0.50 | `None` | 1153.898570 | 2.932820 | 0.932490 | `docs/artifacts/runs/2026-06-13_dingxin-weak-label-sweep-child-r1/runs/20260613T-task-eval-p11-live-influx-r1-01-minimal-cw0p00-tlw0p50-lagnone/multitask_checkpoint.pt` |
| `20260613T-task-eval-p11-live-influx-r1-02-minimal-cw0p00-tlw0p50-lag3` | `minimal` | 0.00 | 0.50 | `3` | 1154.156579 | 2.937639 | 0.934457 | `docs/artifacts/runs/2026-06-13_dingxin-weak-label-sweep-child-r1/runs/20260613T-task-eval-p11-live-influx-r1-02-minimal-cw0p00-tlw0p50-lag3/multitask_checkpoint.pt` |

## Best Run

- child_run_id: `20260613T-task-eval-p11-live-influx-r1-01-minimal-cw0p00-tlw0p50-lagnone`
- physics_constraint_family: `minimal`
- test_total: `1153.8985701851223`
- checkpoint_path: `docs/artifacts/runs/2026-06-13_dingxin-weak-label-sweep-child-r1/runs/20260613T-task-eval-p11-live-influx-r1-01-minimal-cw0p00-tlw0p50-lagnone/multitask_checkpoint.pt`
