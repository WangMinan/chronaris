# Stage I Thesis Weak-Label Multitask Sweep - 20260613T-stage-i-p11-live-influx-r2

- evidence_layer: `thesis_weak_label`
- source_summary.sample_collection.sample_source: `live_influx`
- reused_child_runs_from_partial_run_id: `20260613T-stage-i-p11-live-influx-r1`

## Proxy vs Live Comparison

| sample_source | sample_count | task_entry_count | combination_count | best_child_run_id | best_test_total | best_test_task_total | best_test_causal_total |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `stage_h_window_stats_proxy` | 111 | 333 | 2 | `20260607T-stage-i-evidence-closure-r2-multitask-01-minimal-cw0p00-tlw0p50-lagnone` | 1024.809990 | 2.929292 | 0.934360 |
| `live_influx` | 111 | 333 | 2 | `20260613T-stage-i-p11-live-influx-r1-01-minimal-cw0p00-tlw0p50-lagnone` | 1153.898570 | 2.932820 | 0.932490 |

## Live Ablation Rows

| child_run_id | physics_family | causal_weight | task_loss_weight | lag_window | test_total | test_task_total | test_causal_total | checkpoint |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `20260613T-stage-i-p11-live-influx-r1-01-minimal-cw0p00-tlw0p50-lagnone` | `minimal` | 0.00 | 0.50 | `None` | 1153.898570 | 2.932820 | 0.932490 | `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/runs/20260613T-stage-i-p11-live-influx-r1-01-minimal-cw0p00-tlw0p50-lagnone/multitask_checkpoint.pt` |
| `20260613T-stage-i-p11-live-influx-r1-02-minimal-cw0p00-tlw0p50-lag3` | `minimal` | 0.00 | 0.50 | `3` | 1154.156579 | 2.937639 | 0.934457 | `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/runs/20260613T-stage-i-p11-live-influx-r1-02-minimal-cw0p00-tlw0p50-lag3/multitask_checkpoint.pt` |

## Notes

1. 所有结果仍属于 `thesis weak-label evidence`，任务是 `risk_proxy / workload_proxy / event_replay_tag`，不是人工真值闭环。
2. 本次 stable live 汇总复用了 `20260613T-stage-i-p11-live-influx-r1` 中已完成的两个 child run，与 proxy 现有 `2` 个组合口径对齐。
3. 更大的 `4` 组合 live 尝试已保留 blocker：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/progress.json` 和 `run.log`。
