# Stage I Thesis Weak-Label Evidence - 20260607T-stage-i-multitask-real-closure-r2

- evidence_type: `thesis weak-label evidence`
- interpretation: `mainline closure evidence`
- boundary: 当前结果证明论文主线联合训练已接通，不等价于人工真值任务最优结果。
- checkpoint_path: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_checkpoint.pt`
- multitask_summary_path: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_summary.json`
- thesis_task_manifest_path: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`

## Source Contract

- e_run_manifest_path: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json`
- f_run_manifest_path: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json`
- sample_id contract: `view_id::raw_window_sample_id`
- benchmark_role: `thesis_task_weak_label_benchmark`
- task_role: `thesis_weak_label_task`
- thesis_task_boundary: `weak_label_proxy_not_manual_ground_truth`

## Task Coverage

| task | total_count | valid_label_count |
| --- | ---: | ---: |
| `risk_proxy` | 111 | 111 |
| `workload_proxy` | 111 | 111 |
| `event_replay_tag` | 111 | 111 |

## Test Metrics

- sample_count: `23`
- reconstruction_total: `1.822452`
- alignment: `0.079914`
- physics_total: `472245696.000000`
- causal_total: `0.928897`
- task_total: `4.229608`
- total: `47224576.000000`

## Task Components

| task | weighted_loss |
| --- | ---: |
| `risk_proxy` | 1.195015 |
| `workload_proxy` | 0.004169 |
| `event_replay_tag` | 3.030424 |
