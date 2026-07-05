# Stage I Public Transfer Boundary - 20260607T-stage-i-evidence-closure-r2-transfer-boundary

- evidence_layer: `transfer_boundary`
- public_mainline_status: `public opt closed`
- boundary_note: `公开 UAB/NASA 结果只用于 adapter/calibration 与 transfer-boundary 说明，不能改写为论文鼎新真实双流本体 fully closed。`

## Data Boundary

| corpus | modality_pair | labels | granularity | time_reference | evidence_role |
| --- | --- | --- | --- | --- | --- |
| dingxin_stage_h | `real_physiology + real_vehicle_timeseries` | `risk_proxy/workload_proxy/event_replay_tag or T1/T2/T3` | `window/view` | `Stage H unified timeline with sortie/pilot/view ids` | `thesis_weak_label + dingxin_component_diagnostics` |
| `uab_workload_dataset` | `physiology + task_context_proxy` | `subjective workload / public adapter target` | `window` | `public prepared sequence timeline` | `public_adapter/calibration` |
| `nasa_csm` | `physiology + scenario_context_proxy` | `attention_state / public adapter target` | `window/sequence` | `public prepared sequence timeline` | `public_adapter/calibration` |

## Task Boundary

| layer | task_scope | note |
| --- | --- | --- |
| `thesis_weak_label` | `risk_proxy / workload_proxy / event_replay_tag` | `weak labels built from Dingxin Stage H aligned windows` |
| dingxin_weak_label | 分类任务、回归任务和检索任务 | `Dingxin weak-label component diagnostics only` |
| `public_adapter/calibration` | `UAB/NASA public tasks` | `context-derived second stream only, not real vehicle stream` |

## Performance References

| source_type | dataset | subset | metric | value | source_path |
| --- | --- | --- | --- | ---: | --- |
| `public_adapter_baseline` | `uab_workload_dataset` | `heat_the_chair` | `rmse` | 1.456759 | `docs/artifacts/assets/stage_i_public_opt/20260506T121000Z-stage-i-public-opt-uab/public_opt_summary.json` |
| `calibration_baseline` | `uab_workload_dataset` | `heat_the_chair` | `rmse` | 1.433140 | `docs/artifacts/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/public_opt_summary.json` |
| `legacy_public_opt` | `nasa_csm` | `benchmark_only` | `macro_f1` | 0.744514 | `docs/artifacts/assets/stage_i_public_opt/20260506T161500Z-stage-i-public-opt-nasa-round1/public_opt_summary.json` |
| `torch_uab` | `uab_workload_dataset` | `heat_the_chair` | `rmse` | 1.462995 | `docs/artifacts/assets/stage_i_public_opt_torch/20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1/public_opt_torch_summary.json` |
| `public_mainline_status` | `public_mainline` | `summary` | `status` | 1.000000 | `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt_torch/20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1/public_opt_torch_summary.json` |
