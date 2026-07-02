# Stage I Optimized Model Summary - 20260702T-stage-i-optimized-model-summary-r4-v3-confirm20

- status: `completed`
- runtime_device: `cuda`
- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20`
- boundary: This package summarizes completed optimized P34/P35/P36 evidence with fixed P30/P31/P32 references; public rows remain context-proxy evidence and historical artifacts are read-only.

## P34 Key Deltas
- `T1_maneuver_intensity_class` `balanced_accuracy`: P30=`0.333333` P34=`0.34188` delta=`0.00854701` status=`improved`
- `T1_maneuver_intensity_class` `macro_f1`: P30=`0.173333` P34=`0.216065` delta=`0.0427322` status=`improved`
- `T1_maneuver_intensity_class` `balanced_accuracy`: P30=`0.333333` P34=`0.344729` delta=`0.011396` status=`improved`
- `T1_maneuver_intensity_class` `macro_f1`: P30=`0.173333` P34=`0.187489` delta=`0.0141557` status=`improved`
- `T2_next_window_physiology_response` `mae`: P30=`992.524` P34=`317.41` delta=`675.114` status=`improved`
- `T2_next_window_physiology_response` `nrmse`: P30=`11.3163` P34=`2.05032` delta=`9.26601` status=`improved`
- `T2_next_window_physiology_response` `rmse`: P30=`1111.98` P34=`415.361` delta=`696.618` status=`improved`
- `T2_next_window_physiology_response` `mae`: P30=`762.558` P34=`276.504` delta=`486.054` status=`improved`
- `T2_next_window_physiology_response` `nrmse`: P30=`8.78099` P34=`2.25945` delta=`6.52154` status=`improved`
- `T2_next_window_physiology_response` `rmse`: P30=`838.121` P34=`346.827` delta=`491.294` status=`improved`
- `T3_paired_pilot_window_retrieval` `mrr`: P30=`0.119948` P34=`0.117939` delta=`-0.00200877` status=`regressed`
- `T3_paired_pilot_window_retrieval` `top1`: P30=`0.027027` P34=`0.0315315` delta=`0.0045045` status=`improved`
- `T3_paired_pilot_window_retrieval` `top3`: P30=`0.0945946` P34=`0.0855856` delta=`-0.00900901` status=`regressed`
- `T3_paired_pilot_window_retrieval` `top5`: P30=`0.148649` P34=`0.13964` delta=`-0.00900901` status=`regressed`

## P35 Stream-role Gates
- `private_stage_h` role=`real_vehicle` route=`causal_lagged_vehicle_to_physio` lag=`0.869754` context=`0.101672` vehicle=`0.857364` causal=`0.876729`
- `nasa_csm` role=`scenario_context_proxy` route=`adaptive_context_gate` lag=`0.115751` context=`0.752347` vehicle=`0.155897` causal=`0.121497`
- `uab_workload_dataset` role=`task_context_proxy` route=`adaptive_context_gate` lag=`0.119853` context=`0.755867` vehicle=`0.158099` causal=`0.13321`
- `nasa_csm` role=`scenario_context_proxy` route=`causal_lagged_vehicle_to_physio` lag=`0.865751` context=`0.102347` vehicle=`0.861355` causal=`0.871497`
- `uab_workload_dataset` role=`task_context_proxy` route=`context_adapter_only` lag=`0.0479411` context=`0.855867` vehicle=`0.0451711` causal=`0.0532838`

## GPU Runtime
- `P34` device=`cuda` gpu=`NVIDIA GeForce RTX 4090` cache=`auto` batch=`2048` amp=`bf16` compile=`off` max_mem_gb=`0.0217443` samples_per_sec=`14.144` util_pct=`NA`
- `P35` device=`cuda` gpu=`NVIDIA GeForce RTX 4090` cache=`auto` batch=`NA` amp=`bf16` compile=`off` max_mem_gb=`4.6633` samples_per_sec=`36.1538` util_pct=`1`

## Outputs
- summary: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/optimized_model_summary.csv`
- key metrics: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/key_metric_summary.csv`
- gates: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/stream_role_gate_summary.csv`
- gpu: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/gpu_runtime_summary.csv`
- claim boundary: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/claim_boundary_summary.csv`
- resume commands: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/resume_commands.txt`
