# Stage I Public Adapter Calibration - 20260607T-stage-i-evidence-closure-r2-public-adapter

- evidence_layer: `public_adapter_calibration`
- boundary_note: `UAB/NASA rows only support public adapter or calibration evidence. They do not prove the thesis dual-stream private mainline is fully closed.`

## Best By Category

| category | dataset | subset | metric | value | source_path |
| --- | --- | --- | --- | ---: | --- |
| `public_adapter_baseline` | `uab_workload_dataset` | `heat_the_chair` | `rmse` | 1.456759 | `docs/artifacts/assets/stage_i_public_opt/20260506T121000Z-stage-i-public-opt-uab/public_opt_summary.json` |
| `calibration_baseline` | `uab_workload_dataset` | `heat_the_chair` | `rmse` | 1.433140 | `docs/artifacts/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/public_opt_summary.json` |
| `legacy_public_opt` | `nasa_csm` | `benchmark_only` | `macro_f1` | 0.744514 | `docs/artifacts/assets/stage_i_public_opt/20260506T161500Z-stage-i-public-opt-nasa-round1/public_opt_summary.json` |
| `torch_uab` | `uab_workload_dataset` | `heat_the_chair` | `rmse` | 1.462995 | `docs/artifacts/assets/stage_i_public_opt_torch/20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1/public_opt_torch_summary.json` |

## Rows

| category | dataset | subset | candidate | metric | value |
| --- | --- | --- | --- | --- | ---: |
| `public_adapter_baseline` | `uab_workload_dataset` | `n_back` | `physiology_persistence` | `rmse` | 4.647676 |
| `public_adapter_baseline` | `uab_workload_dataset` | `n_back` | `ridge_residual` | `rmse` | 4.610314 |
| `public_adapter_baseline` | `uab_workload_dataset` | `heat_the_chair` | `physiology_persistence` | `rmse` | 1.456759 |
| `public_adapter_baseline` | `uab_workload_dataset` | `heat_the_chair` | `ridge_residual` | `rmse` | 1.462376 |
| `calibration_baseline` | `uab_workload_dataset` | `heat_the_chair` | `target_prior_median` | `rmse` | 1.433140 |
| `calibration_baseline` | `uab_workload_dataset` | `heat_the_chair` | `target_prior_trimmed_mean` | `rmse` | 1.454298 |
| `calibration_baseline` | `uab_workload_dataset` | `heat_the_chair` | `heat_prior_residual_guarded` | `rmse` | 1.464789 |
| `public_adapter_baseline` | `uab_workload_dataset` | `heat_the_chair` | `physiology_persistence` | `rmse` | 1.456759 |
| `public_adapter_baseline` | `uab_workload_dataset` | `heat_the_chair` | `ridge_heat_physiology_lowdim` | `rmse` | 1.462356 |
| `public_adapter_baseline` | `uab_workload_dataset` | `heat_the_chair` | `huber_heat_physiology_lowdim` | `rmse` | 1.507688 |
| `legacy_public_opt` | `nasa_csm` | `benchmark_only` | `physiology_margin_balanced_logistic` | `macro_f1` | 0.435044 |
| `legacy_public_opt` | `nasa_csm` | `benchmark_only` | `balanced_logistic_context` | `macro_f1` | 0.677300 |
| `legacy_public_opt` | `nasa_csm` | `benchmark_only` | `balanced_linear_svc_context` | `macro_f1` | 0.744514 |
| `legacy_public_opt` | `nasa_csm` | `loft_only` | `physiology_margin_balanced_logistic` | `macro_f1` | 0.300773 |
| `legacy_public_opt` | `nasa_csm` | `loft_only` | `balanced_logistic_context` | `macro_f1` | 0.286348 |
| `legacy_public_opt` | `nasa_csm` | `loft_only` | `balanced_linear_svc_context` | `macro_f1` | 0.364336 |
| `legacy_public_opt` | `nasa_csm` | `combined` | `physiology_margin_balanced_logistic` | `macro_f1` | 0.349639 |
| `legacy_public_opt` | `nasa_csm` | `combined` | `balanced_logistic_context` | `macro_f1` | 0.454992 |
| `legacy_public_opt` | `nasa_csm` | `combined` | `balanced_linear_svc_context` | `macro_f1` | 0.451272 |
| `torch_uab` | `uab_workload_dataset` | `n_back` | `mlp_huber_wide__full__lr0p001__wd0p001` | `rmse` | 4.745978 |
| `torch_uab` | `uab_workload_dataset` | `heat_the_chair` | `mlp_huber_wide__full__lr0p001__wd0p001` | `rmse` | 1.530006 |
| `torch_uab` | `uab_workload_dataset` | `heat_the_chair` | `heat_residual_correction__lr0p0003__wd0p0001` | `rmse` | 1.462995 |
