# Pareto Selection

- selected_candidate_id: `chr_v2_residual_delta_h64`
- rule: T2 RMSE improvement; T1 macro-F1/balanced accuracy within 0.02 of old Chronaris; at least one target E3 metric improved; all dev folds complete.
- old Chronaris T2 RMSE: `838.121039184215`

| candidate_id | T2 RMSE | T2 improved | T1 not degraded | E3 positive signal | Pareto pass |
|---|---:|---|---|---|---|
| chr_v2_residual_h64 | 347.585 | True | True | 0 | False |
| chr_v2_residual_norm_h64 | 347.585 | True | True | 0 | False |
| chr_v2_residual_delta_h64 | 347.585 | True | True | 1 | True |
| chr_v2_residual_h96_wd1e4 | 347.04 | True | True | 0 | False |
| chr_v2_residual_h64_do0p05 | 347.599 | True | True | 0 | False |
| chr_v2_noresidual_h64 | 347.382 | True | True | 0 | False |
| chr_v3_stream_role_h64 | 347.474 | True | True | 0 | False |
| chr_v3_fixed_causal_norm_h64 | 347.474 | True | True | 0 | False |

Dev sweep rows are not thesis confirmed metrics and are not written into the protocol snapshot.
