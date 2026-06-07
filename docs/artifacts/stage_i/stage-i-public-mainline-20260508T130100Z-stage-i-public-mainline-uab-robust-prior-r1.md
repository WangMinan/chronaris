# Stage I Public Mainline Report

- generated_at_utc：`2026-05-08T13:17:50.601487Z`
- public_mainline_status：`public opt closed`

## Mainline Decision

- NASA `public opt round 1` 继续冻结为当前公开主线的已闭合部分。
- UAB 当前 best-of Chronaris 结果已满足严格门槛。
- `chronaris_public_fusion` 继续作为 secondary exploratory branch，不作为当前论文主线。

## NASA

- combined_best_head：`balanced_logistic_context`
- combined_macro_f1：`0.4550`
- combined_balanced_accuracy：`0.5591`
- mainline_closed：`True`

## UAB

- current torch-native branch considered：`True`

| evaluation_group | public_rmse | public_mae | best_source_type | best_public_head | best_deep_model | best_deep_rmse | margin_vs_best_deep | clean_win |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| n_back | 4.6103 | 3.8043 | `legacy_public_opt` | `ridge_residual` | `contiformer` | 4.6541 | 0.0438 | `True` |
| heat_the_chair | 1.4331 | 1.0740 | `uab_public_adapter` | `target_prior_median` | `contiformer` | 1.4568 | 0.0236 | `True` |

- UAB 当前 best-of Chronaris 结果已形成主线胜出，但至少一组领先幅度仍处于 `near-tie` 区间；若要做最严格公平确认，仍建议重跑对应 deep baseline。

## Public Fusion

- NASA coarse best：`fusion_h64_l2_hd4_do01_bias025_lag16_norm1`，combined macro-F1=`0.3558`
- NASA confirm：combined macro-F1=`0.3348`，gate>0.40=`False`
- UAB confirm：mean RMSE=`4.0838`
- 该分支继续保留为 exploratory only。
