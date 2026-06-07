# Stage I Public Mainline Report

- generated_at_utc：`2026-05-06T06:43:02.659076Z`
- public_mainline_status：`NASA closed, UAB partial`

## Mainline Decision

- NASA `public opt round 1` 继续冻结为当前公开主线的已闭合部分。
- UAB 当前仍未形成严格双组 clean win，因此公开主线状态保持 `NASA closed, UAB partial`。
- `chronaris_public_fusion` 继续作为 secondary exploratory branch，不作为当前论文主线。

## NASA

- combined_best_head：`balanced_logistic_context`
- combined_macro_f1：`0.4550`
- combined_balanced_accuracy：`0.5591`
- mainline_closed：`True`

## UAB

| evaluation_group | public_rmse | public_mae | best_deep_model | best_deep_rmse | margin_vs_best_deep | clean_win |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| n_back | 5.0619 | 4.0329 | `contiformer` | 4.6541 | -0.4078 | `False` |
| heat_the_chair | 1.6211 | 1.3230 | `contiformer` | 1.4568 | -0.1644 | `False` |

## Public Fusion

- NASA coarse best：`fusion_h64_l2_hd4_do01_bias025_lag16_norm1`，combined macro-F1=`0.3558`
- NASA confirm：combined macro-F1=`0.3348`，gate>0.40=`False`
- UAB confirm：mean RMSE=`4.0838`
- 该分支继续保留为 exploratory only。
