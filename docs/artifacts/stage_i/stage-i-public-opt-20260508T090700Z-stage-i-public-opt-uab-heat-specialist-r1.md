# Stage I Public Opt UAB Torch Mainline

## 运行口径

- run_id：`20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1`
- dataset_id：`uab_workload_dataset`
- profile：`window_v2`
- runtime_device：`cuda`
- requested_device：`cuda`
- prepared asset root：`/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_opt/20260504T161500Z-stage-i-public-opt-uab-prepared`
- output artifact root：`/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_opt_torch/20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1`
- generated_at_utc：`2026-05-08T09:07:23.742643Z`

## Screen Winner

- candidate_id：`heat_residual_correction__lr0p0003__wd0p0001`
- model_family：`heat_residual_correction`
- feature_profile：`physiology_lowdim`
- hidden_dims：`[0, 0]`
- learning_rate：`0.0003`
- weight_decay：`0.0001`
- full_run_completed：`True`
- ensemble_policy：`none`
- prediction_aggregation_policy：`none`
- supervision_granularity：`window`

## Screen Leaderboard

| rank | candidate_id | feature_profile | mean_rmse | mean_mae | n_back_rmse | heat_the_chair_rmse |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `heat_residual_correction__lr0p0003__wd0p0001` | `physiology_lowdim` | 2.0960 | 1.7883 | 0.0000 | 2.0960 |
| 2 | `heat_residual_correction__lr0p001__wd0p0001` | `physiology_lowdim` | 2.0960 | 1.7883 | 0.0000 | 2.0960 |
| 3 | `heat_residual_correction__lr0p0003__wd0p001` | `physiology_lowdim` | 2.0960 | 1.7883 | 0.0000 | 2.0960 |
| 4 | `heat_residual_correction__lr0p001__wd0p001` | `physiology_lowdim` | 2.0960 | 1.7883 | 0.0000 | 2.0960 |
| 5 | `heat_mlp_lowdim__lr0p0003__wd0p0001` | `physiology_lowdim` | 2.1728 | 1.8863 | 0.0000 | 2.1728 |
| 6 | `heat_mlp_lowdim__lr0p0003__wd0p001` | `physiology_lowdim` | 2.1729 | 1.8864 | 0.0000 | 2.1729 |
| 7 | `heat_mlp_lowdim__lr0p001__wd0p0001` | `physiology_lowdim` | 2.2439 | 1.9685 | 0.0000 | 2.2439 |
| 8 | `heat_mlp_lowdim__lr0p001__wd0p001` | `physiology_lowdim` | 2.2440 | 1.9687 | 0.0000 | 2.2440 |
| 9 | `heat_affine_calibrated_blend__lr0p0003__wd0p001` | `physiology_lowdim` | 2.6045 | 2.3403 | 0.0000 | 2.6045 |
| 10 | `heat_affine_calibrated_blend__lr0p001__wd0p001` | `physiology_lowdim` | 2.6045 | 2.3403 | 0.0000 | 2.6045 |
| 11 | `heat_affine_calibrated_blend__lr0p0003__wd0p0001` | `physiology_lowdim` | 2.6046 | 2.3403 | 0.0000 | 2.6046 |
| 12 | `heat_affine_calibrated_blend__lr0p001__wd0p0001` | `physiology_lowdim` | 2.6046 | 2.3403 | 0.0000 | 2.6046 |
| 13 | `heat_linear_huber_lowdim__lr0p001__wd0p0001` | `physiology_lowdim` | 5.4432 | 5.0891 | 0.0000 | 5.4432 |
| 14 | `heat_linear_huber_lowdim__lr0p001__wd0p001` | `physiology_lowdim` | 5.4433 | 5.0892 | 0.0000 | 5.4433 |
| 15 | `heat_linear_huber_lowdim__lr0p0003__wd0p0001` | `physiology_lowdim` | 5.6848 | 5.3468 | 0.0000 | 5.6848 |
| 16 | `heat_linear_huber_lowdim__lr0p0003__wd0p001` | `physiology_lowdim` | 5.6848 | 5.3468 | 0.0000 | 5.6848 |

## Final LOSO Result

| evaluation_group | rmse | mae | r2 | spearman |
| --- | ---: | ---: | ---: | ---: |
| n_back | not_run | not_run | not_run | not_run |
| heat_the_chair | 1.4630 | 1.1594 | -0.0938 | -0.7272 |
- heat_the_chair selected source：`candidate` / `heat_residual_correction__lr0p0003__wd0p0001`

## Acceptance Gate

| evaluation_group | threshold_rmse | observed_rmse | passed |
| --- | ---: | ---: | --- |
| n_back | 4.6541 | not_run | `False` |
| heat_the_chair | 1.4568 | 1.4630 | `False` |

- public_mainline_status：`NASA closed, UAB partial`

## 历史对照

| evaluation_group | current_public_opt_rmse | MulT rmse | ContiFormer rmse | torch_uab_rmse |
| --- | ---: | ---: | ---: | ---: |
| n_back | 4.6103 | 5.8282 | 4.6541 | 0.0000 |
| heat_the_chair | 1.4568 | 2.8251 | 1.4568 | 1.4630 |

## 结论

- NASA `public opt round 1` 继续冻结为当前公开主线的已闭合部分。
- UAB 由当前 torch-native runner 接替 CPU-heavy `sklearn` 扩搜；若本次 gate 未全过，则主结论仍按 `NASA closed, UAB partial` 维护。
- `chronaris_public_fusion` 保持 secondary exploratory branch，不替代当前 paper-facing public mainline。
