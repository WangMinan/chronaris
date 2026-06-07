# Stage I Public Opt UAB Torch Mainline

## 运行口径

- run_id：`20260507T141500Z-stage-i-public-opt-uab-torch-sessionpooled-r4`
- dataset_id：`uab_workload_dataset`
- profile：`window_v2`
- runtime_device：`cuda`
- requested_device：`cuda`
- prepared asset root：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt/20260504T161500Z-stage-i-public-opt-uab-prepared`
- output artifact root：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt_torch/20260507T141500Z-stage-i-public-opt-uab-torch-sessionpooled-r4`
- generated_at_utc：`2026-05-07T06:14:27.114165Z`

## Screen Winner

- candidate_id：`mlp_huber_wide__physiology_only__lr0p001__wd0p0001`
- model_family：`mlp_huber_wide`
- feature_profile：`physiology_only`
- hidden_dims：`[256, 128]`
- learning_rate：`0.001`
- weight_decay：`0.0001`
- full_run_completed：`True`
- ensemble_policy：`mean_top2`
- prediction_aggregation_policy：`session_mean_broadcast`
- supervision_granularity：`session_pooled_broadcast`

## Screen Leaderboard

| rank | candidate_id | feature_profile | mean_rmse | mean_mae | n_back_rmse | heat_the_chair_rmse |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `mlp_huber_wide__physiology_only__lr0p001__wd0p0001` | `physiology_only` | 4.7710 | 4.4117 | 7.5609 | 1.9810 |
| 2 | `mlp_huber_wide__physiology_only__lr0p001__wd0p001` | `physiology_only` | 4.7710 | 4.4118 | 7.5610 | 1.9811 |
| 3 | `mlp_huber_wide__residual_only__lr0p001__wd0p0001` | `residual_only` | 4.8373 | 4.4740 | 7.5564 | 2.1181 |
| 4 | `mlp_huber_wide__residual_only__lr0p001__wd0p001` | `residual_only` | 4.8373 | 4.4740 | 7.5565 | 2.1181 |
| 5 | `mlp_huber_wide__full__lr0p001__wd0p0001` | `full` | 4.9327 | 4.5541 | 7.6972 | 2.1681 |
| 6 | `mlp_huber_wide__full__lr0p001__wd0p001` | `full` | 4.9327 | 4.5541 | 7.6973 | 2.1682 |
| 7 | `residual_gated_mlp__full__lr0p001__wd0p0001` | `full` | 5.6263 | 4.9358 | 8.4392 | 2.8134 |
| 8 | `residual_gated_mlp__full__lr0p001__wd0p001` | `full` | 5.6264 | 4.9358 | 8.4393 | 2.8134 |
| 9 | `mlp_huber_small__full__lr0p001__wd0p0001` | `full` | 5.6554 | 4.9526 | 8.5406 | 2.7702 |
| 10 | `mlp_huber_small__full__lr0p001__wd0p001` | `full` | 5.6555 | 4.9527 | 8.5407 | 2.7703 |
| 11 | `mlp_huber_wide__physiology_only__lr0p0003__wd0p0001` | `physiology_only` | 5.7015 | 4.9940 | 8.7870 | 2.6161 |
| 12 | `mlp_huber_wide__physiology_only__lr0p0003__wd0p001` | `physiology_only` | 5.7015 | 4.9941 | 8.7870 | 2.6161 |
| 13 | `mlp_huber_small__physiology_only__lr0p001__wd0p0001` | `physiology_only` | 5.7776 | 5.0151 | 8.6964 | 2.8588 |
| 14 | `mlp_huber_small__physiology_only__lr0p001__wd0p001` | `physiology_only` | 5.7777 | 5.0152 | 8.6965 | 2.8589 |
| 15 | `mlp_huber_wide__full__lr0p0003__wd0p0001` | `full` | 5.8343 | 5.0837 | 8.9217 | 2.7469 |
| 16 | `mlp_huber_wide__full__lr0p0003__wd0p001` | `full` | 5.8343 | 5.0837 | 8.9217 | 2.7469 |
| 17 | `mlp_huber_small__residual_only__lr0p001__wd0p0001` | `residual_only` | 5.8352 | 5.0362 | 8.7957 | 2.8747 |
| 18 | `mlp_huber_small__residual_only__lr0p001__wd0p001` | `residual_only` | 5.8353 | 5.0362 | 8.7958 | 2.8747 |
| 19 | `residual_gated_mlp__physiology_only__lr0p001__wd0p0001` | `physiology_only` | 5.8836 | 5.0834 | 8.6753 | 3.0919 |
| 20 | `residual_gated_mlp__physiology_only__lr0p001__wd0p001` | `physiology_only` | 5.8836 | 5.0834 | 8.6753 | 3.0919 |
| 21 | `residual_gated_mlp__residual_only__lr0p001__wd0p0001` | `residual_only` | 5.9940 | 5.1601 | 8.7446 | 3.2434 |
| 22 | `residual_gated_mlp__residual_only__lr0p001__wd0p001` | `residual_only` | 5.9941 | 5.1601 | 8.7446 | 3.2435 |
| 23 | `mlp_huber_wide__residual_only__lr0p0003__wd0p0001` | `residual_only` | 6.0009 | 5.1499 | 9.0034 | 2.9984 |
| 24 | `mlp_huber_wide__residual_only__lr0p0003__wd0p001` | `residual_only` | 6.0009 | 5.1499 | 9.0034 | 2.9984 |
| 25 | `residual_gated_mlp__full__lr0p0003__wd0p0001` | `full` | 6.3578 | 5.4616 | 9.2787 | 3.4370 |
| 26 | `residual_gated_mlp__full__lr0p0003__wd0p001` | `full` | 6.3579 | 5.4616 | 9.2787 | 3.4370 |
| 27 | `mlp_huber_small__full__lr0p0003__wd0p0001` | `full` | 6.6733 | 5.7354 | 9.6113 | 3.7353 |
| 28 | `mlp_huber_small__full__lr0p0003__wd0p001` | `full` | 6.6733 | 5.7354 | 9.6113 | 3.7354 |
| 29 | `mlp_huber_small__physiology_only__lr0p0003__wd0p0001` | `physiology_only` | 6.7157 | 5.8274 | 9.6953 | 3.7362 |
| 30 | `mlp_huber_small__physiology_only__lr0p0003__wd0p001` | `physiology_only` | 6.7157 | 5.8274 | 9.6953 | 3.7362 |
| 31 | `residual_gated_mlp__physiology_only__lr0p0003__wd0p0001` | `physiology_only` | 6.7281 | 5.7602 | 9.6262 | 3.8300 |
| 32 | `residual_gated_mlp__physiology_only__lr0p0003__wd0p001` | `physiology_only` | 6.7281 | 5.7602 | 9.6262 | 3.8301 |
| 33 | `residual_gated_mlp__residual_only__lr0p0003__wd0p0001` | `residual_only` | 6.8443 | 5.7622 | 9.6457 | 4.0429 |
| 34 | `residual_gated_mlp__residual_only__lr0p0003__wd0p001` | `residual_only` | 6.8443 | 5.7622 | 9.6457 | 4.0429 |
| 35 | `mlp_huber_small__residual_only__lr0p0003__wd0p0001` | `residual_only` | 7.0522 | 5.9149 | 10.1614 | 3.9431 |
| 36 | `mlp_huber_small__residual_only__lr0p0003__wd0p001` | `residual_only` | 7.0522 | 5.9149 | 10.1614 | 3.9431 |
| 37 | `linear_huber__physiology_only__lr0p001__wd0p0001` | `physiology_only` | 8.4202 | 7.3397 | 11.1111 | 5.7294 |
| 38 | `linear_huber__physiology_only__lr0p001__wd0p001` | `physiology_only` | 8.4202 | 7.3397 | 11.1111 | 5.7294 |
| 39 | `linear_huber__physiology_only__lr0p0003__wd0p0001` | `physiology_only` | 8.4252 | 7.3458 | 11.1128 | 5.7375 |
| 40 | `linear_huber__physiology_only__lr0p0003__wd0p001` | `physiology_only` | 8.4252 | 7.3458 | 11.1128 | 5.7375 |
| 41 | `linear_huber__full__lr0p001__wd0p0001` | `full` | 8.4318 | 7.3375 | 11.1063 | 5.7573 |
| 42 | `linear_huber__full__lr0p001__wd0p001` | `full` | 8.4318 | 7.3375 | 11.1063 | 5.7573 |
| 43 | `linear_huber__full__lr0p0003__wd0p0001` | `full` | 8.4344 | 7.3415 | 11.1084 | 5.7604 |
| 44 | `linear_huber__full__lr0p0003__wd0p001` | `full` | 8.4344 | 7.3415 | 11.1084 | 5.7604 |
| 45 | `linear_huber__residual_only__lr0p001__wd0p0001` | `residual_only` | 8.5260 | 7.4418 | 11.3125 | 5.7395 |
| 46 | `linear_huber__residual_only__lr0p001__wd0p001` | `residual_only` | 8.5260 | 7.4418 | 11.3125 | 5.7395 |
| 47 | `linear_huber__residual_only__lr0p0003__wd0p0001` | `residual_only` | 8.5306 | 7.4472 | 11.3136 | 5.7476 |
| 48 | `linear_huber__residual_only__lr0p0003__wd0p001` | `residual_only` | 8.5306 | 7.4472 | 11.3136 | 5.7476 |

## Final LOSO Result

| evaluation_group | rmse | mae | r2 | spearman |
| --- | ---: | ---: | ---: | ---: |
| n_back | 6.6951 | 5.8742 | -1.1234 | 0.1390 |
- n_back selected source：`candidate` / `mlp_huber_wide__physiology_only__lr0p001__wd0p0001`
| heat_the_chair | 1.7997 | 1.4798 | -0.6553 | -0.0233 |
- heat_the_chair selected source：`candidate` / `mlp_huber_wide__residual_only__lr0p001__wd0p0001`

## Acceptance Gate

| evaluation_group | threshold_rmse | observed_rmse | passed |
| --- | ---: | ---: | --- |
| n_back | 4.6541 | 6.6951 | `False` |
| heat_the_chair | 1.4568 | 1.7997 | `False` |

- public_mainline_status：`NASA closed, UAB partial`

## 历史对照

| evaluation_group | current_public_opt_rmse | MulT rmse | ContiFormer rmse | torch_uab_rmse |
| --- | ---: | ---: | ---: | ---: |
| n_back | 4.6103 | 5.8282 | 4.6541 | 6.6951 |
| heat_the_chair | 1.4568 | 2.8251 | 1.4568 | 1.7997 |

## 结论

- NASA `public opt round 1` 继续冻结为当前公开主线的已闭合部分。
- UAB 由当前 torch-native runner 接替 CPU-heavy `sklearn` 扩搜；若本次 gate 未全过，则主结论仍按 `NASA closed, UAB partial` 维护。
- `chronaris_public_fusion` 保持 secondary exploratory branch，不替代当前 paper-facing public mainline。
