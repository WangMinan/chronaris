# Stage I Public Opt UAB Torch Mainline

## 运行口径

- run_id：`20260506T165558Z-stage-i-public-opt-uab-torch-gpu`
- dataset_id：`uab_workload_dataset`
- profile：`window_v2`
- runtime_device：`cuda`
- requested_device：`auto`
- prepared asset root：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt/20260504T161500Z-stage-i-public-opt-uab-prepared`
- output artifact root：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt_torch/20260506T165558Z-stage-i-public-opt-uab-torch-gpu`
- generated_at_utc：`2026-05-07T02:40:37.905607Z`

## Screen Winner

- candidate_id：`residual_gated_mlp__full__lr0p0003__wd0p0001`
- model_family：`residual_gated_mlp`
- feature_profile：`full`
- hidden_dims：`[128, 64]`
- learning_rate：`0.0003`
- weight_decay：`0.0001`
- full_run_completed：`True`
- ensemble_policy：`mean_top2`

## Screen Leaderboard

| rank | candidate_id | feature_profile | mean_rmse | mean_mae | n_back_rmse | heat_the_chair_rmse |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `residual_gated_mlp__full__lr0p0003__wd0p0001` | `full` | 4.0273 | 3.4782 | 5.8179 | 2.2367 |
| 2 | `residual_gated_mlp__full__lr0p0003__wd0p001` | `full` | 4.0274 | 3.4783 | 5.8179 | 2.2368 |
| 3 | `mlp_huber_small__full__lr0p0003__wd0p0001` | `full` | 4.0899 | 3.5322 | 5.9307 | 2.2491 |
| 4 | `mlp_huber_small__full__lr0p0003__wd0p001` | `full` | 4.0899 | 3.5323 | 5.9307 | 2.2491 |
| 5 | `mlp_huber_wide__full__lr0p0003__wd0p0001` | `full` | 4.1166 | 3.5691 | 5.9540 | 2.2791 |
| 6 | `mlp_huber_wide__full__lr0p0003__wd0p001` | `full` | 4.1166 | 3.5692 | 5.9540 | 2.2791 |
| 7 | `mlp_huber_small__full__lr0p001__wd0p0001` | `full` | 4.1171 | 3.5697 | 6.0250 | 2.2092 |
| 8 | `mlp_huber_small__full__lr0p001__wd0p001` | `full` | 4.1172 | 3.5699 | 6.0252 | 2.2093 |
| 9 | `mlp_huber_wide__full__lr0p001__wd0p0001` | `full` | 4.1199 | 3.5232 | 6.0612 | 2.1786 |
| 10 | `mlp_huber_wide__full__lr0p001__wd0p001` | `full` | 4.1199 | 3.5232 | 6.0613 | 2.1786 |
| 11 | `residual_gated_mlp__full__lr0p001__wd0p0001` | `full` | 4.1297 | 3.6039 | 5.9238 | 2.3356 |
| 12 | `residual_gated_mlp__full__lr0p001__wd0p001` | `full` | 4.1299 | 3.6042 | 5.9241 | 2.3357 |
| 13 | `mlp_huber_small__residual_only__lr0p0003__wd0p0001` | `residual_only` | 4.1734 | 3.6239 | 6.1315 | 2.2153 |
| 14 | `mlp_huber_small__residual_only__lr0p0003__wd0p001` | `residual_only` | 4.1735 | 3.6240 | 6.1315 | 2.2154 |
| 15 | `residual_gated_mlp__residual_only__lr0p0003__wd0p001` | `residual_only` | 4.1781 | 3.6223 | 6.1634 | 2.1929 |
| 16 | `residual_gated_mlp__residual_only__lr0p001__wd0p0001` | `residual_only` | 4.1801 | 3.6055 | 6.1802 | 2.1801 |
| 17 | `residual_gated_mlp__residual_only__lr0p001__wd0p001` | `residual_only` | 4.1801 | 3.6056 | 6.1801 | 2.1801 |
| 18 | `residual_gated_mlp__residual_only__lr0p0003__wd0p0001` | `residual_only` | 4.2044 | 3.6521 | 6.1634 | 2.2454 |
| 19 | `mlp_huber_wide__residual_only__lr0p0003__wd0p001` | `residual_only` | 4.2310 | 3.6867 | 6.0987 | 2.3632 |
| 20 | `mlp_huber_wide__residual_only__lr0p0003__wd0p0001` | `residual_only` | 4.2310 | 3.6867 | 6.0988 | 2.3632 |
| 21 | `mlp_huber_small__residual_only__lr0p001__wd0p0001` | `residual_only` | 4.2399 | 3.6898 | 6.1376 | 2.3423 |
| 22 | `mlp_huber_small__residual_only__lr0p001__wd0p001` | `residual_only` | 4.2400 | 3.6899 | 6.1377 | 2.3424 |
| 23 | `mlp_huber_wide__residual_only__lr0p001__wd0p0001` | `residual_only` | 4.2818 | 3.7539 | 6.0744 | 2.4891 |
| 24 | `mlp_huber_wide__residual_only__lr0p001__wd0p001` | `residual_only` | 4.2818 | 3.7540 | 6.0745 | 2.4891 |

## Final LOSO Result

| evaluation_group | rmse | mae | r2 | spearman |
| --- | ---: | ---: | ---: | ---: |
| n_back | 5.0619 | 4.0329 | -0.2138 | -0.1148 |
- n_back selected source：`candidate` / `residual_gated_mlp__full__lr0p0003__wd0p0001`
| heat_the_chair | 1.6211 | 1.3230 | -0.3430 | -0.4921 |
- heat_the_chair selected source：`candidate` / `residual_gated_mlp__full__lr0p0003__wd0p0001`

## Acceptance Gate

| evaluation_group | threshold_rmse | observed_rmse | passed |
| --- | ---: | ---: | --- |
| n_back | 4.6541 | 5.0619 | `False` |
| heat_the_chair | 1.4568 | 1.6211 | `False` |

- public_mainline_status：`NASA closed, UAB partial`

## 历史对照

| evaluation_group | current_public_opt_rmse | MulT rmse | ContiFormer rmse | torch_uab_rmse |
| --- | ---: | ---: | ---: | ---: |
| n_back | 4.6103 | 5.8282 | 4.6541 | 5.0619 |
| heat_the_chair | 1.4568 | 2.8251 | 1.4568 | 1.6211 |

## 结论

- NASA `public opt round 1` 继续冻结为当前公开主线的已闭合部分。
- UAB 由当前 torch-native runner 接替 CPU-heavy `sklearn` 扩搜；若本次 gate 未全过，则主结论仍按 `NASA closed, UAB partial` 维护。
- `chronaris_public_fusion` 保持 secondary exploratory branch，不替代当前 paper-facing public mainline。
