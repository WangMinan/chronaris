# Stage I Public Opt UAB Torch Mainline

## 运行口径

- run_id：`20260507T134500Z-stage-i-public-opt-uab-torch-sessionmean-r3`
- dataset_id：`uab_workload_dataset`
- profile：`window_v2`
- runtime_device：`cuda`
- requested_device：`cuda`
- prepared asset root：`/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_opt/20260504T161500Z-stage-i-public-opt-uab-prepared`
- output artifact root：`/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_opt_torch/20260507T134500Z-stage-i-public-opt-uab-torch-sessionmean-r3`
- generated_at_utc：`2026-05-07T05:57:31.442039Z`

## Screen Winner

- candidate_id：`residual_gated_mlp__full__lr0p0003__wd0p0001`
- model_family：`residual_gated_mlp`
- feature_profile：`full`
- hidden_dims：`[128, 64]`
- learning_rate：`0.0003`
- weight_decay：`0.0001`
- full_run_completed：`True`
- ensemble_policy：`mean_top2`
- prediction_aggregation_policy：`session_mean_broadcast`

## Screen Leaderboard

| rank | candidate_id | feature_profile | mean_rmse | mean_mae | n_back_rmse | heat_the_chair_rmse |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `residual_gated_mlp__full__lr0p0003__wd0p0001` | `full` | 3.9226 | 3.4713 | 5.6086 | 2.2367 |
| 2 | `residual_gated_mlp__full__lr0p0003__wd0p001` | `full` | 3.9227 | 3.4714 | 5.6086 | 2.2368 |
| 3 | `mlp_huber_small__full__lr0p0003__wd0p0001` | `full` | 3.9969 | 3.5238 | 5.7448 | 2.2491 |
| 4 | `mlp_huber_small__full__lr0p0003__wd0p001` | `full` | 3.9969 | 3.5239 | 5.7448 | 2.2491 |
| 5 | `mlp_huber_wide__full__lr0p0003__wd0p001` | `full` | 4.0684 | 3.5692 | 5.8578 | 2.2790 |
| 6 | `mlp_huber_wide__full__lr0p0003__wd0p0001` | `full` | 4.0684 | 3.5691 | 5.8578 | 2.2790 |
| 7 | `residual_gated_mlp__full__lr0p001__wd0p0001` | `full` | 4.0772 | 3.6039 | 5.8189 | 2.3356 |
| 8 | `residual_gated_mlp__full__lr0p001__wd0p001` | `full` | 4.0776 | 3.6042 | 5.8195 | 2.3357 |
| 9 | `mlp_huber_small__full__lr0p001__wd0p0001` | `full` | 4.0986 | 3.5697 | 5.9879 | 2.2092 |
| 10 | `mlp_huber_wide__full__lr0p001__wd0p0001` | `full` | 4.0987 | 3.5231 | 6.0190 | 2.1784 |
| 11 | `mlp_huber_small__full__lr0p001__wd0p001` | `full` | 4.0987 | 3.5699 | 5.9882 | 2.2093 |
| 12 | `mlp_huber_wide__full__lr0p001__wd0p001` | `full` | 4.0987 | 3.5231 | 6.0191 | 2.1783 |
| 13 | `mlp_huber_small__residual_only__lr0p0003__wd0p0001` | `residual_only` | 4.1683 | 3.6239 | 6.1213 | 2.2153 |
| 14 | `mlp_huber_small__residual_only__lr0p0003__wd0p001` | `residual_only` | 4.1684 | 3.6240 | 6.1213 | 2.2154 |
| 15 | `residual_gated_mlp__residual_only__lr0p0003__wd0p001` | `residual_only` | 4.1737 | 3.6223 | 6.1546 | 2.1929 |
| 16 | `residual_gated_mlp__residual_only__lr0p001__wd0p0001` | `residual_only` | 4.1768 | 3.6055 | 6.1735 | 2.1801 |
| 17 | `residual_gated_mlp__residual_only__lr0p001__wd0p001` | `residual_only` | 4.1768 | 3.6056 | 6.1734 | 2.1801 |
| 18 | `residual_gated_mlp__residual_only__lr0p0003__wd0p0001` | `residual_only` | 4.2000 | 3.6521 | 6.1545 | 2.2454 |
| 19 | `mlp_huber_wide__residual_only__lr0p0003__wd0p001` | `residual_only` | 4.2233 | 3.6867 | 6.0844 | 2.3623 |
| 20 | `mlp_huber_wide__residual_only__lr0p0003__wd0p0001` | `residual_only` | 4.2234 | 3.6867 | 6.0845 | 2.3622 |
| 21 | `mlp_huber_small__residual_only__lr0p001__wd0p0001` | `residual_only` | 4.2350 | 3.6898 | 6.1277 | 2.3423 |
| 22 | `mlp_huber_small__residual_only__lr0p001__wd0p001` | `residual_only` | 4.2351 | 3.6899 | 6.1278 | 2.3423 |
| 23 | `mlp_huber_wide__residual_only__lr0p001__wd0p0001` | `residual_only` | 4.2693 | 3.7539 | 6.0504 | 2.4881 |
| 24 | `mlp_huber_wide__residual_only__lr0p001__wd0p001` | `residual_only` | 4.2693 | 3.7540 | 6.0505 | 2.4881 |

## Final LOSO Result

| evaluation_group | rmse | mae | r2 | spearman |
| --- | ---: | ---: | ---: | ---: |
| n_back | 4.9450 | 4.0005 | -0.1584 | -0.1111 |
- n_back selected source：`candidate` / `residual_gated_mlp__full__lr0p0003__wd0p0001`
| heat_the_chair | 1.6204 | 1.3230 | -0.3418 | -0.5080 |
- heat_the_chair selected source：`candidate` / `residual_gated_mlp__full__lr0p0003__wd0p0001`

## Acceptance Gate

| evaluation_group | threshold_rmse | observed_rmse | passed |
| --- | ---: | ---: | --- |
| n_back | 4.6541 | 4.9450 | `False` |
| heat_the_chair | 1.4568 | 1.6204 | `False` |

- public_mainline_status：`NASA closed, UAB partial`

## 历史对照

| evaluation_group | current_public_opt_rmse | MulT rmse | ContiFormer rmse | torch_uab_rmse |
| --- | ---: | ---: | ---: | ---: |
| n_back | 4.6103 | 5.8282 | 4.6541 | 4.9450 |
| heat_the_chair | 1.4568 | 2.8251 | 1.4568 | 1.6204 |

## 结论

- NASA `public opt round 1` 继续冻结为当前公开主线的已闭合部分。
- UAB 由当前 torch-native runner 接替 CPU-heavy `sklearn` 扩搜；若本次 gate 未全过，则主结论仍按 `NASA closed, UAB partial` 维护。
- `chronaris_public_fusion` 保持 secondary exploratory branch，不替代当前 paper-facing public mainline。
