# Stage I Public Opt UAB Torch Mainline

## 运行口径

- run_id：`20260506T165558Z-stage-i-public-opt-uab-torch`
- dataset_id：`uab_workload_dataset`
- profile：`window_v2`
- runtime_device：`cpu`
- requested_device：`auto`
- prepared asset root：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt/20260504T161500Z-stage-i-public-opt-uab-prepared`
- output artifact root：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt_torch/20260506T165558Z-stage-i-public-opt-uab-torch`
- generated_at_utc：`2026-05-06T17:06:55.493488Z`

## Screen Winner

- candidate_id：`mlp_huber_wide__full__lr0p001__wd0p001`
- model_family：`mlp_huber_wide`
- feature_profile：`full`
- hidden_dims：`[256, 128]`
- learning_rate：`0.001`
- weight_decay：`0.001`
- full_run_completed：`True`
- ensemble_policy：`mean_top2`

## Screen Leaderboard

| rank | candidate_id | feature_profile | mean_rmse | mean_mae | n_back_rmse | heat_the_chair_rmse |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `mlp_huber_wide__full__lr0p001__wd0p001` | `full` | 4.0524 | 3.4703 | 5.9239 | 2.1809 |
| 2 | `mlp_huber_wide__full__lr0p001__wd0p0001` | `full` | 4.0524 | 3.4703 | 5.9239 | 2.1809 |
| 3 | `mlp_huber_small__full__lr0p001__wd0p0001` | `full` | 4.0660 | 3.5131 | 5.9338 | 2.1981 |
| 4 | `mlp_huber_small__full__lr0p001__wd0p001` | `full` | 4.0660 | 3.5132 | 5.9339 | 2.1982 |
| 5 | `residual_gated_mlp__full__lr0p0003__wd0p0001` | `full` | 4.0682 | 3.5083 | 5.9017 | 2.2347 |
| 6 | `residual_gated_mlp__full__lr0p0003__wd0p001` | `full` | 4.0683 | 3.5084 | 5.9019 | 2.2348 |
| 7 | `residual_gated_mlp__full__lr0p001__wd0p0001` | `full` | 4.0876 | 3.5737 | 5.8369 | 2.3383 |
| 8 | `residual_gated_mlp__full__lr0p001__wd0p001` | `full` | 4.0877 | 3.5741 | 5.8370 | 2.3384 |
| 9 | `mlp_huber_small__full__lr0p0003__wd0p0001` | `full` | 4.1011 | 3.5444 | 5.9542 | 2.2479 |
| 10 | `mlp_huber_small__full__lr0p0003__wd0p001` | `full` | 4.1011 | 3.5444 | 5.9542 | 2.2480 |
| 11 | `mlp_huber_wide__full__lr0p0003__wd0p0001` | `full` | 4.1033 | 3.5444 | 5.9301 | 2.2766 |
| 12 | `mlp_huber_wide__full__lr0p0003__wd0p001` | `full` | 4.1034 | 3.5444 | 5.9301 | 2.2766 |
| 13 | `residual_gated_mlp__residual_only__lr0p001__wd0p0001` | `residual_only` | 4.1695 | 3.6038 | 6.1635 | 2.1754 |
| 14 | `residual_gated_mlp__residual_only__lr0p001__wd0p001` | `residual_only` | 4.1695 | 3.6039 | 6.1634 | 2.1756 |
| 15 | `mlp_huber_small__residual_only__lr0p0003__wd0p0001` | `residual_only` | 4.1730 | 3.6229 | 6.1294 | 2.2166 |
| 16 | `mlp_huber_small__residual_only__lr0p0003__wd0p001` | `residual_only` | 4.1730 | 3.6230 | 6.1294 | 2.2166 |
| 17 | `residual_gated_mlp__residual_only__lr0p0003__wd0p0001` | `residual_only` | 4.1925 | 3.6607 | 6.1394 | 2.2457 |
| 18 | `residual_gated_mlp__residual_only__lr0p0003__wd0p001` | `residual_only` | 4.1926 | 3.6608 | 6.1394 | 2.2457 |
| 19 | `mlp_huber_small__residual_only__lr0p001__wd0p0001` | `residual_only` | 4.2292 | 3.6935 | 6.1114 | 2.3471 |
| 20 | `mlp_huber_small__residual_only__lr0p001__wd0p001` | `residual_only` | 4.2293 | 3.6936 | 6.1115 | 2.3472 |
| 21 | `mlp_huber_wide__residual_only__lr0p0003__wd0p0001` | `residual_only` | 4.2368 | 3.6836 | 6.1108 | 2.3629 |
| 22 | `mlp_huber_wide__residual_only__lr0p0003__wd0p001` | `residual_only` | 4.2369 | 3.6836 | 6.1109 | 2.3629 |
| 23 | `mlp_huber_wide__residual_only__lr0p001__wd0p001` | `residual_only` | 4.2921 | 3.7732 | 6.1066 | 2.4776 |
| 24 | `mlp_huber_wide__residual_only__lr0p001__wd0p0001` | `residual_only` | 4.2921 | 3.7731 | 6.1066 | 2.4776 |

## Final LOSO Result

| evaluation_group | rmse | mae | r2 | spearman |
| --- | ---: | ---: | ---: | ---: |
| n_back | 4.7460 | 3.7743 | -0.0671 | -0.1059 |
- n_back selected source：`candidate` / `mlp_huber_wide__full__lr0p001__wd0p001`
| heat_the_chair | 1.5300 | 1.2150 | -0.1963 | -0.4866 |
- heat_the_chair selected source：`candidate` / `mlp_huber_wide__full__lr0p001__wd0p001`

## Acceptance Gate

| evaluation_group | threshold_rmse | observed_rmse | passed |
| --- | ---: | ---: | --- |
| n_back | 4.6541 | 4.7460 | `False` |
| heat_the_chair | 1.4568 | 1.5300 | `False` |

- public_mainline_status：`NASA closed, UAB partial`

## 历史对照

| evaluation_group | current_public_opt_rmse | MulT rmse | ContiFormer rmse | torch_uab_rmse |
| --- | ---: | ---: | ---: | ---: |
| n_back | 4.6103 | 5.8282 | 4.6541 | 4.7460 |
| heat_the_chair | 1.4568 | 2.8251 | 1.4568 | 1.5300 |

## 结论

- NASA `public opt round 1` 继续冻结为当前公开主线的已闭合部分。
- UAB 由当前 torch-native runner 接替 CPU-heavy `sklearn` 扩搜；若本次 gate 未全过，则主结论仍按 `NASA closed, UAB partial` 维护。
- `chronaris_public_fusion` 保持 secondary exploratory branch，不替代当前 paper-facing public mainline。
