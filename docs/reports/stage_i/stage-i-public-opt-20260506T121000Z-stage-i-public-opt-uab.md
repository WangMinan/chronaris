# Stage I Public Opt UAB Subjective Run

## 运行口径

- run_id：`20260506T121000Z-stage-i-public-opt-uab`
- dataset_id：`uab_workload_dataset`
- profile：`window_v2`
- track：`subjective`
- task_type：`regression`
- prepared asset root：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt/20260504T161500Z-stage-i-public-opt-uab-prepared`
- output artifact root：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt/20260506T121000Z-stage-i-public-opt-uab`
- generated_at_utc：`2026-05-06T03:02:40.010917Z`

## 样本范围

- 总样本数：`33492`
- evaluation groups：`n_back, heat_the_chair`
- raw subsets：`heat_the_chair, n_back`
- split_group 数：`26`
- subject 数：`26`

## Evaluation 指标

| evaluation_group | sample_count | fold_count | best_head | mae | rmse | r2 | spearman |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| n_back | 28052 | 16 | ridge_residual | 3.8043 | 4.6103 | -0.0069 | 0.0621 |
| heat_the_chair | 5440 | 17 | physiology_persistence | 1.1637 | 1.4568 | -0.0845 | -0.7922 |

### n_back

- best_head：`ridge_residual`
- sample_count：`28052`
- fold_count：`16`

| head | mae | rmse | r2 | spearman |
| --- | ---: | ---: | ---: | ---: |
| physiology_persistence | 3.8108 | 4.6477 | -0.0233 | -0.3552 |
| ridge_residual | 3.8043 | 4.6103 | -0.0069 | 0.0621 |

### heat_the_chair

- best_head：`physiology_persistence`
- sample_count：`5440`
- fold_count：`17`

| head | mae | rmse | r2 | spearman |
| --- | ---: | ---: | ---: | ---: |
| physiology_persistence | 1.1637 | 1.4568 | -0.0845 | -0.7922 |
| ridge_residual | 1.1724 | 1.4624 | -0.0929 | -0.7573 |

## 参考对照

- dataset_id：`uab_workload_dataset`
- track：`subjective`

| evaluation_group | public_opt best_head | public_opt rmse | public_opt mae | classical rmse | classical mae | MulT rmse | MulT mae | ContiFormer rmse | ContiFormer mae |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n_back | ridge_residual | 4.6103 | 3.8043 | 10.2234 | 4.7987 | 5.8282 | 4.6705 | 4.6541 | 3.8123 |
| heat_the_chair | physiology_persistence | 1.4568 | 1.1637 | 1.8639 | 1.2785 | 2.8251 | 2.3534 | 1.4568 | 1.1637 |

## 说明

- `public opt` 只迁移 Chronaris 的公开 sequence-contract 思路，不改写既有 Stage I 公开 benchmark 历史事实。
- 当前结果默认直接与 `Phase 3 classical baseline` 和 `MulT / ContiFormer` full LOSO 做同口径对照。
