# Stage I Public Opt Minimal Run

## 运行口径

- run_id：`20260504T170000Z-stage-i-public-opt-uab`
- dataset_id：`uab_workload_dataset`
- profile：`window_v2`
- track：`subjective`
- prepared asset root：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt/20260504T161500Z-stage-i-public-opt-uab-prepared`
- output artifact root：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt/20260504T170000Z-stage-i-public-opt-uab`
- generated_at_utc：`2026-05-04T05:08:21.057092Z`

## 样本范围

- 总样本数：`33492`
- subset：`n_back, heat_the_chair`
- split_group 数：`26`
- subject 数：`26`

## Subset 指标

| subset | sample_count | fold_count | best_head | mae | rmse | r2 | spearman |
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

## 说明

- 本轮仅实现 `chronaris public opt` 的最小可跑版，目标是打通 `UAB subjective regression` 路径并形成可比较工件。
- 本报告不替换既有 Stage I 公开 benchmark 历史结论，也不改写 `Phase 3` / `MulT` / `ContiFormer` 的收口事实。
