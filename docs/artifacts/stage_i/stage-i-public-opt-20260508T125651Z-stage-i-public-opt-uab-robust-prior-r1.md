# Stage I Public Opt UAB Subjective Run

## 运行口径

- run_id：`20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1`
- dataset_id：`uab_workload_dataset`
- profile：`window_v2`
- feature_profile：`full`
- head_catalog：`uab_hybrid`
- train_balance_policy：`class_weight_balanced`
- ensemble_policy：`none`
- prediction_aggregation_policy：`none`
- track：`subjective`
- task_type：`regression`
- prepared asset root：`/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_opt/20260504T161500Z-stage-i-public-opt-uab-prepared`
- output artifact root：`/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1`
- generated_at_utc：`2026-05-08T13:17:24.354524Z`

## 样本范围

- 总样本数：`5440`
- evaluation groups：`heat_the_chair`
- raw subsets：`heat_the_chair`
- split_group 数：`17`
- subject 数：`17`
- feature_group_sizes：`{'full': 580, 'physiology_only': 534, 'physiology_lowdim': 6, 'physiology_scalar_only': 6, 'context_only': 41, 'residual_only': 17}`

## Evaluation 指标

| evaluation_group | sample_count | fold_count | best_head | mae | rmse | r2 | spearman |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| heat_the_chair | 5440 | 17 | target_prior_median | 1.0740 | 1.4331 | -0.0496 | 0.0000 |

### heat_the_chair

- best_head：`target_prior_median`
- sample_count：`5440`
- fold_count：`17`

| head | mae | rmse | r2 | spearman |
| --- | ---: | ---: | ---: | ---: |
| target_prior_median | 1.0740 | 1.4331 | -0.0496 | 0.0000 |
| target_prior_trimmed_mean | 1.1615 | 1.4543 | -0.0808 | -0.7950 |
| heat_prior_residual_guarded | 1.1432 | 1.4648 | -0.0965 | -0.3454 |
| physiology_persistence | 1.1637 | 1.4568 | -0.0845 | -0.7922 |
| ridge_heat_physiology_lowdim | 1.1724 | 1.4624 | -0.0928 | -0.7573 |
| huber_heat_physiology_lowdim | 1.1979 | 1.5077 | -0.1616 | -0.7278 |

## 参考对照

- dataset_id：`uab_workload_dataset`
- track：`subjective`

| evaluation_group | public_opt best_head | public_opt rmse | public_opt mae | MulT rmse | MulT mae | ContiFormer rmse | ContiFormer mae |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| heat_the_chair | target_prior_median | 1.4331 | 1.0740 | 2.8251 | 2.3534 | 1.4568 | 1.1637 |

## Deep 胜出判定

| evaluation_group | best_public_head | best_deep_model | margin_vs_best_deep | gate_passed | needs_deep_rerun |
| --- | --- | --- | ---: | --- | --- |
| heat_the_chair | target_prior_median | contiformer | 0.0236 | `True` | `False` |

- overall_needs_deep_rerun：`False`

## 说明

- `public opt` 只迁移 Chronaris 的公开 sequence-contract 思路，不改写既有 Stage I 公开 benchmark 历史事实。
- 当前主 gate 只看相对 `MulT / ContiFormer` 的胜出情况；`classical baseline` 只保留为历史背景，不作为本轮前进门槛。
