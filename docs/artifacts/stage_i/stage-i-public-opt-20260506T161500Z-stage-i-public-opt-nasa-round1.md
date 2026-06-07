# Stage I Public Opt NASA Attention Run

## 运行口径

- run_id：`20260506T161500Z-stage-i-public-opt-nasa-round1`
- dataset_id：`nasa_csm`
- profile：`window_v2`
- feature_profile：`full`
- head_catalog：`expanded`
- train_balance_policy：`class_weight_balanced`
- ensemble_policy：`none`
- track：`objective`
- task_type：`classification`
- prepared asset root：`/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_opt/20260506T124000Z-stage-i-public-opt-nasa-prepared`
- output artifact root：`/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_opt/20260506T161500Z-stage-i-public-opt-nasa-round1`
- generated_at_utc：`2026-05-06T04:14:21.519837Z`

## 样本范围

- 总样本数：`2810`
- evaluation groups：`benchmark_only, loft_only, combined`
- raw subsets：`benchmark, loft`
- split_group 数：`17`
- subject 数：`17`
- feature_group_sizes：`{'full': 165, 'physiology_only': 129, 'context_only': 31, 'residual_only': 17}`

## Evaluation 指标

| evaluation_group | sample_count | fold_count | best_head | macro_f1 | balanced_accuracy |
| --- | ---: | ---: | --- | ---: | ---: |
| benchmark_only | 1451 | 17 | balanced_linear_svc_context | 0.7445 | 0.7582 |
| loft_only | 1359 | 17 | balanced_linear_svc_context | 0.3643 | 0.3993 |
| combined | 2810 | 17 | balanced_logistic_context | 0.4550 | 0.5591 |

### benchmark_only

- best_head：`balanced_linear_svc_context`
- sample_count：`1451`
- fold_count：`17`

| head | macro_f1 | balanced_accuracy |
| --- | ---: | ---: |
| physiology_margin_balanced_logistic | 0.4350 | 0.5373 |
| balanced_logistic_context | 0.6773 | 0.7305 |
| balanced_linear_svc_context | 0.7445 | 0.7582 |

### loft_only

- best_head：`balanced_linear_svc_context`
- sample_count：`1359`
- fold_count：`17`

| head | macro_f1 | balanced_accuracy |
| --- | ---: | ---: |
| physiology_margin_balanced_logistic | 0.3008 | 0.4080 |
| balanced_logistic_context | 0.2863 | 0.3930 |
| balanced_linear_svc_context | 0.3643 | 0.3993 |

### combined

- best_head：`balanced_logistic_context`
- sample_count：`2810`
- fold_count：`17`

| head | macro_f1 | balanced_accuracy |
| --- | ---: | ---: |
| physiology_margin_balanced_logistic | 0.3496 | 0.4389 |
| balanced_logistic_context | 0.4550 | 0.5591 |
| balanced_linear_svc_context | 0.4513 | 0.4900 |

## 参考对照

- dataset_id：`nasa_csm`
- track：`objective`

| evaluation_group | public_opt best_head | public_opt macro_f1 | public_opt balanced_accuracy | MulT macro_f1 | MulT balanced_accuracy | ContiFormer macro_f1 | ContiFormer balanced_accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| benchmark_only | balanced_linear_svc_context | 0.7445 | 0.7582 | 0.3025 | 0.3333 | 0.3024 | 0.3331 |
| loft_only | balanced_linear_svc_context | 0.3643 | 0.3993 | 0.3022 | 0.3333 | 0.3003 | 0.3292 |
| combined | balanced_logistic_context | 0.4550 | 0.5591 | 0.3023 | 0.3333 | 0.3023 | 0.3330 |

## Deep 胜出判定

| evaluation_group | best_public_head | best_deep_model | margin_vs_best_deep | gate_passed | needs_deep_rerun |
| --- | --- | --- | ---: | --- | --- |
| benchmark_only | balanced_linear_svc_context | mult | 0.4421 | `True` | `False` |
| loft_only | balanced_linear_svc_context | mult | 0.0621 | `True` | `False` |
| combined | balanced_logistic_context | mult | 0.1526 | `True` | `False` |

- overall_needs_deep_rerun：`False`

## 说明

- `public opt` 只迁移 Chronaris 的公开 sequence-contract 思路，不改写既有 Stage I 公开 benchmark 历史事实。
- 当前主 gate 只看相对 `MulT / ContiFormer` 的胜出情况；`classical baseline` 只保留为历史背景，不作为本轮前进门槛。
