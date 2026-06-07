# Stage I Public Opt NASA Attention Run

## 运行口径

- run_id：`20260506T124500Z-stage-i-public-opt-nasa`
- dataset_id：`nasa_csm`
- profile：`window_v2`
- track：`objective`
- task_type：`classification`
- prepared asset root：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt/20260506T124000Z-stage-i-public-opt-nasa-prepared`
- output artifact root：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_public_opt/20260506T124500Z-stage-i-public-opt-nasa`
- generated_at_utc：`2026-05-06T03:01:50.018097Z`

## 样本范围

- 总样本数：`2810`
- evaluation groups：`benchmark_only, loft_only, combined`
- raw subsets：`benchmark, loft`
- split_group 数：`17`
- subject 数：`17`

## Evaluation 指标

| evaluation_group | sample_count | fold_count | best_head | macro_f1 | balanced_accuracy |
| --- | ---: | ---: | --- | ---: | ---: |
| benchmark_only | 1451 | 17 | ridge_context_classifier | 0.3099 | 0.3368 |
| loft_only | 1359 | 17 | physiology_margin | 0.3022 | 0.3333 |
| combined | 2810 | 17 | physiology_margin | 0.3023 | 0.3333 |

### benchmark_only

- best_head：`ridge_context_classifier`
- sample_count：`1451`
- fold_count：`17`

| head | macro_f1 | balanced_accuracy |
| --- | ---: | ---: |
| physiology_margin | 0.3025 | 0.3333 |
| ridge_context_classifier | 0.3099 | 0.3368 |

### loft_only

- best_head：`physiology_margin`
- sample_count：`1359`
- fold_count：`17`

| head | macro_f1 | balanced_accuracy |
| --- | ---: | ---: |
| physiology_margin | 0.3022 | 0.3333 |
| ridge_context_classifier | 0.3010 | 0.3310 |

### combined

- best_head：`physiology_margin`
- sample_count：`2810`
- fold_count：`17`

| head | macro_f1 | balanced_accuracy |
| --- | ---: | ---: |
| physiology_margin | 0.3023 | 0.3333 |
| ridge_context_classifier | 0.3020 | 0.3326 |

## 参考对照

- dataset_id：`nasa_csm`
- track：`objective`

| evaluation_group | public_opt best_head | public_opt macro_f1 | public_opt balanced_accuracy | classical macro_f1 | classical balanced_accuracy | MulT macro_f1 | MulT balanced_accuracy | ContiFormer macro_f1 | ContiFormer balanced_accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| benchmark_only | ridge_context_classifier | 0.3099 | 0.3368 | 0.4642 | 0.4905 | 0.3025 | 0.3333 | 0.3024 | 0.3331 |
| loft_only | physiology_margin | 0.3022 | 0.3333 | 0.3723 | 0.3808 | 0.3022 | 0.3333 | 0.3003 | 0.3292 |
| combined | physiology_margin | 0.3023 | 0.3333 | 0.3741 | 0.3765 | 0.3023 | 0.3333 | 0.3023 | 0.3330 |

## 说明

- `public opt` 只迁移 Chronaris 的公开 sequence-contract 思路，不改写既有 Stage I 公开 benchmark 历史事实。
- 当前结果默认直接与 `Phase 3 classical baseline` 和 `MulT / ContiFormer` full LOSO 做同口径对照。
