# Stage I Causal Support - 20260506T120000Z-stage-i-support

- generated_at_utc: `2026-05-06T02:58:16.462491Z`
- artifact_root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_support/20260506T120000Z-stage-i-support`
- support_matrix: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_support/20260506T120000Z-stage-i-support/support_matrix.csv`
- main ablation matrix: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i_support/20260506T120000Z-stage-i-support/ablation_matrix.csv`

## G(min) Summary

- sample_count: `3`
- mean_attention_entropy: `0.931446`
- mean_max_attention: `0.225623`
- mean_top_event_score: `1.000000`
- mean_top_contribution_score: `2.609973`

## Phase 2 Bundle-Only Ablations

| ablation | mean delta entropy | mean delta top event | mean delta top contribution | mean delta fused L2 | mean delta cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_event_bias` | +0.002143 | +0.000000 | -0.273633 | -0.002576 | -0.000083 |
| `no_state_normalization` | -0.001111 | +0.000000 | +0.073146 | +0.001528 | -0.000064 |
| `vehicle_delta_suppressed` | +0.006814 | -1.000000 | -2.399792 | +0.306467 | -0.150769 |

## Same-Sortie Dual-Pilot

| sortie | delta mean cosine | delta cosine cv | delta top contribution |
| --- | ---: | ---: | ---: |
| `20251002_单01_ACT-8_翼云_J16_12#01` | -0.170068 | +0.192775 | -0.217476 |

## Private No-Mask Comparison

| task | target metrics | no-mask metrics | target_beats_no_mask |
| --- | --- | --- | --- |
| `T1_maneuver_intensity_class` | macro_f1=1.000000, balanced_accuracy=1.000000 | macro_f1=0.173333, balanced_accuracy=0.333333 | `True` |
| `T2_next_window_physiology_response` | rmse=201.489565, mae=113.851926 | rmse=313.232477, mae=173.648719 | `True` |
| `T3_paired_pilot_window_retrieval` | top1_accuracy=1.000000, mrr=1.000000 | top1_accuracy=0.027027, mrr=0.113556 | `True` |

## 辅助 real-sortie deep wrappers

| model | mean event-mask interference | mean attention entropy | pilot delta event-mask interference |
| --- | ---: | ---: | ---: |
| `mult` | 0.110893 | 2.737305 | -0.092728 |
| `contiformer` | 0.114094 | 2.759080 | -0.019390 |

## 因果结论

1. `G(min)` 已经产生稳定的非对称注意力与 top-event/top-contribution 指标。
2. 当前最强 bundle-only 干预是 `vehicle_delta_suppressed`，其 mean delta top contribution 为 `-2.399792`，mean delta top event 为 `-1.000000`。
3. `chronaris_opt_no_causal_mask` 在私有 T1/T2/T3 三任务上都劣于 target variant，说明因果掩码不是可有可无的装饰项。
4. 这条报告回答的是“因果融合是否做出来并给出可解释差异”，不是“因果融合已在公开数据上全面最优”。

## 本结论能支撑什么

- 可以支撑论文中“单向因果约束、关键事件偏置与双 pilot 差异可读性”已经形成真实 sortie 证据链。
- 可以支撑“去掉因果掩码后，私有 proxy 三任务同步退化”的主张。

## 本结论不能支撑什么

- 不能把当前 case-study 级证据写成大规模标签监督下的全面 superiority 证明。
- 不能把 `no_event_bias` 或 `vehicle_delta_suppressed` 的 bundle-only 干预直接等价为完整任务级 ablation 胜负。
