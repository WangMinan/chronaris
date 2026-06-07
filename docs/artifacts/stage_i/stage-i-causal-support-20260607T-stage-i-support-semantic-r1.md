# Stage I Causal Support - 20260607T-stage-i-support-semantic-r1

- generated_at_utc: `2026-06-07T10:13:45.352749Z`
- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r1`
- support_matrix: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r1/support_matrix.csv`
- main ablation matrix: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r1/ablation_matrix.csv`

## G(min) Summary

- sample_count: `3`
- mean_attention_entropy: `0.932071`
- mean_max_attention: `0.224872`
- mean_top_event_score: `1.000000`
- mean_top_contribution_score: `2.597950`

## Semantic Event Fusion

- query_names: `['risk_proxy', 'workload_proxy', 'event_replay_tag']`
- query_count: `3`
- mean_event_token_count: `1.000000`
- mean_query_entropy: `0.000000`
- mean_top_query_score: `2.597950`
- mean_top_event_attribution: `7.793849`

| sample | top query | top query event offset s | top event attribution |
| --- | --- | ---: | ---: |
| `20251005_四01_ACT-4_云_J20_22#01:0020` | `risk_proxy` | 0.742916 | 7.793849 |
| `20251005_四01_ACT-4_云_J20_22#01:0021` | `risk_proxy` | 0.742916 | 7.793849 |
| `20251005_四01_ACT-4_云_J20_22#01:0022` | `risk_proxy` | 0.742916 | 7.793849 |

## Phase 2 Bundle-Only Ablations

- strongest ablation: `vehicle_delta_suppressed`
- strongest delta_mean_top_contribution_score: `-2.399792`

| ablation | mean_attention_entropy | mean_top_event_score | mean_top_contribution_score | delta_mean_attention_entropy | delta_mean_top_event_score | delta_mean_top_contribution_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_event_bias` | 0.932829 | 1.000000 | 2.126160 | +0.002143 | +0.000000 | -0.273633 |
| `no_state_normalization` | 0.929575 | 1.000000 | 2.472938 | -0.001111 | +0.000000 | +0.073146 |
| `vehicle_delta_suppressed` | 0.937500 | 0.000000 | 0.000000 | +0.006814 | -1.000000 | -2.399792 |

## Private No-Mask Comparison

- target_variant: `chronaris_opt`
- no_mask_variant: `chronaris_opt_no_causal_mask`

| task | target metrics | no-mask metrics | target_beats_no_mask |
| --- | --- | --- | ---: |
| `T1_maneuver_intensity_class` | `macro_f1=1.000000, balanced_accuracy=1.000000` | `macro_f1=0.173333, balanced_accuracy=0.333333` | `True` |
| `T2_next_window_physiology_response` | `rmse=201.489565, mae=113.851926` | `rmse=313.232477, mae=173.648719` | `True` |
| `T3_paired_pilot_window_retrieval` | `top1_accuracy=1.000000, mrr=1.000000` | `top1_accuracy=0.027027, mrr=0.113556` | `True` |

## 因果结论

1. `G(min)` 已形成稳定的因果注意力统计，不是只存在于图示。
2. 语义事件融合把时间步注意力进一步折叠成 `event token + query-to-event attribution`，可以把解释粒度从单点权重提升到事件级归因。
3. `Phase 2 bundle-only` 消融已经给出 `no_event_bias / vehicle_delta_suppressed` 两条扰动证据，说明事件偏置与机动上下文都会改变贡献分布。
4. 私有 proxy benchmark 中 `chronaris_opt_no_causal_mask` 三任务同步退化，说明“拿掉掩码”不是无损替换。
5. 因果支撑链回答的是“掩码机制是否有必要”，不是“当前公开 benchmark 已由因果模型接管主线”。
