# task evaluation Causal Support - 20260607T-task-eval-support-semantic-r2

- generated_at_utc: `2026-06-07T10:56:06.555406Z`
- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-06-07_semantic-support`
- support_matrix: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-06-07_semantic-support/support_matrix.csv`
- main ablation matrix: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-06-07_semantic-support/ablation_matrix.csv`

## G(min) Summary

- sample_count: `111`
- mean_attention_entropy: `1.909919`
- mean_max_attention: `0.229867`
- mean_top_event_score: `1.000000`
- mean_top_contribution_score: `2.481295`

## Semantic Event Fusion

- query_names: `['risk_proxy', 'workload_proxy', 'event_replay_tag']`
- query_count: `3`
- view_count: `3`
- top_view_id: `20251005_四01_ACT-4_云_J20_22#01__pilot_10033`
- mean_event_token_count: `1.009009`
- mean_query_entropy: `0.008544`
- mean_top_query_score: `2.473122`
- mean_top_event_attribution: `7.417278`

| sample | top query | top query event offset s | top event attribution |
| --- | --- | ---: | ---: |
| `20251005_四01_ACT-4_云_J20_22#01:0000` | risk_weak_label | 0.626197 | 8.255190 |
| `20251002_单01_ACT-8_翼云_J16_12#01:0000` | risk_weak_label | 0.620689 | 8.179655 |
| `20251005_四01_ACT-4_云_J20_22#01:0002` | risk_weak_label | 0.688106 | 8.058331 |
| `20251005_四01_ACT-4_云_J20_22#01:0003` | risk_weak_label | 0.688106 | 8.058331 |
| `20251005_四01_ACT-4_云_J20_22#01:0004` | risk_weak_label | 0.688106 | 8.058331 |
| `20251005_四01_ACT-4_云_J20_22#01:0005` | risk_weak_label | 0.688106 | 8.058331 |
| `20251005_四01_ACT-4_云_J20_22#01:0006` | risk_weak_label | 0.688106 | 8.058331 |
| `20251005_四01_ACT-4_云_J20_22#01:0007` | risk_weak_label | 0.688106 | 8.058331 |
| `20251005_四01_ACT-4_云_J20_22#01:0008` | risk_weak_label | 0.688106 | 8.058331 |
| `20251005_四01_ACT-4_云_J20_22#01:0009` | risk_weak_label | 0.688106 | 8.058331 |

### View-Level Semantic Ranking

| view | sortie | pilot | state source | dominant query | mean event tokens | mean query entropy | mean top event attribution | top sample | top offset s |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `20251005_四01_ACT-4_云_J20_22#01` | 10033 | `hidden` | risk_weak_label | 1.000000 | 0.000000 | 7.871556 | `20251005_四01_ACT-4_云_J20_22#01:0000` | 0.626197 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `20251002_单01_ACT-8_翼云_J16_12#01` | 10035 | `hidden` | risk_weak_label | 1.027027 | 0.025631 | 7.476044 | `20251002_单01_ACT-8_翼云_J16_12#01:0002` | 0.710754 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `20251002_单01_ACT-8_翼云_J16_12#01` | 10033 | `hidden` | risk_weak_label | 1.000000 | 0.000000 | 6.904233 | `20251002_单01_ACT-8_翼云_J16_12#01:0000` | 0.620689 |

## Phase 2 Bundle-Only Ablations

- strongest ablation: `vehicle_delta_suppressed`
- strongest delta_mean_top_contribution_score: `-2.399792`

| ablation | mean_attention_entropy | mean_top_event_score | mean_top_contribution_score | delta_mean_attention_entropy | delta_mean_top_event_score | delta_mean_top_contribution_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_event_bias` | 0.932829 | 1.000000 | 2.126160 | +0.002143 | +0.000000 | -0.273633 |
| `no_state_normalization` | 0.929575 | 1.000000 | 2.472938 | -0.001111 | +0.000000 | +0.073146 |
| `vehicle_delta_suppressed` | 0.937500 | 0.000000 | 0.000000 | +0.006814 | -1.000000 | -2.399792 |

## Dingxin No-Mask Comparison

- target_variant: `chronaris_opt`
- no_mask_variant: `chronaris_opt_no_causal_mask`

| task | target metrics | no-mask metrics | target_beats_no_mask |
| --- | --- | --- | ---: |
| 分类任务：机动强度分类 | `macro_f1=1.000000, balanced_accuracy=1.000000` | `macro_f1=0.173333, balanced_accuracy=0.333333` | `True` |
| 回归任务：下一窗口生理响应 | `rmse=201.489565, mae=113.851926` | `rmse=313.232477, mae=173.648719` | `True` |
| 检索任务：配对飞行员窗口检索 | `top1_accuracy=1.000000, mrr=1.000000` | `top1_accuracy=0.027027, mrr=0.113556` | `True` |

## 因果结论

1. `G(min)` 已形成稳定的因果注意力统计，不是只存在于图示。
2. 语义事件融合把时间步注意力进一步折叠成 `event token + query-to-event attribution`，可以把解释粒度从单点权重提升到事件级归因。
3. `Phase 2 bundle-only` 消融已经给出 `no_event_bias / vehicle_delta_suppressed` 两条扰动证据，说明事件偏置与机动上下文都会改变贡献分布。
4. 鼎新 weak-label benchmark 中 `chronaris_opt_no_causal_mask` 三任务同步退化，说明“拿掉掩码”不是无损替换。
5. 因果支撑链回答的是“掩码机制是否有必要”，不是“当前公开 benchmark 已由因果模型接管主线”。
