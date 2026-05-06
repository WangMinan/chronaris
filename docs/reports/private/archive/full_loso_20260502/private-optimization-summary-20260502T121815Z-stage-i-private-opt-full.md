# Private Optimization Summary - 20260502T121815Z-stage-i-private-opt-full

- target variant: `chronaris_opt`
- no-mask variant: `chronaris_opt_no_causal_mask`
- private optimality supported: `True`

## Criteria

| check | pass |
| --- | ---: |
| `t1_chronaris_opt_beats_module_baselines` | `True` |
| `t2_chronaris_opt_beats_module_baselines` | `True` |
| `t3_chronaris_opt_beats_module_baselines` | `True` |
| `t1_chronaris_opt_beats_best_deep` | `True` |
| `t2_chronaris_opt_beats_best_deep` | `True` |
| `t1_chronaris_opt_beats_chronaris_opt_no_causal_mask` | `True` |
| `t2_chronaris_opt_beats_chronaris_opt_no_causal_mask` | `True` |
| `t3_chronaris_opt_beats_chronaris_opt_no_causal_mask` | `True` |

## Target Metrics

| task | variant | primary metrics |
| --- | --- | --- |
| `T1_maneuver_intensity_class` | `chronaris_opt` | macro_f1=1.000000, balanced_accuracy=1.000000 |
| `T1_maneuver_intensity_class` | `chronaris_opt_no_causal_mask` | macro_f1=0.173333, balanced_accuracy=0.333333 |
| `T2_next_window_physiology_response` | `chronaris_opt` | rmse=201.489565, mae=113.851926 |
| `T2_next_window_physiology_response` | `chronaris_opt_no_causal_mask` | rmse=313.232477, mae=173.648719 |
| `T3_paired_pilot_window_retrieval` | `chronaris_opt` | top1_accuracy=1.000000, mrr=1.000000 |
| `T3_paired_pilot_window_retrieval` | `chronaris_opt_no_causal_mask` | top1_accuracy=0.027027, mrr=0.113556 |

## Artifacts

- optimized candidate summary: `docs/reports/assets/stage_i_private/20260502T121815Z-stage-i-private-opt-full/optimized_candidate_summary.json`
- optimized candidate metrics: `docs/reports/assets/stage_i_private/20260502T121815Z-stage-i-private-opt-full/optimized_candidate_metrics.csv`
