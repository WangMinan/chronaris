# E3 Result Review Report

## 1. Review Scope

This review audits the current Dingxin four-method E3 run without training, tuning, or rerunning E3. It reads the E3 validation, the deep baseline representation export, and the T1/T2 confirmed context to decide paper usability and next-step optimization scope.

## 2. Input Artifacts

- E3 validation: `docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-four-method-validation`
- Deep baseline export: `docs/artifacts/runs/2026-07-07_deep-baseline-representation-export`
- Thesis protocol snapshot: `docs/artifacts/runs/2026-07-03_thesis-protocol-snapshot`
- Dingxin third-party comparison: `docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison`
- Metric calibration: `docs/artifacts/runs/2026-07-02_metric-calibration`

## 3. Protocol Recap

The E3 validation consumes Chronaris, naive_time_sync, MulT, and ContiFormer fusion streams. MulT and ContiFormer come from `T2_response_lovo_seed17_pooled_embedding`; the E3 validation itself has `training_invoked=false`. E3 remains an unsupervised structure diagnostic and does not replace T1/T2.

## 4. Data Completeness

Input rows and feature dimensions:

```csv
method_name,input_row_count,feature_dimension
chronaris,108,480
contiformer,108,64
mult,108,128
naive_time_sync,108,9710
```

Metric status by method:

```csv
method_name,status,count
chronaris,completed,31
chronaris,unavailable,9
contiformer,completed,31
contiformer,unavailable,9
mult,completed,31
mult,unavailable,9
naive_time_sync,completed,31
naive_time_sync,unavailable,9
```

Metric status by evaluator:

```csv
evaluator,status,count
clap,completed,28
clap,unavailable,4
clasp,completed,28
clasp,unavailable,4
composite_fixed_weights,completed,8
composite_fixed_weights,unavailable,8
stumpy,completed,60
stumpy,unavailable,20
```

No method was silently dropped. Four methods each have 108 input rows. The validation run reports `confirmed_metrics_changed=false` and `training_invoked=false`; only the upstream deep baseline representation export has `training_invoked=true`.

## 5. E3 Metric Summary

Chronaris summary rows:

```csv
metric,metric_group,direction,mean,completed_count,unavailable_count,best_method,chronaris_delta_to_mult,chronaris_delta_to_contiformer,chronaris_delta_to_naive
event_alignment_score,composite_auxiliary,composite_auxiliary,0.1011904761904761,1,0,chronaris;mult;contiformer;naive_time_sync,0.0,0.0,0.0
fragment_replay_recall,composite_auxiliary,composite_auxiliary,,0,1,,,,
replay_consistency_score,composite_auxiliary,direction_uncertain,0.5,1,0,,0.0,0.0,0.0
structure_stability_score,composite_auxiliary,composite_auxiliary,,0,1,,,,
fluss_clasp_agreement,diagnostic,case_only,,0,3,,,,
mp_discord_isolation,diagnostic,case_only,4.0,3,0,,0.0,0.0,1.1294853745071358
snippet_coverage,diagnostic,case_only,0.1111111111111111,3,0,,0.0,0.0,0.0
discord_maneuver_overlap,fragment_replay,higher_is_better,0.0714285714285714,3,0,chronaris;mult;contiformer;naive_time_sync,0.0,0.0,0.0
motif_event_consistency,fragment_replay,higher_is_better,0.0,3,0,chronaris;mult;contiformer;naive_time_sync,0.0,0.0,0.0
nn_segment_cross_view_consistency,fragment_replay,higher_is_better,,0,2,,,,
discord_physio_overlap,fragment_replay_appendix,case_only,0.15,3,0,,0.0,0.0,0.0833333333333333
state_count,neutral_diagnostic,neutral,1.0,3,0,,0.0,0.0,0.0
transition_entropy,neutral_diagnostic,neutral,0.0,3,0,,0.0,0.0,0.0
cp_tolerance_hit_rate,state_segmentation,higher_is_better,0.0,3,0,chronaris;mult;contiformer;naive_time_sync,0.0,0.0,0.0
cross_view_segment_stability,state_segmentation,higher_is_better,1.0,1,1,chronaris;mult;contiformer;naive_time_sync,0.0,0.0,0.0
segment_event_purity,state_segmentation,higher_is_better,0.3333333333333333,3,0,chronaris;mult;contiformer;naive_time_sync,0.0,0.0,0.0
clap_state_replay_consistency,state_segmentation_diagnostic,direction_uncertain,1.0,1,1,,0.0,0.0,0.0
```

Direction-aware summary is stored in `e3_method_metric_summary.csv`.

## 6. Chronaris Relative Performance

Current E3 performance is `mixed`. Chronaris is generally tied with MulT, ContiFormer, and naive_time_sync on the available direction-aware metrics, but the tied values are often weak or non-informative: `cp_tolerance_hit_rate=0`, `motif_event_consistency=0`, and `event_alignment_score=0.10119` for all methods. This does not support a main-text Chronaris advantage.

There are no robust supportive rows; formerly tied-best diagnostics are treated as mixed because they are not differentiating. Mixed rows include neutral, composite, CLaP-sensitive, or zero-signal metrics. Unfavorable E3 rows are not the main issue; the issue is lack of positive differentiating structure signal.

## 7. T1/T2 Consistency

T1/T2 context:

```csv
source,task,split_strategy,metric,direction,chronaris_value,mult_value,contiformer_value,naive_or_classical_value,interpretation,paper_implication
dingxin_thirdparty_comparison,T1_maneuver_intensity_class,leave_one_view_out,balanced_accuracy,higher_is_better,0.3333333333333333,0.3532763532763532,0.356125356125356,0.3504273504273504,does_not_support_chronaris_vs_deep_baselines,T1/T2 remains separate from E3; unfavorable rows argue against using E3 as a main proof.
dingxin_thirdparty_comparison,T1_maneuver_intensity_class,leave_one_view_out,macro_f1,higher_is_better,0.1733333333333333,0.203459684898388,0.2095579035396664,0.211837227644681,does_not_support_chronaris_vs_deep_baselines,T1/T2 remains separate from E3; unfavorable rows argue against using E3 as a main proof.
dingxin_thirdparty_comparison,T2_next_window_physiology_response,leave_one_view_out,mae,lower_is_better,762.5578999733825,273.38187323676215,273.5298546685113,360.9690202958808,does_not_support_chronaris_vs_deep_baselines,T1/T2 remains separate from E3; unfavorable rows argue against using E3 as a main proof.
dingxin_thirdparty_comparison,T2_next_window_physiology_response,leave_one_view_out,nrmse,lower_is_better,8.780987018657159,2.237883859210544,2.23608136177063,4.111481326727145,does_not_support_chronaris_vs_deep_baselines,T1/T2 remains separate from E3; unfavorable rows argue against using E3 as a main proof.
dingxin_thirdparty_comparison,T2_next_window_physiology_response,leave_one_view_out,rmse,lower_is_better,838.121039184215,344.28897466792205,344.3351099592478,463.13663933056176,does_not_support_chronaris_vs_deep_baselines,T1/T2 remains separate from E3; unfavorable rows argue against using E3 as a main proof.
metric_calibration_t1_candidate,T1_maneuver_intensity_class,leave_one_view_out,macro_f1,higher_is_better,0.2046136279570929,,,,supports_later_chronaris_t1_calibration_against_older_chronaris_reference_only,"This supports the confirmed T1 calibration boundary, but it is not an E3 result and does not fix the T2/E3 structure gap."
metric_calibration_t1_candidate,T1_maneuver_intensity_class,leave_one_view_out,balanced_accuracy,higher_is_better,0.3532763532763532,,,,supports_later_chronaris_t1_calibration_against_older_chronaris_reference_only,"This supports the confirmed T1 calibration boundary, but it is not an E3 result and does not fix the T2/E3 structure gap."
metric_calibration_t1_candidate,T1_maneuver_intensity_class,leave_one_view_out,macro_f1,higher_is_better,0.1973148550200222,,,,supports_later_chronaris_t1_calibration_against_older_chronaris_reference_only,"This supports the confirmed T1 calibration boundary, but it is not an E3 result and does not fix the T2/E3 structure gap."
metric_calibration_t1_candidate,T1_maneuver_intensity_class,leave_one_view_out,balanced_accuracy,higher_is_better,0.3447293447293447,,,,supports_later_chronaris_t1_calibration_against_older_chronaris_reference_only,"This supports the confirmed T1 calibration boundary, but it is not an E3 result and does not fix the T2/E3 structure gap."
```

The confirmed third-party comparison has Chronaris behind MulT and ContiFormer on T2 regression. Later T1 calibration improves the Chronaris T1 boundary, but E3 does not show corresponding structural superiority. This mismatch argues for controlled Chronaris optimization rather than using E3 as a result claim.

## 8. Figure Review

Figure inventory rows: 24. Candidate table rows: 4.

```csv
candidate_label,file_path,method_name,figure_type,paper_candidate_level,reason
four_method_state_timeline_comparison,,,,reject,No four-method combined state timeline exists in the current run; individual timelines are not enough for a main figure.
chronaris_fragment_replay_case,docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-four-method-validation/plots/fragment_replay_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png,chronaris,fragment_replay,appendix,Best available Chronaris case figure; use only as diagnostic appendix if needed.
deep_baseline_fragment_replay_case,docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-four-method-validation/plots/fragment_replay_contiformer_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png,contiformer,fragment_replay,appendix,Best available MulT/ContiFormer case figure for qualitative comparison.
clap_unavailable_boundary_figure,docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-four-method-validation/plots/state_timeline_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png,chronaris,state_timeline,reject,Use textual limitation instead of a main figure; single-state fallback can be misleading.
```

No main-text E3 figure is recommended. Fragment replay plots can be appendix diagnostics; state timelines and transition graphs mostly show the CLaP short-sequence limitation.

## 9. Paper Use Decision

Decision: `D. needs_chronaris_optimization`.

E3 should not enter the main experiment as proof of Chronaris advantage. At most, it can appear as an appendix diagnostic or as motivation for a controlled Chronaris candidate optimization round.

## 10. Negative-result Handling

This is not treated as a failure to hide. The result indicates: current E3 has limited differentiating power on 36-window streams, CLaP is unavailable across streams, Chronaris does not gain structural advantage over T2-trained deep baseline embeddings, and T2 confirmed context remains unfavorable against MulT / ContiFormer.

## 11. Recommended Next Step

Use `next_prompt_for_chronaris_controlled_optimization.md`. The next round should optimize Chronaris candidates under fixed baselines and fixed evaluators, using Pareto filtering across T1, T2, and E3.

## 12. What Was Not Done

- No model training.
- No E3 rerun.
- No evaluator parameter changes.
- No confirmed metric edits.
- No thesis protocol snapshot edits.
- No T3 artifact deletion.
