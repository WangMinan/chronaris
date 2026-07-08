# E3 Paper Use Decision

Decision: `D. needs_chronaris_optimization`

## Reason

The current E3 validation is complete and reproducible, but it does not provide a positive Chronaris advantage claim. Four methods each contribute 108 input rows and the validation run has `training_invoked=false`, yet the direction-aware E3 metrics are mostly tied or non-informative: `cp_tolerance_hit_rate=0`, `motif_event_consistency=0`, `segment_event_purity=0.3333`, `discord_maneuver_overlap=0.0714`, and `event_alignment_score=0.10119` are effectively identical for Chronaris, MulT, ContiFormer, and naive_time_sync. CLaP is unavailable for all 12 streams, so CLaP-dependent replay rows are not suitable for main claims.

The T1/T2 context strengthens the optimization decision. The confirmed third-party comparison has Chronaris behind MulT and ContiFormer on T2 RMSE/MAE/NRMSE under leave-one-view-out. T1 later has an accepted calibration candidate, but that does not convert into E3 structural advantage and does not resolve T2.

## Citable Metrics

- `cross_view_segment_stability`: completed only where two views exist; all methods tie at 1.0, so it can describe evaluator behavior but not a Chronaris advantage.
- `segment_event_purity`: all methods tie at 0.3333; it can be listed in an appendix table as a weak-label diagnostic.
- `discord_maneuver_overlap`: all methods tie near 0.0714; only weak-label post-hoc diagnostic, not expert truth.
- `event_alignment_score`: fixed-weight auxiliary score, all methods tie at 0.10119; not a main conclusion.

## Non-citable Or Risky Metrics

- `cp_tolerance_hit_rate` and `motif_event_consistency`: all methods are zero, so they do not support a positive claim.
- `replay_consistency_score` and `clap_state_replay_consistency`: not main-table candidates because CLaP is unavailable and state sequences are fallback-derived.
- `state_count`, `transition_entropy`, `mp_discord_isolation`, `snippet_coverage`: diagnostic or appendix-only; they are not winner metrics.
- `structure_stability_score` and `fragment_replay_recall`: unavailable because `nn_segment_cross_view_consistency` is unavailable.

## Recommended Paper Location

Do not use E3 as a main experimental proof in the current form. Mention it, if needed, as an appendix diagnostic or as motivation for the next controlled Chronaris optimization. The main thesis argument should continue to rely on the confirmed T1/T2/T3 boundaries already present in the protocol snapshot.

## Recommended Table

Use `e3_method_metric_summary.csv` only as an appendix diagnostic table. Do not merge it into `result_matrix_long.csv`, `experiment_registry.csv`, or any confirmed metric table.

## Recommended Figures

No main-text E3 figure is recommended. The best available fragment replay figures can be used in the appendix as qualitative diagnostics; state timeline and transition graph plots are not recommended because CLaP did not form stable multi-state outputs.

## Risks

- MulT and ContiFormer inputs are T2-trained held-out pooled embeddings, while Chronaris and naive_time_sync are reconstructed from existing feature values; this is a valid recorded protocol but not an identical representation-family comparison.
- Current Dingxin E3 coverage has only 3 streams per method and 36 windows per stream.
- Weak event and physiology flags are post-hoc proxies, not expert truth.
- Tuning E3 evaluator parameters would invalidate this review boundary; optimization should target Chronaris candidates under a new run root.
