# Stage I LLM Preprocessing Comparison - 20260614T-stage-i-p21-llm-comparison-r1

- status: `success`
- artifact_root: `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1`
- summary_path: `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/llm_comparison_summary.json`
- condition_manifest_path: `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/condition_manifest.json`
- midterm_summary_path: `docs/midterm/llm-preprocessing-comparison-summary-2026-06-14.md`

## Boundary

LLM outputs are used only as preprocessing context, whitelisted semantic hints, runtime explanations, and human-review material. They do not overwrite weak-label values, do not become manual truth, and do not prove the core causal-fusion claim.

## Conditions

| condition | name | status | key gate |
| --- | --- | --- | --- |
| `A0` | `baseline` | `completed` | `control` |
| `A1` | `llm_context` | `completed` | `label_unchanged=true` |
| `A2` | `llm_semantic_hints` | `completed_coverage_only` | `recipe_whitelist` |
| `A3` | `llm_runtime_explanation` | `success_bounded_explained_subset` | `model_prediction / semantic_attribution / schema_gap_note / weak_label_boundary` |
| `A4` | `human_review_packet` | `pending_human_review` | `human_review_completed=false` |

## Result Summary

| area | result | boundary |
| --- | --- | --- |
| task context | `333/333` entries attached, `label_unchanged=true` | `no_label_overwrite` |
| semantic hints | query coverage `3 -> 7`, added `4` whitelisted hints | `ranking_not_recomputed_from_summary_only` |
| runtime explanations | `4/12` cases have LLM explanations, complete explained cases `4` | `explanation_not_expert_truth` |
| human review packet | `15` items generated, `human_review_completed=false` | `pending_human_review` |

## Task Context Comparison

| task | baseline entries | attached entries | label unchanged | review decision | human review |
| --- | ---: | ---: | --- | --- | --- |
| `event_replay_tag` | 111 | 111 | `true` | `keep_current_rule` | `true` |
| `risk_proxy` | 111 | 111 | `true` | `keep_current_rule` | `true` |
| `workload_proxy` | 111 | 111 | `true` | `keep_current_rule` | `true` |

## Semantic Hint Comparison

| query | condition | recipe | source | whitelisted | ranking status |
| --- | --- | --- | --- | --- | --- |
| `risk_proxy` | `A0_baseline` | `gap_plus_event` | `built_in_query_bank` | `true` | `not_recomputed_from_summary_only` |
| `workload_proxy` | `A0_baseline` | `physiology_plus_gap` | `built_in_query_bank` | `true` | `not_recomputed_from_summary_only` |
| `event_replay_tag` | `A0_baseline` | `vehicle_plus_event` | `built_in_query_bank` | `true` | `not_recomputed_from_summary_only` |
| `pilot_vehicle_event_gap` | `A2_llm_semantic_hints` | `vehicle_plus_event` | `P20_llm_semantic_hints` | `true` | `not_recomputed_from_summary_only` |
| `event_gap_analysis` | `A2_llm_semantic_hints` | `gap_plus_event` | `P20_llm_semantic_hints` | `true` | `not_recomputed_from_summary_only` |
| `pilot_physiology_gap` | `A2_llm_semantic_hints` | `physiology_plus_gap` | `P20_llm_semantic_hints` | `true` | `not_recomputed_from_summary_only` |
| `multi_aircraft_coordination_gap` | `A2_llm_semantic_hints` | `coordination_gap` | `P20_llm_semantic_hints` | `true` | `not_recomputed_from_summary_only` |

## Runtime Explanation Comparison

| sample | has LLM | without LLM | with LLM | delta |
| --- | --- | ---: | ---: | ---: |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033::20251005_四01_ACT-4_云_J20_22#01:0000` | `true` | 0.75 | 1.00 | 0.25 |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033::20251005_四01_ACT-4_云_J20_22#01:0001` | `true` | 0.75 | 1.00 | 0.25 |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033::20251005_四01_ACT-4_云_J20_22#01:0002` | `true` | 0.75 | 1.00 | 0.25 |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033::20251005_四01_ACT-4_云_J20_22#01:0003` | `true` | 0.75 | 1.00 | 0.25 |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033::20251005_四01_ACT-4_云_J20_22#01:0004` | `false` | 0.75 | 0.00 | 0.00 |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033::20251005_四01_ACT-4_云_J20_22#01:0005` | `false` | 0.75 | 0.00 | 0.00 |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033::20251005_四01_ACT-4_云_J20_22#01:0006` | `false` | 0.75 | 0.00 | 0.00 |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033::20251005_四01_ACT-4_云_J20_22#01:0007` | `false` | 0.75 | 0.00 | 0.00 |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033::20251005_四01_ACT-4_云_J20_22#01:0008` | `false` | 0.75 | 0.00 | 0.00 |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033::20251005_四01_ACT-4_云_J20_22#01:0009` | `false` | 0.75 | 0.00 | 0.00 |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033::20251005_四01_ACT-4_云_J20_22#01:0010` | `false` | 0.75 | 0.00 | 0.00 |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033::20251005_四01_ACT-4_云_J20_22#01:0011` | `false` | 0.75 | 0.00 | 0.00 |

## Human Review Packet

| item type | count | validation status |
| --- | ---: | --- |
| `field_semantic` | 6 | `pending_human_review` |
| `schema_gap_policy` | 6 | `pending_human_review` |
| `weak_label_rule` | 3 | `pending_human_review` |

The packet contains empty human reviewer, decision, and notes fields. Until those fields are filled by a reviewer, this artifact is review material only, not completed validation.

## Output Paths

- task_context_comparison_path: `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/task_context_comparison.csv`
- semantic_hint_comparison_path: `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/semantic_hint_comparison.csv`
- runtime_explanation_comparison_path: `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/runtime_explanation_comparison.csv`
- human_review_packet_path: `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/human_review_packet.csv`
- midterm_claims_payload_path: `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/midterm_claims_payload.json`

## Source Paths

- llm_context_path: `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_preprocessing_context.json`
- task_manifest_path: `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`
- support_summary_path: `docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`
- runtime_schema_contract_path: `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json`
- runtime_case_table_path: `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv`
- runtime_explanations_path: `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/runtime_llm_explanations.jsonl`
