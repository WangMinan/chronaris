# task evaluation LLM Preprocessing - 20260614T-task-eval-p20-deepseek-llm-preprocessing-r3-sliced

- status: `success`
- mode: `build-and-evaluate`
- prompt_version: `task_eval_llm_preprocessing.agent_guardrails.v2`
- schema_version: `task_eval_llm_preprocessing_context.v2`
- artifact_root: `docs/artifacts/runs/2026-06-14_llm-preprocessing-context`
- context_path: `docs/artifacts/runs/2026-06-14_llm-preprocessing-context/llm_preprocessing_context.json`
- harness_summary_path: `docs/artifacts/runs/2026-06-14_llm-preprocessing-context/llm_harness_summary.json`
- request_count: `8`
- error_count: `0`

## Boundary

LLM output is preprocessing context, semantic hints, rule review, and runtime explanation. It is not manual ground truth, does not fabricate missing BUS values, and does not convert canonical exact into native exact evidence.

## Output Counts

- field_semantic_count: `24`
- weak_label_review_count: `3`
- semantic_query_hint_count: `4`
- runtime_explanation_count: `4`

## Harness Gates

- attempt_count: `8`
- provider_failure_attempt_count: `0`
- schema_repair_attempt_count: `0`
- failed_initial_attempt_count: `0`
- selected_invalid_task_count: `0`

| task_name | attempt | valid | output_count | failed_gates |
| --- | --- | --- | ---: | --- |
| `field_semantics` | `initial` | `True` | 12 | `` |
| `field_semantics` | `initial` | `True` | 12 | `` |
| `weak_label_review` | `initial` | `True` | 3 | `` |
| `semantic_query_hints` | `initial` | `True` | 4 | `` |
| `schema_gap_policy` | `initial` | `True` | 3 | `` |
| `schema_gap_policy` | `initial` | `True` | 3 | `` |
| `runtime_explanations` | `initial` | `True` | 2 | `` |
| `runtime_explanations` | `initial` | `True` | 2 | `` |

## Payload Slicing

- initial_call_count: `8`
- sliced_task_count: `3`
- sliced_tasks: `field_semantics, runtime_explanations, schema_gap_policy`

| task_name | item_key | slice | item_count | total_item_count | sliced |
| --- | --- | ---: | ---: | ---: | --- |
| `field_semantics` | `fields` | 1/2 | 12 | 24 | `True` |
| `field_semantics` | `fields` | 2/2 | 12 | 24 | `True` |
| `weak_label_review` | `task_summaries` | 1/1 | 3 | 3 | `False` |
| `semantic_query_hints` | `global_card` | 1/1 | 1 | 1 | `False` |
| `schema_gap_policy` | `missing_measurement_groups` | 1/2 | 3 | 6 | `True` |
| `schema_gap_policy` | `missing_measurement_groups` | 2/2 | 3 | 6 | `True` |
| `runtime_explanations` | `runtime_cases` | 1/2 | 2 | 4 | `True` |
| `runtime_explanations` | `runtime_cases` | 2/2 | 2 | 4 | `True` |

## Pipeline Integration

- `field_semantic_dictionary` provides audited field-role hints for schema review before task evaluation task building.
- `weak_label_rule_review` is attached to thesis task entries as optional context; it does not overwrite labels.
- `semantic_query_hints` are converted only through whitelisted semantic recipes before entering event-level fusion support.
- `schema_gap_policy` keeps native-aligned versus canonical-exact runtime boundaries explicit.
- `runtime_case_explanations` summarize predictions, schema gaps, semantic attribution, and weak-label boundaries for replay cases.

## Next Comparison Work

- Compare baseline task evaluation task entries against LLM-context-attached entries with label values held fixed.
- Compare semantic support with built-in query bank only versus built-in plus whitelisted LLM semantic hints.
- Compare runtime explanation/report completeness with and without LLM preprocessing context.
- Add human review on a small field/rule sample to measure whether LLM review reduces manual audit effort.

## Weak-Label Comparison

| task_name | decision | sample_count | agreement | conflict | human_review | agreement_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 事件回放标签 | `keep_current_rule` | 111 | 111 | 0 | 111 | 1.000000 |
| risk_weak_label | `keep_current_rule` | 111 | 111 | 0 | 111 | 1.000000 |
| workload_weak_label | `keep_current_rule` | 111 | 111 | 0 | 111 | 1.000000 |

## Source Paths

- multitask_summary_path: `docs/artifacts/runs/2026-06-07_dingxin-multitask-real-closure/multitask_summary.json`
- task_manifest_path: `docs/artifacts/runs/2026-06-07_dingxin-multitask-real-closure/thesis_task_manifest.jsonl`
- live_sweep_summary_path: `docs/artifacts/runs/2026-06-13_dingxin-weak-label-sweep-resume/multitask_sweep_summary.json`
- support_summary_path: `docs/artifacts/runs/2026-06-07_semantic-support/support_summary.json`
- runtime_schema_contract_path: `docs/artifacts/runs/2026-06-13_runtime-schema-contract/runtime_schema_contract.json`
- runtime_service_summary_path: `docs/artifacts/runs/2026-06-13_runtime-schema-contract/runtime_service_smoke_summary.json`
- runtime_case_table_path: `docs/artifacts/runs/2026-06-21_thesis-materials-report-figures/runtime_semantic_case.csv`
