# Stage I LLM Preprocessing - 20260614T-stage-i-p20-deepseek-llm-preprocessing-r1

- status: `success`
- mode: `build-and-evaluate`
- artifact_root: `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r1`
- context_path: `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r1/llm_preprocessing_context.json`
- request_count: `5`
- error_count: `0`

## Boundary

LLM output is preprocessing context, semantic hints, rule review, and runtime explanation. It is not manual ground truth, does not fabricate missing BUS values, and does not convert canonical exact into native exact evidence.

## Output Counts

- field_semantic_count: `24`
- weak_label_review_count: `3`
- semantic_query_hint_count: `4`
- runtime_explanation_count: `4`

## Weak-Label Comparison

| task_name | decision | sample_count | agreement | conflict | human_review | agreement_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `event_replay_tag` | `needs_human_review` | 111 | 0 | 0 | 111 | 0.000000 |
| `risk_proxy` | `keep_current_rule` | 111 | 111 | 0 | 0 | 1.000000 |
| `workload_proxy` | `keep_current_rule` | 111 | 111 | 0 | 0 | 1.000000 |

## Source Paths

- multitask_summary_path: `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_summary.json`
- task_manifest_path: `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`
- live_sweep_summary_path: `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json`
- support_summary_path: `docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`
- runtime_schema_contract_path: `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json`
- runtime_service_summary_path: `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_service_smoke_summary.json`
- runtime_case_table_path: `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv`
