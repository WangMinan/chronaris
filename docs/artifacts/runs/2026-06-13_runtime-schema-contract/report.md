# task evaluation Runtime Service Smoke - 20260613T-task-eval-runtime-service-smoke-r2-contract

- checkpoint_path: `docs/artifacts/runs/2026-06-07_dingxin-multitask-real-closure/multitask_checkpoint.pt`
- sample_jsonl_path: raw input JSONL pruned from docs/LFS on 2026-06-19; see `docs/artifacts/cleanup/20260619-lfs-docs-prune.md`
- input_sample_count: `37`
- view_ids: `['20251005_四01_ACT-4_云_J20_22#01__pilot_10033']`
- runtime_summary_path: `docs/artifacts/runs/2026-06-13_runtime-schema-contract/runtime_inference/20260613T-task-eval-runtime-service-smoke-r2-contract-runtime/runtime_inference_summary.json`
- canonical_runtime_summary_path: `docs/artifacts/runs/2026-06-13_runtime-schema-contract/runtime_inference/20260613T-task-eval-runtime-service-smoke-r2-contract-canonical/runtime_inference_summary.json`
- schema_contract_path: `docs/artifacts/runs/2026-06-13_runtime-schema-contract/runtime_schema_contract.json`
- native_feature_schema_status: `aligned`
- canonical_feature_schema_status: `exact`
- vehicle_feature_gap: `965 -> 1930`

## Error Cases

- `missing_checkpoint`: `expected_failure` | `FileNotFoundError` | `checkpoint not found: docs/artifacts/runs/2026-06-07_dingxin-multitask-real-closure/missing-checkpoint.pt`
- `missing_fields`: `expected_failure` | `ValueError` | `runtime payload missing required fields: ['vehicle']`
- `empty_window`: `expected_failure` | `ValueError` | `empty window detected in `physiology` stream`
- `schema_mismatch`: `expected_failure` | `ValueError` | `runtime feature schema mismatch: missing_physiology_count=0, extra_physiology_count=0, missing_vehicle_count=965, extra_vehicle_count=1, extra_vehicle_names=['schema.extra']`
- `native_strict_feature_schema`: `expected_failure` | `ValueError` | `native strict feature schema mismatch: missing_vehicle_count=965, extra_vehicle_count=0, missing_measurement_groups={'BUS6000019110021': 221, 'BUS6000019110022': 140, 'BUS6000019110023': 560, 'BUS6000019110024': 21, 'BUS6000019110025': 2, 'BUS6000019110026': 21}`

## Figures

- `runtime_service_flow`: `docs/artifacts/runs/2026-06-13_runtime-schema-contract/runtime_service_flow.png`
- `runtime_payload_schema`: `docs/artifacts/runs/2026-06-13_runtime-schema-contract/runtime_payload_schema.png`
- `runtime_error_cases`: `docs/artifacts/runs/2026-06-13_runtime-schema-contract/runtime_error_cases.png`
