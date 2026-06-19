# Stage I Runtime Service Smoke - 20260613T-stage-i-runtime-service-smoke-r1

- checkpoint_path: `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_checkpoint.pt`
- sample_jsonl_path: raw input JSONL pruned from docs/LFS on 2026-06-19; see `docs/artifacts/cleanup/20260619-lfs-docs-prune.md`
- input_sample_count: `37`
- view_ids: `['20251005_四01_ACT-4_云_J20_22#01__pilot_10033']`
- predictions_jsonl_path: `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/runtime_inference/20260613T-stage-i-runtime-service-smoke-r1-runtime/runtime_inference_predictions.jsonl`
- runtime_summary_path: `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/runtime_inference/20260613T-stage-i-runtime-service-smoke-r1-runtime/runtime_inference_summary.json`

## Error Cases

- `missing_checkpoint`: `expected_failure` | `FileNotFoundError` | `checkpoint not found: docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/missing-checkpoint.pt`
- `missing_fields`: `expected_failure` | `ValueError` | `runtime payload missing required fields: ['vehicle']`
- `empty_window`: `expected_failure` | `ValueError` | `empty window detected in `physiology` stream`
- `schema_mismatch`: `expected_failure` | `ValueError` | `runtime feature schema mismatch: missing_physiology_count=0, extra_physiology_count=0, missing_vehicle_count=965, extra_vehicle_count=1, extra_vehicle_names=['schema.extra']`

## Figures

- `runtime_service_flow`: `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/runtime_service_flow.png`
- `runtime_payload_schema`: `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/runtime_payload_schema.png`
- `runtime_error_cases`: `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/runtime_error_cases.png`
