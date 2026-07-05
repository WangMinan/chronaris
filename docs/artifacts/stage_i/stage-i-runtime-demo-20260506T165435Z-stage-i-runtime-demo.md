# Stage I Runtime Demo - 20260506T165435Z-stage-i-runtime-demo

- generated_at_utc: `2026-05-07T02:39:29.890909Z`
- source_type: `optimized_candidate_package`
- source_path: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json`

## Optimized Package Overview

- package_path: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json`
- target_variant_name: `chronaris_opt`
- source_run_id: `20260504T120000Z-stage-i-private-opt-package`
- record_sample_count: `111`
- record_view_count: `3`
- selected_vehicle_field_count: `1930`
- selected_physiology_field_count: `12`

## Dependency Contract

- requires_stage_h_all_window_contract: `True`
- requires_f_full_reference_hidden: `True`
- requires_stage_g_causal_fusion: `True`
- use_causal_mask: `True`
- fusion_output_mode: `pooled_with_residual`
- lag_window_points: `3`
- residual_mode: `raw_window_stats`

## Task Export Summary

| task | task_type | head_family | recommended_head | prediction_contract_available | metric snapshot |
| --- | --- | --- | --- | --- | --- |
| 分类任务：机动强度分类 | `classification` | `class_balanced_threshold` | `class_balanced_threshold` | `True` | `macro_f1=1.000000, balanced_accuracy=1.000000` |
| 回归任务：下一窗口生理响应 | `regression` | `n/a` | `physiology_persistence` | `True` | `rmse=201.489565, mae=113.851926` |
| 检索任务：配对飞行员窗口检索 | `retrieval` | `chronaris_time_residual_retrieval` | `chronaris_time_residual_retrieval` | `True` | `top1_accuracy=1.000000, mrr=1.000000` |

## Diagnostics

- mean_attention_entropy: `0.934591421135911`
- mean_top_event_concentration: `0.3974255199904914`
- mean_event_mask_interference: `0.0021227399508158364`
- mean_causal_residual_gate: `1.0`
- lag_window_points: `3`
- residual_mode: `raw_window_stats`
- use_causal_mask: `True`
