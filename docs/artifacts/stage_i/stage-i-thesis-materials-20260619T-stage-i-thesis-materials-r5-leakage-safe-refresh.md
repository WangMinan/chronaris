# Stage I Thesis Materials - 20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh

## 概览

- 本轮将中期报告/PPT 图表刷新为 `11` 张 PNG 与对应 `11` 张 CSV，所有数值来自已有 artifact JSON/CSV 或本轮 rotation metadata audit。
- 证据层级继续分开：thesis weak-label、private proxy、public adapter、runtime/schema、semantic support、rigid-body/rotation、LLM preprocessing/comparison 不合并为同一层结论。
- `live_influx` weak-label sweep: sample_count=`111`, task_entry_count=`333`, best_test_total=`1153.898570`。
- `stage_h_window_stats_proxy` weak-label sweep: sample_count=`111`, task_entry_count=`333`, best_test_total=`1024.809990`。
- runtime/schema: native=`aligned` with vehicle `965`, canonical=`exact` with vehicle `1930`。
- runtime semantic case: view_id=`20251005_四01_ACT-4_云_J20_22#01__pilot_10033`，windows=`8`，query_types=`risk_proxy,workload_proxy`，schema=`native aligned / canonical exact`。
- semantic support: view_count=`4`, query_count=`3`，主图改为 event token / query-to-event attribution。
- LLM comparison: `3->7; added=4`；human review packet `15` 条，仍为 pending review。
- rotation audit: `disabled`；current sortie still lacks paired rate fields for pitch/roll/yaw。

## 图表替换说明

| figure_id | PNG | CSV | 替代的问题 |
| --- | --- | --- | --- |
| `evidence_layer_overview` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/evidence_layer_overview.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/evidence_layer_overview.csv` | evidence_layer_overview no longer uses all-one artifact-present bars. |
| `runtime_payload_schema` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/runtime_payload_schema.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/runtime_payload_schema.csv` | runtime_payload_schema is rendered as a contract comparison instead of a generic schema plot. |
| `runtime_semantic_case` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/runtime_semantic_case.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/runtime_semantic_case.csv` | uses a compact selected-window axis and replaces near-flat/repeated attribution bars with case-level semantic attribution and task-output ranges. |
| `rigid_body_rotation_audit` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/rigid_body_rotation_audit.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/rigid_body_rotation_audit.csv` | rotation rate absence is shown as a matrix, not empty bars. |
| `weak_label_sweep_ablation` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/weak_label_sweep_ablation.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/weak_label_sweep_ablation.csv` | weak-label sweep no longer uses two near-identical bars. |
| `chronaris_opt_component_ablation` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/chronaris_opt_component_ablation.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/chronaris_opt_component_ablation.csv` | mixed-unit delta bars are replaced by task facets plus normalized contribution. |
| `model_backbone_ablation` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/model_backbone_ablation.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/model_backbone_ablation.csv` |  |
| `task_adapter_ablation` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/task_adapter_ablation.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/task_adapter_ablation.csv` |  |
| `public_transfer_boundary` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/public_transfer_boundary.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/public_transfer_boundary.csv` | defensive boundary copy is replaced by positive Chinese role wording. |
| `semantic_event_fusion_overview` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/semantic_event_fusion_overview.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/semantic_event_fusion_overview.csv` | replaces the old Stage G minimal causal-mask heatmap as the main semantic fusion figure. |
| `llm_comparison_a0_a4` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/llm_comparison_a0_a4.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/llm_comparison_a0_a4.csv` | adds LLM preprocessing/comparison evidence without treating LLM output as truth. |

## Tables

- `evidence_layer_overview`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/evidence_layer_overview.csv` (`7` rows, `12` columns)
- `runtime_payload_schema`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/runtime_payload_schema.csv` (`2` rows, `14` columns)
- `runtime_semantic_case`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/runtime_semantic_case.csv` (`8` rows, `23` columns)
- `rigid_body_rotation_audit`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/rigid_body_rotation_audit.csv` (`15` rows, `15` columns)
- `weak_label_sweep_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/weak_label_sweep_ablation.csv` (`5` rows, `25` columns)
- `chronaris_opt_component_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/chronaris_opt_component_ablation.csv` (`36` rows, `27` columns)
- `model_backbone_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/model_backbone_ablation.csv` (`18` rows, `27` columns)
- `task_adapter_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/task_adapter_ablation.csv` (`18` rows, `27` columns)
- `public_transfer_boundary`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/public_transfer_boundary.csv` (`3` rows, `10` columns)
- `semantic_event_fusion_overview`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/semantic_event_fusion_overview.csv` (`4` rows, `18` columns)
- `llm_comparison_a0_a4`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/llm_comparison_a0_a4.csv` (`6` rows, `9` columns)

## Figures

- `evidence_layer_overview`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/evidence_layer_overview.png` | evidence_layer=`cross_layer_index`
- `runtime_payload_schema`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/runtime_payload_schema.png` | evidence_layer=`runtime_schema`
- `runtime_semantic_case`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/runtime_semantic_case.png` | evidence_layer=`runtime_semantic_support`
- `rigid_body_rotation_audit`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/rigid_body_rotation_audit.png` | evidence_layer=`rigid_body_rotation_diagnostics`
- `weak_label_sweep_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/weak_label_sweep_ablation.png` | evidence_layer=`thesis_weak_label`
- `chronaris_opt_component_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/chronaris_opt_component_ablation.png` | evidence_layer=`private_proxy`
- `model_backbone_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/model_backbone_ablation.png` | evidence_layer=`private_proxy_leakage_safe`
- `task_adapter_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/task_adapter_ablation.png` | evidence_layer=`private_proxy_leakage_safe`
- `public_transfer_boundary`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/public_transfer_boundary.png` | evidence_layer=`transfer_boundary`
- `semantic_event_fusion_overview`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/semantic_event_fusion_overview.png` | evidence_layer=`semantic_support`
- `llm_comparison_a0_a4`: `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/llm_comparison_a0_a4.png` | evidence_layer=`llm_preprocessing_comparison`

## 仍受数据限制的边界

- rotation：本轮已重新检查 MySQL metadata，pitch/roll/yaw angle 有候选，pitch_rate/roll_rate/yaw_rate 仍缺失，因此不复跑 rotation-enabled rigid-body 对照。
- runtime：当前保持 native aligned / canonical exact；未声称生产级在线服务，也未声称原生 replay payload 已 exact。
- weak-label sweep：本轮使用已有 stable/partial artifacts 重绘，不包装成大规模搜索。
- LLM semantic hints：当前对比只证明 query coverage `3 -> 7`，没有从 summary 倒推出 view ranking 或 attribution 改善。

- Plot font: `using CJK font WenQuanYi Zen Hei`
