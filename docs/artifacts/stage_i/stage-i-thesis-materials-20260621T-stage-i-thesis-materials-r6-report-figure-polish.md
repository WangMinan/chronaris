# 中期图表材料 - 20260621T-stage-i-thesis-materials-r6-report-figure-polish

## 概览

- 本轮将中期报告图表刷新为 `12` 张 PNG 与对应 `12` 张 CSV，所有数值来自已有 JSON/CSV summary 或本轮旋转字段审计。
- 证据层级继续分开：论文弱标注、私有代理、公开适配、运行字段契约、语义融合支撑、刚体/旋转诊断和大语言模型预处理对比分别解读。
- `live_influx` weak-label sweep: sample_count=`111`, task_entry_count=`333`, best_test_total=`1153.898570`。
- `stage_h_window_stats_proxy` weak-label sweep: sample_count=`111`, task_entry_count=`333`, best_test_total=`1024.809990`。
- 运行字段契约：原始输入状态=`aligned`，飞机状态字段=`965`；统一输入状态=`exact`，字段维度=`1930`。
- 代表性窗口案例：窗口数=`8`，查询类型=`risk_proxy,workload_proxy`，字段检查=`原始 aligned / 统一 exact`。
- 语义融合支撑：视图记录=`4`，查询类型=`3`；缺少完整归因矩阵时仅展示覆盖/支撑状态。
- 大语言模型预处理对比：`3->7; added=4`；复核材料 `15` 条，状态为待人工复核。
- 旋转字段诊断：`disabled`；current sortie still lacks paired rate fields for pitch/roll/yaw。

## 图表替换说明

| figure_id | PNG | CSV | 替代的问题 |
| --- | --- | --- | --- |
| `evidence_layer_overview` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/evidence_layer_overview.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/evidence_layer_overview.csv` | evidence_layer_overview is redrawn as a 2x4 report-readable card overview. |
| `runtime_payload_schema` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/runtime_payload_schema.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/runtime_payload_schema.csv` | runtime_payload_schema is rendered as a contract comparison instead of a generic schema plot. |
| `runtime_service_flow` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/runtime_service_flow.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/runtime_service_flow.csv` | runtime_service_flow is redrawn as a 2x2 Chinese process diagram without internal runtime labels. |
| `runtime_semantic_case` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/runtime_semantic_case.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/runtime_semantic_case.csv` | uses a compact selected-window axis and replaces near-flat/repeated attribution bars with case-level semantic attribution and task-output ranges. |
| `rigid_body_rotation_audit` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/rigid_body_rotation_audit.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/rigid_body_rotation_audit.csv` | rotation rate absence is shown as a matrix, not empty bars. |
| `weak_label_sweep_ablation` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/weak_label_sweep_ablation.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/weak_label_sweep_ablation.csv` | weak-label sweep uses a lag-window trend plus task-record coverage instead of a repeated heatmap. |
| `chronaris_opt_component_ablation` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/chronaris_opt_component_ablation.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/chronaris_opt_component_ablation.csv` | redundant full-candidate bars are replaced by baseline metric cards and a compact signed relative-change overview. |
| `model_backbone_ablation` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/model_backbone_ablation.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/model_backbone_ablation.csv` | model_backbone_ablation uses full task names, full model first, and a zero-centered diverging heatmap. |
| `task_adapter_ablation` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/task_adapter_ablation.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/task_adapter_ablation.csv` | task_adapter_ablation uses readable task-input labels and a zero-centered diverging heatmap. |
| `public_transfer_boundary` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/public_transfer_boundary.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/public_transfer_boundary.csv` | defensive boundary copy is replaced by positive Chinese role wording. |
| `semantic_event_fusion_overview` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/semantic_event_fusion_overview.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/semantic_event_fusion_overview.csv` | semantic fusion overview is redrawn as a report-readable flow and coverage/status matrix. |
| `llm_comparison_a0_a4` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/llm_comparison_a0_a4.png` | `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/llm_comparison_a0_a4.csv` | adds LLM preprocessing/comparison evidence without treating LLM output as truth. |

## Tables

- `evidence_layer_overview`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/evidence_layer_overview.csv` (`7` rows, `12` columns)
- `runtime_payload_schema`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/runtime_payload_schema.csv` (`2` rows, `14` columns)
- `runtime_service_flow`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/runtime_service_flow.csv` (`4` rows, `8` columns)
- `runtime_semantic_case`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/runtime_semantic_case.csv` (`8` rows, `24` columns)
- `rigid_body_rotation_audit`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/rigid_body_rotation_audit.csv` (`15` rows, `16` columns)
- `weak_label_sweep_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/weak_label_sweep_ablation.csv` (`5` rows, `28` columns)
- `chronaris_opt_component_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/chronaris_opt_component_ablation.csv` (`36` rows, `39` columns)
- `model_backbone_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/model_backbone_ablation.csv` (`18` rows, `39` columns)
- `task_adapter_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/task_adapter_ablation.csv` (`18` rows, `39` columns)
- `public_transfer_boundary`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/public_transfer_boundary.csv` (`3` rows, `10` columns)
- `semantic_event_fusion_overview`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/semantic_event_fusion_overview.csv` (`4` rows, `19` columns)
- `llm_comparison_a0_a4`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/llm_comparison_a0_a4.csv` (`6` rows, `9` columns)

## Figures

- `evidence_layer_overview`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/evidence_layer_overview.png` | evidence_layer=`cross_layer_index`
- `runtime_payload_schema`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/runtime_payload_schema.png` | evidence_layer=`runtime_schema`
- `runtime_service_flow`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/runtime_service_flow.png` | evidence_layer=`runtime_schema`
- `runtime_semantic_case`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/runtime_semantic_case.png` | evidence_layer=`runtime_semantic_support`
- `rigid_body_rotation_audit`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/rigid_body_rotation_audit.png` | evidence_layer=`rigid_body_rotation_diagnostics`
- `weak_label_sweep_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/weak_label_sweep_ablation.png` | evidence_layer=`thesis_weak_label`
- `chronaris_opt_component_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/chronaris_opt_component_ablation.png` | evidence_layer=`private_proxy`
- `model_backbone_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/model_backbone_ablation.png` | evidence_layer=`private_proxy_strict_protocol`
- `task_adapter_ablation`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/task_adapter_ablation.png` | evidence_layer=`private_proxy_strict_protocol`
- `public_transfer_boundary`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/public_transfer_boundary.png` | evidence_layer=`transfer_boundary`
- `semantic_event_fusion_overview`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/semantic_event_fusion_overview.png` | evidence_layer=`semantic_support`
- `llm_comparison_a0_a4`: `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/llm_comparison_a0_a4.png` | evidence_layer=`llm_preprocessing_comparison`

## 仍受数据限制的边界

- 旋转诊断：本轮已重新检查 MySQL metadata，pitch/roll/yaw angle 有候选，pitch_rate/roll_rate/yaw_rate 仍缺失，因此不复跑启用旋转残差的刚体对照。
- 运行字段契约：当前保持原始输入已对齐、统一输入已校验；未声称生产级在线服务，也未声称原始回放输入已经完全补齐。
- 弱标注 sweep：本轮使用已有稳定/部分执行产物重绘，不包装成大规模搜索。
- 大语言模型查询建议：当前对比只证明查询覆盖 `3 -> 7`，没有从 summary 倒推出视图排序或归因改善。

- Plot font: `using CJK font WenQuanYi Zen Hei`
