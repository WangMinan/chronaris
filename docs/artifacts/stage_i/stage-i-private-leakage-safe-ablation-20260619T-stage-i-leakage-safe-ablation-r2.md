# Stage I Leakage-Safe Private Proxy Ablation - 20260619T-stage-i-leakage-safe-ablation-r2

- protocol: `leakage_safe_v1`
- leakage_safe: `True`
- audit_status: `pass`
- seeds: `[17, 29, 43, 71, 97]`
- split_strategy: `['leave_one_view_out', 'leave_one_sortie_out']`
- label-feature audit: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/label_feature_overlap_audit.json`

## 读取边界

`T1/T2/T3` 仍属于 private proxy 组件诊断。`leakage_safe_v1` 不覆盖历史结果，而是新增排除标签源字段、确定性派生特征、样本身份与窗口位置的评价协议。

## 消融汇总

| group | task | component | metric | mean | std | relative_delta_percent |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `model_backbone` | `T1_maneuver_intensity_class` | 双流连续表示 | `macro_f1` | 0.276605 | 0.000000 | -59.580 |
| `model_backbone` | `T2_next_window_physiology_response` | 双流连续表示 | `rmse` | 498.662419 | 0.000000 | -42.197 |
| `model_backbone` | `T3_paired_pilot_window_retrieval` | 双流连续表示 | `top1_accuracy` | 0.000000 | 0.000000 | 0.000 |
| `task_adapter` | `T1_maneuver_intensity_class` | 完整leakage-safe任务输入 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `task_adapter` | `T2_next_window_physiology_response` | 完整leakage-safe任务输入 | `rmse` | 862.694175 | 0.000000 | 0.000 |
| `task_adapter` | `T3_paired_pilot_window_retrieval` | 完整leakage-safe任务输入 | `top1_accuracy` | 0.000000 | 0.000000 | 0.000 |
| `model_backbone` | `T1_maneuver_intensity_class` | 完整方案 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `model_backbone` | `T2_next_window_physiology_response` | 完整方案 | `rmse` | 862.694175 | 0.000000 | 0.000 |
| `model_backbone` | `T3_paired_pilot_window_retrieval` | 完整方案 | `top1_accuracy` | 0.000000 | 0.000000 | 0.000 |
| `model_backbone` | `T1_maneuver_intensity_class` | 朴素时间同步 | `macro_f1` | 0.276605 | 0.000000 | -59.580 |
| `model_backbone` | `T2_next_window_physiology_response` | 朴素时间同步 | `rmse` | 498.662419 | 0.000000 | -42.197 |
| `model_backbone` | `T3_paired_pilot_window_retrieval` | 朴素时间同步 | `top1_accuracy` | 0.000000 | 0.000000 | 0.000 |
| `task_adapter` | `T1_maneuver_intensity_class` | 仅融合潜态 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `task_adapter` | `T2_next_window_physiology_response` | 仅融合潜态 | `rmse` | 1047.354460 | 0.000000 | 21.405 |
| `task_adapter` | `T3_paired_pilot_window_retrieval` | 仅融合潜态 | `top1_accuracy` | 0.000000 | 0.000000 | 0.000 |
| `model_backbone` | `T1_maneuver_intensity_class` | 移除因果掩码 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `model_backbone` | `T2_next_window_physiology_response` | 移除因果掩码 | `rmse` | 862.823712 | 0.000000 | 0.015 |
| `model_backbone` | `T3_paired_pilot_window_retrieval` | 移除因果掩码 | `top1_accuracy` | 0.000000 | 0.000000 | 0.000 |
| `model_backbone` | `T1_maneuver_intensity_class` | 移除物理约束 | `macro_f1` | 0.170022 | 0.000000 | 1.910 |
| `model_backbone` | `T2_next_window_physiology_response` | 移除物理约束 | `rmse` | 3281.980439 | 0.000000 | 280.434 |
| `model_backbone` | `T3_paired_pilot_window_retrieval` | 移除物理约束 | `top1_accuracy` | 0.000000 | 0.000000 | 0.000 |
| `task_adapter` | `T1_maneuver_intensity_class` | 移除原始窗口统计残差 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `task_adapter` | `T2_next_window_physiology_response` | 移除原始窗口统计残差 | `rmse` | 862.694175 | 0.000000 | 0.000 |
| `task_adapter` | `T3_paired_pilot_window_retrieval` | 移除原始窗口统计残差 | `top1_accuracy` | 0.000000 | 0.000000 | 0.000 |
| `model_backbone` | `T1_maneuver_intensity_class` | 移除语义事件融合 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `model_backbone` | `T2_next_window_physiology_response` | 移除语义事件融合 | `rmse` | 821.828927 | 0.000000 | -4.737 |
| `model_backbone` | `T3_paired_pilot_window_retrieval` | 移除语义事件融合 | `top1_accuracy` | 0.000000 | 0.000000 | 0.000 |
| `task_adapter` | `T1_maneuver_intensity_class` | 移除任务头 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `task_adapter` | `T2_next_window_physiology_response` | 移除任务头 | `rmse` | 201.489565 | 0.000000 | -76.644 |
| `task_adapter` | `T3_paired_pilot_window_retrieval` | 移除任务头 | `top1_accuracy` | 0.000000 | 0.000000 | 0.000 |
| `task_adapter` | `T1_maneuver_intensity_class` | 移除时间位置特征 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `task_adapter` | `T2_next_window_physiology_response` | 移除时间位置特征 | `rmse` | 862.694175 | 0.000000 | 0.000 |
| `task_adapter` | `T3_paired_pilot_window_retrieval` | 移除时间位置特征 | `top1_accuracy` | 0.000000 | 0.000000 | 0.000 |
| `task_adapter` | `T1_maneuver_intensity_class` | 仅单模态表示 | `macro_f1` | 0.262981 | 0.000000 | -51.720 |
| `task_adapter` | `T2_next_window_physiology_response` | 仅单模态表示 | `rmse` | 665.161615 | 0.000000 | -22.897 |
| `task_adapter` | `T3_paired_pilot_window_retrieval` | 仅单模态表示 | `top1_accuracy` | 0.000000 | 0.000000 | 0.000 |

## 产物

- label_feature_overlap_audit_json: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/label_feature_overlap_audit.json`
- label_feature_overlap_audit_csv: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/label_feature_overlap_audit.csv`
- seed_metrics_csv: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/seed_metrics.csv`
- split_manifest_json: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/split_manifest.json`
- cross_view_metrics_csv: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/cross_view_metrics.csv`
- cross_sortie_metrics_csv: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/cross_sortie_metrics.csv`
- model_backbone_ablation_csv: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/model_backbone_ablation.csv`
- model_backbone_ablation_json: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/model_backbone_ablation.json`
- model_backbone_ablation_png: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/model_backbone_ablation.png`
- task_adapter_ablation_csv: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/task_adapter_ablation.csv`
- task_adapter_ablation_json: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/task_adapter_ablation.json`
- task_adapter_ablation_png: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/task_adapter_ablation.png`
- t2_error_distribution_png: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/t2_error_distribution.png`
- t3_similarity_distribution_csv: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/t3_similarity_distribution.csv`
- t3_similarity_distribution_png: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/t3_similarity_distribution.png`
