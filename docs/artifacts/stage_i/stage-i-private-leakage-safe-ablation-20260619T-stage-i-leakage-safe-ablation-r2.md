# Stage I Leakage-Safe Dingxin Component-Diagnostic Ablation - 20260619T-stage-i-leakage-safe-ablation-r2

- protocol: `leakage_safe_v1`
- leakage_safe: `True`
- audit_status: `pass`
- seeds: `[17, 29, 43, 71, 97]`
- split_strategy: `['leave_one_view_out', 'leave_one_sortie_out']`
- label-feature audit: `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/label_feature_overlap_audit.json`

## 读取边界

分类任务、回归任务和检索任务均属于从现有鼎新数据派生的弱监督组件诊断。`leakage_safe_v1` 不覆盖历史结果，而是新增排除标签源字段、确定性派生特征、样本身份与窗口位置的评价协议。

## 消融汇总

| group | task | component | metric | mean | std | relative_delta_percent |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `model_backbone` | 分类任务 宏平均F1 | 双流连续表示 | `macro_f1` | 0.276605 | 0.000000 | -59.580 |
| `model_backbone` | 回归任务 RMSE | 双流连续表示 | `rmse` | 498.662419 | 0.000000 | -42.197 |
| `model_backbone` | 检索任务 Top-1 | 双流连续表示 | `top1_accuracy` | 0.067568 | 0.000000 | -150.000 |
| `task_adapter` | 分类任务 宏平均F1 | 完整防泄漏任务输入 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `task_adapter` | 回归任务 RMSE | 完整防泄漏任务输入 | `rmse` | 862.694175 | 0.000000 | 0.000 |
| `task_adapter` | 检索任务 Top-1 | 完整防泄漏任务输入 | `top1_accuracy` | 0.027027 | 0.000000 | 0.000 |
| `model_backbone` | 分类任务 宏平均F1 | 完整方案 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `model_backbone` | 回归任务 RMSE | 完整方案 | `rmse` | 862.694175 | 0.000000 | 0.000 |
| `model_backbone` | 检索任务 Top-1 | 完整方案 | `top1_accuracy` | 0.027027 | 0.000000 | 0.000 |
| `model_backbone` | 分类任务 宏平均F1 | 朴素时间同步 | `macro_f1` | 0.276605 | 0.000000 | -59.580 |
| `model_backbone` | 回归任务 RMSE | 朴素时间同步 | `rmse` | 498.662419 | 0.000000 | -42.197 |
| `model_backbone` | 检索任务 Top-1 | 朴素时间同步 | `top1_accuracy` | 0.067568 | 0.000000 | -150.000 |
| `task_adapter` | 分类任务 宏平均F1 | 仅融合潜态 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `task_adapter` | 回归任务 RMSE | 仅融合潜态 | `rmse` | 1047.354460 | 0.000000 | 21.405 |
| `task_adapter` | 检索任务 Top-1 | 仅融合潜态 | `top1_accuracy` | 0.027027 | 0.000000 | 0.000 |
| `model_backbone` | 分类任务 宏平均F1 | 移除因果掩码 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `model_backbone` | 回归任务 RMSE | 移除因果掩码 | `rmse` | 862.823712 | 0.000000 | 0.015 |
| `model_backbone` | 检索任务 Top-1 | 移除因果掩码 | `top1_accuracy` | 0.027027 | 0.000000 | 0.000 |
| `model_backbone` | 分类任务 宏平均F1 | 移除物理约束 | `macro_f1` | 0.170022 | 0.000000 | 1.910 |
| `model_backbone` | 回归任务 RMSE | 移除物理约束 | `rmse` | 3281.980439 | 0.000000 | 280.434 |
| `model_backbone` | 检索任务 Top-1 | 移除物理约束 | `top1_accuracy` | 0.027027 | 0.000000 | 0.000 |
| `task_adapter` | 分类任务 宏平均F1 | 移除原始窗口统计残差 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `task_adapter` | 回归任务 RMSE | 移除原始窗口统计残差 | `rmse` | 862.694175 | 0.000000 | 0.000 |
| `task_adapter` | 检索任务 Top-1 | 移除原始窗口统计残差 | `top1_accuracy` | 0.027027 | 0.000000 | 0.000 |
| `model_backbone` | 分类任务 宏平均F1 | 移除语义事件融合 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `model_backbone` | 回归任务 RMSE | 移除语义事件融合 | `rmse` | 821.828927 | 0.000000 | -4.737 |
| `model_backbone` | 检索任务 Top-1 | 移除语义事件融合 | `top1_accuracy` | 0.027027 | 0.000000 | 0.000 |
| `task_adapter` | 分类任务 宏平均F1 | 移除任务头 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `task_adapter` | 回归任务 RMSE | 移除任务头 | `rmse` | 201.489565 | 0.000000 | -76.644 |
| `task_adapter` | 检索任务 Top-1 | 移除任务头 | `top1_accuracy` | 0.027027 | 0.000000 | 0.000 |
| `task_adapter` | 分类任务 宏平均F1 | 移除时间位置特征 | `macro_f1` | 0.173333 | 0.000000 | 0.000 |
| `task_adapter` | 回归任务 RMSE | 移除时间位置特征 | `rmse` | 862.694175 | 0.000000 | 0.000 |
| `task_adapter` | 检索任务 Top-1 | 移除时间位置特征 | `top1_accuracy` | 0.027027 | 0.000000 | 0.000 |
| `task_adapter` | 分类任务 宏平均F1 | 仅单模态表示 | `macro_f1` | 0.262981 | 0.000000 | -51.720 |
| `task_adapter` | 回归任务 RMSE | 仅单模态表示 | `rmse` | 665.161615 | 0.000000 | -22.897 |
| `task_adapter` | 检索任务 Top-1 | 仅单模态表示 | `top1_accuracy` | 0.027027 | 0.000000 | 0.000 |

## 检索任务诊断

检索任务使用 `same_sortie_cross_pilot` 候选池：候选集合限定为同一 sortie 的另一名飞行员窗口；`pilot_id/window_index` 仍不进入特征向量。

| component | candidate_policy | query_count | candidate_count | top1 | top3 | top5 | mrr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 双流连续表示 | `same_sortie_cross_pilot` | 74 | 2738 | 0.067568 | 0.121622 | 0.175676 | 0.152959 |
| 朴素时间同步 | `same_sortie_cross_pilot` | 74 | 2738 | 0.067568 | 0.121622 | 0.175676 | 0.152959 |
| 完整防泄漏任务输入 | `same_sortie_cross_pilot` | 74 | 2738 | 0.027027 | 0.094595 | 0.148649 | 0.119948 |
| 完整方案 | `same_sortie_cross_pilot` | 74 | 2738 | 0.027027 | 0.094595 | 0.148649 | 0.119948 |
| 仅融合潜态 | `same_sortie_cross_pilot` | 74 | 2738 | 0.027027 | 0.081081 | 0.135135 | 0.113556 |
| 移除因果掩码 | `same_sortie_cross_pilot` | 74 | 2738 | 0.027027 | 0.108108 | 0.162162 | 0.124087 |
| 移除物理约束 | `same_sortie_cross_pilot` | 74 | 2738 | 0.027027 | 0.081081 | 0.135135 | 0.113546 |
| 移除原始窗口统计残差 | `same_sortie_cross_pilot` | 74 | 2738 | 0.027027 | 0.094595 | 0.148649 | 0.119948 |
| 移除语义事件融合 | `same_sortie_cross_pilot` | 74 | 2738 | 0.027027 | 0.067568 | 0.121622 | 0.107175 |
| 移除任务头 | `same_sortie_cross_pilot` | 74 | 2738 | 0.027027 | 0.094595 | 0.148649 | 0.119948 |
| 移除时间位置特征 | `same_sortie_cross_pilot` | 74 | 2738 | 0.027027 | 0.094595 | 0.148649 | 0.119948 |
| 仅单模态表示 | `same_sortie_cross_pilot` | 74 | 2738 | 0.027027 | 0.094595 | 0.148649 | 0.119948 |

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
