# task evaluation Dingxin Real-Data Third-party Comparison - 20260702T-task-eval-private-thirdparty-comparison-gpuopt-r1

## Executive Summary

On the Dingxin / feature export real dual-stream dataset, Chronaris is compared with MulT and ContiFormer under the same leakage-safe split manifest. The comparison uses real physiology and real vehicle time-series streams, with label-source fields and identity/time-position features excluded from model inputs. Across the classification, regression and retrieval component-diagnostic tasks, the report provides model-level leaderboard, fold-level stability and Chronaris-vs-third-party deltas.

## Dataset and protocol

- evidence_role: `private_real_dual_stream`
- sample_facts: `{'sample_count': 111, 'view_count': 3, 'sortie_count': 2}`
- split_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/split_manifest.json`

## Leakage-safe audit

- audit_json: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/label_feature_overlap_audit.json`
- audit_csv: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/label_feature_overlap_audit.csv`

## Main leaderboard

| task | split_strategy | model | metric_display | value_mean | value_std | seed_count |
| --- | --- | --- | --- | --- | --- | --- |
| 分类任务 | leave_one_view_out | Chronaris完整模型 | macro-F1 | 0.1733 | 0.0000 | 3 |
| 分类任务 | leave_one_view_out | Chronaris完整模型 | balanced accuracy | 0.3333 | 0.0000 | 3 |
| 分类任务 | leave_one_sortie_out | Chronaris完整模型 | macro-F1 | 0.1733 | 0.0000 | 3 |
| 分类任务 | leave_one_sortie_out | Chronaris完整模型 | balanced accuracy | 0.3333 | 0.0000 | 3 |
| 分类任务 | leave_one_view_out | MulT | macro-F1 | 0.2035 | 0.0156 | 3 |
| 分类任务 | leave_one_view_out | MulT | balanced accuracy | 0.3533 | 0.0081 | 3 |
| 分类任务 | leave_one_sortie_out | MulT | macro-F1 | 0.2132 | 0.0148 | 3 |
| 分类任务 | leave_one_sortie_out | MulT | balanced accuracy | 0.3397 | 0.0052 | 3 |
| 分类任务 | leave_one_view_out | ContiFormer | macro-F1 | 0.2096 | 0.0255 | 3 |
| 分类任务 | leave_one_view_out | ContiFormer | balanced accuracy | 0.3561 | 0.0145 | 3 |
| 分类任务 | leave_one_sortie_out | ContiFormer | macro-F1 | 0.2230 | 0.0122 | 3 |
| 分类任务 | leave_one_sortie_out | ContiFormer | balanced accuracy | 0.3376 | 0.0060 | 3 |
| 分类任务 | leave_one_view_out | 传统特征基线 | macro-F1 | 0.2118 | 0.0000 | 3 |
| 分类任务 | leave_one_view_out | 传统特征基线 | balanced accuracy | 0.3504 | 0.0000 | 3 |
| 分类任务 | leave_one_sortie_out | 传统特征基线 | macro-F1 | 0.1888 | 0.0000 | 3 |
| 分类任务 | leave_one_sortie_out | 传统特征基线 | balanced accuracy | 0.3333 | 0.0000 | 3 |
| 回归任务 | leave_one_view_out | Chronaris完整模型 | RMSE | 838.1210 | 0.0000 | 3 |
| 回归任务 | leave_one_view_out | Chronaris完整模型 | MAE | 762.5579 | 0.0000 | 3 |
| 回归任务 | leave_one_view_out | Chronaris完整模型 | NRMSE | 8.7810 | 0.0000 | 3 |
| 回归任务 | leave_one_sortie_out | Chronaris完整模型 | RMSE | 1111.9789 | 0.0000 | 3 |
| 回归任务 | leave_one_sortie_out | Chronaris完整模型 | MAE | 992.5239 | 0.0000 | 3 |
| 回归任务 | leave_one_sortie_out | Chronaris完整模型 | NRMSE | 11.3163 | 0.0000 | 3 |
| 回归任务 | leave_one_view_out | MulT | RMSE | 344.2890 | 0.3901 | 3 |
| 回归任务 | leave_one_view_out | MulT | MAE | 273.3819 | 0.4420 | 3 |
| 回归任务 | leave_one_view_out | MulT | NRMSE | 2.2379 | 0.0045 | 3 |
| 回归任务 | leave_one_sortie_out | MulT | RMSE | 413.4298 | 0.4768 | 3 |
| 回归任务 | leave_one_sortie_out | MulT | MAE | 314.7452 | 0.6979 | 3 |
| 回归任务 | leave_one_sortie_out | MulT | NRMSE | 2.0426 | 0.0011 | 3 |
| 回归任务 | leave_one_view_out | ContiFormer | RMSE | 344.3351 | 0.0458 | 3 |
| 回归任务 | leave_one_view_out | ContiFormer | MAE | 273.5299 | 0.0409 | 3 |
| 回归任务 | leave_one_view_out | ContiFormer | NRMSE | 2.2361 | 0.0014 | 3 |
| 回归任务 | leave_one_sortie_out | ContiFormer | RMSE | 414.2889 | 0.2751 | 3 |
| 回归任务 | leave_one_sortie_out | ContiFormer | MAE | 315.9286 | 0.3334 | 3 |
| 回归任务 | leave_one_sortie_out | ContiFormer | NRMSE | 2.0463 | 0.0027 | 3 |
| 回归任务 | leave_one_view_out | 传统特征基线 | RMSE | 463.1366 | 0.0000 | 3 |
| 回归任务 | leave_one_view_out | 传统特征基线 | MAE | 360.9690 | 0.0000 | 3 |
| 回归任务 | leave_one_view_out | 传统特征基线 | NRMSE | 4.1115 | 0.0000 | 3 |
| 回归任务 | leave_one_sortie_out | 传统特征基线 | RMSE | 11284.2992 | 0.0000 | 3 |
| 回归任务 | leave_one_sortie_out | 传统特征基线 | MAE | 8571.0432 | 0.0000 | 3 |
| 回归任务 | leave_one_sortie_out | 传统特征基线 | NRMSE | 141.4624 | 0.0000 | 3 |
| 检索任务 | leave_one_view_out | Chronaris完整模型 | Top-1 | 0.0270 | 0.0000 | 3 |
| 检索任务 | leave_one_view_out | Chronaris完整模型 | Top-3 | 0.0946 | 0.0000 | 3 |
| 检索任务 | leave_one_view_out | Chronaris完整模型 | Top-5 | 0.1486 | 0.0000 | 3 |
| 检索任务 | leave_one_view_out | Chronaris完整模型 | MRR | 0.1199 | 0.0000 | 3 |
| 检索任务 | leave_one_sortie_out | Chronaris完整模型 | Top-1 | 0.0270 | 0.0000 | 3 |
| 检索任务 | leave_one_sortie_out | Chronaris完整模型 | Top-3 | 0.0946 | 0.0000 | 3 |
| 检索任务 | leave_one_sortie_out | Chronaris完整模型 | Top-5 | 0.1486 | 0.0000 | 3 |
| 检索任务 | leave_one_sortie_out | Chronaris完整模型 | MRR | 0.1199 | 0.0000 | 3 |
| 检索任务 | leave_one_view_out | MulT | Top-1 | 0.0270 | 0.0000 | 3 |
| 检索任务 | leave_one_view_out | MulT | Top-3 | 0.0901 | 0.0064 | 3 |
| 检索任务 | leave_one_view_out | MulT | Top-5 | 0.1441 | 0.0064 | 3 |
| 检索任务 | leave_one_view_out | MulT | MRR | 0.1178 | 0.0030 | 3 |
| 检索任务 | leave_one_view_out | ContiFormer | Top-1 | 0.0270 | 0.0000 | 3 |
| 检索任务 | leave_one_view_out | ContiFormer | Top-3 | 0.0856 | 0.0064 | 3 |
| 检索任务 | leave_one_view_out | ContiFormer | Top-5 | 0.1396 | 0.0064 | 3 |
| 检索任务 | leave_one_view_out | ContiFormer | MRR | 0.1157 | 0.0030 | 3 |
| 检索任务 | leave_one_view_out | 朴素时间同步基线 | Top-1 | 0.0270 | 0.0000 | 3 |
| 检索任务 | leave_one_view_out | 朴素时间同步基线 | Top-3 | 0.0811 | 0.0000 | 3 |
| 检索任务 | leave_one_view_out | 朴素时间同步基线 | Top-5 | 0.1351 | 0.0000 | 3 |
| 检索任务 | leave_one_view_out | 朴素时间同步基线 | MRR | 0.1136 | 0.0000 | 3 |

## Improvement over third-party baselines

| task | split_strategy | metric_display | baseline | chronaris_value | baseline_value | delta_abs | delta_rel_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 分类任务 | leave_one_view_out | macro-F1 | MulT | 0.1733 | 0.2035 | -0.0301 | -14.8070 |
| 分类任务 | leave_one_view_out | macro-F1 | ContiFormer | 0.1733 | 0.2096 | -0.0362 | -17.2862 |
| 分类任务 | leave_one_view_out | macro-F1 | 传统特征基线 | 0.1733 | 0.2118 | -0.0385 | -18.1762 |
| 分类任务 | leave_one_view_out | balanced accuracy | MulT | 0.3333 | 0.3533 | -0.0199 | -5.6452 |
| 分类任务 | leave_one_view_out | balanced accuracy | ContiFormer | 0.3333 | 0.3561 | -0.0228 | -6.4000 |
| 分类任务 | leave_one_view_out | balanced accuracy | 传统特征基线 | 0.3333 | 0.3504 | -0.0171 | -4.8780 |
| 分类任务 | leave_one_sortie_out | macro-F1 | MulT | 0.1733 | 0.2132 | -0.0399 | -18.6946 |
| 分类任务 | leave_one_sortie_out | macro-F1 | ContiFormer | 0.1733 | 0.2230 | -0.0497 | -22.2691 |
| 分类任务 | leave_one_sortie_out | macro-F1 | 传统特征基线 | 0.1733 | 0.1888 | -0.0154 | -8.1802 |
| 分类任务 | leave_one_sortie_out | balanced accuracy | MulT | 0.3333 | 0.3397 | -0.0064 | -1.8868 |
| 分类任务 | leave_one_sortie_out | balanced accuracy | ContiFormer | 0.3333 | 0.3376 | -0.0043 | -1.2658 |
| 分类任务 | leave_one_sortie_out | balanced accuracy | 传统特征基线 | 0.3333 | 0.3333 | 0.0000 | 0.0000 |
| 回归任务 | leave_one_view_out | RMSE | MulT | 838.1210 | 344.2890 | -493.8321 | -143.4353 |
| 回归任务 | leave_one_view_out | RMSE | ContiFormer | 838.1210 | 344.3351 | -493.7859 | -143.4027 |
| 回归任务 | leave_one_view_out | RMSE | 传统特征基线 | 838.1210 | 463.1366 | -374.9844 | -80.9663 |
| 回归任务 | leave_one_view_out | MAE | MulT | 762.5579 | 273.3819 | -489.1760 | -178.9351 |
| 回归任务 | leave_one_view_out | MAE | ContiFormer | 762.5579 | 273.5299 | -489.0280 | -178.7842 |
| 回归任务 | leave_one_view_out | MAE | 传统特征基线 | 762.5579 | 360.9690 | -401.5889 | -111.2530 |
| 回归任务 | leave_one_view_out | NRMSE | MulT | 8.7810 | 2.2379 | -6.5431 | -292.3790 |
| 回归任务 | leave_one_view_out | NRMSE | ContiFormer | 8.7810 | 2.2361 | -6.5449 | -292.6953 |
| 回归任务 | leave_one_view_out | NRMSE | 传统特征基线 | 8.7810 | 4.1115 | -4.6695 | -113.5723 |
| 回归任务 | leave_one_sortie_out | RMSE | MulT | 1111.9789 | 413.4298 | -698.5492 | -168.9644 |
| 回归任务 | leave_one_sortie_out | RMSE | ContiFormer | 1111.9789 | 414.2889 | -697.6900 | -168.4066 |
| 回归任务 | leave_one_sortie_out | RMSE | 传统特征基线 | 1111.9789 | 11284.2992 | 10172.3202 | 90.1458 |
| 回归任务 | leave_one_sortie_out | MAE | MulT | 992.5239 | 314.7452 | -677.7786 | -215.3420 |
| 回归任务 | leave_one_sortie_out | MAE | ContiFormer | 992.5239 | 315.9286 | -676.5953 | -214.1608 |
| 回归任务 | leave_one_sortie_out | MAE | 传统特征基线 | 992.5239 | 8571.0432 | 7578.5194 | 88.4200 |
| 回归任务 | leave_one_sortie_out | NRMSE | MulT | 11.3163 | 2.0426 | -9.2738 | -454.0296 |
| 回归任务 | leave_one_sortie_out | NRMSE | ContiFormer | 11.3163 | 2.0463 | -9.2701 | -453.0185 |
| 回归任务 | leave_one_sortie_out | NRMSE | 传统特征基线 | 11.3163 | 141.4624 | 130.1461 | 92.0005 |
| 检索任务 | leave_one_view_out | Top-1 | MulT | 0.0270 | 0.0270 | 0.0000 | 0.0000 |
| 检索任务 | leave_one_view_out | Top-1 | ContiFormer | 0.0270 | 0.0270 | 0.0000 | 0.0000 |
| 检索任务 | leave_one_view_out | Top-1 | 朴素时间同步基线 | 0.0270 | 0.0270 | 0.0000 | 0.0000 |
| 检索任务 | leave_one_view_out | Top-1 | 传统特征基线 | 0.0270 | 0.0676 | -0.0405 | -60.0000 |
| 检索任务 | leave_one_view_out | Top-3 | MulT | 0.0946 | 0.0901 | 0.0045 | 5.0000 |
| 检索任务 | leave_one_view_out | Top-3 | ContiFormer | 0.0946 | 0.0856 | 0.0090 | 10.5263 |
| 检索任务 | leave_one_view_out | Top-3 | 朴素时间同步基线 | 0.0946 | 0.0811 | 0.0135 | 16.6667 |
| 检索任务 | leave_one_view_out | Top-3 | 传统特征基线 | 0.0946 | 0.1216 | -0.0270 | -22.2222 |
| 检索任务 | leave_one_view_out | Top-5 | MulT | 0.1486 | 0.1441 | 0.0045 | 3.1250 |
| 检索任务 | leave_one_view_out | Top-5 | ContiFormer | 0.1486 | 0.1396 | 0.0090 | 6.4516 |
| 检索任务 | leave_one_view_out | Top-5 | 朴素时间同步基线 | 0.1486 | 0.1351 | 0.0135 | 10.0000 |
| 检索任务 | leave_one_view_out | Top-5 | 传统特征基线 | 0.1486 | 0.1757 | -0.0270 | -15.3846 |
| 检索任务 | leave_one_view_out | MRR | MulT | 0.1199 | 0.1178 | 0.0021 | 1.8083 |
| 检索任务 | leave_one_view_out | MRR | ContiFormer | 0.1199 | 0.1157 | 0.0043 | 3.6832 |
| 检索任务 | leave_one_view_out | MRR | 朴素时间同步基线 | 0.1199 | 0.1136 | 0.0064 | 5.6285 |
| 检索任务 | leave_one_view_out | MRR | 传统特征基线 | 0.1199 | 0.1530 | -0.0330 | -21.5819 |
| 检索任务 | leave_one_sortie_out | Top-1 | 朴素时间同步基线 | 0.0270 | 0.0270 | 0.0000 | 0.0000 |
| 检索任务 | leave_one_sortie_out | Top-1 | 传统特征基线 | 0.0270 | 0.0676 | -0.0405 | -60.0000 |
| 检索任务 | leave_one_sortie_out | Top-3 | 朴素时间同步基线 | 0.0946 | 0.0811 | 0.0135 | 16.6667 |
| 检索任务 | leave_one_sortie_out | Top-3 | 传统特征基线 | 0.0946 | 0.1216 | -0.0270 | -22.2222 |
| 检索任务 | leave_one_sortie_out | Top-5 | 朴素时间同步基线 | 0.1486 | 0.1351 | 0.0135 | 10.0000 |
| 检索任务 | leave_one_sortie_out | Top-5 | 传统特征基线 | 0.1486 | 0.1757 | -0.0270 | -15.3846 |
| 检索任务 | leave_one_sortie_out | MRR | 朴素时间同步基线 | 0.1199 | 0.1136 | 0.0064 | 5.6285 |
| 检索任务 | leave_one_sortie_out | MRR | 传统特征基线 | 0.1199 | 0.1530 | -0.0330 | -21.5819 |

## Figure index

- `fig_private_third_party_task_leaderboard`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_third_party_task_leaderboard.png`
- `fig_private_third_party_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_third_party_delta_heatmap.png`
- `fig_private_third_party_fold_variance`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_third_party_fold_variance.png`
- `fig_private_third_party_training_curves`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_third_party_training_curves.png`
- `fig_private_third_party_retrieval_topk`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_third_party_retrieval_topk.png`
- `fig_private_third_party_gpu_throughput`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_third_party_gpu_throughput.png`
- `fig_private_thirdparty_t1_macro_f1`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_thirdparty_t1_macro_f1.png`
- `fig_private_thirdparty_t2_rmse`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_thirdparty_t2_rmse.png`
- `fig_private_thirdparty_t3_retrieval`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_thirdparty_t3_retrieval.png`
- `fig_private_thirdparty_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_thirdparty_delta_heatmap.png`
- `fig_private_thirdparty_fold_stability`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_thirdparty_fold_stability.png`
- `fig_private_thirdparty_confusion_t1`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_thirdparty_confusion_t1.png`
- `fig_private_thirdparty_t2_error_distribution`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_thirdparty_t2_error_distribution.png`
- `fig_private_thirdparty_t3_retrieval_curve`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/fig_private_thirdparty_t3_retrieval_curve.png`

## Reproducibility

- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison`
- config: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/private_thirdparty_config.json`
- evidence_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/evidence_manifest.json`
- run_log: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/run.log`
- progress: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/progress.json`

## Midterm-ready wording

鼎新 / feature export 分支在同一 leakage-safe 任务协议下比较 Chronaris、MulT 与 ContiFormer，量化真实生理流和真实航电流连续对齐场景中的模型适配性。分类任务、回归任务和检索任务均属于从现有鼎新数据派生的组件诊断任务，结果以 fold-level stability、mean/std 和 Chronaris-vs-baseline delta 展示。
