# task evaluation Dingxin Real-Data Third-party Comparison - 20260702T-task-eval-metric-calibration-r1-t3-screen

## Executive Summary

On the Dingxin / feature export real dual-stream dataset, Chronaris is compared with MulT and ContiFormer under the same leakage-safe split manifest. The comparison uses real physiology and real vehicle time-series streams, with label-source fields and identity/time-position features excluded from model inputs. Across 分类任务、回归任务和检索任务 weak-label constructed tasks, the report provides model-level leaderboard, fold-level stability and Chronaris-vs-third-party deltas.

## Dataset and protocol

- evidence_role: dingxin_real_dual_stream
- sample_facts: `{'sample_count': 111, 'view_count': 3, 'sortie_count': 2}`
- split_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/split_manifest.json`

## Leakage-safe audit

- audit_json: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/label_feature_overlap_audit.json`
- audit_csv: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/label_feature_overlap_audit.csv`

## Main leaderboard

| task_name | split_strategy | model_name | metric | value_mean | value_std | seed_count |
| --- | --- | --- | --- | --- | --- | --- |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_info_nce_temp0p05_hardw2 | top1 | 0.0270 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_info_nce_temp0p05_hardw2 | top3 | 0.0946 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_info_nce_temp0p05_hardw2 | top5 | 0.1486 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_info_nce_temp0p05_hardw2 | mrr | 0.1199 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_info_nce_temp0p10_hardw2 | top1 | 0.0270 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_info_nce_temp0p10_hardw2 | top3 | 0.0946 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_info_nce_temp0p10_hardw2 | top5 | 0.1486 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_info_nce_temp0p10_hardw2 | mrr | 0.1199 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_info_nce_temp0p20_hardw1 | top1 | 0.0270 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_info_nce_temp0p20_hardw1 | top3 | 0.0946 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_info_nce_temp0p20_hardw1 | top5 | 0.1486 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_info_nce_temp0p20_hardw1 | mrr | 0.1199 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_supcon_margin0p10_hardw2 | top1 | 0.0270 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_supcon_margin0p10_hardw2 | top3 | 0.0946 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_supcon_margin0p10_hardw2 | top5 | 0.1486 | 0.0000 | 1 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | p37_t3_supcon_margin0p10_hardw2 | mrr | 0.1199 | 0.0000 | 1 |

## Improvement over third-party baselines

_No rows._

## Figure index

- `fig_private_third_party_task_leaderboard`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_third_party_task_leaderboard.png`
- `fig_private_third_party_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_third_party_delta_heatmap.png`
- `fig_private_third_party_fold_variance`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_third_party_fold_variance.png`
- `fig_private_third_party_training_curves`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_third_party_training_curves.png`
- `fig_private_third_party_retrieval_topk`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_third_party_retrieval_topk.png`
- `fig_private_third_party_gpu_throughput`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_third_party_gpu_throughput.png`
- `fig_private_thirdparty_t1_macro_f1`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_thirdparty_t1_macro_f1.png`
- `fig_private_thirdparty_t2_rmse`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_thirdparty_t2_rmse.png`
- `fig_private_thirdparty_t3_retrieval`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_thirdparty_t3_retrieval.png`
- `fig_private_thirdparty_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_thirdparty_delta_heatmap.png`
- `fig_private_thirdparty_fold_stability`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_thirdparty_fold_stability.png`
- `fig_private_thirdparty_confusion_t1`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_thirdparty_confusion_t1.png`
- `fig_private_thirdparty_t2_error_distribution`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_thirdparty_t2_error_distribution.png`
- `fig_private_thirdparty_t3_retrieval_curve`: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/fig_private_thirdparty_t3_retrieval_curve.png`

## Reproducibility

- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen`
- config: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/private_thirdparty_config.json`
- evidence_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/evidence_manifest.json`
- run_log: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/run.log`
- progress: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/nested_private/20260702T-task-eval-metric-calibration-r1-t3-screen/progress.json`

## Midterm-ready wording

自有鼎新 / feature export 分支在同一 leakage-safe 任务协议下比较 Chronaris、MulT 与 ContiFormer，量化真实生理流和真实航电流连续对齐场景中的模型适配性。分类任务、回归任务和检索任务 均保持 weak-label constructed task 标注边界，结果以 fold-level stability、mean/std 和 Chronaris-vs-baseline delta 展示。
