# task evaluation Optimized Selected Polish - 20260702T-task-eval-metric-calibration-r1

## Executive Summary
- status: `completed`
- runtime_device: `cuda`
- boundary: earlier comparison and ablation runs are fixed references; public rows use a context-derived second input stream.
- decision: accept the classification-task calibration and public-data route calibration where they improve confirmed references; keep the confirmed retrieval-task result where the new candidate does not improve it.

## Fixed references and protocol boundary
- task-head confirmed reference: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_task-head-calibration`
- stream-role confirmed reference: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_stream-role-fusion`

## Retrieval Task Selected Polish
- metrics: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/t3_metric_calibration_metrics.csv`
- delta vs confirmed retrieval reference: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/p37_delta_vs_p34.csv`
| metric_display | confirmed_reference_value | candidate_model | candidate_value | delta_positive_is_better |
| --- | --- | --- | --- | --- |
| Top-1 | 0.0315315 | 检索任务：InfoNCE低温候选 | 0.0315315 | 0 |
| Top-3 | 0.0855856 | 检索任务：InfoNCE低温候选 | 0.0855856 | 0 |
| Top-5 | 0.13964 | 检索任务：InfoNCE低温候选 | 0.13964 | 0 |
| MRR | 0.117939 | 检索任务：InfoNCE低温候选 | 0.117939 | 0 |

## Classification Task Calibration
- metrics: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/t1_calibration_metrics.csv`
| split_strategy | metric_display | confirmed_reference_value | candidate_model | candidate_value | delta_positive_is_better |
| --- | --- | --- | --- | --- | --- |
| leave_one_view_out | macro-F1 | 0.187489 | 分类任务：焦点损失+门控候选 | 0.204614 | 0.0171246 |
| leave_one_view_out | balanced accuracy | 0.344729 | 分类任务：焦点损失+门控候选 | 0.353276 | 0.00854701 |
| leave_one_sortie_out | macro-F1 | 0.216065 | 分类任务：轻门控候选 | 0.220099 | 0.00403356 |
| leave_one_sortie_out | balanced accuracy | 0.34188 | 分类任务：轻门控候选 | 0.337607 | -0.0042735 |
| task | split_strategy | model | metric_display | value_mean |
| --- | --- | --- | --- | --- |
| 分类任务 | leave_one_view_out | 分类任务：焦点损失+门控候选 | macro-F1 | 0.204614 |
| 分类任务 | leave_one_view_out | 分类任务：焦点损失+门控候选 | balanced accuracy | 0.353276 |
| 分类任务 | leave_one_sortie_out | 分类任务：焦点损失+门控候选 | macro-F1 | 0.202212 |
| 分类任务 | leave_one_sortie_out | 分类任务：焦点损失+门控候选 | balanced accuracy | 0.33547 |
| 分类任务 | leave_one_view_out | 分类任务：轻门控候选 | macro-F1 | 0.197315 |
| 分类任务 | leave_one_view_out | 分类任务：轻门控候选 | balanced accuracy | 0.344729 |
| 分类任务 | leave_one_sortie_out | 分类任务：轻门控候选 | macro-F1 | 0.220099 |
| 分类任务 | leave_one_sortie_out | 分类任务：轻门控候选 | balanced accuracy | 0.337607 |

## Public route calibration
- metrics: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/public_route_calibration_metrics.csv`
- gate calibration: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/route_gate_calibration.csv`
| dataset_id | variant | primary_metric | selection_score | combined_macro_f1 | mean_rmse |
| --- | --- | --- | --- | --- | --- |
| nasa_csm | 公开数据路线：自适应上下文门控 | combined_macro_f1 | 0.443919 | 0.443919 |  |
| nasa_csm | 公开数据路线：上下文适配增强 | combined_macro_f1 | 0.404797 | 0.404797 |  |
| uab_workload_dataset | 公开数据路线：自适应上下文门控 | mean_rmse | 3.19181 |  | 3.19181 |
| uab_workload_dataset | 公开数据路线：上下文适配增强 | mean_rmse | 3.32712 |  | 3.32712 |
| dataset_id | metric_display | candidate_variant | candidate_value | reference_value | delta_positive_is_better |
| --- | --- | --- | --- | --- | --- |
| nasa_csm | combined_macro_f1 | 公开数据路线：自适应上下文门控 | 0.443919 | 0.432193 | 0.0117263 |
| nasa_csm | combined_macro_f1 | 公开数据路线：上下文适配增强 | 0.404797 | 0.432193 | -0.027396 |
| uab_workload_dataset | mean_rmse | 公开数据路线：自适应上下文门控 | 3.19181 | 3.37478 | 0.182968 |
| uab_workload_dataset | mean_rmse | 公开数据路线：上下文适配增强 | 3.32712 | 3.37478 | 0.0476603 |
| scope_display | dataset_id | metric_display | reference_display | candidate_value | reference_value | delta_positive_is_better |
| --- | --- | --- | --- | --- | --- | --- |
| 鼎新真实数据 |  | macro-F1 | 已确认 Chronaris 鼎新基线 | 0.204614 | 0.173333 | 0.0312803 |
| 鼎新真实数据 |  | balanced accuracy | 已确认 Chronaris 鼎新基线 | 0.353276 | 0.333333 | 0.019943 |
| 鼎新真实数据 |  | macro-F1 | 已确认 Chronaris 鼎新基线 | 0.220099 | 0.173333 | 0.0467657 |
| 鼎新真实数据 |  | balanced accuracy | 已确认 Chronaris 鼎新基线 | 0.337607 | 0.333333 | 0.0042735 |
| 鼎新真实数据 |  | Top-1 | 已确认 Chronaris 鼎新基线 | 0.0315315 | 0.027027 | 0.0045045 |
| 鼎新真实数据 |  | Top-3 | 已确认 Chronaris 鼎新基线 | 0.0855856 | 0.0945946 | -0.00900901 |
| 鼎新真实数据 |  | Top-5 | 已确认 Chronaris 鼎新基线 | 0.13964 | 0.148649 | -0.00900901 |
| 鼎新真实数据 |  | MRR | 已确认 Chronaris 鼎新基线 | 0.117939 | 0.119948 | -0.00200877 |
| nan | nasa_csm | combined_macro_f1 | 已确认公开数据无滞后窗口基线 | 0.443919 | 0.394103 | 0.0498162 |
| nan | nasa_csm | combined_macro_f1 | 已确认公开数据无滞后窗口基线 | 0.404797 | 0.394103 | 0.0106939 |
| nan | uab_workload_dataset | mean_rmse | 已确认公开数据上下文输入基线 | 3.19181 | 4.6763 | 1.48448 |
| nan | uab_workload_dataset | mean_rmse | 已确认公开数据上下文输入基线 | 3.32712 | 4.6763 | 1.34918 |

## Accepted / rejected candidates
- accepted: 分类任务；current candidate improves macro-F1 on at least one split.
- accepted: 公开数据路线；current candidate public route improves at least one confirmed stream-role reference public metric.
- rejected: 检索任务；No current candidate retrieval metric exceeded confirmed retrieval reference in available rows.

## GPU runtime summary
- GPU summary: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/gpu_perf_summary.json`

## Figure index
- retrieval task retrieval leaderboard: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_t3_retrieval_leaderboard.png`
- retrieval task delta vs confirmed retrieval reference: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_t3_delta_vs_p34.png`
- classification task macro f1 leaderboard: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_t1_macro_f1_leaderboard.png`
- classification task gate sweep: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_t1_gate_sweep.png`
- classification task confusion matrix best: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_t1_confusion_matrix_best.png`
- public data route nasa macro f1: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_public_route_nasa_macro_f1.png`
- public data route uab rmse: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_public_route_uab_rmse.png`
- route gate profile: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_route_gate_profile.png`
- public data route delta heatmap: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_public_route_delta_heatmap.png`
- overall acceptance summary: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_overall_acceptance_summary.png`
- gpu runtime: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_gpu_runtime.png`
- retrieval task hard negative margin: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_t3_hard_negative_margin.png`
- retrieval task similarity distribution: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/fig_p37_t3_similarity_distribution.png`

## Reproducibility manifest
- manifest: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/evidence_manifest.json`
- resume: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_metric-calibration/resume_command.txt`

## Midterm / thesis-ready wording
The selected polish focuses on the two remaining diagnostic gaps after the confirmed references: retrieval ranking and public-data routing with a context-derived second input stream. A new candidate is used only where it improves the confirmed metric under the same split and leakage boundary; otherwise the existing confirmed result remains the thesis-facing result.
