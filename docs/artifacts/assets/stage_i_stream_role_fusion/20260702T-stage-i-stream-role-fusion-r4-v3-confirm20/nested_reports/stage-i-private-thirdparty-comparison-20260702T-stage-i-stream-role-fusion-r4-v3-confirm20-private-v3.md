# Stage I Dingxin Real-Data Third-party Comparison - 20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3

## Executive Summary

On the Dingxin / Stage H real dual-stream dataset, Chronaris is compared with MulT and ContiFormer under the same leakage-safe split manifest. The comparison uses real physiology and real vehicle time-series streams, with label-source fields and identity/time-position features excluded from model inputs. Across 分类任务、回归任务和检索任务 weak-label constructed tasks, the report provides model-level leaderboard, fold-level stability and Chronaris-vs-third-party deltas.

## Dataset and protocol

- evidence_role: dingxin_real_dual_stream
- sample_facts: `{'sample_count': 111, 'view_count': 3, 'sortie_count': 2}`
- split_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/split_manifest.json`

## Leakage-safe audit

- audit_json: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/label_feature_overlap_audit.json`
- audit_csv: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/label_feature_overlap_audit.csv`

## Main leaderboard

| task_name | split_strategy | model_name | metric | value_mean | value_std | seed_count |
| --- | --- | --- | --- | --- | --- | --- |
| 分类任务：机动强度分类 | leave_one_view_out | chronaris_v3_stream_role_fusion | macro_f1 | 0.1882 | 0.0168 | 3 |
| 分类任务：机动强度分类 | leave_one_view_out | chronaris_v3_stream_role_fusion | balanced_accuracy | 0.3447 | 0.0081 | 3 |
| 分类任务：机动强度分类 | leave_one_sortie_out | chronaris_v3_stream_role_fusion | macro_f1 | 0.2439 | 0.0155 | 3 |
| 分类任务：机动强度分类 | leave_one_sortie_out | chronaris_v3_stream_role_fusion | balanced_accuracy | 0.3483 | 0.0080 | 3 |
| 分类任务：机动强度分类 | leave_one_view_out | v3_no_role_gate | macro_f1 | 0.1886 | 0.0163 | 3 |
| 分类任务：机动强度分类 | leave_one_view_out | v3_no_role_gate | balanced_accuracy | 0.3447 | 0.0081 | 3 |
| 分类任务：机动强度分类 | leave_one_sortie_out | v3_no_role_gate | macro_f1 | 0.2439 | 0.0155 | 3 |
| 分类任务：机动强度分类 | leave_one_sortie_out | v3_no_role_gate | balanced_accuracy | 0.3483 | 0.0080 | 3 |
| 分类任务：机动强度分类 | leave_one_view_out | v3_fixed_causal_lag | macro_f1 | 0.1886 | 0.0163 | 3 |
| 分类任务：机动强度分类 | leave_one_view_out | v3_fixed_causal_lag | balanced_accuracy | 0.3447 | 0.0081 | 3 |
| 分类任务：机动强度分类 | leave_one_sortie_out | v3_fixed_causal_lag | macro_f1 | 0.2439 | 0.0155 | 3 |
| 分类任务：机动强度分类 | leave_one_sortie_out | v3_fixed_causal_lag | balanced_accuracy | 0.3483 | 0.0080 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_view_out | chronaris_v3_stream_role_fusion | rmse | 342.6909 | 0.3254 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_view_out | chronaris_v3_stream_role_fusion | mae | 271.3299 | 0.3583 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_view_out | chronaris_v3_stream_role_fusion | nrmse | 2.2268 | 0.0035 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_sortie_out | chronaris_v3_stream_role_fusion | rmse | 412.6902 | 0.2021 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_sortie_out | chronaris_v3_stream_role_fusion | mae | 313.6325 | 0.2857 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_sortie_out | chronaris_v3_stream_role_fusion | nrmse | 2.0416 | 0.0023 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_view_out | v3_no_role_gate | rmse | 342.6909 | 0.3254 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_view_out | v3_no_role_gate | mae | 271.3299 | 0.3583 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_view_out | v3_no_role_gate | nrmse | 2.2268 | 0.0035 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_sortie_out | v3_no_role_gate | rmse | 412.6902 | 0.2021 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_sortie_out | v3_no_role_gate | mae | 313.6325 | 0.2857 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_sortie_out | v3_no_role_gate | nrmse | 2.0416 | 0.0023 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_view_out | v3_fixed_causal_lag | rmse | 342.6909 | 0.3254 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_view_out | v3_fixed_causal_lag | mae | 271.3299 | 0.3583 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_view_out | v3_fixed_causal_lag | nrmse | 2.2268 | 0.0035 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_sortie_out | v3_fixed_causal_lag | rmse | 412.6902 | 0.2021 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_sortie_out | v3_fixed_causal_lag | mae | 313.6325 | 0.2857 | 3 |
| 回归任务：下一窗口生理响应 | leave_one_sortie_out | v3_fixed_causal_lag | nrmse | 2.0416 | 0.0023 | 3 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | chronaris_v3_stream_role_fusion | top1 | 0.0315 | 0.0064 | 3 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | chronaris_v3_stream_role_fusion | top3 | 0.0901 | 0.0064 | 3 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | chronaris_v3_stream_role_fusion | top5 | 0.1441 | 0.0064 | 3 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | chronaris_v3_stream_role_fusion | mrr | 0.1201 | 0.0054 | 3 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | v3_no_role_gate | top1 | 0.0315 | 0.0064 | 3 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | v3_no_role_gate | top3 | 0.0901 | 0.0064 | 3 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | v3_no_role_gate | top5 | 0.1441 | 0.0064 | 3 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | v3_no_role_gate | mrr | 0.1201 | 0.0054 | 3 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | v3_fixed_causal_lag | top1 | 0.0315 | 0.0064 | 3 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | v3_fixed_causal_lag | top3 | 0.0901 | 0.0064 | 3 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | v3_fixed_causal_lag | top5 | 0.1441 | 0.0064 | 3 |
| 检索任务：配对飞行员窗口检索 | leave_one_view_out | v3_fixed_causal_lag | mrr | 0.1201 | 0.0054 | 3 |

## Improvement over third-party baselines

_No rows._

## Figure index

- `fig_private_third_party_task_leaderboard`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_third_party_task_leaderboard.png`
- `fig_private_third_party_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_third_party_delta_heatmap.png`
- `fig_private_third_party_fold_variance`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_third_party_fold_variance.png`
- `fig_private_third_party_training_curves`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_third_party_training_curves.png`
- `fig_private_third_party_retrieval_topk`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_third_party_retrieval_topk.png`
- `fig_private_third_party_gpu_throughput`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_third_party_gpu_throughput.png`
- `fig_private_thirdparty_t1_macro_f1`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_thirdparty_t1_macro_f1.png`
- `fig_private_thirdparty_t2_rmse`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_thirdparty_t2_rmse.png`
- `fig_private_thirdparty_t3_retrieval`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_thirdparty_t3_retrieval.png`
- `fig_private_thirdparty_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_thirdparty_delta_heatmap.png`
- `fig_private_thirdparty_fold_stability`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_thirdparty_fold_stability.png`
- `fig_private_thirdparty_confusion_t1`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_thirdparty_confusion_t1.png`
- `fig_private_thirdparty_t2_error_distribution`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_thirdparty_t2_error_distribution.png`
- `fig_private_thirdparty_t3_retrieval_curve`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/fig_private_thirdparty_t3_retrieval_curve.png`

## Reproducibility

- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3`
- config: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/private_thirdparty_config.json`
- evidence_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/evidence_manifest.json`
- run_log: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/run.log`
- progress: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/progress.json`

## Midterm-ready wording

自有鼎新 / Stage H 分支在同一 leakage-safe 任务协议下比较 Chronaris、MulT 与 ContiFormer，量化真实生理流和真实航电流连续对齐场景中的模型适配性。分类任务、回归任务和检索任务 均保持 weak-label constructed task 标注边界，结果以 fold-level stability、mean/std 和 Chronaris-vs-baseline delta 展示。
