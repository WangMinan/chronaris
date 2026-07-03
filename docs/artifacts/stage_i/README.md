# Stage I Artifacts Index

更新时间：2026-07-03

本目录保存当前 Stage I 报告集合；当前 AI coding 入口请先看 [../ARTIFACTS.md](../ARTIFACTS.md) 与 [../stage/stage-i/README.md](../stage/stage-i/README.md)。

## 当前主入口

- 中期证据整编：`stage-i-midterm-20260607T-stage-i-midterm-r3.md`
- 主动 evidence runner：`stage-i-evidence-closure-20260607T-stage-i-evidence-closure-r2.md`
- bounded weak-label sweep：`stage-i-thesis-weak-label-multitask-sweep-20260607T-stage-i-evidence-closure-r2-multitask.md`
- live weak-label stable resume：`stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r3-resume.md`
- thesis materials r6 report figure polish：`stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md`
- leakage-safe private proxy ablation r2：`stage-i-private-leakage-safe-ablation-20260619T-stage-i-leakage-safe-ablation-r2.md`
- runtime service smoke r2 contract：`stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r2-contract.md`
- DeepSeek LLM preprocessing r3 sliced：`stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md`
- LLM preprocessing comparison r1：`stage-i-llm-comparison-20260614T-stage-i-p21-llm-comparison-r1.md`
- Public model comparison r1：`stage-i-public-model-comparison-20260701T-stage-i-public-model-comparison-r1.md`
- Public fusion refresh r1：`stage-i-public-fusion-refresh-20260701T-stage-i-public-fusion-refresh-r1.md`
- Public fusion GPU optimization r1：`stage-i-public-fusion-gpu-optimization-20260701T-stage-i-public-fusion-refresh-r1-gpuopt-r1.md`
- Private third-party comparison GPUOPT r1：`stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md`
- Public fusion ablation GPUOPT r1：`stage-i-public-fusion-ablation-20260702T-stage-i-public-fusion-ablation-gpuopt-r1.md`
- Cross-evidence matrix GPUOPT r1：`stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md`
- Optimized final polish P37 r1：`stage-i-optimized-final-polish-20260702T-stage-i-optimized-final-polish-r1.md`
- Thesis protocol freeze P38 r1：`stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md`
- 中期事实清单：`../../midterm/midterm-fact-sheet-2026-06-13.md`
- 中期边界说明：`../../midterm/boundaries-and-risks-2026-06-13.md`
- P21 中期结果摘要：`../../midterm/llm-preprocessing-comparison-summary-2026-06-14.md`
- docs LFS 清理记录：`../cleanup/20260619-lfs-docs-prune.md`
- docs 深度清理记录：`../cleanup/20260621-deep-cleanup.md`
- src/docs artifact prune 记录：`../cleanup/20260701-src-docs-artifact-prune.md`
- P42 thesis prep cleanup 记录：`../cleanup/20260703-thesis-prep-cleanup.md`
- 历史 Stage I closure：`stage-i-closure-2026-04-30.md`
- Phase C thesis weak-label evidence：`thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md`
- 当前公开主线：`stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md`
- UAB robust-prior adapter：`stage-i-public-opt-20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1.md`
- NASA enhanced round 1：`stage-i-public-opt-20260506T161500Z-stage-i-public-opt-nasa-round1.md`
- UAB torch auto-cuda confirm：`stage-i-public-opt-20260506T165558Z-stage-i-public-opt-uab-torch-gpu.md`
- Public-P27 模型对比资产：
  - assets root：`../assets/stage_i_public_model_comparison/20260701T-stage-i-public-model-comparison-r1/`
  - CSV：`model_comparison_long.csv`、`model_comparison_wide.csv`、`improvement_summary.csv`
  - figures：`fig_public_model_leaderboard_nasa_macro_f1.png`、`fig_public_model_delta_heatmap.png`、`fig_uab_subjective_rmse_comparison.png`、`fig_public_model_win_summary.png`
  - manifest：`evidence_manifest.json`
- Public-P28 fusion refresh 资产：
  - assets root：`../assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/`
  - status：`completed`
  - CSV/JSON：`screen_leaderboard.csv`、`confirm_leaderboard.csv`、`fold_metrics.csv`、`training_curves.csv`、`best_by_dataset_task.json`、`fusion_refresh_summary.json`
  - figures：`fig_public_fusion_refresh_confirm_vs_baselines.png`、`fig_public_fusion_refresh_delta_heatmap.png`、`fig_public_fusion_config_sensitivity.png`、`fig_public_fusion_training_curves_best.png`、`fig_public_fusion_win_summary.png`
  - manifest/log/progress：`evidence_manifest.json`、`run.log`、`progress.json`
- Public-P28 GPUOPT 效率资产：
  - assets root：`../assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/`
  - base run：`20260701T-stage-i-public-fusion-refresh-r1`
  - report：`stage-i-public-fusion-gpu-optimization-20260701T-stage-i-public-fusion-refresh-r1-gpuopt-r1.md`
  - summary/log/progress：`optimization_summary.json`、`gpu_perf_summary.json`、`optimization_run.log`、`optimization_progress.json`
  - CSV：`gpu_perf_batches.csv`、`gpu_perf_fold_summary.csv`、`fold_metrics.gpuopt.csv`、`training_curves.gpuopt.csv`
  - figures：`plots/fig_gpu_throughput_before_after.png`、`plots/fig_gpu_batch_timing_breakdown.png`、`plots/fig_gpu_memory_and_batch_size.png`、`plots/fig_gpu_cache_effect.png`、`plots/fig_gpu_training_progress_heartbeat.png`
- Private-P30 third-party comparison 资产：
  - assets root：`../assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/`
  - status：`completed`
  - report：`stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md`
  - CSV/JSON：`private_thirdparty_summary.json`、`model_comparison_wide.csv`、`private_thirdparty_config.json`、`gpu_perf_summary.json`
  - figures：`fig_private_thirdparty_t1_macro_f1.png`、`fig_private_thirdparty_t2_rmse.png`、`fig_private_thirdparty_t3_retrieval.png`、`fig_private_thirdparty_delta_heatmap.png`
  - manifest/log/progress：`evidence_manifest.json`、`run.log`、`progress.json`
- Public-P31 fusion ablation 资产：
  - assets root：`../assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/`
  - status：`completed`
  - report：`stage-i-public-fusion-ablation-20260702T-stage-i-public-fusion-ablation-gpuopt-r1.md`
  - CSV/JSON：`public_fusion_ablation_summary.json`、`ablation_summary.csv`、`component_contribution.csv`、`gpu_perf_summary.json`
  - figures：`fig_public_ablation_nasa_macro_f1.png`、`fig_public_ablation_nasa_balanced_accuracy.png`、`fig_public_ablation_uab_rmse.png`、`fig_public_ablation_delta_heatmap.png`、`fig_public_ablation_gpu_throughput.png`
  - manifest/log/progress：`evidence_manifest.json`、`run.log`、`progress.json`
- P32 cross-evidence matrix 资产：
  - assets root：`../assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/`
  - status：`completed`
  - report：`stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md`
  - CSV/JSON/MD：`cross_evidence_matrix.csv`、`cross_evidence_matrix.json`、`cross_evidence_summary.md`
  - figures：`fig_cross_evidence_matrix.png`、`fig_cross_evidence_metric_overview.png`、`fig_private_public_evidence_roles.png`、`fig_private_public_result_summary.png`、`fig_method_claim_support_map.png`
  - manifest：`evidence_manifest.json`
- P34 task-aware heads confirm 资产：
  - assets root：`../assets/stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20/`
  - status：`completed`
  - report：`stage-i-task-aware-heads-20260702T-stage-i-task-heads-optimization-r3-confirm20.md`
  - CSV/JSON：`task_head_metrics_long.csv`、`task_head_metrics_wide.csv`、`improvement_vs_p30.csv`、`gate_contribution_summary.csv`、`t2_residual_decomposition.csv`、`contrastive_diagnostics.csv`、`gpu_perf_summary.json`
  - figures：`fig_p34_t1_macro_f1_leaderboard.png`、`fig_p34_t2_rmse_leaderboard.png`、`fig_p34_t3_retrieval_leaderboard.png`、`fig_p34_delta_vs_p30_heatmap.png`
  - manifest/log/progress：`evidence_manifest.json`、`run.log`、`progress.json`
- P35 stream-role-aware fusion completed 资产：
  - assets root：`../assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/`
  - status：`completed`
  - report：`stage-i-stream-role-aware-fusion-20260702T-stage-i-stream-role-fusion-r4-v3-confirm20.md`
  - CSV/JSON：`route_manifest.json`、`gate_statistics.csv`、`route_decision_summary.csv`、`private_metrics.csv`、`public_metrics.csv`、`comparison_vs_p31.csv`、`comparison_vs_p34.csv`、`fold_metrics.csv`、`training_curves.csv`、`gpu_perf_summary.json`
  - figures：`fig_p35_route_gate_statistics.png`、`fig_p35_private_task_delta.png`、`fig_p35_public_ablation_comparison.png`、`fig_p35_context_vs_vehicle_gate.png`、`fig_p35_nasa_macro_f1_route_comparison.png`、`fig_p35_uab_rmse_route_comparison.png`、`fig_p35_stream_role_decision_map.png`
  - manifest/log/progress：`evidence_manifest.json`、`run.log`、`progress.json`
  - 2026-07-03 P42 cleanup 后，nested public/private confirm 的 per-candidate byproduct、重复 task manifest、GPU batch 明细和 child run 日志不再作为 git 资产；aggregate metrics、figures、summary 和顶层 logs 保留，外置备份见 `../cleanup/20260703-thesis-prep-cleanup.md`。
- P36 optimized Chronaris re-evaluation completed 资产：
  - assets root：`../assets/stage_i_optimized_reevaluation/20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20/`
  - status：`completed`
  - report：`stage-i-optimized-chronaris-reevaluation-20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20.md`
  - CSV/JSON：`optimized_private_comparison.csv`、`optimized_public_comparison.csv`、`optimized_delta_vs_p30.csv`、`optimized_delta_vs_p31.csv`、`optimized_delta_vs_p27_p28.csv`、`optimized_cross_evidence_matrix.csv`、`model_selection_summary.json`
  - figures：`fig_p36_private_before_after.png`、`fig_p36_private_vs_thirdparty_delta.png`、`fig_p36_public_before_after.png`、`fig_p36_taskwise_win_summary.png`、`fig_p36_cross_evidence_matrix_v2.png`、`fig_p36_method_claim_support_map.png`
  - manifest/log/progress：`evidence_manifest.json`、`run.log`、`progress.json`、`resume_command.txt`
- Optimized model summary completed 资产：
  - assets root：`../assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/`
  - status：`completed`
  - report：`stage-i-optimized-model-summary-20260702T-stage-i-optimized-model-summary-r4-v3-confirm20.md`
  - CSV/JSON：`optimized_model_summary.json`、`optimized_model_summary.csv`、`key_metric_summary.csv`、`stream_role_gate_summary.csv`、`gpu_runtime_summary.csv`、`claim_boundary_summary.csv`
  - figures：`fig_model_summary_status.png`、`fig_model_summary_metric_delta.png`、`fig_model_summary_gate_profile.png`、`fig_model_summary_gpu_runtime.png`
  - manifest/log/progress：`evidence_manifest.json`、`run.log`、`progress.json`、`resume_commands.txt`
- P37 optimized final polish completed 资产：
  - assets root：`../assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/`
  - status：`completed`
  - report：`stage-i-optimized-final-polish-20260702T-stage-i-optimized-final-polish-r1.md`
  - CSV/JSON：`optimized_final_polish_summary.json`、`t3_final_polish_metrics.csv`、`t1_calibration_metrics.csv`、`public_route_calibration_metrics.csv`、`p37_delta_vs_p34.csv`、`p37_delta_vs_p35.csv`、`p37_delta_vs_p30_p31.csv`、`accepted_candidate_summary.json`、`rejected_candidate_summary.json`、`gpu_perf_summary.json`
  - figures：`fig_p37_t3_retrieval_leaderboard.png`、`fig_p37_t3_delta_vs_p34.png`、`fig_p37_t1_macro_f1_leaderboard.png`、`fig_p37_public_route_delta_heatmap.png`、`fig_p37_gpu_runtime.png`
  - manifest/log/progress/resume：`evidence_manifest.json`、`run.log`、`progress.json`、`resume_command.txt`
  - boundary：P30/P31/P34/P35/P36 是固定 reference；T1 与 public route accepted，T3 rejected 并沿用 P34 retrieval；public 仍是 context proxy。
  - 2026-07-03 P42 cleanup 后，nested screen/confirm 的可再生成训练曲线、child run 日志、GPU batch、逐候选 deep summary 和顶层稠密预测副本已外置备份并从 git 当前树删除；P37 顶层 summary/metrics/figures/manifest 保留。
- P38 thesis protocol freeze completed 资产：
  - assets root：`../assets/stage_i_thesis_protocol/20260703T-stage-i-thesis-protocol-r1/`
  - status：`completed`
  - report：`stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md`
  - CSV/JSON：`experiment_registry.csv`、`result_matrix_long.csv`、`result_matrix_summary.csv`、`claim_boundary_table.csv`、`thesis_protocol_summary.json`
  - manifest/log/progress/resume：`evidence_manifest.json`、`run.log`、`progress.json`、`resume_command.txt`
  - boundary：P38 只读聚合 P30/P31/P32/P34/P35/P36/P37，不重跑训练、不删除 artifact、不改写历史；public 仍是 context proxy，private 仍是 proxy/weak-label evidence。
- private component ablation（历史协议）：`stage-i-private-component-ablation-20260607T-stage-i-evidence-closure-r2-private-proxy.md`
- public adapter calibration：`stage-i-public-adapter-calibration-20260607T-stage-i-evidence-closure-r2-public-adapter.md`
- public transfer boundary：`stage-i-public-transfer-boundary-20260607T-stage-i-evidence-closure-r2-transfer-boundary.md`
- rigid-body rotation audit r3：`stage-i-rigid-body-rotation-audit-20260619T-stage-i-rotation-audit-r3-figure-refresh.md`
- Phase D/E/F r2：
  - `stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md`
  - `stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md`
  - `stage-i-semantic-event-support-20260607T-stage-i-semantic-support-r2.md`
  - `stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md`
- 历史 support：
  - `stage-i-alignment-support-20260506T120000Z-stage-i-support.md`
  - `stage-i-causal-support-20260506T120000Z-stage-i-support.md`
  - `stage-i-ablation-support-20260506T120000Z-stage-i-support.md`
- Thesis-facing demo：
  - `stage-i-runtime-demo-20260506T165435Z-stage-i-runtime-demo.md`
  - `stage-i-anchor-20260506T165435Z-stage-i-anchor.md`

历史 P16/P17 图表与 runtime：

- `stage-i-rigid-body-rotation-audit-20260607T-stage-i-rotation-audit-r2.md`
- 旧 P16/P18 thesis materials r2-p18/r3/r4 图包已被后续图包接管；r4 runtime case refresh 已从 docs 产物目录清理，`../assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv` 仅作为 P20/P21 LLM preprocessing 历史输入表。
- P25 thesis materials r5 图包与报告已由 P26 r6 接管，并已按 `../cleanup/20260621-deep-cleanup.md` 从 docs 产物目录清理；当前图表入口使用 r6。
- 早期 public torch/mainline/fusion screen archive-only 迭代已按 `../cleanup/20260701-src-docs-artifact-prune.md` 从 docs 当前产物目录清理；当前 public adapter 与模型对比入口使用 P27/P28、UAB robust-prior、NASA round1、UAB torch auto-cuda confirm 和 fusion screen round2。
- runtime service smoke r1、runtime replay r1、semantic support r1、rigid-body r1、P11 live r2 和 P20 preprocessing r1 仅保留在 git 历史；当前入口使用 r2/r3/r6/r2-contract 产物。

历史 P11 live 首轮：

- `20260613T-stage-i-p11-live-influx-r1` 的子运行和 blocker log 被 r3 resume summary 引用，因此保留为可复现依赖；`p11-live-influx-r2` 汇总已由 r3 resume 接管并清理。

## 引用规则

- `public opt closed` 仍是公开支撑证据，但 UAB `target_prior_median` 只能写成 `public adapter / calibration evidence`。
- UAB/NASA 第二模态统一写成 `context proxy / public adapter evidence`。
- 私有分支需要区分 `T1/T2/T3 = private proxy benchmark / proxy evidence` 与 `risk_proxy / workload_proxy / event_replay_tag = thesis weak-label evidence`。
- `stage_i_private_leakage_safe_ablation` r2 是 `protocol=leakage_safe_v1` 的新增协议；历史 private component ablation 只能作为历史代理诊断，不与 r2 防泄漏结果混成同一实验结论。
- P20/P21 DeepSeek LLM preprocessing 只能写成 preprocessing context、rule review、whitelisted semantic hints、runtime explanation、bounded comparison 和 pending human review packet；不能写成人工真值、OpenAI 默认接入、核心因果证据或人工验证完成。
- P30 private third-party comparison 只能写成 T1/T2/T3 private proxy task 的第三方对比；当前结果是混合结果，不能写成 Chronaris 全面胜出。
- P31 public fusion ablation 只能写成 public adapter / context proxy 上的组件敏感性诊断；不替代 P28 confirmed metrics，也不证明 public 第二模态等价于私有航电流。
- P32 cross-evidence matrix 用于 private/public/proxy/component 四层证据分工；不要把四层指标混成单一排行榜。
- P34/P35/P36 与 optimized model summary 当前为 optimized v2/v3 CUDA confirm / completed aggregation / summary，不替代 P30/P31/P32 completed 结果；引用时必须写明 P34 是 20-epoch confirm、P35 是 requested private/public v3 confirm、public 第二流仍是 context proxy，也不得写成 optimized Chronaris 全面超过全部 baseline。
- P37 final polish 当前为固定 P30/P31/P34/P35/P36 reference 上的局部收束：T1 calibration 与 public route calibration accepted，T3 retrieval rejected 并沿用 P34 confirmed retrieval。引用时不要把 public context proxy 写成私有真实航电流，也不要把 P37 写成全面胜出。
- P38 thesis protocol freeze 是论文协议矩阵入口；论文图表若引用 P30/P31/P32/P34/P35/P36/P37，应优先从 P38 的 `experiment_registry.csv` 或 `result_matrix_long.csv` 反查原始 artifact path 和 claim boundary。
- 2026-07-02 后，本轮 P34/P35/P36/summary 柱状图已基于现有 CSV/JSON 重绘并增加短数值标签；future run 默认使用 `checkpoint_policy=last`，checkpoint binary 和 dense prediction CSV 不进入 git。清理/profiling 入口：`../cleanup/20260702-p34-p36-gpu-and-docs-cleanup.md`。
- 历史 archive 只用于追溯，不作为当前状态入口。
- 历史 raw replay、prepared bundle 和大型 manifest 已按 `../cleanup/20260619-lfs-docs-prune.md` 从 docs/LFS 中清理；引用旧实验时优先看报告、summary、plots 和当前索引。
- 2026-07-01 后，不要继续链接已清理的早期 public torch/mainline/fusion screen round1 archive-only 路径；如需复跑，使用当前 `scripts/stage_i/public/` 入口重新生成。
