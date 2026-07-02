# Chronaris 当前状态

更新时间：2026-07-02

## 一句话状态

项目已经具备中期答辩可用的历史实验资产与最新主动证据闭环：真实链路 `Stage E/F/G(min)/H`、Stage I 历史公开 benchmark、`chronaris_opt` 私有代理证据、public adapter 支撑线、Phase C 真实 Stage H multitask 联合训练证据和中期证据包都已形成并进入 git 历史；同时，`P10 evidence runner`、`P11 live_influx thesis weak-label sweep`、`P12 chronaris_opt` 组件诊断、`P13 public adapter calibration`、`P14 public transfer boundary`、`P15 rigid_body rotation audit`、`P16 thesis materials`、`P17 runtime service smoke`、`P18 partial/resume + runtime schema contract` 已新增落盘。当前新增 `P24 leakage-safe private component ablation`、`P25 thesis materials r5 leakage-safe refresh`、`P26 thesis materials r6 report figure polish`、`Public-P27/Public-P28` 与本轮 `P30/P31/P32/P34/P35/P36`：P24 把历史 private proxy 满分结果升级为带标签-特征同源审计的 `leakage_safe_v1` 组件诊断；P26 基于已有 summary/CSV/manifest 重绘 12 张 300dpi 中期报告图件与 12 张配套 CSV，并新增 `runtime_service_flow.png`；Public-P27/P28 完成 public model comparison、chronaris_public_fusion CUDA screen + full LOSO confirm、P28 后 P27 重新整编和中期报告图表；P30/P31/P32 进一步补齐 private third-party comparison、public fusion ablation 和跨证据矩阵；P34/P35/P36 进一步把诊断转化为 optimized Chronaris CUDA confirm、stream-role v3 confirm 和统一聚合 summary。2026-06-21 深度清理已删除被 r6 接管的 r5 图包/报告、空 artifact 目录和本地 Python 编译缓存；2026-07-01 src/docs artifact prune 已删除本地编译缓存和 archive-only 早期 public torch/mainline/fusion screen 旧产物，当前公开入口统一使用 P27/P28、P31 与仍被代码/报告引用的 public adapter 基线。`P20 DeepSeek 在线时序数据预处理` 已完成 provider contract、agent-style prompt/harness v2、切片整合、mock/repair 测试和小样本真实 DeepSeek run，并作为 Stage H 到 Stage I 之间的可选 preprocessing context 接入；`P21 LLM preprocessing 对比实验` 已完成 A0-A4 本地对比、工程资产、Stage I 报告和中期 summary。

## 当前阶段

- 阶段 A/B/C：已完成。
- 阶段 E0：已完成 preview 路径。
- 阶段 E/F/G(min)：已完成真实链路收口，作为历史基线保留。
- 阶段 H：已完成标准化特征导出收口，`validation` profile 可稳定导出 3 个双流 view。
- 阶段 I 历史公开 benchmark：`Phase 0/1/2/3` 已完成并收口。
- 阶段 I thesis mainline：
  - `Phase A/B` 已经进入 git 历史，主线边界校准、统一骨干训练入口、checkpoint inference export contract 已具备。
  - `Phase C` 已进入 git 历史，内容包括统一任务头、任务监督损失、因果正则接入、`risk_proxy / workload_proxy / event_replay_tag` weak-label thesis task builder、`stage_i_multitask_train` 联合训练入口，以及 private benchmark 中 `proxy_evidence / thesis_task_evidence` 分层。
  - `Phase D` 已完成二轮真实 smoke / ablation：
    - 首轮 r1 产物已清理，仅保留在 git 历史中。
    - 二轮：`docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/`
    - 最新汇总报告：`docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md`
    - 当前 `rigid_body` 已经在真实链路上启用 `translation + vertical`：
      - `vehicle_rigid_body_translation=1.133332371711731`
      - `vehicle_rigid_body_vertical=3.9466116428375244`
    - `rotation` 仍为 `0`，当前主要原因是缺少成对角速度字段，而不是 MySQL 元数据问题。
  - `Phase E` 已完成真实语义事件融合 support：
    - 单 view preview：`docs/artifacts/stage_i/stage-i-semantic-event-20260607T-stage-i-semantic-event-r1.md`
    - 多 view support summary：`docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`
    - 多 view support 报告：`docs/artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md`
    - 当前 semantic support 已覆盖 `3` 个双流 view、`111` 个样本，并给出 view-level ranking。
  - `Phase F` 已完成真实 runtime replay：
    - 首轮 replay r1 产物已清理，仅保留在 git 历史中。
    - 服务化补强 replay：`docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/`
    - 最新报告：`docs/artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md`
    - 当前 runtime replay 已支持 `batch / incremental / both`，并输出 `latency / throughput / feature_schema_status / feature_schema_source`。
  - 中期前主动证据：
    - `P10 evidence runner` 已完成：
      - 稳定 manifest：`docs/artifacts/assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/evidence_manifest.json`
      - 稳定报告：`docs/artifacts/stage_i/stage-i-evidence-closure-20260607T-stage-i-evidence-closure-r2.md`
      - 当前 `r2` 已纳入 `multitask / rigid_body / semantic / runtime / private_proxy / public_adapter / rotation` 七项 evidence task。
    - `P11 thesis weak-label multitask sweep` 已完成 bounded 版本：
      - 产物：`docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/`
      - 报告：`docs/artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260607T-stage-i-evidence-closure-r2-multitask.md`
      - 当前 `skip-heavy` runner 路线使用 `stage_h_window_stats_proxy` 样本源，仍明确写成 `thesis weak-label evidence`，不包装成人工真值任务。
    - `P11+ live_influx thesis weak-label sweep` 已完成稳定汇总：
      - 当前稳定汇总：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json`
      - 当前报告：`docs/artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r3-resume.md`
      - 当前 `sample_source=live_influx`、`sample_count=111`、`task_entry_count=333`、`combination_count=2`；r3 resume 复用了 `20260613T-stage-i-p11-live-influx-r1` 中已完成的两个 live child run，并保留了更大 `4` 组合尝试的 blocker 日志。被 r3 接管的 r2 汇总已清理。
    - `P12 chronaris_opt component ablation` 已完成：
      - 产物：`docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/`
      - 报告：`docs/artifacts/stage_i/stage-i-private-component-ablation-20260607T-stage-i-evidence-closure-r2-private-proxy.md`
      - 历史协议只作为 private proxy 组件诊断追溯入口。
    - `P12+ leakage-safe private component ablation` 已完成：
      - 产物：`docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/`
      - 报告：`docs/artifacts/stage_i/stage-i-private-leakage-safe-ablation-20260619T-stage-i-leakage-safe-ablation-r2.md`
      - 当前 `protocol=leakage_safe_v1`、`audit_status=pass`、`seed_count=5`，输出 label-feature overlap audit、seed metrics、cross-view/cross-sortie metrics、model/task ablation CSV/JSON/PNG 和 T2/T3 分布图。
    - `P13 public adapter calibration` 已完成：
      - 产物：`docs/artifacts/assets/stage_i_public_adapter_calibration/20260607T-stage-i-evidence-closure-r2-public-adapter/public_adapter_calibration_summary.json`
      - 报告：`docs/artifacts/stage_i/stage-i-public-adapter-calibration-20260607T-stage-i-evidence-closure-r2-public-adapter.md`
    - `P14 public transfer boundary` 已完成：
      - 产物：`docs/artifacts/assets/stage_i_public_transfer_boundary/20260607T-stage-i-evidence-closure-r2-transfer-boundary/public_transfer_boundary_summary.json`
      - 报告：`docs/artifacts/stage_i/stage-i-public-transfer-boundary-20260607T-stage-i-evidence-closure-r2-transfer-boundary.md`
    - `P15 rigid_body rotation audit` 已完成：
      - 产物：`docs/artifacts/assets/stage_i_rotation_audit/20260607T-stage-i-rotation-audit-r2/rigid_body_rotation_audit_summary.json`
      - 报告：`docs/artifacts/stage_i/stage-i-rigid-body-rotation-audit-20260607T-stage-i-rotation-audit-r2.md`
      - 当前已确认 `BUS6000019110020.code1031 = 真航向` 可映射到 `yaw`，但 `yaw_rate` 仍缺失，因此 `rotation_status=disabled`。
    - `P16 thesis materials` 已由 2026-06-21 的 `r6-report-figure-polish` 接管为当前入口：
      - 旧 r1 图包已清理，不再作为 docs 下可打开产物。
      - 当前写作入口以 `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_manifest.json` 为准。
    - `P17 runtime service smoke` 首轮已完成：
      - 首轮 r1 产物已由 P18 `r2-contract` 接管并清理，仅保留在 git 历史中。
      - 已支持 checkpoint 冷启动、单 view replay JSONL -> predictions JSONL / summary JSON，并固化 `missing_checkpoint / missing_fields / empty_window / schema_mismatch` 四类错误样例；当前 schema contract 入口以 P18 `r2-contract` 为准。
    - `P18 P11/P17 风险收口优化` 已完成：
      - P11 stable resume：
        - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json`
        - `docs/artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r3-resume.md`
      - P11 partial blocked：
        - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/partial_summary.json`
        - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/thesis_weak_label_multitask_ablation.partial.csv`
      - P17 runtime schema contract：
        - `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json`
        - `docs/artifacts/stage_i/stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r2-contract.md`
        - raw canonical payload 已按 `docs/artifacts/cleanup/20260619-lfs-docs-prune.md` 从 docs/LFS 清理；exact contract 证据保留在 schema contract 与 canonical runtime summary。
      - P16/P18 旧刷新图包：
        - r3/r4 图包曾由 P25/r5 接管并从 docs 产物目录清理；当前入口已由 `20260621T-stage-i-thesis-materials-r6-report-figure-polish` 接管，r5 也已从 docs 当前产物目录清理。
        - 仅保留 `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv` 作为 P20/P21 LLM preprocessing 的历史输入表。
      - P16/P21/P24/P26 中期图表质量刷新：
        - `docs/artifacts/assets/stage_i_rotation_audit/20260619T-stage-i-rotation-audit-r3-figure-refresh/rigid_body_rotation_audit_summary.json`
        - `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_manifest.json`
        - `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/table_manifest.json`
        - `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_quality_audit.csv`
        - `docs/artifacts/stage_i/stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md`
        - 当前结果：`12` 张 PNG 与 `12` 张 CSV，覆盖 evidence layer overview、runtime payload schema、runtime service flow、runtime semantic case、rigid-body/rotation、weak-label sweep、private component overview、model backbone ablation、task adapter ablation、public transfer boundary、semantic event fusion 和 LLM comparison。
    - `P20 DeepSeek 在线时序数据预处理` 已完成 agent-style harness v2、切片整合与真实小样本 run：
      - 计划入口：`docs/implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md`
      - 当前真实 run：`docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_preprocessing_summary.json`
      - 当前报告：`docs/artifacts/stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md`
      - 当前结果：`request_count=8`、`error_count=0`、`field_semantic_count=24`、`weak_label_review_count=3`、`semantic_query_hint_count=4`、`runtime_explanation_count=4`。
      - 当前 harness：`prompt_version=stage_i_llm_preprocessing.agent_guardrails.v2`、`schema_repair_attempt_count=0`、`final_invalid_task_count=0`。
      - 当前切片：`field_semantics` 按 12+12、`schema_gap_policy` 按 3+3、`runtime_explanations` 按 2+2 切片，并在本地按 stable identifier 合并。
      - 当前定位：DeepSeek v4-pro 只作为字段语义归一、weak-label 规则复核、schema gap 预处理建议和 runtime 解释层；不替代人工真值、物理约束或因果融合主线。
    - `P21 LLM preprocessing 对比实验` 已完成：
      - 工程资产：`docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/llm_comparison_summary.json`
      - 条件 manifest：`docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/condition_manifest.json`
      - 工程报告：`docs/artifacts/stage_i/stage-i-llm-comparison-20260614T-stage-i-p21-llm-comparison-r1.md`
      - 中期 summary：`docs/midterm/llm-preprocessing-comparison-summary-2026-06-14.md`
      - 当前结果：A1 `333/333` task entries 已 attach P20 context，`label_unchanged=true`；A2 semantic query coverage 从 `3` 扩到 `7`，且仅使用 whitelisted recipes；A3 `4/12` runtime cases 有 LLM explanation，解释子集四项完整性达到 `1.0`；A4 生成 `15` 条 `pending_human_review` 复核 packet。
      - 当前边界：A2 没有从 summary 伪造 view ranking/top attribution 重算；A4 没有写成人工验证完成。
    - `Public-P27/Public-P28 public model comparison + fusion refresh` 已完成：
      - P28 refresh 产物：`docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fusion_refresh_summary.json`
      - P28 报告：`docs/artifacts/stage_i/stage-i-public-fusion-refresh-20260701T-stage-i-public-fusion-refresh-r1.md`
      - P28 状态：`status=completed`，CUDA screen 后 top-1 full LOSO confirm；`run.log` 与 `progress.json` 已落盘。
      - P27 comparison 产物：`docs/artifacts/assets/stage_i_public_model_comparison/20260701T-stage-i-public-model-comparison-r1/model_comparison_wide.csv`
      - P27 报告：`docs/artifacts/stage_i/stage-i-public-model-comparison-20260701T-stage-i-public-model-comparison-r1.md`
      - P27 manifest：`p28_refresh_included=true`、`missing_metrics=[]`，图件覆盖 NASA macro-F1/BA、UAB RMSE、delta heatmap 和 W/T/L summary。
    - `P28-GPUOPT chronaris_public_fusion training efficiency profiling` 已完成：
      - 产物：`docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/optimization_summary.json`
      - 报告：`docs/artifacts/stage_i/stage-i-public-fusion-gpu-optimization-20260701T-stage-i-public-fusion-refresh-r1-gpuopt-r1.md`
      - 当前结果：代表性 fold profiling 中 tensor cache `auto` + auto batch `2048` + AMP `bf16` 把吞吐从 `559.52` 提升到 `5090.93` samples/sec，speedup `9.10x`；显存峰值从 `0.276 GB` 提升到 `3.913 GB`，nvidia-smi GPU utilization 快照从 `11.67%` 到 `15.83%`。
      - 边界：本轮未重跑完整 P28 full LOSO，不替代 P28 confirmed metrics；`gpu_optimization/resume_command.txt` 只用于恢复 GPUOPT profiling。
    - `P30 private third-party comparison` 已完成：
      - 产物：`docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/private_thirdparty_summary.json`
      - 报告：`docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md`
      - 当前结果：`status=completed`，private real dual-stream Stage H 样本为 `111` windows / `3` views / `2` sorties，完成 `159/195` protocol fold rows；T1/T2/T3 与 MulT、ContiFormer、naive time sync、classical baseline 对比为混合结果，不包装成 Chronaris 全面胜出。
      - GPUOPT：CUDA required，RTX 4090，torch `2.11.0+cu130` / CUDA `13.0`，tensor cache `auto(cuda)`、AMP `bf16`、auto batch best `2048`，OOM/cache fallback 均为 `0`。
    - `P31 public fusion ablation` 已完成：
      - 产物：`docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/public_fusion_ablation_summary.json`
      - 报告：`docs/artifacts/stage_i/stage-i-public-fusion-ablation-20260702T-stage-i-public-fusion-ablation-gpuopt-r1.md`
      - 当前结果：`status=completed`，screen `112` fold rows，confirm `936/952` fold rows；NASA `full` combined macro-F1=`0.327192`、BA=`0.346014`，`no_lag_window` 在 NASA combined macro-F1=`0.394103`；UAB `full` mean RMSE=`3.279090`，`context_only` mean RMSE=`3.066633`。
      - GPUOPT：P31 正式 run 经 tensor cache、AMP `bf16`、吞吐型 auto batch、heartbeat/progress/resume 收口；候选级多进程并发曾触发 CUDA launch failure，最终完成版采用稳定串行 candidate 执行，保留 blocked/resume 日志在 run.log 中。
    - `P32 cross-evidence matrix` 已完成：
      - 产物：`docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/evidence_manifest.json`
      - 报告：`docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md`
      - 当前结果：`status=completed`，矩阵共 `292` 行，覆盖 `private_thirdparty_comparison=72`、`private_component_ablation=36`、`public_model_comparison=80`、`public_component_ablation=104` 四个象限。
    - `P34/P35/P36 optimized Chronaris v2/v3 confirm` 与 optimized model summary 已新增，当前入口为 P34 r3 + P35/P36 r4 confirm20：
      - P34 task-aware heads 产物：`docs/artifacts/assets/stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20/evidence_manifest.json`
      - P34 报告：`docs/artifacts/stage_i/stage-i-task-aware-heads-20260702T-stage-i-task-heads-optimization-r3-confirm20.md`
      - P35 stream-role fusion 产物：`docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/evidence_manifest.json`
      - P35 报告：`docs/artifacts/stage_i/stage-i-stream-role-aware-fusion-20260702T-stage-i-stream-role-fusion-r4-v3-confirm20.md`
      - P36 optimized re-evaluation 产物：`docs/artifacts/assets/stage_i_optimized_reevaluation/20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20/evidence_manifest.json`
      - P36 报告：`docs/artifacts/stage_i/stage-i-optimized-chronaris-reevaluation-20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20.md`
      - Optimized model summary 产物：`docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/evidence_manifest.json`
      - Optimized model summary 报告：`docs/artifacts/stage_i/stage-i-optimized-model-summary-20260702T-stage-i-optimized-model-summary-r4-v3-confirm20.md`
      - 当前结果：P34 `status=completed`，为 CUDA 20-epoch confirm（3 seeds / `leave_one_view_out` + `leave_one_sortie_out` / 20 epochs），T1 两个 split 小幅改善，T2 在两个 split 上明显改善，T3 仍为混合结果；P35 `status=completed`，完成 requested private/public v3 confirm，private 分支比较 `chronaris_v3_stream_role_fusion`、`v3_no_role_gate`、`v3_fixed_causal_lag` 在 T1/T2/T3 上的结果，public 分支比较 `v3_stream_role`、`v3_no_role_gate`、`v3_force_private_causal`、`v3_context_adapter_only` 在 NASA/UAB context-proxy 上的结果，24 个 public dataset/variant/seed 组合无缺失；P36 与 optimized model summary `status=completed`，P36 private comparison 汇总 P30=72、P34=56、P35_v3_confirm=42 行，public comparison 汇总 P31=6、P35_v3_confirm=8 行。它们不覆盖 P30/P31/P32 confirmed metrics；公开第二流仍是 context proxy，不能写成私有航电流，也不能写成 optimized Chronaris 全面胜出。
      - 2026-07-02 后处理：基于既有 CSV/JSON 重绘 P34/P35/P36/summary 柱状图并增加短数值标签；新增 future-run `checkpoint_policy`（`off` / `last` / `epoch_and_fold`），默认只保留 last checkpoint；清理记录见 `docs/artifacts/cleanup/20260702-p34-p36-gpu-and-docs-cleanup.md`。本次不改变 P34/P35/P36 metrics。

## Git 与工作区核对

- 当前分支为 `main`；本轮 P34/P35/P36 与 optimized model summary r4 confirm20 资产已完成图表重绘、large docs cleanup 和 future-run GPU/checkpoint 输出策略补强，提交前以本地 diff 为准。本轮不改变 P27/P28/P30/P31/P32 confirmed metrics；P34 是 completed CUDA 20-epoch confirm，P35/P36/optimized model summary 是 requested CUDA v3 confirm / aggregation completed，但 public 证据仍是 context proxy，private T1/T2/T3 仍是 proxy benchmark。
- 本轮 2026-07-02 清理删除 P34/P35/P36 过渡 run、P31 superseded r1，以及 current r4 nested dense prediction/checkpoint/partial CSV；压缩 P35/P31 training curves 和大日志；dense prediction CSV 已备份到远程开发机仓库外 `/home/wangminan/projects/chronaris-local-artifacts/dense-predictions-pruned-20260702/`，重复 weak-label manifest 副本已回指 canonical；`.pt` checkpoint 已备份到 `/home/wangminan/projects/chronaris-local-artifacts/checkpoints-history-20260702/`，提交后使用 `git filter-repo` 从 git 历史移除 checkpoint binary。
- `P10-P15` 主动证据工具、测试、报告、索引和可引用汇总资产已经进入远端历史；`Phase D/E/F` 代码、文档与资产作为历史基线保留。
- 最新进入历史的 `P10-P15` 主动证据提交：
  - `70b651a feat: add stage i evidence closure tools`
- 最新进入历史的 Phase D/E/F 相关提交：
  - `57ca739 feat: expand stage i rigid-body support and runtime service`
  - `890a315 docs: record stage i runtime semantic rigid-body artifacts`
  - `0f4db72 feat: add stage i runtime sample exporter`
  - `9ef4f64 feat: add rigid-body physics semantic event runtime inference`
- 关键实现提交 `a5fda40` 覆盖 Stage I thesis mainline `Phase C`：真实 Stage H multitask 联合训练、private/thesis 分层资产和中期证据包。
- 更早的关键实现提交 `2055dec` 覆盖 Stage I thesis mainline `Phase A/B`：public adapter/proxy 边界、backbone train、Stage H checkpoint inference contract 和相关测试。
- 当前 `Phase D/E/F` 主代码、runtime sample exporter、r2 产物和状态文档已经进入 git 历史；`P10-P15` evidence runner、bounded sweep、private component ablation、public adapter calibration、transfer boundary、rotation audit 已进入 git 历史。
  - 刚体物理：`physics_state_mapping.py`、`physics_residuals.py`、`physics.py`、`physics_features.py`、`run_stage_e_relative_preview.py`。
  - 语义事件融合：`semantic_event.py`、`causal_fusion.py`、`src/chronaris/pipelines/stage_i/evidence/support_builders.py`、`src/chronaris/pipelines/stage_i/evidence/support_reporting.py`。
  - runtime inference：`streaming_windows.py`、`runtime_inference.py`、`scripts/stage_i/runtime/run_inference.py`、`scripts/stage_i/runtime/export_runtime_samples.py`、`scripts/stage_i/evidence/build_semantic_event_support.py`。
  - 测试覆盖：`tests/test_alignment_model_losses.py`、`tests/test_stage_i_support.py`、`tests/test_runtime_inference.py`。
- 本轮工作区已收口为 `P10-P15` 主动证据闭环提交；主体功能与资产状态以 `70b651a` 为准，后续纯文档同步提交不改变该证据事实。
- 已进入历史的关键前置产物：
  - Phase C 真实联合训练：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_summary.json`
  - thesis weak-label 报告：`docs/artifacts/stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md`
  - private benchmark 分层资产：`docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json`
  - 最新中期证据包：`docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md`
  - 最新刚体约束 r2：`docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/rigid_body_ablation_summary.json`
  - 最新 semantic support r2：`docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`
  - 最新 runtime service r2：`docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/runtime_inference_summary.json`
  - 最新主动 evidence runner：`docs/artifacts/assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/evidence_manifest.json`
  - 最新 bounded weak-label sweep：`docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/multitask_sweep_summary.json`
  - 最新 private component ablation：`docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/chronaris_opt_component_ablation.json`
  - 最新 public adapter calibration：`docs/artifacts/assets/stage_i_public_adapter_calibration/20260607T-stage-i-evidence-closure-r2-public-adapter/public_adapter_calibration_summary.json`
  - 最新 public transfer boundary：`docs/artifacts/assets/stage_i_public_transfer_boundary/20260607T-stage-i-evidence-closure-r2-transfer-boundary/public_transfer_boundary_summary.json`
  - 最新 rotation audit：`docs/artifacts/assets/stage_i_rotation_audit/20260619T-stage-i-rotation-audit-r3-figure-refresh/rigid_body_rotation_audit_summary.json`
  - 最新 live weak-label stable resume：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json`
  - 最新 live weak-label partial blocked：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/partial_summary.json`
  - 最新 runtime schema contract：`docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json`
  - 最新 docs LFS 清理记录：`docs/artifacts/cleanup/20260619-lfs-docs-prune.md`
  - 最新 src/docs artifact prune 记录：`docs/artifacts/cleanup/20260701-src-docs-artifact-prune.md`
  - P20/P21 LLM runtime case 输入表：`docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv`
  - 最新 thesis materials r6 report figure polish：`docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_manifest.json`
  - 最新 leakage-safe private component ablation：`docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/ablation_summary.json`
  - 最新中期事实清单：`docs/midterm/midterm-fact-sheet-2026-06-13.md`
  - 最新中期边界说明：`docs/midterm/boundaries-and-risks-2026-06-13.md`
  - 最新 P20 DeepSeek LLM preprocessing package：`docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_preprocessing_summary.json`
  - 最新 P21 LLM preprocessing comparison package：`docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/llm_comparison_summary.json`
  - 最新 public fusion refresh package：`docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fusion_refresh_summary.json`
  - 最新 public fusion GPU optimization package：`docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/optimization_summary.json`
  - 最新 public model comparison package：`docs/artifacts/assets/stage_i_public_model_comparison/20260701T-stage-i-public-model-comparison-r1/evidence_manifest.json`
  - 最新 private third-party comparison package：`docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/evidence_manifest.json`
  - 最新 public fusion ablation package：`docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/evidence_manifest.json`
  - 最新 cross-evidence matrix package：`docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/evidence_manifest.json`

## 当前主线事实

- 当前鼎新私有任务验证主线仍是 `chronaris_opt`，但它属于 `private proxy benchmark / proxy evidence`。
- 当前公开支撑线为 `public opt closed`，但 `UAB robust-prior adapter / target_prior_median` 只能写成 `public adapter / calibration evidence`，不能写成双流连续对齐或因果融合模块本体的直接胜利。
- 当前公开第二模态应写成 `context proxy / public adapter evidence`，不是论文严格意义上的真实航电流。
- P30/P31/P32 已把 private real dual-stream、private leakage-safe proxy、public model comparison、public component ablation 四层证据放入同一 cross-evidence matrix；该矩阵用于中期报告的证据分层，不用于把 public context proxy 伪写成私有航电流。
- `T1/T2/T3` 是私有代理任务；`risk_proxy / workload_proxy / event_replay_tag` 是 thesis weak-label task builder，不等价于人工真值任务。
- `20251110_单01_ACT-2_涛_J20_26#01` 仍是 vehicle-only partial-data，不是双流 Stage H view。
- 中期 P20/P21 LLM preprocessing 已按 DeepSeek v4-pro 完成小样本真实 run 与 A0-A4 对比实验；不默认使用 OpenAI，且 LLM 仅作为在线时序数据预处理、规则复核、whitelisted semantic hints、runtime explanation 和人工复核 packet，不替代物理约束、因果融合或人工真值。

## 当前代码组织事实

- Stage I pipeline 源码已按职责拆分到 `src/chronaris/pipelines/stage_i/common/`、`training/`、`public/`、`private/`、`evidence/`、`llm/`、`legacy/`，不再继续新增单层 `stage_i_*.py` 主实现文件。
- Stage I CLI 入口已按用途拆分到 `scripts/stage_i/<category>/`；根目录不再保留旧 `run_stage_i_*.py` / `build_stage_i_*.py` / `prepare_stage_i_*.py` / `export_stage_i_*.py` 文件。
- 旧 Python 模块 import 路径通过 `chronaris.pipelines.stage_i` 的包级兼容映射解析到新子包；旧脚本命令不保留 wrapper，后续命令以 `scripts/README.md` 的 canonical 路径为准。

## 编码层面还需要做什么

1. P21 已完成 LLM preprocessing 融入管线的 A0-A4 对比实验；后续若要升级结论强度，优先填写 `human_review_packet.csv` 并基于 Stage H tensor 复跑带 LLM query specs 的 semantic support，而不是从现有 summary 倒推 ranking 变化。
2. 维护当前 `P11 stable resume / partial blocked / blocker log` 三段证据链，避免后续又退回到“完成两点 + 口头说明”的状态。
3. 若后续要补更大的 `live_influx` 网格，先明确预算，再从当前 `2` 组合 stable resume 版继续扩展，而不是覆盖现有 stable summary。
4. 若后续发现可用角速度字段，需要在 `rotation audit` 的基础上补 `minimal / full / rigid_body` 复跑；若没有，则继续保持 `rotation disabled` 的 diagnostics 口径。
5. 若后续要把 runtime/service 收紧到 `native exact schema`，优先补齐上游 view replay payload 的 vehicle measurement groups；当前 `native aligned / canonical exact` 已经把部署边界写清，不需要重建上游接收器。

## 实验层面还需要做什么

1. 已完成 Phase C 真实联合训练证据：
   - 输入：`docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json`
   - 输入：`docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json`
   - 输出：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/`
2. 已完成 private benchmark 分层资产刷新：
   - 输出：`docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/`
   - `private_benchmark_summary.json` 已包含 `evidence_layers.proxy_evidence / thesis_task_evidence`
3. 已完成中期证据包重建：
   - 输出：`docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/`
   - 报告：`docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md`
4. 本轮新增的 `Phase D/E/F` 产物已经落盘并已更新到 `r2`：
   - runtime service-style replay：`docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/`
     - `sample_count=40`
     - `replay_mode=both`
     - `sample_count_match=True`
     - `feature_schema_status=aligned`
     - `feature_schema_source=input_normalization_stats`
   - semantic event support：`docs/artifacts/assets/stage_i_semantic_event_support/20260607T-stage-i-semantic-support-r2/`
     - `view_count=3`
     - `top_view_id=20251005_四01_ACT-4_云_J20_22#01__pilot_10033`
   - support 聚合：`docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/`
     - `support_summary.json` 已包含 `causal_support.semantic_event.view_rows`
   - rigid_body ablation：`docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/`
     - `vehicle_field_metadata.status=loaded`
     - `enabled_residuals=['translation','vertical']`
     - `vehicle_rigid_body_translation=1.133332371711731`
     - `vehicle_rigid_body_vertical=3.9466116428375244`
     - `vehicle_rigid_body_rotation=0`
5. 本轮新增主动 evidence 已落盘：
   - `P10 evidence runner`：
     - `docs/artifacts/assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/evidence_manifest.json`
   - `P11 thesis weak-label multitask sweep`：
     - `docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/`
     - 当前 `sample_source=stage_h_window_stats_proxy`
   - `P12 chronaris_opt component ablation`：
     - `docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/`
   - `P13 public adapter calibration`：
     - `docs/artifacts/assets/stage_i_public_adapter_calibration/20260607T-stage-i-evidence-closure-r2-public-adapter/`
   - `P14 public transfer boundary`：
     - `docs/artifacts/assets/stage_i_public_transfer_boundary/20260607T-stage-i-evidence-closure-r2-transfer-boundary/`
  - `P15 rigid_body rotation audit`：
    - `docs/artifacts/assets/stage_i_rotation_audit/20260607T-stage-i-rotation-audit-r2/`
  - `P11+ live_influx thesis weak-label sweep`：
    - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/`
    - 当前 `sample_source=live_influx`
    - 当前 `sample_count=111`
    - 当前 `task_entry_count=333`
    - 当前 `combination_count=2`
  - `P16 thesis materials` 首轮：
    - 旧 r1 图包已由 P22/r3 替代并清理。
    - 当前中期图表只引用 `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/`。
  - `P17 runtime service smoke`：
    - 首轮 r1 已清理；当前入口为 `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/`
    - 当前 `input_sample_count=37`
    - 当前 `view_count=1`
    - 当前 `feature_schema_status=aligned`
    - 当前 `runtime_error_cases.json` 已覆盖 `missing_checkpoint / missing_fields / empty_window / schema_mismatch`
  - `P18 P11 partial/resume`：
    - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/`
    - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/`
    - 当前 stable resume 已记录 `derived_from_run_id / completed_child_run_paths / blocked_attempt_log_paths / blocked_at_run_index`
    - 当前 partial blocked 已记录 `status=partial_blocked`
  - `P18 runtime schema contract`：
    - `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/`
    - 当前 `native_feature_schema_status=aligned`
    - 当前 `canonical_feature_schema_status=exact`
    - 当前 `expected_vehicle_feature_count=1930`
    - 当前 `input_vehicle_feature_count=965`
    - 当前 `missing_vehicle_feature_count=965`
    - 当前 missing groups 已覆盖 `BUS6000019110021` 到 `BUS6000019110026`
  - `P18 thesis materials 刷新`：
    - 旧 r2-p18/r3/r4 图包已由 P25 r5 替代并清理；r5 又已由 P26 r6 接管并在 2026-06-21 深度清理中从 docs 产物目录删除。
    - 保留 `runtime_semantic_case.csv` 作为 P20/P21 LLM preprocessing 历史输入表。
  - `P16/P21/P24 thesis materials r5 图表质量刷新`：
    - r5 曾把低信息图替换为 evidence matrix、schema contract 对照、runtime semantic case 复盘、rotation availability matrix、weak-label 小网格、分任务 private ablation、model backbone ablation、task adapter ablation、中文 public transfer、semantic event fusion 和 LLM A0-A4 review flow。
    - 当前 r5 图包/报告已由 P26 r6 接管并按 `docs/artifacts/cleanup/20260621-deep-cleanup.md` 清理，仅保留在 git 历史中。
  - `P26 thesis materials r6 报告图重绘`：
    - `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/`
    - 当前 `figure_manifest.json` 与 `figure_quality_audit.csv` 已记录 12 张 PNG、12 张 CSV、图片宽高、最小字号、中文标签策略、内部词检查和长 ID 处理策略；新增 `runtime_service_flow.png`，并把语义融合缺源情况降级为覆盖/支撑状态展示。
  - `2026-06-21 深度清理`：
    - 清理记录：`docs/artifacts/cleanup/20260621-deep-cleanup.md`
    - 已删除 P25 r5 thesis figure 图包和对应 Stage I 报告。
    - 已删除 P11 r3/r4 下两个空 `runs/` 目录，并清理本地 `src/tests/third_party` Python 编译缓存。
    - 明确保留 P11 live r1 child run、P20/P21 runtime case 输入表、public adapter baseline summary 和 canonical Stage I src/scripts 入口。
  - `中期报告写作材料`：
    - `docs/midterm/midterm-fact-sheet-2026-06-13.md`
    - `docs/midterm/boundaries-and-risks-2026-06-13.md`
    - `docs/midterm/claims-matrix-2026-06-13.md`

## 当前关键入口

- 当前执行入口与任务队列：[implementation/TASKS.md](implementation/TASKS.md)
- 论文需求入口：[requirements/SPEC.md](requirements/SPEC.md)
- 产物索引：[artifacts/ARTIFACTS.md](artifacts/ARTIFACTS.md)
- 中期前目标笔记：[implementation/notes/midterm-goal-2026-06-07.md](implementation/notes/midterm-goal-2026-06-07.md)
- P20 DeepSeek 在线时序数据预处理计划：[implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md](implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md)
- P20 DeepSeek 在线时序数据预处理 run：[artifacts/stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md](artifacts/stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md)

## 本轮验证

使用指定 conda 解释器并显式启用 torch runtime 测试执行：

```bash
CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 \
  /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest \
  tests.test_alignment_model_losses \
  tests.test_alignment_pipeline \
  tests.test_stage_i_support \
  tests.test_stage_i_multitask_train \
  tests.test_runtime_inference
```

结果：历史主线回归 `Ran 33 tests in 7.962s`，`OK`。本轮新增回归与主动 evidence 命令已完成：

```bash
CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 \
  /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest \
  tests.test_alignment_model_losses \
  tests.test_stage_i_multitask_sweep \
  tests.test_stage_i_private_component_ablation \
  tests.test_stage_i_public_transfer_boundary \
  tests.test_stage_i_rotation_audit \
  tests.test_stage_i_evidence_runner \
  tests.test_stage_i_multitask_train \
  tests.test_stage_i_private_optimization \
  tests.test_stage_i_public_opt
```

结果：`Ran 56 tests`，`OK`。历史 P10-P15 命令已真实执行；2026-06-19 深度清理后，runtime replay r1、support r1、rigid-body r1 等首轮产物不再作为 docs 下可打开入口，当前入口以 r2/r3/r6 summary、manifest 和报告为准。

本轮新增回归与 `P11+/P16/P17` 验证已完成：

```bash
CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 \
  /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest \
  tests.test_stage_i_multitask_sweep \
  tests.test_runtime_inference \
  tests.test_runtime_service_smoke
```

结果：`Ran 7 tests`，`OK`。历史 P17 smoke r1 命令已真实执行；当前 runtime service 入口以 `20260613T-stage-i-runtime-service-smoke-r2-contract` 为准。

当前 `P17` smoke 关键结果：

- `checkpoint cold-start = success`
- `input_sample_count = 37`
- `view_count = 1`
- `predictions_jsonl = generated`
- `feature_schema_status = aligned`
- `error_cases = missing_checkpoint / missing_fields / empty_window / schema_mismatch`

本轮 `P18` 新增验证已完成：

```bash
CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 \
  /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest \
  tests.test_stage_i_multitask_sweep \
  tests.test_runtime_service_smoke \
  tests.test_runtime_schema_contract
```

结果：`Ran 6 tests`，`OK`。此外，本轮真实命令已完成：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/stage_i/evidence/run_weak_label_sweep.py \
  --run-id 20260613T-stage-i-p11-live-influx-r3-resume \
  --e-run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json \
  --f-run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json \
  --sample-source live_influx \
  --resume-existing \
  --resume-run-root docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1 \
  --max-runs 2 \
  --epoch-count 1 \
  --batch-size 8 \
  --device cpu

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/stage_i/evidence/run_weak_label_sweep.py \
  --run-id 20260613T-stage-i-p11-live-influx-r4-partial \
  --e-run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json \
  --f-run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json \
  --sample-source stage_h_window_stats_proxy \
  --resume-existing \
  --resume-run-root docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1 \
  --max-runs 4 \
  --max-runtime-seconds 0 \
  --epoch-count 1 \
  --batch-size 8 \
  --device cpu

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/stage_i/runtime/run_smoke.py \
  --run-id 20260613T-stage-i-runtime-service-smoke-r2-contract \
  --checkpoint-path <local-only-checkpoint-backup-or-regenerated-multitask-checkpoint.pt> \
  --sample-jsonl <regenerated-input-view-runtime-samples-jsonl> \
  --artifact-root docs/artifacts/assets/stage_i_runtime_service \
  --report-root docs/artifacts/stage_i \
  --device cpu \
  --replay-mode both \
  --strict-feature-schema
```

当前 `P18` 关键结果：

- `P11 stable resume = completed`
- `P11 partial blocked = expected blocker`
- `P17 native_feature_schema_status = aligned`
- `P17 canonical_feature_schema_status = exact`

刚体对比首轮 r1 已清理，当前可打开入口为 `docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/rigid_body_ablation_summary.json` 和 `docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md`；当前已确认 `vehicle_field_metadata.status=loaded`，且 `vehicle_rigid_body_translation` 已经非零。

## 中期前边界管理

下面几类工作现在纳入中期前主动任务，但必须按边界清楚、证据分层、可复现的方式推进。

- CPU-heavy `sklearn` 或 UAB torch 候选搜索：中期前允许有限预算补跑；论文中只能作为公开 adapter baseline / calibration baseline。
- NASA/UAB 公开数据适配器结果：中期前整理成 public adapter evidence 和 transfer boundary；不能改写成论文双流本体闭环。
- `chronaris_opt` 与 `T1/T2/T3`：中期前补机制诊断；仍只能写成 private proxy benchmark evidence，不能写成人工真值 thesis task fully closed。
- `risk_proxy / workload_proxy / event_replay_tag`：中期前补小网格和消融；仍只能写成 thesis weak-label evidence。
- DeepSeek 在线 LLM 预处理：P20 已接入字段语义归一、weak-label 复核、schema gap policy、runtime 解释和切片整合；仍不能写成 OpenAI 接入、人工真值替代、原始全量数据外发或因果证据。
- `rigid_body rotation`：中期前必须核验真实字段；启用或缺失都要以 diagnostics 形式固化。
- 上游接收器、入库链路和原始大文件入仓：中期前不重建；论文系统封装时可说明现有 MySQL / InfluxDB 接入边界，必要时补轻量接口说明或部署文档。
