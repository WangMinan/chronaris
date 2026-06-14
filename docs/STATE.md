# Chronaris 当前状态

更新时间：2026-06-14

## 一句话状态

项目已经具备中期答辩可用的历史实验资产与最新主动证据闭环：真实链路 `Stage E/F/G(min)/H`、Stage I 历史公开 benchmark、`chronaris_opt` 私有代理证据、public adapter 支撑线、Phase C 真实 Stage H multitask 联合训练证据和中期证据包都已形成并进入 git 历史；同时，`P10 evidence runner`、`P11 live_influx thesis weak-label sweep`、`P12 chronaris_opt` 组件诊断、`P13 public adapter calibration`、`P14 public transfer boundary`、`P15 rigid_body rotation audit`、`P16 thesis materials`、`P17 runtime service smoke`、`P18 partial/resume + runtime schema contract` 已新增落盘。`P20 DeepSeek 在线时序数据预处理` 已完成 provider contract、agent-style prompt/harness v2、切片整合、mock/repair 测试和小样本真实 DeepSeek run，并作为 Stage H 到 Stage I 之间的可选 preprocessing context 接入。

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
    - 首轮：`docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r1/`
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
    - 首轮 replay：`docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-replay-r1/`
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
      - 稳定汇总：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r2/multitask_sweep_summary.json`
      - 比较表：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r2/proxy_vs_live_influx_comparison.csv`
      - 报告：`docs/artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r2.md`
      - 当前 `sample_source=live_influx`、`sample_count=111`、`task_entry_count=333`、`combination_count=2`；稳定版复用了 `20260613T-stage-i-p11-live-influx-r1` 中已完成的两个 live child run，并保留了更大 `4` 组合尝试的 blocker 日志。
    - `P12 chronaris_opt component ablation` 已完成：
      - 产物：`docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/`
      - 报告：`docs/artifacts/stage_i/stage-i-private-component-ablation-20260607T-stage-i-evidence-closure-r2-private-proxy.md`
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
    - `P16 thesis materials` 首轮已完成：
      - 首轮 root：`docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r1/`
      - 首轮报告：`docs/artifacts/stage_i/stage-i-thesis-materials-20260613T-stage-i-thesis-materials-r1.md`
      - 已导出 `6` 张稳定表、`6` 张 PNG 说明图，以及 `table_manifest.json / figure_manifest.json`；当前写作入口以 P18 刷新后的 `r2-p18` 为准。
    - `P17 runtime service smoke` 首轮已完成：
      - 首轮 root：`docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/`
      - 首轮报告：`docs/artifacts/stage_i/stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r1.md`
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
        - `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/canonical_runtime_samples.jsonl`
        - `docs/artifacts/stage_i/stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r2-contract.md`
      - P16 刷新图表：
        - `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/figure_manifest.json`
        - `docs/artifacts/stage_i/stage-i-thesis-materials-20260613T-stage-i-thesis-materials-r2-p18.md`
    - `P20 DeepSeek 在线时序数据预处理` 已完成 agent-style harness v2、切片整合与真实小样本 run：
      - 计划入口：`docs/implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md`
      - 当前真实 run：`docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_preprocessing_summary.json`
      - 当前报告：`docs/artifacts/stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md`
      - 当前结果：`request_count=8`、`error_count=0`、`field_semantic_count=24`、`weak_label_review_count=3`、`semantic_query_hint_count=4`、`runtime_explanation_count=4`。
      - 当前 harness：`prompt_version=stage_i_llm_preprocessing.agent_guardrails.v2`、`schema_repair_attempt_count=0`、`final_invalid_task_count=0`。
      - 当前切片：`field_semantics` 按 12+12、`schema_gap_policy` 按 3+3、`runtime_explanations` 按 2+2 切片，并在本地按 stable identifier 合并。
      - 当前定位：DeepSeek v4-pro 只作为字段语义归一、weak-label 规则复核、schema gap 预处理建议和 runtime 解释层；不替代人工真值、物理约束或因果融合主线。

## Git 与工作区核对

- 当前分支为 `main`，本地 `main` 相对 `origin/main` 存在未推送提交；最新 P20 r3-sliced 代码与产物以本文件列出的本地路径为准，推送后进入远端历史。
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
  - 语义事件融合：`semantic_event.py`、`causal_fusion.py`、`stage_i_support_builders.py`、`stage_i_support_reporting.py`。
  - runtime inference：`streaming_windows.py`、`runtime_inference.py`、`run_stage_i_runtime_inference.py`、`export_stage_i_runtime_samples.py`、`run_stage_i_semantic_event_support.py`。
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
  - 历史 live weak-label sweep r2：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r2/multitask_sweep_summary.json`
  - 最新 private component ablation：`docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/chronaris_opt_component_ablation.json`
  - 最新 public adapter calibration：`docs/artifacts/assets/stage_i_public_adapter_calibration/20260607T-stage-i-evidence-closure-r2-public-adapter/public_adapter_calibration_summary.json`
  - 最新 public transfer boundary：`docs/artifacts/assets/stage_i_public_transfer_boundary/20260607T-stage-i-evidence-closure-r2-transfer-boundary/public_transfer_boundary_summary.json`
  - 最新 rotation audit：`docs/artifacts/assets/stage_i_rotation_audit/20260607T-stage-i-rotation-audit-r2/rigid_body_rotation_audit_summary.json`
  - 历史 thesis materials r1：`docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r1/figure_manifest.json`
  - 历史 runtime service smoke r1：`docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/runtime_service_smoke_summary.json`
  - 最新 live weak-label stable resume：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json`
  - 最新 live weak-label partial blocked：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/partial_summary.json`
  - 最新 runtime schema contract：`docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json`
  - 最新 thesis materials p18：`docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/figure_manifest.json`
  - 最新中期事实清单：`docs/midterm/midterm-fact-sheet-2026-06-13.md`
  - 最新中期边界说明：`docs/midterm/boundaries-and-risks-2026-06-13.md`
  - 最新 P20 DeepSeek LLM preprocessing package：`docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_preprocessing_summary.json`

## 当前主线事实

- 当前鼎新私有任务验证主线仍是 `chronaris_opt`，但它属于 `private proxy benchmark / proxy evidence`。
- 当前公开支撑线为 `public opt closed`，但 `UAB robust-prior adapter / target_prior_median` 只能写成 `public adapter / calibration evidence`，不能写成双流连续对齐或因果融合模块本体的直接胜利。
- 当前公开第二模态应写成 `context proxy / public adapter evidence`，不是论文严格意义上的真实航电流。
- `T1/T2/T3` 是私有代理任务；`risk_proxy / workload_proxy / event_replay_tag` 是 thesis weak-label task builder，不等价于人工真值任务。
- `20251110_单01_ACT-2_涛_J20_26#01` 仍是 vehicle-only partial-data，不是双流 Stage H view。
- 中期 P20 LLM preprocessing 已按 DeepSeek v4-pro 完成小样本真实 run，并新增 agent-style prompt/harness v2 与切片整合；不默认使用 OpenAI，且 LLM 仅作为在线时序数据预处理、规则复核和解释层，不替代物理约束、因果融合或人工真值。

## 编码层面还需要做什么

1. P20 后续优先做对比实验：baseline Stage I task entries vs LLM-context-attached entries、内置 semantic query bank vs 内置+LLM whitelisted hints、runtime report with/without LLM explanation，以及小样本人工复核节省量；当前 provider contract、agent-style schema harness、切片整合、mock/repair 测试和小样本真实 DeepSeek run 已完成。
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
    - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r2/`
    - 当前 `sample_source=live_influx`
    - 当前 `sample_count=111`
    - 当前 `task_entry_count=333`
    - 当前 `combination_count=2`
  - `P16 thesis materials` 首轮：
    - `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r1/`
    - `figure_manifest.json` 已列出 `evidence_layer_overview / weak_label_sweep_ablation / chronaris_opt_component_ablation / public_transfer_boundary / runtime_semantic_case / rigid_body_rotation_audit`
  - `P17 runtime service smoke` 首轮：
    - `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/`
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
    - `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/`
    - 当前 `figure_manifest.json` 已把 P11 stable/partial blocker 与 P17 native aligned/canonical exact schema contract 纳入图表源。
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

结果：`Ran 56 tests`，`OK`。此外，本轮真实命令已完成：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/export_stage_i_runtime_samples.py \
  --run-id 20260607T-stage-i-runtime-replay-r1 \
  --run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json \
  --output-root docs/artifacts/assets/stage_i_runtime_inference

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_runtime_inference.py \
  --run-id 20260607T-stage-i-runtime-replay-r1 \
  --checkpoint-path docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_checkpoint.pt \
  --sample-jsonl docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-replay-r1/runtime_samples.jsonl \
  --artifact-root docs/artifacts/assets/stage_i_runtime_inference \
  --report-root docs/artifacts/stage_i \
  --device cpu

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_support.py \
  --run-id 20260607T-stage-i-support-semantic-r1 \
  --artifact-root docs/artifacts/assets/stage_i_support \
  --report-root docs/artifacts/stage_i \
  --causal-g-summary-path docs/artifacts/stage_i/assets/stage-i-semantic-event-20260607T-stage-i-semantic-event-r1/causal_fusion_summary.json

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_private_component_ablation.py \
  --run-id 20260607T-stage-i-private-component-r1 \
  --e-run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json \
  --f-run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_public_adapter_calibration.py \
  --run-id 20260607T-stage-i-evidence-closure-r2-public-adapter

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/build_stage_i_public_transfer_boundary.py \
  --run-id 20260607T-stage-i-evidence-closure-r2-transfer-boundary \
  --calibration-summary-path docs/artifacts/assets/stage_i_public_adapter_calibration/20260607T-stage-i-evidence-closure-r2-public-adapter/public_adapter_calibration_summary.json

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_rigid_body_rotation_audit.py \
  --run-id 20260607T-stage-i-rotation-audit-r2

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_evidence_closure.py \
  --run-id 20260607T-stage-i-evidence-closure-r2 \
  --skip-heavy \
  --test-summary "Ran 56 tests across P10-P15 and related suites; OK"
```

本轮新增回归与 `P11+/P16/P17` 验证已完成：

```bash
CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 \
  /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest \
  tests.test_stage_i_multitask_sweep \
  tests.test_runtime_inference \
  tests.test_runtime_service_smoke
```

结果：`Ran 7 tests`，`OK`。此外，本轮真实命令已完成：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_thesis_materials.py \
  --run-id 20260613T-stage-i-thesis-materials-r1 \
  --live-sweep-summary-path docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r2/multitask_sweep_summary.json

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_runtime_smoke.py \
  --run-id 20260613T-stage-i-runtime-service-smoke-r1 \
  --checkpoint-path docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_checkpoint.pt \
  --sample-jsonl docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/input_view_runtime_samples.jsonl \
  --artifact-root docs/artifacts/assets/stage_i_runtime_service \
  --report-root docs/artifacts/stage_i \
  --device cpu \
  --replay-mode both
```

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
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_multitask_sweep.py \
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

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_multitask_sweep.py \
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

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_runtime_smoke.py \
  --run-id 20260613T-stage-i-runtime-service-smoke-r2-contract \
  --checkpoint-path docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_checkpoint.pt \
  --sample-jsonl docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/input_view_runtime_samples.jsonl \
  --artifact-root docs/artifacts/assets/stage_i_runtime_service \
  --report-root docs/artifacts/stage_i \
  --device cpu \
  --replay-mode both \
  --strict-feature-schema

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_thesis_materials.py \
  --run-id 20260613T-stage-i-thesis-materials-r2-p18 \
  --live-sweep-summary-path docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json \
  --live-partial-summary-path docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/partial_summary.json \
  --runtime-summary-path docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_inference/20260613T-stage-i-runtime-service-smoke-r2-contract-runtime/runtime_inference_summary.json \
  --runtime-service-summary-path docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_service_smoke_summary.json \
  --runtime-schema-contract-path docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json
```

当前 `P18` 关键结果：

- `P11 stable resume = completed`
- `P11 partial blocked = expected blocker`
- `P17 native_feature_schema_status = aligned`
- `P17 canonical_feature_schema_status = exact`

刚体对比另行落到了 `docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r1/` 和 `docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r1.md`；与上一版不同的是，本次已确认 `vehicle_field_metadata.status=loaded`，且 `vehicle_rigid_body_translation` 已经非零。

## 中期前边界管理

下面几类工作现在纳入中期前主动任务，但必须按边界清楚、证据分层、可复现的方式推进。

- CPU-heavy `sklearn` 或 UAB torch 候选搜索：中期前允许有限预算补跑；论文中只能作为公开 adapter baseline / calibration baseline。
- NASA/UAB 公开数据适配器结果：中期前整理成 public adapter evidence 和 transfer boundary；不能改写成论文双流本体闭环。
- `chronaris_opt` 与 `T1/T2/T3`：中期前补机制诊断；仍只能写成 private proxy benchmark evidence，不能写成人工真值 thesis task fully closed。
- `risk_proxy / workload_proxy / event_replay_tag`：中期前补小网格和消融；仍只能写成 thesis weak-label evidence。
- DeepSeek 在线 LLM 预处理：P20 已接入字段语义归一、weak-label 复核、schema gap policy、runtime 解释和切片整合；仍不能写成 OpenAI 接入、人工真值替代、原始全量数据外发或因果证据。
- `rigid_body rotation`：中期前必须核验真实字段；启用或缺失都要以 diagnostics 形式固化。
- 上游接收器、入库链路和原始大文件入仓：中期前不重建；论文系统封装时可说明现有 MySQL / InfluxDB 接入边界，必要时补轻量接口说明或部署文档。
