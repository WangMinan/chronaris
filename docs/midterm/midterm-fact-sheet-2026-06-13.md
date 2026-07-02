# Chronaris 中期事实清单

更新时间：2026-07-02

本清单冻结当前中期报告可引用事实。所有实验事实必须能追溯到 `docs/artifacts/` 下的报告、CSV、JSON 或 PNG。写中期报告时，优先引用本清单中的当前入口；历史报告只作为追溯材料，不从旧报告倒推当前状态。

## 1. 总判断

当前仓库已经形成中期答辩可用的证据闭环：

- 数据链路层：Stage E/F/G(min)/H 历史真实链路已经收口，Stage H 可稳定导出标准化融合特征。
- 模型主线层：Stage I Phase A/B/C 已完成统一骨干、真实 Stage H weak-label 联合训练、checkpoint 导出和 private/thesis 分层资产。
- 证据补强层：Phase D/E/F 已完成刚体物理约束、语义事件融合 support、runtime replay/service 补强。
- 主动证据层：P10-P18、P27/P28 与 P30/P31/P32 已完成，覆盖 evidence runner、P11 weak-label sweep、private component ablation、public adapter/model comparison、private third-party comparison、public fusion ablation、transfer boundary、rotation audit、thesis figures、runtime smoke、schema contract 和 cross-evidence matrix。
- 文档与索引层：当前状态、任务队列、产物索引和 Stage I 报告入口已经同步到 `docs/STATE.md`、`docs/implementation/TASKS.md`、`docs/artifacts/ARTIFACTS.md`、`docs/artifacts/stage_i/README.md`。

## 2. 当前论文目标与仓库能力

论文方向来自 [../requirements/SPEC.md](../requirements/SPEC.md)：航空人机异构时序数据连续对齐与语义融合。当前仓库应支撑：

| 能力 | 当前状态 | 证据入口 |
| --- | --- | --- |
| 从 MySQL / InfluxDB 读取指定架次多源数据与元信息 | 已完成接入与 Stage H 资产化 | [../STATE.md](../STATE.md) |
| 统一 schema、时间参考和样本组织 | 已完成 Stage H 标准化 view 与 sample contract | [../artifacts/stage_h/stage-h-closure-2026-04-27.md](../artifacts/stage_h/stage-h-closure-2026-04-27.md) |
| 双流连续潜态建模 | 已完成 Stage I Phase C 真实 weak-label 联合训练 | [../artifacts/stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md](../artifacts/stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md) |
| 物理一致性约束时间对齐 | translation + vertical 约束已在真实链路启用；rotation disabled diagnostics 已落盘 | [../artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md](../artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md) |
| 因果掩码与语义事件融合 | 语义 support 覆盖 3 个双流 view | [../artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md](../artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md) |
| 标准化融合特征与中间态接口 | Stage H 与 runtime replay 已形成样本、checkpoint、prediction 输出 | [../artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md](../artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md) |
| 面向风险、负荷、事件复盘的验证 | risk/workload/event weak-label 任务已完成训练、sweep 和 r6 图表；private T1/T2/T3 另有 leakage-safe 组件诊断、third-party comparison 与 cross-evidence matrix | [../artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md](../artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md) |

## 3. 数据与样本事实

当前 Stage I thesis 主线使用 Stage H all-window clean 资产：

- E 流 manifest：`docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json`
- F 流 manifest：`docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json`
- 当前真实样本集合：
  - sortie_count=`2`
  - view_count=`3`
  - sample_count=`111`
  - sample_id_mode=`view_prefixed`
  - 每个 view `37` 个窗口样本

当前三个 view：

| sortie_id | view_id | pilot_id | sample_count | export_start_utc | export_stop_utc |
| --- | --- | --- | --- | --- | --- |
| `20251005_四01_ACT-4_云_J20_22#01` | `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `10033` | `37` | `2025-10-05T01:35:00+00:00` | `2025-10-05T01:38:01+00:00` |
| `20251002_单01_ACT-8_翼云_J16_12#01` | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `10035` | `37` | `2025-10-02T08:35:00+00:00` | `2025-10-02T08:38:01+00:00` |
| `20251002_单01_ACT-8_翼云_J16_12#01` | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `10033` | `37` | `2025-10-02T08:35:00+00:00` | `2025-10-02T08:38:01+00:00` |

写法建议：

- 可以写“当前中期阶段已在 2 个 sortie、3 个双流 view、111 个窗口样本上形成可复现实验闭环”。
- 不应写“覆盖全部飞行数据”或“大规模临床/实飞泛化验证完成”。

## 4. Stage I Phase C 真实 weak-label 联合训练

当前 Phase C 主线产物：

- 报告：[../artifacts/stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md](../artifacts/stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md)
- Summary：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_summary.json`
- Checkpoint：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_checkpoint.pt`

关键指标：

| 项 | 值 |
| --- | --- |
| run_id | `20260607T-stage-i-multitask-real-closure-r2` |
| sample_count | `111` |
| task_entry_count | `333` |
| split_counts | train=`66`, validation=`22`, test=`23` |
| task heads | `risk_proxy` classification, `workload_proxy` regression, `event_replay_tag` retrieval |
| risk labels | low=`38`, high=`37`, medium=`36` |
| event tags | maneuver_dominant=`111` |
| workload normalized range | `0` to `1` |

测试集摘要：

| metric | value |
| --- | --- |
| test sample_count | `23` |
| reconstruction_total | `1.8224518299102783` |
| alignment | `0.07991387695074081` |
| task_total | `4.229607582092285` |
| causal_total | `0.9288970232009888` |
| total | `47224576` |
| task risk_proxy loss | `1.1950148344039917` |
| task workload_proxy loss | `0.004169390071183443` |
| task event_replay_tag loss | `3.030423641204834` |

边界：

- `risk_proxy / workload_proxy / event_replay_tag` 是 thesis weak-label task，不是人工标注真值。
- `risk_proxy` 的来源是 vehicle intensity + physiology variation。
- `workload_proxy` 的来源是 physiology variation + vehicle intensity。
- `event_replay_tag` 是 derived event tag group pairing，不是专家复盘标注。

## 5. P10 evidence runner

当前 P10 稳定入口：

- Manifest：`docs/artifacts/assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/evidence_manifest.json`
- 报告：[../artifacts/stage_i/stage-i-evidence-closure-20260607T-stage-i-evidence-closure-r2.md](../artifacts/stage_i/stage-i-evidence-closure-20260607T-stage-i-evidence-closure-r2.md)
- run_id：`20260607T-stage-i-evidence-closure-r2`
- status：`completed`

七项 evidence task：

| task | evidence_layer | status | 说明 |
| --- | --- | --- | --- |
| multitask | thesis_weak_label | completed | weak-label multitask sweep |
| rigid_body | rigid_body_support | completed | 复用 Phase D r2 真实 ablation |
| semantic | semantic_support | completed | 复用 Phase E semantic support |
| runtime | runtime_replay | completed | 复用 Phase F runtime service r2 |
| private_proxy | private_proxy | completed | chronaris_opt component ablation |
| public_adapter | public_adapter_closure | completed | public adapter calibration |
| rotation | rotation_diagnostics | completed | rigid_body rotation audit |

写法建议：

- 可以写“P10 将 Stage I 分散证据组织为统一 manifest/report 入口，支撑中期材料追溯”。
- 不应把 P10 写成新的模型贡献；它是证据工程和复现实验入口。

## 6. P11 thesis weak-label sweep

### 6.1 bounded proxy sweep

当前 bounded 版本：

- Summary：`docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/multitask_sweep_summary.json`
- 报告：[../artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260607T-stage-i-evidence-closure-r2-multitask.md](../artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260607T-stage-i-evidence-closure-r2-multitask.md)
- sample_source：`stage_h_window_stats_proxy`
- sample_count：`111`
- task_entry_count：`333`
- combination_count：`2`
- best_test_total：`1024.8099895974865`

### 6.2 live_influx stable resume

当前 live stable/resume 版本：

- Summary：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json`
- 报告：[../artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r3-resume.md](../artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r3-resume.md)
- run_id：`20260613T-stage-i-p11-live-influx-r3-resume`
- status：`completed`
- evidence_layer：`thesis_weak_label`
- sample_source：`live_influx`
- sample_count：`111`
- task_entry_count：`333`
- combination_count：`2`
- resume_existing：`true`
- resume_run_root：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1`
- completed_child_summary_count：`2`
- blocked_at_run_index：`3`
- blocker log：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/run.log`
- best_child_run_id：`20260613T-stage-i-p11-live-influx-r1-01-minimal-cw0p00-tlw0p50-lagnone`
- best_test_total：`1153.8985701851223`

live stable 的两个完成 child run：

1. `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/runs/20260613T-stage-i-p11-live-influx-r1-01-minimal-cw0p00-tlw0p50-lagnone/multitask_summary.json`
2. `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/runs/20260613T-stage-i-p11-live-influx-r1-02-minimal-cw0p00-tlw0p50-lag3/multitask_summary.json`

### 6.3 partial blocked evidence

当前 partial blocked 版本：

- Partial summary：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/partial_summary.json`
- Partial CSV：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/thesis_weak_label_multitask_ablation.partial.csv`
- run_id：`20260613T-stage-i-p11-live-influx-r4-partial`
- status：`partial_blocked`
- evidence_layer：`thesis_weak_label`
- sample_count：`111`
- task_entry_count：`333`
- completed_child_runs：
  - `20260613T-stage-i-p11-live-influx-r1-01-minimal-cw0p00-tlw0p50-lagnone`
  - `20260613T-stage-i-p11-live-influx-r1-02-minimal-cw0p00-tlw0p50-lag3`
- blocked_at_run_index：`3`
- blocker_log_path：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/run.log`

写法建议：

- 可以写“P11 已完成 bounded proxy 与 live_influx stable/resume 双路线验证，并保留 larger grid partial blocker 证据，避免伪造成完整成功”。
- 不应写“4 组合 live_influx sweep 全部完成”。
- 不应写“weak-label 任务等价于人工标注风险/负荷/事件真值”。

## 7. P12 chronaris_opt private component ablation

当前入口：

- Summary：`docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/chronaris_opt_component_ablation.json`
- 报告：[../artifacts/stage_i/stage-i-private-component-ablation-20260607T-stage-i-evidence-closure-r2-private-proxy.md](../artifacts/stage_i/stage-i-private-component-ablation-20260607T-stage-i-evidence-closure-r2-private-proxy.md)
- evidence_layer：`private_proxy`
- task_boundary：`t1_t2_t3_are_proxy_tasks_not_direct_thesis_tasks`
- records：sample_count=`111`, view_count=`3`, sortie_count=`2`
- variants：`naive_sync`, `e_baseline`, `f_full`, `g_min`, `g_no_causal_mask`, `chronaris_opt`, `chronaris_opt_no_causal_mask`, `chronaris_opt_no_time_residual`, `chronaris_opt_no_task_head`

`chronaris_opt` 三项 private proxy 指标：

| task | metric | value |
| --- | --- | --- |
| T1_maneuver_intensity_class | macro_f1 | `1.0` |
| T2_next_window_physiology_response | rmse | `201.4895651832178` |
| T3_paired_pilot_window_retrieval | top1_accuracy | `1.0` |

机制诊断摘要：

- `chronaris_opt` mean_attention_entropy=`0.9345914001937385`
- mean_top_event_concentration=`0.3974255199904914`
- mean_event_mask_interference=`0.002122758745073198`
- lag_window_points=`3`
- residual_mode=`raw_window_stats`
- use_causal_mask=`true`

边界：

- T1/T2/T3 是 private proxy benchmark，不是论文三类 weak-label task。
- 这组结果可用于说明模块组合与消融价值，不能写成公开泛化或人工真值验证。

### P12.2 leakage-safe private component ablation

当前新增协议入口：

- Summary：`docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/ablation_summary.json`
- 报告：[../artifacts/stage_i/stage-i-private-leakage-safe-ablation-20260619T-stage-i-leakage-safe-ablation-r2.md](../artifacts/stage_i/stage-i-private-leakage-safe-ablation-20260619T-stage-i-leakage-safe-ablation-r2.md)
- protocol：`leakage_safe_v1`
- audit_status：`pass`
- records：sample_count=`111`, view_count=`3`, sortie_count=`2`
- seeds：`17, 29, 43, 71, 97`
- split_strategy：T1/T2 使用 leave-one-view-out 与 leave-one-sortie-out；聚合图表默认引用 leave-one-view-out。T3 只使用具备跨飞行员正样本的数据视图，不为单飞行员视图生成伪配对；候选池限定为 `same_sortie_cross_pilot`，即同一 sortie 的另一名飞行员窗口，`pilot_id/window_index` 不进入特征向量。

标签-特征同源审计：

- 直接字段重叠、标签确定性派生特征、样本/飞行员/窗口位置和原始时间身份字段均已审计。
- 审计产物：`label_feature_overlap_audit.json` 与 `label_feature_overlap_audit.csv`。
- T3 正负样本相似度分布原始 `164280` 行，结构化写出 `20000` 行；保留全部正样本并确定性抽样负样本。

完整防泄漏任务输入的当前指标：

| task | metric | value | 说明 |
| --- | --- | --- | --- |
| T1_maneuver_intensity_class | macro_f1 | `0.17333333333333334` | balanced_accuracy=`0.3333333333333333` |
| T2_next_window_physiology_response | rmse | `862.6941748579226` | persistence_rmse=`201.4895651832178`, nrmse=`0.3695127256456088`, persistence_improvement_rate=`-3.281582393973904` |
| T3_paired_pilot_window_retrieval | top1_accuracy | `0.02702702702702703` | top3=`0.0945945945945946`, top5=`0.14864864864864866`, mrr=`0.11994791108940953`, valid_query_count=`74`, candidate_count=`2738`, candidate_pool_policy=`same_sortie_cross_pilot` |

当前组件总览中 T3 最佳为 `continuous_dual_state / naive_time_sync`，Top-1=`0.06756756756756757`，Top-5=`0.17567567567567569`，MRR=`0.15295949410099255`。

写法建议：

- 可以写“历史 private proxy 满分结果已新增 leakage-safe 审计与消融协议复核，排除了标签源同源特征和身份/时间位置泄漏，结果更适合论文实验章节作为组件诊断引用”。
- 不应写“leakage-safe T3 已经达到历史满分”或“历史满分指标仍可直接作为论文主实验结果”。当前 T3 在同 sortie 跨飞行员候选池下已经有非零命中，但 Top-1 仍只有 `0.0270` 到 `0.0676`，应写成严格协议下的初步补齐和后续改进方向。

## 8. P13/P14 public adapter 与 transfer boundary

### P13 public adapter calibration

当前入口：

- Summary：`docs/artifacts/assets/stage_i_public_adapter_calibration/20260607T-stage-i-evidence-closure-r2-public-adapter/public_adapter_calibration_summary.json`
- 报告：[../artifacts/stage_i/stage-i-public-adapter-calibration-20260607T-stage-i-evidence-closure-r2-public-adapter.md](../artifacts/stage_i/stage-i-public-adapter-calibration-20260607T-stage-i-evidence-closure-r2-public-adapter.md)
- evidence_layer：`public_adapter_calibration`
- source_count：`5`
- row_count：`22`

best_by_category：

| category | dataset | candidate | metric | value | secondary |
| --- | --- | --- | --- | --- | --- |
| public_adapter_baseline | uab_workload_dataset / heat_the_chair | physiology_persistence | rmse | `1.4567585657677036` | mae=`1.1636927899776721` |
| calibration_baseline | uab_workload_dataset / heat_the_chair | target_prior_median | rmse | `1.433140268927459` | mae=`1.073958345604878` |
| legacy_public_opt | nasa_csm | balanced_linear_svc_context | macro_f1 | `0.7445135090687961` | balanced_accuracy=`0.758165694976977` |
| torch_uab | uab_workload_dataset / heat_the_chair | heat_residual_correction__lr0p0003__wd0p0001 | rmse | `1.4629949608517474` | mae=`1.15944445165349` |

### P14 transfer boundary

当前入口：

- Summary：`docs/artifacts/assets/stage_i_public_transfer_boundary/20260607T-stage-i-evidence-closure-r2-transfer-boundary/public_transfer_boundary_summary.json`
- 报告：[../artifacts/stage_i/stage-i-public-transfer-boundary-20260607T-stage-i-evidence-closure-r2-transfer-boundary.md](../artifacts/stage_i/stage-i-public-transfer-boundary-20260607T-stage-i-evidence-closure-r2-transfer-boundary.md)
- public_mainline_status：`public opt closed`

数据边界：

| corpus | modality_pair | labels | evidence_role |
| --- | --- | --- | --- |
| private_stage_h | real_physiology + real_vehicle_timeseries | risk_proxy/workload_proxy/event_replay_tag or T1/T2/T3 | thesis_weak_label + private_proxy |
| uab_workload_dataset | physiology + task_context_proxy | subjective workload / public adapter target | public_adapter/calibration |
| nasa_csm | physiology + scenario_context_proxy | attention_state / public adapter target | public_adapter/calibration |

边界：

- UAB/NASA 结果只能写成 public adapter / calibration evidence。
- UAB/NASA 第二模态是 context proxy，不是论文严格意义上的真实航电流。
- 不能把 public adapter 结果改写为私有双流连续对齐主线 fully closed。

### 8.5 P27/P28/P30/P31/P32 public/private comparison 与 cross-evidence matrix

当前新增比较实验入口：

- P27 public model comparison：`docs/artifacts/assets/stage_i_public_model_comparison/20260701T-stage-i-public-model-comparison-r1/evidence_manifest.json`
- P28 public fusion refresh：`docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fusion_refresh_summary.json`
- P30 private third-party comparison：`docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/private_thirdparty_summary.json`
- P31 public fusion ablation：`docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/public_fusion_ablation_summary.json`
- P32 cross-evidence matrix：`docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/evidence_manifest.json`

P30 private third-party comparison 使用 private real dual-stream Stage H 的 `111` 个窗口、`3` 个 view、`2` 个 sortie，对 `chronaris_full`、`mult`、`contiformer`、`naive_time_sync` 和 `classical_baseline` 做 T1/T2/T3 proxy task 对比。当前结果是混合对比：`chronaris_full` 在 T3 top3/top5/MRR 上优于 `naive_time_sync`，但 T1 macro-F1 低于 MulT/ContiFormer/classical，T2 RMSE 也未全面领先 MulT/ContiFormer；不能写成 Chronaris 全面胜出或人工真值任务闭环。

P31 public fusion ablation 在 NASA CSM 与 UAB workload public context proxy 上比较 `full`、`no_lag_window`、`no_event_bias`、`physiology_only`、`context_only`、`no_causal_fusion`、`no_target_transform` 和 `mse_loss`。关键读数如下：

| dataset / task | full | 最佳 ablation | 说明 |
| --- | --- | --- | --- |
| NASA combined macro-F1 | `0.327192` | `no_lag_window=0.394103` | full 不是 public context proxy 上的最优项 |
| NASA combined balanced_accuracy | `0.346014` | `no_lag_window=0.385870` | 体现 lag-window 组件敏感性 |
| UAB mean RMSE | `3.279090` | `context_only=3.066633` | lower is better，context proxy 在 UAB 上占主导 |

P32 cross-evidence matrix 将 private/public/proxy/component 证据统一成 `292` 行矩阵：`private_thirdparty_comparison=72`、`private_component_ablation=36`、`public_model_comparison=80`、`public_component_ablation=104`。中期报告可以用它说明证据层级和边界，但不能把 private Stage H、private proxy、public adapter、public context proxy 结果混成同一个胜负排行榜。

写法建议：

- 可以写“本阶段新增第三方对比、公开组件消融和跨证据矩阵，把 private real dual-stream、private proxy、public adapter 和 public component ablation 分层组织为可追溯证据”。
- 不应写“公开数据已证明私有航电双流泛化成功”或“P30/P31 证明 Chronaris 在所有任务上优于第三方模型”。

## 9. P15/Phase D rigid-body 与 rotation audit

### Phase D rigid-body r2

当前入口：

- Summary：`docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/rigid_body_ablation_summary.json`
- 报告：[../artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md](../artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md)

关键 family 指标：

| family | test_total | test_alignment | test_physics_total | metadata_status |
| --- | --- | --- | --- | --- |
| minimal | `1110.7086181640625` | `0.09487661719322205` | `11088.6591796875` | loaded |
| full | `2.1334598064422607` | `0.10848616063594818` | `2.6374268531799316` | loaded |
| rigid_body | `2.484066963195801` | `0.11162154376506805` | `6.105468273162842` | loaded |

rigid_body 已启用 residual：

- enabled_residuals：`translation`, `vertical`
- vehicle_rigid_body_translation=`1.133332371711731`
- vehicle_rigid_body_vertical=`3.9466116428375244`
- vehicle_rigid_body_rotation=`0`
- metadata measurement：`BUS6000019110020`
- metadata field_count：`96`

### P15 rotation audit

当前入口：

- Summary：`docs/artifacts/assets/stage_i_rotation_audit/20260619T-stage-i-rotation-audit-r3-figure-refresh/rigid_body_rotation_audit_summary.json`
- 报告：[../artifacts/stage_i/stage-i-rigid-body-rotation-audit-20260619T-stage-i-rotation-audit-r3-figure-refresh.md](../artifacts/stage_i/stage-i-rigid-body-rotation-audit-20260619T-stage-i-rotation-audit-r3-figure-refresh.md)
- evidence_layer：`rotation_diagnostics`
- rotation_enabled：`false`
- rotation_status：`disabled`
- rotation_reading：`current sortie still lacks paired rate fields for pitch/roll/yaw`

已找到的角度字段：

- pitch：`BUS6000019110020.code1030`，载机俯仰角
- roll：`BUS6000019110020.code1032`，载机横滚角
- yaw：`BUS6000019110020.code1031`，载机真航向

缺失字段：

- pitch_rate：缺失
- roll_rate：缺失
- yaw_rate：缺失

写法建议：

- 可以写“当前已完成刚体平移与垂向约束；旋转项由于缺少成对角速度字段，在中期阶段作为数据字段诊断保留”。
- 不应写“完整 6DoF 刚体旋转约束已经启用”。

## 10. Phase E semantic event support

当前入口：

- Summary：`docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`
- 报告：[../artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md](../artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md)
- semantic event report：[../artifacts/stage_i/stage-i-semantic-event-support-20260607T-stage-i-semantic-support-r2.md](../artifacts/stage_i/stage-i-semantic-event-support-20260607T-stage-i-semantic-support-r2.md)

语义 support 摘要：

- query_names：`risk_proxy`, `workload_proxy`, `event_replay_tag`
- query_count：`3`
- view_count：`3`
- top_view_id：`20251005_四01_ACT-4_云_J20_22#01__pilot_10033`
- mean_event_token_count=`1.0090090036392212`
- mean_query_entropy=`0.00854379249115785`
- mean_top_query_score=`2.4731220890272847`
- mean_top_event_attribution=`7.417277874173345`

三 view 摘要：

| view_id | sample_count | dominant_query_name | mean_top_query_score | mean_top_event_attribution | top_sample_id |
| --- | --- | --- | --- | --- | --- |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `37` | risk_proxy | `2.623851899359677` | `7.8715556988845` | `20251005_四01_ACT-4_云_J20_22#01:0000` |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `37` | risk_proxy | `2.4941032794681757` | `7.476044445424466` | `20251002_单01_ACT-8_翼云_J16_12#01:0002` |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `37` | risk_proxy | `2.301411088254001` | `6.904233478211068` | `20251002_单01_ACT-8_翼云_J16_12#01:0000` |

写法建议：

- 可以写“语义事件融合 support 已在当前 3 个双流 view 上形成 view-level ranking 与样本级 attribution”。
- 不应写“已完成专家语义事件标签验证”。

## 11. Phase F / P17 runtime service

### 历史 runtime service replay r2

当前入口：

- Summary：`docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/runtime_inference_summary.json`
- 报告：[../artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md](../artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md)
- sample_count：`40`
- replay_mode：`batch`
- mean_attention_entropy=`1.9059074968099594`
- mean_top_event_score=`1`
- mean_top_contribution_score=`2.718562451004982`
- semantic query_count=`3`
- semantic mean_top_event_attribution=`8.04756212234497`

### P17/P18 runtime schema contract

当前入口：

- Summary：`docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_service_smoke_summary.json`
- Schema contract：`docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json`
- canonical payload route：已验证 `canonical_feature_schema_status=exact`；raw JSONL 已按 `docs/artifacts/cleanup/20260619-lfs-docs-prune.md` 从 docs/LFS 清理，当前引用以 `runtime_schema_contract.json` 和 r2 contract 报告为准。
- 报告：[../artifacts/stage_i/stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r2-contract.md](../artifacts/stage_i/stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r2-contract.md)
- checkpoint：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_checkpoint.pt`
- input_sample_count：`37`
- view_count：`1`
- view_id：`20251005_四01_ACT-4_云_J20_22#01__pilot_10033`
- native_feature_schema_status：`aligned`
- canonical_feature_schema_status：`exact`
- schema_source：`input_normalization_stats`
- schema_hash：`47c8cbb3328ece97e7dd3b373a0fd55af36bd1d4ae9ec9475f2d83e22acc078a`

Schema contract 计数：

| schema | physiology_count | vehicle_count | status |
| --- | --- | --- | --- |
| expected checkpoint schema | `12` | `1930` | baseline |
| native runtime input | `12` | `965` | aligned |
| canonical payload | `12` | `1930` | exact |

Native 缺失的 vehicle measurement groups：

- `BUS6000019110021`
- `BUS6000019110022`
- `BUS6000019110023`
- `BUS6000019110024`
- `BUS6000019110025`
- `BUS6000019110026`

错误样例：

| case_id | status | error_type | 说明 |
| --- | --- | --- | --- |
| missing_checkpoint | expected_failure | FileNotFoundError | checkpoint path does not exist |
| missing_fields | expected_failure | ValueError | drop vehicle from first payload |
| empty_window | expected_failure | ValueError | zero-length stream values |
| schema_mismatch | expected_failure | ValueError | extra feature and missing native vehicle features |
| native_strict_feature_schema | expected_failure | ValueError | native strict schema mismatch with six missing groups |

写法建议：

- 可以写“runtime smoke 已完成 checkpoint 冷启动、JSONL 输入、prediction 输出、错误样例和 schema contract”。
- 可以写“canonical payload 路线可以达到 service contract exact schema”。
- 必须同时写“native runtime input 仍是 aligned，缺少 965 个 vehicle features，集中在 6 个 BUS measurement groups”。
- 不应写“原始上游输入已 native exact”。

## 12. P16/P18/P26 thesis figures

当前图表入口：

- 图表报告：[../artifacts/stage_i/stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md](../artifacts/stage_i/stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md)
- figure manifest：`docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_manifest.json`
- table manifest：`docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/table_manifest.json`
- QA 清单：`docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_quality_audit.csv`

当前 12 张 PNG 说明图：

| figure_id | PNG | CSV | evidence_layer | 用途 |
| --- | --- | --- | --- | --- |
| evidence_layer_overview | `evidence_layer_overview.png` | `evidence_layer_overview.csv` | cross_layer_index | 展示 Stage I 证据层级矩阵，替代 artifact present=1 柱状图 |
| runtime_payload_schema | `runtime_payload_schema.png` | `runtime_payload_schema.csv` | runtime_schema | 对照 native replay payload 与 canonical service payload 字段契约 |
| runtime_service_flow | `runtime_service_flow.png` | `runtime_service_flow.csv` | runtime_schema | 展示当前 runtime service 的输入、契约、推理和边界闭环 |
| runtime_semantic_case | `runtime_semantic_case.png` | `runtime_semantic_case.csv` | runtime_semantic_support | 展示 runtime case card、窗口级语义归因、query 分布、范围 chip 与 schema 状态摘要 |
| rigid_body_rotation_audit | `rigid_body_rotation_audit.png` | `rigid_body_rotation_audit.csv` | rigid_body_rotation_diagnostics | 展示 family loss 与 pitch/roll/yaw angle/rate 可用性矩阵 |
| weak_label_sweep_ablation | `weak_label_sweep_ablation.png` | `weak_label_sweep_ablation.csv` | thesis_weak_label | 对比 proxy/live best metrics、运行状态和小网格 lag |
| chronaris_opt_component_ablation | `chronaris_opt_component_ablation.png` | `chronaris_opt_component_ablation.csv` | private_proxy | 兼容总览图，分任务尺度展示防泄漏组件消融 |
| model_backbone_ablation | `model_backbone_ablation.png` | `model_backbone_ablation.csv` | private_proxy_leakage_safe | 模型骨干结构消融，推荐用于论文实验章节 |
| task_adapter_ablation | `task_adapter_ablation.png` | `task_adapter_ablation.csv` | private_proxy_leakage_safe | 任务适配层消融，推荐用于论文实验章节 |
| public_transfer_boundary | `public_transfer_boundary.png` | `public_transfer_boundary.csv` | transfer_boundary | 中文展示公开适配、私有弱标注主线、私有代理消融的正向分工 |
| semantic_event_fusion_overview | `semantic_event_fusion_overview.png` | `semantic_event_fusion_overview.csv` | semantic_support | 展示双流 semantic event fusion 与 query-to-event attribution |
| llm_comparison_a0_a4 | `llm_comparison_a0_a4.png` | `llm_comparison_a0_a4.csv` | llm_preprocessing_comparison | 展示 LLM A0-A4 对比与待人工复核边界 |

建议用于中期报告的图：

1. 总览：`evidence_layer_overview.png`。
2. 数据链路 / 运行时 schema：`runtime_payload_schema.png`。
3. 模型链路 / 语义融合：`semantic_event_fusion_overview.png`。
4. 实验链路 / weak-label：`weak_label_sweep_ablation.png`。
5. 组件诊断总览：`chronaris_opt_component_ablation.png`。
6. 模型骨干结构消融：`model_backbone_ablation.png`。
7. 任务适配层消融：`task_adapter_ablation.png`。
8. 物理约束与字段边界：`rigid_body_rotation_audit.png`。
9. 公开适配与私有主线分工：`public_transfer_boundary.png`。
10. LLM 预处理复核材料链路：`llm_comparison_a0_a4.png`。

## 13. 中期历史证据包

当前历史中期证据包：

- 报告：[../artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md](../artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md)
- 资产目录：`docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/`
- Manifest：`docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/midterm_evidence_manifest.json`
- Figure index：`docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/midterm_figure_index.csv`
- Metrics：`docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/midterm_metrics.csv`

该证据包形成于 P10-P18 之前，仍可作为历史整编入口；正式写当前中期报告时，应优先使用本清单与 `stage_i_thesis_figures r6-report-figure-polish`。

## 14. 当前可直接写入报告的贡献描述

推荐写成四个层次：

1. 数据与样本组织：面向私有航空人机时序数据，完成 MySQL/InfluxDB 接入、跨源时间基准、Stage H 标准化窗口和双流 view 组织。
2. 模型主线：建立生理流和飞机流双流连续潜态建模框架，接入物理约束、因果掩码、语义事件融合和任务头。
3. 实验与验证：在 2 个 sortie、3 个双流 view、111 个窗口样本上完成 weak-label thesis task 训练、live/proxy sweep、private proxy 消融、private third-party comparison、public adapter/model comparison、public fusion ablation、cross-evidence matrix 和 runtime smoke。
4. 工程与可复现：形成 evidence runner、manifest、schema contract、错误样例、图表和文档索引，使中期报告材料可追溯、可复跑、可解释。

## 15. 当前不应扩写成结论的内容

- 不应声称当前已完成人工标注风险、负荷、事件复盘真值验证。
- 不应声称 public adapter 证明私有双流主线泛化成功。
- 不应声称 public fusion ablation 证明 public context proxy 等价于私有航电流。
- 不应声称 P30/P31 证明 Chronaris 在全部任务上全面胜出。
- 不应声称 T1/T2/T3 是论文最终任务，只能作为 private proxy。
- 不应声称原始 runtime upstream input 已 native exact schema。
- 不应声称完整 6DoF rotation residual 已启用。
- 不应声称当前覆盖所有架次或全部飞行数据。

## 16. 后续写中期报告的建议材料顺序

1. 研究背景与问题：从 `../requirements/SPEC.md` 提炼。
2. 技术路线：从 `../implementation/TASKS.md` 的总路线和 Stage 结构提炼。
3. 数据与样本：引用本清单第 3 节。
4. 方法实现：按 Stage H、Phase C、Phase D/E/F、P10-P18 展开。
5. 实验结果：引用第 4-12 节表格与图。
6. 边界与风险：引用 [boundaries-and-risks-2026-06-13.md](boundaries-and-risks-2026-06-13.md)。
7. 后续计划：围绕文献综述、模板适配、native exact schema、可用角速度字段、新 sortie 扩展和人工/专家标注计划展开。
