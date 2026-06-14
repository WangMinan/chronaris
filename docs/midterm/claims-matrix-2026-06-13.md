# Chronaris 中期 Claims Matrix

更新时间：2026-06-14

本矩阵用于写中期报告、PPT 和后续论文综述时核对表述强度。每条 claim 都应同时满足：有证据路径、有证据层级、有禁止表述。

| claim | 可写强度 | evidence_layer | 主要证据 | 禁止表述 |
| --- | --- | --- | --- | --- |
| 仓库已完成从私有 MySQL/InfluxDB 到 Stage H 双流 view 的样本组织 | 强 | data_contract / stage_h | `docs/artifacts/stage_h/stage-h-closure-2026-04-27.md`; `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json`; `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json` | 全量数据集工程化完成 |
| 当前中期私有样本覆盖 2 个 sortie、3 个双流 view、111 个窗口样本 | 强 | private_stage_h | `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_summary.json`; `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json` | 覆盖全部架次和全部飞行数据 |
| Phase C 已完成真实 Stage H weak-label 联合训练 | 强 | thesis_weak_label | `docs/artifacts/stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md`; `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_checkpoint.pt` | 人工真值风险/负荷/事件任务已完成 |
| risk_proxy/workload_proxy/event_replay_tag 可用于验证论文任务原型 | 中强 | thesis_weak_label | `multitask_summary.json`; `stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r3-resume.md` | 三个任务就是最终人工标注任务 |
| P11 已完成 bounded proxy 与 live_influx stable/resume sweep | 强 | thesis_weak_label | `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json`; `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/partial_summary.json` | 4 组合 live_influx 大网格全部完成 |
| P11 对更大 live_influx 尝试没有伪造成成功，保留 partial blocker | 强 | reproducibility_boundary | `partial_summary.json`; `run.log`; `blocked_at_run_index=3` | blocker 已经被解决或可以忽略 |
| chronaris_opt 在 T1/T2/T3 private proxy 上表现强 | 强但限域 | private_proxy | `docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/chronaris_opt_component_ablation.json`; `docs/artifacts/stage_i/stage-i-private-component-ablation-20260607T-stage-i-evidence-closure-r2-private-proxy.md` | T1/T2/T3 是论文最终任务 |
| 因果掩码、时间残差、任务头对 private proxy 有诊断价值 | 中强 | private_proxy | `chronaris_opt_component_ablation.json`; `chronaris_opt_component_ablation.png` | 这些组件已在公开数据上证明泛化 |
| UAB/NASA 公开数据可作为 adapter/calibration 支撑 | 中强 | public_adapter_calibration | `public_adapter_calibration_summary.json`; `public_transfer_boundary_summary.json` | 公开数据证明私有航空双流主线 fully closed |
| public 第二模态是 context proxy，不是真实航电流 | 强 | transfer_boundary | `docs/artifacts/assets/stage_i_public_transfer_boundary/20260607T-stage-i-evidence-closure-r2-transfer-boundary/public_transfer_boundary_summary.json` | public 数据具有与私有航电流等价的模态结构 |
| rigid_body translation + vertical 已启用 | 强 | rigid_body_support | `docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/rigid_body_ablation_summary.json` | 完整 6DoF 已启用 |
| rotation disabled 是字段缺失诊断，不是忘做 | 强 | rotation_diagnostics | `docs/artifacts/assets/stage_i_rotation_audit/20260607T-stage-i-rotation-audit-r2/rigid_body_rotation_audit_summary.json` | rotation 已完整验证 |
| semantic event support 已覆盖 3 个双流 view | 强 | semantic_support | `docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`; `stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md` | 已完成专家语义事件标注体系 |
| runtime replay 已能输出 prediction 与 semantic support case | 强 | runtime_replay | `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/runtime_inference_summary.json`; `runtime_semantic_case.csv` | 已达到生产级在线服务 |
| runtime native input 当前是 aligned schema | 强 | runtime_service_contract | `runtime_service_smoke_summary.json`; `runtime_schema_contract.json` | native exact 已完成 |
| canonical payload 可达到 exact service contract | 强但限域 | runtime_service_contract | `canonical_runtime_samples.jsonl`; `runtime_schema_contract.json`; `stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r2-contract.md` | 原始上游输入 exact |
| P16/P18 已产出中期说明图表 | 强 | thesis_materials | `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/figure_manifest.json`; `stage-i-thesis-materials-20260613T-stage-i-thesis-materials-r2-p18.md` | 图表覆盖所有可能实验 |
| P20 已形成 DeepSeek 在线时序数据预处理方案 | 计划 | llm_preprocessing_plan | `docs/implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md` | DeepSeek 版本已经代码实现；已经调用 API；OpenAI 是默认 provider；LLM 输出等同人工真值 |
| 当前仓库具备中期报告材料基础 | 强 | cross_layer_index | `docs/STATE.md`; `docs/artifacts/ARTIFACTS.md`; `docs/midterm/midterm-fact-sheet-2026-06-13.md` | 毕业论文最终实验已全部完成 |

## 建议正文映射

| 报告章节 | 推荐使用 claims |
| --- | --- |
| 研究背景与问题 | 数据接入、双流 view、异构时序对齐需求 |
| 研究内容与技术路线 | Stage H、Phase C、Phase D/E/F、P10-P18 |
| 已完成工作 | Phase C 训练、P11 sweep、P12-P15、P16-P18 |
| 实验结果与分析 | P11、P12、P13/P14、rigid_body、semantic、runtime |
| 存在问题 | weak-label、partial live grid、native aligned、rotation disabled、样本规模 |
| 下一步计划 | 文献综述、更多 sortie/view、专家复核、native exact、角速度字段核验、DeepSeek 在线时序预处理 |

## 文献检索映射

后续搜索论文时，建议按 claim 反向组织文献：

| 需要文献支撑的点 | 检索方向 |
| --- | --- |
| 异构多源时序对齐 | multimodal time series alignment, heterogeneous sensor temporal alignment |
| 连续潜态建模 | continuous latent state model, neural state space model, temporal representation learning |
| 物理一致性约束 | physics-informed temporal model, kinematic constraint learning, rigid body dynamics residual |
| 因果掩码和跨模态融合 | causal attention mask, cross-modal fusion, causal multimodal transformer |
| 飞行员风险/负荷/事件复盘 | pilot workload assessment, physiological monitoring aviation, flight event replay |
| weak-label 与代理任务 | weak supervision, proxy labels, self-supervised event labeling |
| runtime schema contract | model serving schema validation, ML data contract, online inference contract |
| LLM 辅助时序数据预处理 | LLM for time series preprocessing, language model assisted imputation, time series natural language alignment, automated labeling with LLM |
