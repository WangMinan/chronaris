# Chronaris 产物索引

更新时间：2026-06-21

## 1. 目录定位

本目录用于组织报告、图、CSV、JSON、checkpoint、manifest 等可引用产物。

LFS 配额清理记录：[cleanup/20260619-lfs-docs-prune.md](cleanup/20260619-lfs-docs-prune.md)。该清理删除了历史 raw replay / prepared bundle / 大型 JSONL 与 NPZ 载荷，保留报告、summary、schema contract 和中期写作入口。

## 2. 阶段产物

阶段产物按 [../implementation/TASKS.md](../implementation/TASKS.md) 的阶段划分：

- [stage/stage-a/](stage/stage-a/)：仓库初始化与最小设计。
- [stage/stage-b/](stage/stage-b/)：真实元信息与数据访问接入。
- [stage/stage-c/](stage/stage-c/)：统一样本组织与数据核验。
- [stage/stage-d/](stage/stage-d/)：数据集工程化与批量构建。
- [stage/stage-e0/](stage/stage-e0/)：单架次最小训练输入适配。
- [stage/stage-e/](stage/stage-e/)：双流连续潜态对齐。
- [stage/stage-f/](stage/stage-f/)：物理一致性约束。
- [stage/stage-g/](stage/stage-g/)：因果掩码与语义融合。
- [stage/stage-h/](stage/stage-h/)：标准化融合特征导出。
- [stage/stage-i/](stage/stage-i/)：典型任务评测、论文证据与运行时。

## 3. 中期答辩产物

- [mid-term/](mid-term/)：中期答辩证据包、图件、指标表和运行日志。
- [../midterm/](../midterm/)：中期报告写作事实清单、边界风险说明和 claims matrix。

当前中期主入口：

- [stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md](stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md)
- [stage_i/stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md](stage_i/stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md)
- [stage_i/stage-i-private-leakage-safe-ablation-20260619T-stage-i-leakage-safe-ablation-r2.md](stage_i/stage-i-private-leakage-safe-ablation-20260619T-stage-i-leakage-safe-ablation-r2.md)
- [../midterm/midterm-fact-sheet-2026-06-13.md](../midterm/midterm-fact-sheet-2026-06-13.md)
- [../midterm/boundaries-and-risks-2026-06-13.md](../midterm/boundaries-and-risks-2026-06-13.md)
- [../midterm/claims-matrix-2026-06-13.md](../midterm/claims-matrix-2026-06-13.md)
- [../implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md](../implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md)
- [../midterm/llm-preprocessing-comparison-plan-2026-06-14.md](../midterm/llm-preprocessing-comparison-plan-2026-06-14.md)
- [../midterm/llm-preprocessing-comparison-summary-2026-06-14.md](../midterm/llm-preprocessing-comparison-summary-2026-06-14.md)

## 4. 当前最常引用产物

- Stage H 收口：[stage_h/stage-h-closure-2026-04-27.md](stage_h/stage-h-closure-2026-04-27.md)
- Stage I 历史公开收口：[stage_i/stage-i-closure-2026-04-30.md](stage_i/stage-i-closure-2026-04-30.md)
- Stage I thesis weak-label evidence：[stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md](stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md)
- Stage I evidence runner r2：[stage_i/stage-i-evidence-closure-20260607T-stage-i-evidence-closure-r2.md](stage_i/stage-i-evidence-closure-20260607T-stage-i-evidence-closure-r2.md)
- Stage I bounded weak-label sweep r2：[stage_i/stage-i-thesis-weak-label-multitask-sweep-20260607T-stage-i-evidence-closure-r2-multitask.md](stage_i/stage-i-thesis-weak-label-multitask-sweep-20260607T-stage-i-evidence-closure-r2-multitask.md)
- Stage I live weak-label stable resume：[stage_i/stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r3-resume.md](stage_i/stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r3-resume.md)
- Stage I thesis materials r6 report figure polish：[stage_i/stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md](stage_i/stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md)
- Stage I leakage-safe private proxy ablation r2：[stage_i/stage-i-private-leakage-safe-ablation-20260619T-stage-i-leakage-safe-ablation-r2.md](stage_i/stage-i-private-leakage-safe-ablation-20260619T-stage-i-leakage-safe-ablation-r2.md)
- Stage I public mainline：[stage_i/stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md](stage_i/stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md)
- Stage I support ablation：[stage_i/stage-i-ablation-support-20260506T120000Z-stage-i-support.md](stage_i/stage-i-ablation-support-20260506T120000Z-stage-i-support.md)
- Stage I rigid-body r2：[stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md](stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md)
- Stage I semantic support r2：[stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md](stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md)
- Stage I runtime service r2：[stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md](stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md)
- Private optimized package summary：[private-optimization-summary-20260607T-stage-i-private-opt-package-r2.md](private-optimization-summary-20260607T-stage-i-private-opt-package-r2.md)
- Stage I private component ablation r2：[stage_i/stage-i-private-component-ablation-20260607T-stage-i-evidence-closure-r2-private-proxy.md](stage_i/stage-i-private-component-ablation-20260607T-stage-i-evidence-closure-r2-private-proxy.md)
- Stage I public adapter calibration r2：[stage_i/stage-i-public-adapter-calibration-20260607T-stage-i-evidence-closure-r2-public-adapter.md](stage_i/stage-i-public-adapter-calibration-20260607T-stage-i-evidence-closure-r2-public-adapter.md)
- Stage I public transfer boundary r2：[stage_i/stage-i-public-transfer-boundary-20260607T-stage-i-evidence-closure-r2-transfer-boundary.md](stage_i/stage-i-public-transfer-boundary-20260607T-stage-i-evidence-closure-r2-transfer-boundary.md)
- Stage I rotation audit r3 figure refresh：[stage_i/stage-i-rigid-body-rotation-audit-20260619T-stage-i-rotation-audit-r3-figure-refresh.md](stage_i/stage-i-rigid-body-rotation-audit-20260619T-stage-i-rotation-audit-r3-figure-refresh.md)
- Stage I rotation audit r2 历史入口：[stage_i/stage-i-rigid-body-rotation-audit-20260607T-stage-i-rotation-audit-r2.md](stage_i/stage-i-rigid-body-rotation-audit-20260607T-stage-i-rotation-audit-r2.md)
- Stage I runtime service smoke r2 contract：[stage_i/stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r2-contract.md](stage_i/stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r2-contract.md)
- Midterm fact sheet：[../midterm/midterm-fact-sheet-2026-06-13.md](../midterm/midterm-fact-sheet-2026-06-13.md)
- Midterm boundaries and risks：[../midterm/boundaries-and-risks-2026-06-13.md](../midterm/boundaries-and-risks-2026-06-13.md)
- Midterm claims matrix：[../midterm/claims-matrix-2026-06-13.md](../midterm/claims-matrix-2026-06-13.md)
- P20 DeepSeek 在线时序数据预处理计划：[../implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md](../implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md)
- P20 DeepSeek 在线时序数据预处理 run：[stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md](stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md)
- P21 LLM preprocessing 对比实验计划：[../midterm/llm-preprocessing-comparison-plan-2026-06-14.md](../midterm/llm-preprocessing-comparison-plan-2026-06-14.md)
- P21 LLM preprocessing 对比实验 run：[stage_i/stage-i-llm-comparison-20260614T-stage-i-p21-llm-comparison-r1.md](stage_i/stage-i-llm-comparison-20260614T-stage-i-p21-llm-comparison-r1.md)
- P21 中期结果摘要：[../midterm/llm-preprocessing-comparison-summary-2026-06-14.md](../midterm/llm-preprocessing-comparison-summary-2026-06-14.md)
- P21 执行 prompt（历史追溯）：[../implementation/notes/goal-prompt-stage-i-p21-llm-comparison-2026-06-14.md](../implementation/notes/goal-prompt-stage-i-p21-llm-comparison-2026-06-14.md)
- 当前 `chronaris_opt` package：[assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_package.json](assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_package.json)
- 当前 evidence manifest：[assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/evidence_manifest.json](assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/evidence_manifest.json)
- 当前 thesis figure manifest：[assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_manifest.json](assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_manifest.json)
- 当前 thesis figure quality audit：[assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_quality_audit.csv](assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_quality_audit.csv)
- 当前 leakage-safe private ablation summary：[assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/ablation_summary.json](assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/ablation_summary.json)
- 当前 runtime service smoke summary：[assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_service_smoke_summary.json](assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_service_smoke_summary.json)
- 当前 live weak-label stable resume summary：[assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json](assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json)
- 当前 live weak-label partial summary：[assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/partial_summary.json](assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/partial_summary.json)
- 当前 runtime schema contract：[assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json](assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json)
- 当前 docs LFS 清理记录：[cleanup/20260619-lfs-docs-prune.md](cleanup/20260619-lfs-docs-prune.md)
- P20/P21 LLM runtime case 输入表：[assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv](assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv)
- 当前 P21 LLM comparison summary：[assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/llm_comparison_summary.json](assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/llm_comparison_summary.json)

## 5. 引用规则

- 引用当前状态先看 [../STATE.md](../STATE.md)，不要从历史报告倒推当前阶段。
- 引用执行入口先看 [../implementation/TASKS.md](../implementation/TASKS.md)。
- 引用论文能力要求先看 [../requirements/SPEC.md](../requirements/SPEC.md)。
- 历史报告可以引用，但必须说明是历史快照、公开适配器证据、私有代理证据还是 thesis weak-label evidence。
- 2026-06-19 起，若历史报告提到已清理的 raw JSONL / NPZ / prepared bundle，应按 [cleanup/20260619-lfs-docs-prune.md](cleanup/20260619-lfs-docs-prune.md) 处理：引用保留的 summary/report，复跑时重新生成 raw payload。
- 2026-06-19 深度清理后，P20 LLM preprocessing r1、P11 live r2、rigid-body r1、runtime replay r1、semantic support r1、runtime service smoke r1 和 thesis materials r4 仅保留在 git 历史中；当前文档入口统一使用本索引列出的 r3/r6/r2-contract 稳定产物。
- P20 DeepSeek 已生成小样本真实切片 run 产物：`docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/` 和 `docs/artifacts/stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md`。引用时必须写成 LLM preprocessing context / rule review / semantic hints / runtime explanation / bounded slicing，不得写成人工真值、核心因果证据或原始全量高频时序外发。
- P21 LLM comparison 已生成 A0-A4 对比产物：`docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/`、`docs/artifacts/stage_i/stage-i-llm-comparison-20260614T-stage-i-p21-llm-comparison-r1.md` 和 `docs/midterm/llm-preprocessing-comparison-summary-2026-06-14.md`。引用时必须保留 `label_unchanged=true`、semantic hints whitelist、runtime explanation bounded subset、`human_review_completed=false` 四个边界。
- `stage_i_private_component_ablation` 的历史满分结果与 `stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/` 是不同协议。论文实验章节优先引用 `protocol=leakage_safe_v1` 的 r2 消融；历史结果只能作为 private proxy 历史对照，不得与防泄漏结果混成同一柱状结论。
