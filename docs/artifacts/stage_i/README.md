# Stage I Artifacts Index

更新时间：2026-06-19

本目录保存当前 Stage I 报告集合；当前 AI coding 入口请先看 [../ARTIFACTS.md](../ARTIFACTS.md) 与 [../stage/stage-i/README.md](../stage/stage-i/README.md)。

## 当前主入口

- 中期证据整编：`stage-i-midterm-20260607T-stage-i-midterm-r3.md`
- 主动 evidence runner：`stage-i-evidence-closure-20260607T-stage-i-evidence-closure-r2.md`
- bounded weak-label sweep：`stage-i-thesis-weak-label-multitask-sweep-20260607T-stage-i-evidence-closure-r2-multitask.md`
- live weak-label stable resume：`stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r3-resume.md`
- thesis materials r5 leakage-safe refresh：`stage-i-thesis-materials-20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh.md`
- leakage-safe private proxy ablation r2：`stage-i-private-leakage-safe-ablation-20260619T-stage-i-leakage-safe-ablation-r2.md`
- runtime service smoke r2 contract：`stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r2-contract.md`
- DeepSeek LLM preprocessing r3 sliced：`stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md`
- LLM preprocessing comparison r1：`stage-i-llm-comparison-20260614T-stage-i-p21-llm-comparison-r1.md`
- 中期事实清单：`../../midterm/midterm-fact-sheet-2026-06-13.md`
- 中期边界说明：`../../midterm/boundaries-and-risks-2026-06-13.md`
- P21 中期结果摘要：`../../midterm/llm-preprocessing-comparison-summary-2026-06-14.md`
- docs LFS 清理记录：`../cleanup/20260619-lfs-docs-prune.md`
- 历史 Stage I closure：`stage-i-closure-2026-04-30.md`
- Phase C thesis weak-label evidence：`thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md`
- 当前公开主线：`stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md`
- UAB robust-prior adapter：`stage-i-public-opt-20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1.md`
- NASA enhanced round 1：`stage-i-public-opt-20260506T161500Z-stage-i-public-opt-nasa-round1.md`
- UAB torch auto-cuda confirm：`stage-i-public-opt-20260506T165558Z-stage-i-public-opt-uab-torch-gpu.md`
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
- 旧 P16/P18 thesis materials 图包已由 r5 接管为当前入口；r4 runtime case refresh 已从 docs 产物目录清理，`../assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv` 仅作为 P20/P21 LLM preprocessing 历史输入表。
- runtime service smoke r1、runtime replay r1、semantic support r1、rigid-body r1、P11 live r2 和 P20 preprocessing r1 仅保留在 git 历史；当前入口使用 r2/r3/r5 产物。

历史 P11 live 首轮：

- `20260613T-stage-i-p11-live-influx-r1` 的子运行和 blocker log 被 r3 resume summary 引用，因此保留为可复现依赖；`p11-live-influx-r2` 汇总已由 r3 resume 接管并清理。

## 引用规则

- `public opt closed` 仍是公开支撑证据，但 UAB `target_prior_median` 只能写成 `public adapter / calibration evidence`。
- UAB/NASA 第二模态统一写成 `context proxy / public adapter evidence`。
- 私有分支需要区分 `T1/T2/T3 = private proxy benchmark / proxy evidence` 与 `risk_proxy / workload_proxy / event_replay_tag = thesis weak-label evidence`。
- `stage_i_private_leakage_safe_ablation` r2 是 `protocol=leakage_safe_v1` 的新增协议；历史 private component ablation 只能作为历史代理诊断，不与 r2 防泄漏结果混成同一实验结论。
- P20/P21 DeepSeek LLM preprocessing 只能写成 preprocessing context、rule review、whitelisted semantic hints、runtime explanation、bounded comparison 和 pending human review packet；不能写成人工真值、OpenAI 默认接入、核心因果证据或人工验证完成。
- 历史 archive 只用于追溯，不作为当前状态入口。
- 历史 raw replay、prepared bundle 和大型 manifest 已按 `../cleanup/20260619-lfs-docs-prune.md` 从 docs/LFS 中清理；引用旧实验时优先看报告、summary、plots 和当前索引。
