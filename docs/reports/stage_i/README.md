# Stage I Reports

更新时间：2026-05-09

当前顶层只保留以下几类文档：

- 当前中期证据整编
  - `stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md`
- 当前主线与主判断
  - `stage-i-closure-2026-04-30.md`
  - `stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md`
  - `stage-i-public-opt-20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1.md`
  - `stage-i-public-opt-20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1.md`
  - `stage-i-public-opt-20260506T165558Z-stage-i-public-opt-uab-torch-gpu.md`
  - `stage-i-public-opt-20260506T161500Z-stage-i-public-opt-nasa-round1.md`
  - `stage-i-public-fusion-screen-20260506T-stage-i-public-fusion-screen-round2.md`
- 当前仍直接被论文证据或 frozen 对照引用的主报告
  - `stage-i-case-study-phase2-2026-04-29.md`
  - `stage-i-deep-comparison-full-loso-2026-05-01.md`
  - `stage-i-alignment-support-20260506T120000Z-stage-i-support.md`
  - `stage-i-causal-support-20260506T120000Z-stage-i-support.md`
  - `stage-i-ablation-support-20260506T120000Z-stage-i-support.md`
  - `stage-i-runtime-demo-20260506T165435Z-stage-i-runtime-demo.md`
  - `stage-i-anchor-20260506T165435Z-stage-i-anchor.md`

已归档历史快照：

- `archive/baselines/`
  - Phase 1 / Phase 3 的旧 baseline 主报告
- `archive/deep_history/`
  - 早期 deep comparison / thesis-support 历史快照
- `archive/public_history/`
  - `public opt` 旧 baseline、`20260507/20260508` 的 UAB torch 迭代 / mainline 历史快照、`public_fusion` round 1、以及第一版中期证据包

引用规则：

- 若要回答“当前中期整编证据包是什么”，优先看 `stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md`。
- 若要回答“当前公开主线是什么”，优先看 `stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md`；当前状态仍为 `public opt closed`。
- 若要回答“NASA public fusion 这轮有没有正向结果”，优先看 `stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md` 与其引用的 `docs/reports/assets/stage_i_midterm_runtime/20260509T065500Z-stage-i-public-fusion-nasa-full-confirm/`；这是补充证据，不代表公开主线切换。
- 若要回答“公开 deep frozen 对照是什么”，优先看 `stage-i-deep-comparison-full-loso-2026-05-01.md`。
- 若要回答“UAB 为什么能写成 public opt closed”，优先看 `stage-i-public-opt-20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1.md`；`target_prior_median` 仍只能写成 `uab_public_adapter` / calibration baseline。
- 若只是追溯 `20260507/20260508` 期间的 UAB torch 迭代、旧 mainline 判定或第一版中期包，请进入 `archive/public_history/`，不要把这些快照当成当前状态文档。
