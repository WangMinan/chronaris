# Stage I Reports

更新时间：2026-05-08

当前顶层只保留以下两类文档：

- 当前主线与主判断
  - `stage-i-closure-2026-04-30.md`
  - `stage-i-public-mainline-20260508T091000Z-stage-i-public-mainline-uab-heat-specialist-r1.md`
  - `stage-i-public-opt-20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1.md`
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
  - `public opt` 旧 baseline、早期 UAB torch 迭代快照与 `public_fusion` round 1 历史报告

引用规则：

- 若要回答“当前公开主线是什么”，优先看 `stage-i-public-mainline-20260508T091000Z-stage-i-public-mainline-uab-heat-specialist-r1.md`。
- 若要回答“公开 deep frozen 对照是什么”，优先看 `stage-i-deep-comparison-full-loso-2026-05-01.md`。
- 若要回答“当前 UAB torch 迭代做到哪一步”，优先看 `stage-i-public-opt-20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1.md`；这轮 `heat_residual_correction` 已把 `heat_the_chair` 推到 `RMSE=1.4630 / MAE=1.1594`，但仍未超过 legacy `1.4568` RMSE gate。
- `stage-i-public-opt-20260507T133000Z-stage-i-public-opt-uab-torch-mainline-r2.md` 与 `stage-i-public-opt-20260507T141500Z-stage-i-public-opt-uab-torch-sessionpooled-r4.md` 已下沉到 `archive/public_history/`，不再留在当前层级。
- 若继续执行公开主线实验，新的运行证据必须同时包含 `run.log` 与 `progress.json`；UAB 新默认是 `heat_the_chair` heat-only `heat_specialist` CUDA route，NASA 只做 prepared asset 校验与可选 GPU confirm。
- 当前公开主线仍冻结为 `NASA closed, UAB partial`；`heat_specialist` 已完成真实 full LOSO 验证，但 best-of 仍由 legacy `physiology_persistence` 保持，不能写成 UAB closure。
- 若只是回溯阶段启动期或旧 baseline，请到 `archive/`，不要把历史快照当成当前状态文档。
