# 2026-06-21 深度清理记录

## 清理范围

本轮清理面向 `src/`、`scripts/` 和 `docs/artifacts/`，目标是删除确认无用或已被当前入口接管的内容，同时保留当前证据链仍直接依赖的历史产物。

## 已清理内容

- 删除 `docs/artifacts/assets/stage_i_thesis_figures/20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh/`。
- 删除 `docs/artifacts/stage_i/stage-i-thesis-materials-20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh.md`。
- 删除空目录：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/runs/`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/runs/`
- 清理本地 Python 编译缓存：`src/**/__pycache__/`、`tests/**/__pycache__/` 和 `third_party/**/__pycache__/`。这些缓存由 `.gitignore` 覆盖，不属于可提交源码。
- 删除本地构建元数据目录 `src/chronaris.egg-info/`。该目录由 `*.egg-info/` ignore 规则覆盖，不属于可提交源码。

## 当前替代入口

- r5 thesis figures 已由 论文图表报告级重绘 r6 接管：
  - `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_manifest.json`
  - `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/table_manifest.json`
  - `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_quality_audit.csv`
  - `docs/artifacts/stage_i/stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md`

## 明确保留内容

- `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/`：r3 stable resume 与 r4 partial summary 仍引用其中两个 live child run、checkpoint 和 blocker log。
- `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv`：DeepSeek/LLM preprocessing 与 comparison 仍把它作为 runtime case 输入表。
- `docs/artifacts/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/` 与 `docs/artifacts/assets/stage_i_public_opt_torch/20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1/`：public adapter calibration、public transfer boundary 和代码常量仍直接引用其 summary。
- `scripts/stage_i/<category>/` canonical CLI 与 `src/chronaris/pipelines/stage_i/<subpackage>/` 源码：本轮引用扫描和测试入口检查未发现可安全删除的 tracked canonical 脚本或源码文件；清理只移除本地编译缓存。

## 引用规则

- 新写中期报告、PPT 或状态文档时，当前 thesis figure 入口统一使用 r6。
- 若需要追溯 r5，只能从 git 历史读取；docs 当前工作树不再保留 r5 图包或 r5 报告。
- 不要把已清理的 r5 路径重新加入 `ARTIFACTS.md`、`stage_i/README.md` 或 `docs/midterm/`。
