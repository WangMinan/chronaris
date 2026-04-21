# Stage E Closure - 2026-04-21

## 1. 结论

阶段 E 已完成，满足进入阶段 F 的开发条件。

依据：

1. 真实环境下完成 `none` 与 `zscore_train` 两组实跑对照。
2. 两组运行均完成 `train/validation/test`，并导出中间态与诊断产物。
3. 阶段 E 默认阈值模板评估均为 `PASS`。
4. 阶段 E 相关单测与 runtime 测试通过。

## 2. 收口实跑记录

主报告：

- `docs/reports/alignment-preview-stage-e-closure-2026-04-21.md`

核心结果（seed=`20260421`）：

| 配置 | train total | validation total | test total | threshold verdict |
| --- | ---: | ---: | ---: | :---: |
| `none` | `2.045169` | `2.036023` | `2.036023` | `PASS` |
| `zscore_train` | `1.559523` | `1.460789` | `1.561467` | `PASS` |

阶段 F 默认建议采用：`zscore_train`（test total 更低）。

## 3. 阈值模板（默认）

当前默认阈值配置：

- `min_sample_count = 1`
- `min_mean_projection_cosine = 0.65`
- `enforce_min_projection_cosine = false`
- `max_mean_projection_l2_gap = 0.25`
- `max_mean_projection_l2_ratio_deviation = 0.30`
- `max_projection_cosine_cv = 0.15`
- `max_projection_l2_gap_cv = 0.25`

说明：

- `min projection cosine` 单点门槛默认不强制，避免少量异常点导致整轮误判。
- 若阶段 F 需要更严格门槛，可通过 `--threshold-enforce-min-cosine` 启用。

## 4. 产物与入口

脚本入口：

- `python scripts/run_stage_e_relative_preview.py --compare-with-zscore-train --report-path docs/reports/alignment-preview-stage-e-closure-2026-04-21.md`

关键产物：

- 对照主报告：`docs/reports/alignment-preview-stage-e-closure-2026-04-21.md`
- 模型 checkpoint：
  - `docs/reports/assets/alignment-preview-stage-e-closure-2026-04-21-none/alignment_model_checkpoint.pt`
  - `docs/reports/assets/alignment-preview-stage-e-closure-2026-04-21-zscore_train/alignment_model_checkpoint.pt`
- 诊断 JSON/CSV：
  - `docs/reports/assets/alignment-preview-stage-e-closure-2026-04-21-none/`
  - `docs/reports/assets/alignment-preview-stage-e-closure-2026-04-21-zscore_train/`
- 可视化图像：同上 assets 目录

## 5. 测试闭环

执行命令：

- `CHRONARIS_ENABLE_NUMPY_RUNTIME_TESTS=1 CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 python -m unittest discover -s tests -p 'test_*.py'`

结果：

- `Ran 63 tests ... OK`

附注：

- 本轮对 `tests` 做了域级合并，`test_*.py` 总数收敛到 9 个，保持可读性与定位效率。
