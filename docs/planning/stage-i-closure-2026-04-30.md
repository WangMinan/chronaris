# 阶段 I 收口记录（2026-04-30）

## 1. 收口结论

阶段 I 已完成收口。

本轮收口完成了四个 gate：

1. `UAB window_v2` workload 主线已在真实数据上完成 `Leave-One-Subject-Out` 分类/回归与三路模态消融。
2. `NASA CSM` attention-state 主线已复用已完成资产纳入统一 closure，并保留 `benchmark_only / loft_only / combined` 三组结果。
3. `UAB session vs window` 与 `cross-dataset window count` 对比图已生成，明确说明 window-scale 主线与历史 session-level 参照的关系。
4. `tests.test_stage_i_pipeline` 与全量 `discover` 已在 `chronaris` 环境下通过，`AGENTS.md / docs/README.md / coding-roadmap.md / stage-i-closure` 状态已同步。

## 2. 收口策略

- 最终 closure run 根：`docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/`
- UAB 资产来源：此前的 prepared dataset 与本轮真跑结果已吸收到 `docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/uab_window/`
- NASA 资产来源：此前已完成的 attention baseline 资产已吸收到 `docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/nasa_attention/`
- closure 组装方式：`run_stage_i_phase3.py --reuse-existing-artifacts`

这样做的目的不是继续信任半成品 run 根，而是把“UAB 已补齐 + NASA 已完成”的两套资产收束到一个新的最终事实源。

## 3. 主结果摘要

UAB `window_v2` workload：

- `n_back` 最优客观模型：`logistic_regression`
  - `macro-F1=0.3478`
  - `balanced_accuracy=0.3625`
- `heat_the_chair` 最优客观模型：`linear_svc`
  - `macro-F1=0.5405`
  - `balanced_accuracy=0.5405`
- `n_back` 最优主观模型：`linear_svr`
  - `RMSE=10.2234`
  - `MAE=4.7987`
- `heat_the_chair` 最优主观模型：`linear_svr`
  - `RMSE=1.8639`
  - `MAE=1.2785`

NASA CSM attention-state：

- `benchmark_only`：`macro-F1=0.4642`
- `loft_only`：`macro-F1=0.3723`
- `combined`：`macro-F1=0.3741`

UAB session vs window：

- 客观任务两组都低于历史 session-level 结果，但 window-scale 保留了 subject-wise 评测、模态消融和跨数据集统一 contract
- 主观回归中 `ecg_only` 在两组任务上都优于 `eeg_ecg`，说明当前 window-scale regression 对模态裁剪较敏感，后续若扩展模型应优先从回归稳健性切入

## 4. 收口实跑资产

主运行：

- run root：`docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/`
- closure summary：`docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/closure_summary.json`
- closure report：`docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/stage_i_phase3_closure_report.md`
- 主报告：`docs/reports/stage-i-closure-2026-04-30.md`

阶段主报告：

- `Phase 1`：`docs/reports/stage-i-uab-baseline-2026-04-29.md`
- `Phase 2`：`docs/reports/stage-i-case-study-phase2-2026-04-29.md`
- `Phase 3 / UAB window`：`docs/reports/stage-i-uab-window-baseline-2026-04-29.md`
- `Phase 3 / NASA attention`：`docs/reports/stage-i-nasa-attention-baseline-2026-04-29.md`

图表：

- `phase3_window_counts.png`
- `uab_session_vs_window_objective.png`
- `uab_session_vs_window_subjective.png`
- UAB / NASA 各自的 subset count、label distribution、ablation、confusion matrix、regression scatter 图

## 5. 测试闭环

执行命令：

- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_pipeline`
- `CHRONARIS_ENABLE_NUMPY_RUNTIME_TESTS=1 CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest discover -s tests -p 'test_*.py'`

结果：

- 定向测试：`Ran 7 tests ... OK`
- 全量测试：`Ran 100 tests ... OK`

附注：

- `sklearn 1.8.0` 下出现的 `n_jobs` future warning 与 `SimpleImputer` missing-feature warning 不影响当前收口判定；本轮以真实结果、可复现命令和 green tests 为 gate 依据。

## 6. 后续边界

- 阶段 I 已收口，默认冻结 `Phase 0 + Phase 1 + Phase 2 + Phase 3` 的主线事实。
- `20251110_单01_ACT-2_涛_J20_26#01` 仍保持 vehicle-only partial-data，不提升为双流 Stage H / Stage I 样本。
- MATB-II / DS007262 / EEGMAT / Braindecode / EEGNet / MulT / ContiFormer 等只作为阶段 I 之后的增强实验，不再计入当前阶段收口条件。
