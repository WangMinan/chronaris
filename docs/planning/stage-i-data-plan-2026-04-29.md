# Stage I 数据计划

更新时间：2026-04-30

说明：

- 本文档保留为阶段 I 启动期的数据计划与开工顺序记录。
- 当前阶段状态与收口结果已由 `docs/planning/stage-i-closure-2026-04-30.md` 接管。

## 0. 总分期

阶段 I 最终按 `4` 个 phase 收口：

1. `Phase 0`：公开数据 contract、环境依赖、评测/脚本入口固化
2. `Phase 1`：UAB 主数据集双轨 baseline
3. `Phase 2`：Stage H 真实双流资产 case study
4. `Phase 3`：NASA CSM 第二公开数据集、对比/消融补齐与阶段 I 收口

当前进度：

- `Phase 0`：已完成
- `Phase 1`：已完成
- `Phase 2`：已完成
- `Phase 3`：已完成

当前不纳入阶段 I 主线完成条件：

- MATB-II / DS007262 / EEGMAT 等补充数据
- Braindecode / EEGNet / 更重深度模型
- MulT / ContiFormer 这类更重对照模型

这些内容默认在阶段 I 收口后再评估是否作为增强实验继续推进。

## 1. 已完成的 Phase 0 + Phase 1

当前已完成：

1. 固化公开数据的 Stage I task manifest contract
2. 固化 UAB 数据集适配与 session 级特征导出
3. 跑通“客观任务标签分类 + 主观负荷回归”双轨 baseline
4. 产出可复用机器资产、主报告和测试闭环

## 2. 已固化的数据范围

- 数据根目录：`/home/wangminan/dataset/chronaris`
- 主数据集：`uab_workload_dataset`
- 主线子集：
  - `n_back`：`48` 个 session
  - `heat_the_chair`：`34` 个 session
- 辅助子集：
  - `flight_simulator`：`5` 个 session
- Stage I manifest 当前总条目数：`87`
- 训练角色：
  - `primary=82`
  - `auxiliary=5`

当前 contract 固定为“每个有标签 session 产一个样本”，不在 `Phase 0 + Phase 1` 引入密集滑窗监督。

## 3. Phase 0 + Phase 1 落地产物

代码入口：

- `src/chronaris/dataset/stage_i_contracts.py`
- `src/chronaris/dataset/uab_stage_i.py`
- `src/chronaris/features/stage_i_features.py`
- `src/chronaris/evaluation/stage_i_metrics.py`
- `src/chronaris/pipelines/stage_i_baseline.py`
- `scripts/prepare_stage_i_dataset.py`
- `scripts/run_stage_i_baseline.py`

机器资产根目录：

- `docs/reports/assets/stage_i/20260429T000000Z-stage-i-phase0-1-uab/`

关键资产：

- `task_manifest.jsonl`
- `feature_table.parquet`
- `feature_schema.json`
- `dataset_summary.json`
- `objective_metrics.json`
- `subjective_metrics.json`
- `fold_predictions.csv`

## 4. Phase 0 + Phase 1 实跑结果摘要

特征导出：

- 总特征维度：`416`
- EEG 统计特征：`404`
- ECG 统计特征：`12`
- `n_back` 缺失 ECG 的 session 数：`9`

客观任务标签分类：

- `heat_the_chair` 最优：`logistic_regression`
  - `macro-F1=0.6173`
  - `balanced_accuracy=0.6176`
- `n_back` 最优：`random_forest_classifier`
  - `macro-F1=0.4966`
  - `balanced_accuracy=0.5000`

主观负荷回归：

- `heat_the_chair` 最优：`random_forest_regressor`
  - `RMSE=1.4664`
  - `MAE=1.2519`
- `n_back` 最优：`random_forest_regressor`
  - `RMSE=4.5161`
  - `MAE=3.6536`

主报告：

- `docs/reports/stage-i-uab-baseline-2026-04-29.md`

测试闭环：

- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_pipeline`
- `CHRONARIS_ENABLE_NUMPY_RUNTIME_TESTS=1 CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest discover -s tests -p 'test_*.py'`

## 5. Phase 2 实跑结果摘要

Phase 2 当前输入固定为：

- `docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/run_manifest.json`

Phase 2 当前产物根目录：

- `docs/reports/assets/stage_i/20260429T000000Z-stage-i-phase2-case-study/`

Phase 2 当前主报告：

- `docs/reports/stage-i-case-study-phase2-2026-04-29.md`

Phase 2 当前已验证：

- 已纳入 `3` 个真实双流 view，其中 `PASS=2`、`WARN=1`
- 已完成同 sortie 双 pilot 对比：`20251002_单01_ACT-8_翼云_J16_12#01`
- 已完成 `4` 条 bundle-only 路径：
  - `projection_refusion_baseline`
  - `no_event_bias`
  - `no_state_normalization`
  - `vehicle_delta_suppressed`
- `WARN` view 已给出主线解释，不再作为附录处理
- `vehicle_delta_suppressed` 在 `3` 个 view 上都将 `mean_top_event_score` 压到 `0.0`，验证了事件驱动消融路径可执行

Phase 2 关键结论：

- `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` 的 `WARN` 主要来自：
  - `mean_projection_cosine` 偏低
  - `projection_cosine_cv` 偏高
  - `projection_l2_gap_cv` 偏高
- 同 sortie 对比下，该 `WARN` pilot 相比 `PASS` pilot 的：
  - `delta mean projection cosine = -0.170068`
  - `delta projection cosine cv = +0.192775`
  - `delta projection l2 gap cv = +0.392164`

补充说明：

- `hidden-vs-projection cosine` 当前使用 fused `L2 norm profile` 的 cosine，而不是直接逐向量余弦；原因是 Stage H 导出的 hidden fused 为 `96` 维，而 projection rerun baseline 为 `48` 维，二者维度不同。

## 6. 下一步

阶段 I 已于 `2026-04-30` 完成收口。后续默认按下面顺序推进：

1. 保持 `docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/` 作为阶段 I 机器事实根目录。
2. 在不破坏 frozen Stage I contract 的前提下，再考虑 MATB-II / DS007262 / EEGMAT 等补充公开数据。
3. 在阶段 I 收口后，再评估更重的 EEG 深度模型与多模态对照，而不是回退修改已完成的 Phase 3 主线。
