# Stage I UAB Baseline

- 生成时间：2026-04-29 02:51:43 UTC
- 机器产物根目录：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260429T000000Z-stage-i-phase0-1-uab`

## 数据摘要

- 样本总数：`87`
- 主线样本：`82`
- 辅助 flight 样本：`5`
- 特征总数：`416`（EEG `404` / ECG `12`）

## 客观任务标签分类

### heat_the_chair

- 最优模型：`logistic_regression`
- macro-F1：`0.6173`
- balanced accuracy：`0.6176`
- 样本数：`34`

### n_back

- 最优模型：`random_forest_classifier`
- macro-F1：`0.4966`
- balanced accuracy：`0.5000`
- 样本数：`48`

## 主观负荷回归

### heat_the_chair

- 最优模型：`random_forest_regressor`
- MAE：`1.2519`
- RMSE：`1.4664`
- Spearman：`0.2645`

### n_back

- 最优模型：`random_forest_regressor`
- MAE：`3.6536`
- RMSE：`4.5161`
- Spearman：`0.1610`

## Flight 辅助摘要

- flight session 数：`5`
- 理论难度分布：`{"1": 1, "2": 1, "3": 1, "4": 1}`
- 感知难度均值：`1.5952`

## 图表

- `objective_confusion_matrix_heat_the_chair`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260429T000000Z-stage-i-phase0-1-uab/plots/objective_confusion_matrix_heat_the_chair.png`
- `objective_confusion_matrix_n_back`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260429T000000Z-stage-i-phase0-1-uab/plots/objective_confusion_matrix_n_back.png`
- `subjective_regression_heat_the_chair`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260429T000000Z-stage-i-phase0-1-uab/plots/subjective_regression_heat_the_chair.png`
- `subjective_regression_n_back`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260429T000000Z-stage-i-phase0-1-uab/plots/subjective_regression_n_back.png`

