# Stage I UAB Baseline

- 生成时间：2026-04-30 01:44:15 UTC
- 机器产物根目录：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/uab_window`

## 数据摘要

- 样本总数：`34748`
- recording 数：`87`
- window 数：`34748`
- split_group 数：`28`
- 特征总数：`214`
- 模态特征数：`{"eeg": 202, "ecg": 12}`
- subset 计数：`{"n_back": 28052, "heat_the_chair": 5440, "flight_simulator": 1256}`
- 标签分布：`{"workload_level": {"0": 11593, "1": 12317, "2": 9582}, "theoretical_difficulty": {"2.0": 434, "4.0": 114, "1.0": 331, "3.0": 121}}`

## 实验设置

- 窗口策略：`5s / 5s`，优先使用 EEG/ECG overlap；完全缺 ECG 的 recording 保留为 EEG-only window。
- 模型策略：window-scale 主线默认收敛到线性 classical baselines；历史 session-level `Phase 1` 结果保留原三模型对照作为参照。
- 任务设置：`n_back / heat_the_chair` 为 workload 主线，`flight_simulator` 仅保留辅助摘要，不参与主收口 gate。
- 评测切分：`Leave-One-Subject-Out`。
- 缺失 ECG window 计数：`{"flight_simulator": 380, "heat_the_chair": 654, "n_back": 9044}`

## 客观/分类主结果

- 主特征集：`eeg_ecg`
- 消融列定义：`{"eeg_ecg": 214, "eeg_only": 202, "ecg_only": 12}`

### n_back

- 最优模型：`logistic_regression`
- macro_f1：`0.3478`
- balanced_accuracy：`0.3625`
- 样本数：`28052`
- ablation：`{"eeg_ecg": 0.34776469149030814, "eeg_only": 0.3276273109204963, "ecg_only": 0.34182320896064994}`
- 结果说明：`eeg_ecg` 是该组最优消融。

### heat_the_chair

- 最优模型：`linear_svc`
- macro_f1：`0.5405`
- balanced_accuracy：`0.5405`
- 样本数：`5440`
- ablation：`{"eeg_ecg": 0.5404767483230504, "eeg_only": 0.5215350709229751, "ecg_only": 0.513365691403931}`
- 结果说明：`eeg_ecg` 是该组最优消融。

## 主观/回归主结果

- 主特征集：`eeg_ecg`
- 消融列定义：`{"eeg_ecg": 214, "eeg_only": 202, "ecg_only": 12}`

### n_back

- 最优模型：`linear_svr`
- rmse：`10.2234`
- mae：`4.7987`
- spearman：`0.1209`
- 样本数：`28052`
- ablation：`{"eeg_ecg": 10.223413421909578, "eeg_only": 11.985013124136149, "ecg_only": 5.4682413286903575}`
- 结果说明：`ecg_only` 优于主特征集 `eeg_ecg`，说明该组对当前模态裁剪更敏感。

### heat_the_chair

- 最优模型：`linear_svr`
- rmse：`1.8639`
- mae：`1.2785`
- spearman：`0.1721`
- 样本数：`5440`
- ablation：`{"eeg_ecg": 1.8639105911346199, "eeg_only": 2.4076468046550064, "ecg_only": 1.4579560165642602}`
- 结果说明：`ecg_only` 优于主特征集 `eeg_ecg`，说明该组对当前模态裁剪更敏感。

## Flight 辅助摘要

- flight window 数：`1256`
- flight split_group 数：`2`
- 理论难度分布：`{"1": 331, "2": 434, "3": 121, "4": 114}`
- 感知难度均值：`1.4347`

## 图表

- `dataset_subset_counts`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/uab_window/plots/dataset_subset_counts.png`
- `label_distribution`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/uab_window/plots/label_distribution.png`
- `objective_ablation_primary_metric`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/uab_window/plots/objective_ablation_macro_f1.png`
- `objective_confusion_matrix_heat_the_chair`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/uab_window/plots/objective_confusion_matrix_heat_the_chair.png`
- `objective_confusion_matrix_n_back`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/uab_window/plots/objective_confusion_matrix_n_back.png`
- `subjective_ablation_primary_metric`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/uab_window/plots/subjective_ablation_rmse.png`
- `subjective_regression_heat_the_chair`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/uab_window/plots/subjective_regression_heat_the_chair.png`
- `subjective_regression_n_back`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/uab_window/plots/subjective_regression_n_back.png`

