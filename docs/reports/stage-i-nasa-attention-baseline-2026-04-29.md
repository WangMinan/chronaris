# Stage I NASA CSM Attention Baseline

- 生成时间：2026-04-29 08:33:49 UTC
- 机器产物根目录：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/nasa_attention`

## 数据摘要

- 样本总数：`16609`
- recording 数：`68`
- window 数：`16609`
- split_group 数：`17`
- 特征总数：`92`
- 模态特征数：`{"eeg": 80, "peripheral": 12}`
- subset 计数：`{"benchmark": 3441, "loft": 13168}`
- 标签分布：`{"attention_state": {"2": 2332, "0": 13799, "5": 294, "1": 184}}`

## 实验设置

- 窗口策略：`5s / 5s`，仅保留完整落在单一非零事件段中的窗口；背景 `event=0` 只进入盘点，不参与主分类。
- 任务设置：`benchmark_only / loft_only / combined` 三套 attention-state 分类主结果。
- 评测切分：`Leave-One-Subject-Out`，combined 中同一 subject 的 benchmark 和 LOFT 窗口共享同一 fold。

## 客观/分类主结果

- 主特征集：`all_sensors`
- 消融列定义：`{"all_sensors": 92, "eeg_only": 80, "peripheral_only": 12}`

### benchmark_only

- 最优模型：`linear_svc`
- macro_f1：`0.4642`
- balanced_accuracy：`0.4905`
- 样本数：`1451`
- ablation：`{"all_sensors": 0.4641561353985333, "eeg_only": 0.4460816461835107, "peripheral_only": 0.4172284863448296}`
- 结果说明：`all_sensors` 是该组最优消融。

### loft_only

- 最优模型：`linear_svc`
- macro_f1：`0.3723`
- balanced_accuracy：`0.3808`
- 样本数：`1359`
- ablation：`{"all_sensors": 0.3723186247618541, "eeg_only": 0.34126707789549715, "peripheral_only": 0.38141408844016006}`
- 结果说明：`peripheral_only` 优于主特征集 `all_sensors`，说明该组对当前模态裁剪更敏感。

### combined

- 最优模型：`linear_svc`
- macro_f1：`0.3741`
- balanced_accuracy：`0.3765`
- 样本数：`2810`
- ablation：`{"all_sensors": 0.3741024976765464, "eeg_only": 0.3306928521069402, "peripheral_only": 0.3879762965458353}`
- 结果说明：`peripheral_only` 优于主特征集 `all_sensors`，说明该组对当前模态裁剪更敏感。

## 图表

- `dataset_subset_counts`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/nasa_attention/plots/dataset_subset_counts.png`
- `label_distribution`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/nasa_attention/plots/label_distribution.png`
- `objective_ablation_primary_metric`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/nasa_attention/plots/objective_ablation_macro_f1.png`
- `objective_confusion_matrix_benchmark_only`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/nasa_attention/plots/objective_confusion_matrix_benchmark_only.png`
- `objective_confusion_matrix_combined`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/nasa_attention/plots/objective_confusion_matrix_combined.png`
- `objective_confusion_matrix_loft_only`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/nasa_attention/plots/objective_confusion_matrix_loft_only.png`

