# Stage I Phase 3 Closure

- 生成时间：`2026-04-30T03:51:02.713471Z`
- 机器产物根目录：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure`

## 收口说明

- 本轮收口把 Stage I 主线从 session-level UAB 切换到 window-level UAB，并补齐 NASA CSM attention-state 第二公开数据集。
- `Phase 0 + Phase 1` 的 session-level UAB 资产继续保留为历史参照；当前主线事实以本次 window-level UAB + NASA CSM closure 为准。
- 两条任务保持独立训练与独立评测，只统一数据 contract、pipeline 接口、artifact contract 和闭环文档。
- window-scale baseline 只保留 `LogisticRegression / LinearSVC / Ridge / LinearSVR` 线性 classical baselines；随机森林与核 SVR 不再进入主收口。
- UAB 与 NASA 都使用 `5s / 5s` 固定窗口和 `Leave-One-Subject-Out` 切分；NASA 背景窗口不进入主分类。

## UAB Window 主线

- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/uab_window`
- entry_count: `34748`
- recording_count: `87`
- window_count: `34748`
- subset_counts: `{"n_back": 28052, "heat_the_chair": 5440, "flight_simulator": 1256}`
- label_distribution: `{"workload_level": {"0": 11593, "1": 12317, "2": 9582}, "theoretical_difficulty": {"2.0": 434, "4.0": 114, "1.0": 331, "3.0": 121}}`

### 主结果

- n_back: macro-F1 `0.3478`, balanced accuracy `0.3625`
- heat_the_chair: macro-F1 `0.5405`, balanced accuracy `0.5405`
- n_back: RMSE `10.2234`, MAE `4.7987`
- heat_the_chair: RMSE `1.8639`, MAE `1.2785`

### 客观消融摘要

- n_back: `{"eeg_ecg": 0.34776469149030814, "eeg_only": 0.3276273109204963, "ecg_only": 0.34182320896064994}`
- heat_the_chair: `{"eeg_ecg": 0.5404767483230504, "eeg_only": 0.5215350709229751, "ecg_only": 0.513365691403931}`

### 主观消融摘要

- n_back: `{"eeg_ecg": 10.223413421909578, "eeg_only": 11.985013124136149, "ecg_only": 5.4682413286903575}`
- heat_the_chair: `{"eeg_ecg": 1.8639105911346199, "eeg_only": 2.4076468046550064, "ecg_only": 1.4579560165642602}`

## NASA Attention 主线

- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/nasa_attention`
- entry_count: `16609`
- recording_count: `68`
- window_count: `16609`
- subset_counts: `{"benchmark": 3441, "loft": 13168}`
- label_distribution: `{"attention_state": {"2": 2332, "0": 13799, "5": 294, "1": 184}}`

### 主结果

- benchmark_only: macro-F1 `0.4642`, balanced accuracy `0.4905`
- loft_only: macro-F1 `0.3723`, balanced accuracy `0.3808`
- combined: macro-F1 `0.3741`, balanced accuracy `0.3765`

### 客观消融摘要

- benchmark_only: `{"all_sensors": 0.4641561353985333, "eeg_only": 0.4460816461835107, "peripheral_only": 0.4172284863448296}`
- loft_only: `{"all_sensors": 0.3723186247618541, "eeg_only": 0.34126707789549715, "peripheral_only": 0.38141408844016006}`
- combined: `{"all_sensors": 0.3741024976765464, "eeg_only": 0.3306928521069402, "peripheral_only": 0.3879762965458353}`

## UAB Session vs Window

### objective

- n_back: session `0.4966` / window `0.3478` / delta `-0.1488`
- heat_the_chair: session `0.6173` / window `0.5405` / delta `-0.0768`

### subjective

- n_back: session `4.5161` / window `10.2234` / delta `5.7073`
- heat_the_chair: session `1.4664` / window `1.8639` / delta `0.3975`

## Closure 图表

- `phase3_window_counts`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/plots/phase3_window_counts.png`
- `uab_session_vs_window_objective`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/plots/uab_session_vs_window_objective.png`
- `uab_session_vs_window_subjective`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/plots/uab_session_vs_window_subjective.png`

