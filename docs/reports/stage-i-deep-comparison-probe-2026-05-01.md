# Stage I Deep Comparison Probe

更新时间：2026-05-01

- 机器资产根目录：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison`
- UAB sequence 资产：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/uab_sequences`
- NASA sequence 资产：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/nasa_sequences`
- unified comparison probe：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/comparison_probe_all`

## 1. 范围

本轮完成的是增强实验第二批的可执行 probe，而不是最终 full LOSO 收口：

1. `UAB window_v2` sequence 导出
2. `NASA attention_state` sequence 导出
3. `Stage H -> UAB -> NASA` 三段 unified comparison
4. 对比模型固定为 `MulT` 与 `ContiFormer`
5. 公开数据评估参数固定为：
   - `epochs=1`
   - `batch_size=256`
   - `hidden_dim=32`
   - `num_heads=2`
   - `layers=1`
   - `max-folds=1`

## 2. 资产事实

### 2.1 UAB

- `entry_count = 34748`
- `recording_count = 87`
- `split_group_count = 28`
- `sequence_length = 64`
- `subset_counts`
  - `n_back = 28052`
  - `heat_the_chair = 5440`
  - `flight_simulator = 1256`
- `ecg_zero_mask_samples`
  - `n_back = 9044`
  - `heat_the_chair = 654`
  - `flight_simulator = 380`
- sequence 导出耗时：`4m48s`

### 2.2 NASA

- `entry_count = 16609`
- `recording_count = 68`
- `split_group_count = 17`
- `sequence_length = 64`
- `subset_counts`
  - `benchmark = 3441`
  - `loft = 13168`
- `inventory_only_background_count = 13799`

## 3. Probe 结果

### 3.1 真实 sortie 层

真实 sortie 仍复用第一批 `stage_h_case` 资产，并进入新的统一 summary。当前 `3 views / 24 samples` 都已写入：

- `comparison_probe_all/stage_h_case/mult`
- `comparison_probe_all/stage_h_case/contiformer`

### 3.2 UAB

当前只完成 `1` 个 LOSO fold 的 probe：

| model | group | macro-F1 | balanced accuracy | RMSE | MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| `MulT` | `n_back` | 0.160081 | 0.204060 | 6.865815 | 5.779193 |
| `MulT` | `heat_the_chair` | 0.336066 | 0.500000 | 1.245245 | 0.946868 |
| `ContiFormer` | `n_back` | 0.170151 | 0.333333 | 5.892185 | 5.217401 |
| `ContiFormer` | `heat_the_chair` | 0.330579 | 0.500000 | 2.018593 | 1.874114 |

对照 classical baseline：

- `n_back` objective classical `macro-F1 = 0.347765`
- `heat_the_chair` objective classical `macro-F1 = 0.540477`

### 3.3 NASA

当前也只完成 `1` 个 LOSO fold 的 probe：

| model | group | macro-F1 | balanced accuracy |
| --- | --- | ---: | ---: |
| `MulT` | `benchmark_only` | 0.299578 | 0.333333 |
| `MulT` | `loft_only` | 0.308244 | 0.333333 |
| `MulT` | `combined` | 0.304264 | 0.333333 |
| `ContiFormer` | `benchmark_only` | 0.299578 | 0.333333 |
| `ContiFormer` | `loft_only` | 0.308244 | 0.333333 |
| `ContiFormer` | `combined` | 0.304264 | 0.333333 |

对照 classical baseline：

- `benchmark_only` classical `macro-F1 = 0.464156`
- `loft_only` classical `macro-F1 = 0.372319`
- `combined` classical `macro-F1 = 0.374102`

## 4. 当前判断

1. 深度基线 sequence contract、CLI、wrapper、summary/report 链路已经能覆盖 `Stage H + UAB + NASA` 三段。
2. `UAB` 的 sequence 导出和 deep baseline 路径都已经在真实本地数据上跑通。
3. 当前最重的瓶颈已经从“能不能跑通”变成“full LOSO 双模型的 wall time 是否接受”。
4. 因此当前事实应表述为：
   - `公开数据增强实验 probe 已完成`
   - `公开数据 full LOSO 收口仍待长跑`

## 5. 本轮未完成

1. 尚未完成 `UAB` 双模型 full LOSO。
2. 尚未完成 `NASA` 双模型 full LOSO。
3. 尚未把 probe comparison 升级为最终收口报告。
