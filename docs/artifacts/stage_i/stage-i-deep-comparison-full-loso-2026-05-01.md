# Stage I Deep Comparison Full LOSO

更新时间：2026-05-01

- 机器资产根目录：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T-full-loso-deep-comparison`
- sequence 资产继续复用：
  - `Stage H`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/stage_h_case_sequences`
  - `UAB`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/uab_sequences`
  - `NASA`: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/nasa_sequences`

## 1. 本轮口径

本轮完成的是增强实验第二批的 full LOSO，而不是新的数据 contract 扩展。

固定配置：

- `epochs = 1`
- `batch_size = 256`
- `hidden_dim = 32`
- `num_heads = 2`
- `layers = 1`
- `dropout = 0.1`

统一 comparison 覆盖：

1. `stage_h_case`
2. `uab_workload_dataset`
3. `nasa_csm`

模型固定为：

1. `MulT`
2. `ContiFormer`

## 2. Stage H Real Sortie

真实 sortie 层维持第一批结论，并已被纳入 full LOSO comparison summary：

- `3 views`
- `24 sequence samples`
- `2 PASS + 1 WARN`

同 sortie 双 pilot 差异依然可读：

- `MulT`
  - `delta stability = -0.019888`
  - `delta entropy = -0.024434`
  - `delta top concentration = +0.012759`
  - `delta event-mask interference = -0.092728`
- `ContiFormer`
  - `delta stability = +0.015861`
  - `delta entropy = -0.018965`
  - `delta top concentration = +0.006944`
  - `delta event-mask interference = -0.019390`

## 3. UAB Full LOSO

### 3.1 Objective

| model | group | macro-F1 | balanced accuracy | classical macro-F1 | classical balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `MulT` | `n_back` | 0.262130 | 0.304878 | 0.347765 | 0.362466 |
| `MulT` | `heat_the_chair` | 0.516098 | 0.521129 | 0.540477 | 0.540470 |
| `ContiFormer` | `n_back` | 0.162030 | 0.333333 | 0.347765 | 0.362466 |
| `ContiFormer` | `heat_the_chair` | 0.322203 | 0.500000 | 0.540477 | 0.540470 |

### 3.2 Subjective

| model | group | RMSE | MAE | classical RMSE | classical MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| `MulT` | `n_back` | 5.828201 | 4.670543 | 10.223413 | 4.798697 |
| `MulT` | `heat_the_chair` | 2.825106 | 2.353416 | 1.863911 | 1.278513 |
| `ContiFormer` | `n_back` | 4.654091 | 3.812269 | 10.223413 | 4.798697 |
| `ContiFormer` | `heat_the_chair` | 1.456759 | 1.163693 | 1.863911 | 1.278513 |

### 3.3 当前判断

1. `UAB objective` 上，当前轻量 deep baseline 没有超过现有 classical baseline。
2. `UAB subjective` 上，`ContiFormer` 在 `n_back` 和 `heat_the_chair` 两组都优于现有 classical baseline；`MulT` 只在 `n_back RMSE` 上更好。
3. 因此当前最稳妥的论文式表述是：
   - 深度基线对 workload classification 没有带来收益
   - 连续时间建模在 subjective regression 上存在局部价值

## 4. NASA Full LOSO

| model | group | macro-F1 | balanced accuracy | classical macro-F1 | classical balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `MulT` | `benchmark_only` | 0.302460 | 0.333333 | 0.464156 | 0.490542 |
| `MulT` | `loft_only` | 0.302226 | 0.333333 | 0.372319 | 0.380775 |
| `MulT` | `combined` | 0.302347 | 0.333333 | 0.374102 | 0.376498 |
| `ContiFormer` | `benchmark_only` | 0.302437 | 0.333057 | 0.464156 | 0.490542 |
| `ContiFormer` | `loft_only` | 0.300283 | 0.329193 | 0.372319 | 0.380775 |
| `ContiFormer` | `combined` | 0.302264 | 0.333047 | 0.374102 | 0.376498 |

当前判断：

1. `NASA attention_state` 上，两种 deep baseline 都没有超过现有 classical baseline。
2. `MulT` 与 `ContiFormer` 的差异非常小，说明当前受限于轻量参数与标签结构，模型容量并不是主决定因素。

## 5. 总结

1. 增强实验第二批的 full LOSO 已完成。
2. 当前 sequence contract、CLI、wrapper、summary/report 路径已经能稳定覆盖 `Stage H + UAB + NASA`。
3. 在当前已验证配置下：
   - `Stage H` 的真实 sortie deep comparison 主要承担系统级解释价值。
   - `UAB / NASA objective` 上，deep baseline 未优于 existing classical baseline。
   - `UAB subjective` 上，`ContiFormer` 显示出更强的连续时间回归潜力。
4. 如果后续不继续调参，当前最合理的收口口径是：
   - 将 `MulT / ContiFormer` 作为对照与负证据保留
   - quantitative mainline 继续以现有 classical / Chronaris 路径为主
